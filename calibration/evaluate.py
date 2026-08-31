#!/usr/bin/env python3
"""Primary evaluation on the held-out test split (protocol §2.2, §3, §8.4).

This is the **only** script allowed to read the test split for publication
(protocol §2.2): it must be run exactly once per release, and every run is
logged (timestamp + git sha) in ``calibration/evaluation_runs_<lang>.jsonl``.

Scores are **reused** from ``calibration/scores_<lang>_v01.json`` (written by
``calibrate.py``) so the test split is never re-scored — only the OOD Mistral
corpus is scored here. Thresholds are read from the published profile, i.e.
they were fitted on ``train`` only.

Outputs:

- ``calibration/evaluation_<lang>_v01.json`` — traceability header (git sha,
  corpus SHA-256, seeds, versions) + primary metrics with 95 % bootstrap CIs,
  stratified confusion tables, OOD metrics, 5-fold complement (protocol §2.3),
- ``calibration/evaluation_runs_<lang>.jsonl`` — one append-only line per run.

Usage::

    python -m calibration.evaluate \
        --corpus calibration/corpus/binoculars-eu-corpus-fr-v01.jsonl \
        --splits calibration/splits_fr_v01.json \
        --scores calibration/scores_fr_v01.json \
        --ood-corpus calibration/corpus/binoculars-eu-corpus-fr-v01-ood.jsonl \
        --profile fr
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, auc, f1_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from binoculars_eu import Binoculars
from binoculars_eu.profiles import get_profile
from calibration.build_splits import merged_strata
from calibration.calibrate import (
    SEED_TORCH,
    environment_report,
    git_state,
    library_versions,
    load_corpus,
    load_splits,
    write_json,
)

SEED_BOOTSTRAP = 100   # protocol §1: numpy.random.default_rng for bootstrap
SEED_5FOLD = 42        # protocol §1: StratifiedKFold
N_BOOT = 1000          # protocol §3.3
ECE_BINS = 10
LENGTH_BINS = (150, 300, 500)  # same edges as build_splits.stratum_key


# --------------------------------------------------------------------------
# Metrics (protocol §3.3, §3.4 — verbatim)
# --------------------------------------------------------------------------
def bootstrap_metric(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_boot: int = N_BOOT,
    seed: int = SEED_BOOTSTRAP,
) -> dict:
    """Bootstrap CI for a metric, following protocol §3.3 exactly."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            values.append(metric_fn(y_true[idx], y_score[idx]))
        except ValueError:
            continue  # resample with a single class — skip, per protocol
    values_arr = np.array(values)
    return {
        "point": float(metric_fn(y_true, y_score)),
        "ci_low": float(np.percentile(values_arr, 2.5)),
        "ci_high": float(np.percentile(values_arr, 97.5)),
        "n_boot_valid": int(len(values_arr)),
    }


def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    """TPR at the target FPR, linear interpolation (protocol §3.4)."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.interp(target_fpr, fpr, tpr))


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray,
                               n_bins: int = ECE_BINS) -> float:
    """Expected Calibration Error over equal-width confidence bins."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > low) & (confidence <= high) if high < 1.0 else (confidence >= low)
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(
            float(correct[mask].mean()) - float(confidence[mask].mean())
        )
    return ece


def minmax_confidence(neg_train: np.ndarray, neg_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pseudo-probability and confidence from a min-max fit on **train** only.

    TODO(spec): the Binoculars score is not probabilistic; the protocol sets
    an ECE target (§3.1) without defining a calibration map. Closest intent:
    min-max fitted on train only (§2.2 discipline), confidence = max(p, 1-p).
    """
    lo, hi = float(neg_train.min()), float(neg_train.max())
    span = max(hi - lo, 1e-12)
    p = np.clip((neg_eval - lo) / span, 0.0, 1.0)  # P(AI), non-monotone-safe
    return p, np.maximum(p, 1.0 - p)


def point_metrics(y_true: np.ndarray, neg: np.ndarray, thr_accuracy: float,
                  thr_low_fpr: float) -> dict[str, float]:
    """Point estimates of the primary metrics on one score array.

    ``neg`` is the negated Binoculars score (higher = more AI-like, as in
    calibrate.py); profile thresholds live on the raw score scale, so the
    decision boundary on ``neg`` is ``-threshold``.
    """
    fpr, tpr, _ = roc_curve(y_true, neg)
    return {
        "auc": float(auc(fpr, tpr)),
        "tpr_at_fpr_1": tpr_at_fpr(y_true, neg, 0.01),
        "tpr_at_fpr_5": tpr_at_fpr(y_true, neg, 0.05),
        "f1_at_accuracy": float(f1_score(y_true, (neg >= -thr_accuracy).astype(int))),
        "accuracy_at_accuracy": float(accuracy_score(y_true, (neg >= -thr_accuracy).astype(int))),
        "f1_at_low_fpr": float(f1_score(y_true, (neg >= -thr_low_fpr).astype(int))),
    }


# --------------------------------------------------------------------------
# Arrays and stratified confusion (protocol §3.2)
# --------------------------------------------------------------------------
def split_arrays(records: list[dict], scores: dict[str, float],
                 ids: set[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true with 1=AI, negated scores) for one split."""
    subset = [r for r in records if r["id"] in ids]
    y = np.array([1 if r["label"] == "ai" else 0 for r in subset], dtype=int)
    neg = np.array([-scores[r["id"]] for r in subset], dtype=float)
    return y, neg


def length_bin(length_tokens: int) -> str:
    """Same binning as build_splits.stratum_key."""
    for i, edge in enumerate(LENGTH_BINS):
        if length_tokens < edge:
            return f"L{i + 1}"
    return "L4"


def stratified_confusion(records: list[dict], scores: dict[str, float],
                         ids: set[str], threshold: float) -> dict[str, dict]:
    """Confusion counts per human source, AI generator and length bin."""
    subset = [r for r in records if r["id"] in ids]

    def counts(key_fn) -> dict[str, dict[str, int]]:
        table: dict[str, dict[str, int]] = {}
        for r in subset:
            key = key_fn(r)
            cell = table.setdefault(key, {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
            predicted_ai = scores[r["id"]] <= threshold  # low score = AI
            if r["label"] == "ai":
                cell["tp" if predicted_ai else "fn"] += 1
            else:
                cell["fp" if predicted_ai else "tn"] += 1
        return table

    return {
        "by_source": counts(lambda r: r["source"]),
        "by_generator": counts(lambda r: r["meta"].get("generator", "human")),
        "by_length_bin": counts(lambda r: length_bin(r["meta"]["length_tokens"])),
    }


# --------------------------------------------------------------------------
# OOD Mistral scoring (only corpus scored by this script)
# --------------------------------------------------------------------------
def evaluate_ood(detector: Binoculars, ood_path: Path | None,
                 human_test: list[dict], batch_size: int) -> dict:
    """AUC of test humans vs OOD Mistral text, with latency measured here.

    TODO(spec): protocol §3.1 sets an OOD AUC target without specifying the
    human comparator; closest intent: the 100 test-split humans (the same
    population the headline metrics are published on).
    """
    if ood_path is None:
        return {"note": "no --ood-corpus given", "auc_ood": None}
    ood_records, ood_sha = load_corpus(ood_path)
    texts = [r["text"] for r in ood_records] + [r["text"] for r in human_test]
    torch.manual_seed(SEED_TORCH)
    started = time.perf_counter()
    values = detector.compute_score(texts)
    if isinstance(values, float):
        values = [values]
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    latencies = [elapsed_ms / len(texts)] * len(texts)  # batch-level granularity
    neg = -np.array(values, dtype=float)
    y = np.array([1] * len(ood_records) + [0] * len(human_test), dtype=int)
    metrics = bootstrap_metric(y, neg, lambda a, b: float(auc(*roc_curve(a, b)[:2])))
    vram_mb = None
    if torch.cuda.is_available():
        vram_mb = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    return {
        "corpus_sha256": ood_sha,
        "n_ood": len(ood_records),
        "n_human": len(human_test),
        "auc_ood": metrics,
        # Latency/VRAM are only measurable on the corpus scored here (the
        # hold-out scores are reused from calibrate.py, which does not log
        # per-text timing — TODO(spec) protocol §3.2 wants P50/P99 per length
        # bin on the corpus; recorded for OOD now, hold-out left null.
        "latency_ms_p50": float(np.percentile(latencies, 50)),
        "latency_ms_p99": float(np.percentile(latencies, 99)),
        "vram_peak_mb": vram_mb,
    }


# --------------------------------------------------------------------------
# 5-fold complement (protocol §2.3)
# --------------------------------------------------------------------------
def five_fold_metrics(records: list[dict], scores: dict[str, float],
                      thr_accuracy: float, thr_low_fpr: float) -> dict:
    """Mean ± std of primary metrics over a stratified 5-fold of the corpus.

    Thresholds stay frozen (fitted on train by calibrate.py) — the fold loop
    measures split fragility, never refits anything.
    """
    y = np.array([1 if r["label"] == "ai" else 0 for r in records], dtype=int)
    neg = np.array([-scores[r["id"]] for r in records], dtype=float)
    strata = merged_strata(records)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED_5FOLD)
    rows = [point_metrics(y[idx], neg[idx], thr_accuracy, thr_low_fpr)
            for idx, _ in kfold.split(neg, strata)]
    keys = rows[0].keys()
    return {
        name: {"mean": float(np.mean([r[name] for r in rows])),
               "std": float(np.std([r[name] for r in rows]))}
        for name in keys
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True, help="calibration JSONL")
    parser.add_argument("--splits", type=Path, required=True, help="split manifest JSON")
    parser.add_argument("--scores", type=Path, required=True,
                        help="scores_<lang>_v01.json written by calibrate.py")
    parser.add_argument("--ood-corpus", type=Path, default=None, help="OOD JSONL (optional)")
    parser.add_argument("--profile", default="fr", help="registered profile code")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--config", default="v01", help="protocol config tag")
    parser.add_argument("--git-sha", default=None,
                        help="override git sha tracing (e.g. on runners without "
                             "a git checkout of the exact evaluated revision)")
    parser.add_argument("--output-dir", type=Path, default=Path("calibration"))
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    print("=== evaluate environment ===")
    print(environment_report(args.profile))
    print("============================\n")

    records, corpus_sha256 = load_corpus(args.corpus)
    splits = load_splits(args.splits, {r["id"] for r in records})
    scores = json.loads(args.scores.read_text(encoding="utf-8"))
    missing = {r["id"] for r in records} - set(scores)
    if missing:
        raise ValueError(
            f"scores file misses {len(missing)} corpus ids, e.g. {sorted(missing)[:3]}"
        )

    profile = get_profile(args.profile)
    thr_acc, thr_lf = profile.threshold_accuracy, profile.threshold_low_fpr
    print(f"profile {args.profile!r}: accuracy={thr_acc}, low_fpr={thr_lf} "
          f"(calibrated {profile.calibration_date}, corpus {profile.corpus_sha256[:12]}…)")
    if profile.corpus_sha256 != corpus_sha256:
        raise ValueError("profile corpus_sha256 does not match the corpus on disk")
    print("splits: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    # --- test split: primary metrics, single pass (protocol §2.2) ------------
    y_test, neg_test = split_arrays(records, scores, splits["test"])
    print(f"\nscoring reused for all splits; test n={len(y_test)} (NOT re-scored)")
    y_train, neg_train = split_arrays(records, scores, splits["train"])

    auc_fn = lambda a, b: float(auc(*roc_curve(a, b)[:2]))  # noqa: E731

    def f1_at_thr(a: np.ndarray, b: np.ndarray) -> float:
        return float(f1_score(a, (b >= -thr_acc).astype(int)))

    def acc_at_thr(a: np.ndarray, b: np.ndarray) -> float:
        return float(accuracy_score(a, (b >= -thr_acc).astype(int)))

    metrics: dict = {
        "auc": bootstrap_metric(y_test, neg_test, auc_fn),
        "tpr_at_fpr_1": bootstrap_metric(y_test, neg_test, lambda a, b: tpr_at_fpr(a, b, 0.01)),
        "tpr_at_fpr_5": bootstrap_metric(y_test, neg_test, lambda a, b: tpr_at_fpr(a, b, 0.05)),
        "f1_at_accuracy": bootstrap_metric(y_test, neg_test, f1_at_thr),
        "accuracy_at_accuracy": bootstrap_metric(y_test, neg_test, acc_at_thr),
    }
    for name, value in metrics.items():
        print(f"  {name:24s} = {value['point']:.4f} "
              f"[{value['ci_low']:.4f}, {value['ci_high']:.4f}]")

    # --- ECE (test), calibration map fitted on train only ---------------------
    p_test, conf_test = minmax_confidence(neg_train, neg_test)
    correct_test = ((p_test >= 0.5).astype(int) == y_test)
    metrics["ece"] = {
        "point": expected_calibration_error(conf_test, correct_test),
        "note": "min-max map fitted on train (protocol §2.2); see TODO in minmax_confidence",
    }
    print(f"  {'ece':24s} = {metrics['ece']['point']:.4f} (min-max on train)")

    # --- OOD Mistral (scored here; the only network/model work of this run) ---
    human_test = [r for r in records if r["id"] in splits["test"] and r["label"] == "human"]
    ood: dict = {"note": "no --ood-corpus given", "auc_ood": None}
    if args.ood_corpus is not None:
        detector = Binoculars.for_language(args.profile, mode="accuracy")
        ood = evaluate_ood(detector, args.ood_corpus, human_test, args.batch_size)
    metrics["ood_mistral"] = ood
    if ood.get("auc_ood"):
        print(f"  auc_ood (test humans vs Mistral) = {ood['auc_ood']['point']:.4f} "
              f"[{ood['auc_ood']['ci_low']:.4f}, {ood['auc_ood']['ci_high']:.4f}] "
              f"(n={ood['n_ood']} ood + {ood['n_human']} human)")

    # --- 5-fold complement over the full corpus (protocol §2.3) ---------------
    folds = five_fold_metrics(records, scores, thr_acc, thr_lf)
    metrics["five_fold"] = folds
    print(f"\n5-fold (thresholds frozen): auc mean={folds['auc']['mean']:.4f} "
          f"± {folds['auc']['std']:.4f} | tpr@1% mean={folds['tpr_at_fpr_1']['mean']:.4f} "
          f"± {folds['tpr_at_fpr_1']['std']:.4f}")

    # --- artefacts ------------------------------------------------------------
    git_sha, git_dirty = git_state()
    if args.git_sha is not None:
        git_sha, git_dirty = args.git_sha, None
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "corpus_sha256": corpus_sha256,
        "seeds": {"split": SEED_5FOLD, "bootstrap": SEED_BOOTSTRAP, "torch": SEED_TORCH},
        "config": args.config,
        "versions": library_versions(),
        "n_test": int(len(y_test)),
        "thresholds": {"accuracy": thr_acc, "low_fpr": thr_lf,
                       "tpr_at_fpr_1": profile.threshold_tpr_at_fpr_1},
        "metrics": metrics,
        "confusion_test_at_accuracy": stratified_confusion(
            records, scores, splits["test"], thr_acc
        ),
    }
    lang = args.profile
    write_json(args.output_dir / f"evaluation_{lang}_v01.json", payload)
    run_line = {"timestamp": payload["timestamp"], "git_sha": git_sha,
                "git_dirty": git_dirty, "config": args.config,
                "headline_tpr_at_fpr_1": metrics["tpr_at_fpr_1"]["point"],
                "auc": metrics["auc"]["point"]}
    with (args.output_dir / f"evaluation_runs_{lang}.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(run_line, ensure_ascii=False) + "\n")
    print(f"\nartefacts: evaluation_{lang}_v01.json + append evaluation_runs_{lang}.jsonl")
    print("test metrics published once — per protocol §2.2 do not re-run for tuning")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
