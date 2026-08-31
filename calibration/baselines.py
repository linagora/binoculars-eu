#!/usr/bin/env python3
"""Baseline comparison on the held-out test split (protocol §4).

Evaluates the five mandatory baselines on the same hold-out as evaluate.py,
with the same 95 % bootstrap CIs, and writes
``calibration/baselines_<lang>_v01.json``:

- **B0 Random** — ``numpy.random.rand`` (uniform pseudo-scores),
- **B1 Length** — logistic regression on token length, fitted on train,
- **B2 Shallow features** — LR on length, comma rate, lexical TTR,
  punctuation ratio, mean word length, fitted on train,
- **B3 Binoculars-Falcon-EN** — the original English detector
  (``Binoculars.from_legacy``) applied to the French texts, upstream
  thresholds unchanged,
- **B4 Profile under evaluation (ours)** — reused from
  ``scores_<lang>_v01.json``.

Convention (protocol §4.3): every baseline reports the primary metric set
with CIs; each baseline uses its natural decision threshold — 0.5 for B0,
train-fitted F1-optimal for B1/B2, the published upstream threshold for B3,
the calibrated profile thresholds for B4. B3 requires downloading the Falcon
weights (~15 GB) — pass ``--skip-falcon`` for a partial run.

Usage::

    python -m calibration.baselines \
        --corpus calibration/corpus/binoculars-eu-corpus-fr-v01.jsonl \
        --splits calibration/splits_fr_v01.json \
        --scores calibration/scores_fr_v01.json \
        --profile fr
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, f1_score, precision_recall_curve, roc_curve

from binoculars_eu import Binoculars
from calibration.calibrate import SEED_TORCH, git_state, library_versions
from calibration.evaluate import (
    SEED_BOOTSTRAP,
    bootstrap_metric,
    load_corpus,
    load_splits,
    tpr_at_fpr,
    write_json,
)

# TODO(spec): protocol §1 freezes no seed for B0 random scores; using the
# bootstrap seed (100) keeps the run reproducible while remaining un-anchored
# on the test labels.
SEED_B0 = 100
FALCON_OBSERVER = "tiiuae/falcon-7b"
FALCON_PERFORMER = "tiiuae/falcon-7b-instruct"


# --------------------------------------------------------------------------
# Baseline scorers: each returns scores oriented "higher = more AI-like"
# --------------------------------------------------------------------------
def b0_random(n: int) -> np.ndarray:
    """Uniform random pseudo-scores (B0)."""
    return np.random.default_rng(SEED_B0).random(n)


def shallow_features(text: str) -> list[float]:
    """Length, comma rate, lexical TTR, punctuation ratio, mean word length."""
    words = re.findall(r"\w+", text, flags=re.UNICODE)
    n_words = max(len(words), 1)
    chars = max(len(text), 1)
    puncts = len(re.findall(r"[.,;:!?«»\"'()\[\]—–-]", text))
    return [
        float(n_words),
        text.count(",") / n_words,
        len({w.lower() for w in words}) / n_words,
        puncts / chars,
        sum(len(w) for w in words) / n_words,
    ]


def b_length_train_test(
    records_train: list[dict], records_test: list[dict]
) -> np.ndarray:
    """B1: logistic regression on token length, fitted on train."""
    x_train = np.array([[r["meta"]["length_tokens"]] for r in records_train])
    x_test = np.array([[r["meta"]["length_tokens"]] for r in records_test])
    model = LogisticRegression(random_state=SEED_TORCH, max_iter=1000)
    model.fit(x_train, [r["label"] for r in records_train])
    return model.predict_proba(x_test)[:, 1]


def b_features_train_test(
    records_train: list[dict], records_test: list[dict]
) -> np.ndarray:
    """B2: logistic regression on the shallow feature set, fitted on train."""
    x_train = np.array([shallow_features(r["text"]) for r in records_train])
    x_test = np.array([shallow_features(r["text"]) for r in records_test])
    model = LogisticRegression(random_state=SEED_TORCH, max_iter=1000)
    model.fit(x_train, [r["label"] for r in records_train])
    return model.predict_proba(x_test)[:, 1]


def b3_falcon_scores(records_test: list[dict], batch_size: int,
                     load_in_8bit: bool) -> np.ndarray:
    """B3: original English Binoculars applied to the profile's texts.

    TODO(spec): protocol §4.1 does not fix a precision for B3; on 24 GB cards
    two Falcon-7B copies only fit in 8-bit (PRD §16.1) — the precision used
    is recorded in the output artefact.
    """
    detector = Binoculars.from_legacy(
        FALCON_OBSERVER, FALCON_PERFORMER, mode="accuracy",
        load_in_8bit=load_in_8bit,
    )
    scores: list[float] = []
    for start in range(0, len(records_test), batch_size):
        batch = [r["text"] for r in records_test[start : start + batch_size]]
        values = detector.compute_score(batch)
        scores.extend([values] if isinstance(values, float) else values)
    return -np.array(scores, dtype=float)  # higher = more AI-like


def b4_profile_scores(
    records_test: list[dict], scores: dict[str, float]
) -> np.ndarray:
    """B4: the evaluated profile, reused from calibrate.py artefacts."""
    return -np.array([scores[r["id"]] for r in records_test], dtype=float)


# --------------------------------------------------------------------------
# Metrics per baseline
# --------------------------------------------------------------------------
def train_f1_threshold(y_train: np.ndarray, s_train: np.ndarray) -> float:
    """F1-optimal threshold fitted on train (B1/B2 natural threshold)."""
    precision, recall, thresholds = precision_recall_curve(y_train, s_train)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best = int(np.argmax(f1[:-1]))
    return float(thresholds[best])


def baseline_metrics(
    y_test: np.ndarray, s_test: np.ndarray, threshold: float
) -> dict:
    """Primary metric set with bootstrap CIs for one baseline's scores."""
    auc_fn = lambda a, b: float(auc(*roc_curve(a, b)[:2]))  # noqa: E731

    def f1_at_thr(a: np.ndarray, b: np.ndarray) -> float:
        return float(f1_score(a, (b >= threshold).astype(int)))

    def acc_at_thr(a: np.ndarray, b: np.ndarray) -> float:
        return float(accuracy_score(a, (b >= threshold).astype(int)))

    return {
        "threshold": threshold,
        "auc": bootstrap_metric(y_test, s_test, auc_fn),
        "tpr_at_fpr_1": bootstrap_metric(y_test, s_test, lambda a, b: tpr_at_fpr(a, b, 0.01)),
        "tpr_at_fpr_5": bootstrap_metric(y_test, s_test, lambda a, b: tpr_at_fpr(a, b, 0.05)),
        "f1_at_threshold": bootstrap_metric(y_test, s_test, f1_at_thr),
        "accuracy_at_threshold": bootstrap_metric(y_test, s_test, acc_at_thr),
    }


def print_row(name: str, m: dict) -> None:
    """One console row of the Baseline × Metric table."""
    auc_m, tpr1 = m["auc"], m["tpr_at_fpr_1"]
    tpr5, f1_m = m["tpr_at_fpr_5"], m["f1_at_threshold"]
    acc_m = m["accuracy_at_threshold"]
    print(
        f"  {name:10s} AUC={auc_m['point']:.4f} [{auc_m['ci_low']:.4f},{auc_m['ci_high']:.4f}]"
        f"  TPR@1%={tpr1['point']:.4f}  TPR@5%={tpr5['point']:.4f}"
        f"  F1={f1_m['point']:.4f}  Acc={acc_m['point']:.4f}"
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True, help="calibration JSONL")
    parser.add_argument("--splits", type=Path, required=True, help="split manifest JSON")
    parser.add_argument("--scores", type=Path, required=True,
                        help="scores_<lang>_v01.json written by calibrate.py")
    parser.add_argument("--profile", default="fr", help="registered profile code")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--skip-falcon", action="store_true",
                        help="skip B3 (avoids the ~15 GB Falcon download)")
    parser.add_argument("--falcon-precision", choices=("8bit", "bfloat16"),
                        default="8bit",
                        help="B3 model precision (8bit fits 24 GB cards, PRD §16.1)")
    parser.add_argument("--output-dir", type=Path, default=Path("calibration"))
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    records, corpus_sha256 = load_corpus(args.corpus)
    splits = load_splits(args.splits, {r["id"] for r in records})
    scores = json.loads(args.scores.read_text(encoding="utf-8"))
    missing = {r["id"] for r in records} - set(scores)
    if missing:
        raise ValueError(f"scores file misses {len(missing)} corpus ids")

    train = [r for r in records if r["id"] in splits["train"]]
    test = [r for r in records if r["id"] in splits["test"]]
    y_train = np.array([1 if r["label"] == "ai" else 0 for r in train], dtype=int)
    y_test = np.array([1 if r["label"] == "ai" else 0 for r in test], dtype=int)
    print(f"baselines on test n={len(test)} (train-fitted baselines use n={len(train)})")

    results: dict[str, dict] = {}

    # B0 — Random (no threshold to fit; 0.5 is the natural decision point)
    results["B0_random"] = baseline_metrics(y_test, b0_random(len(test)), 0.5)
    # B1 — Length LR / B2 — Shallow-feature LR (threshold fitted on train)
    s_b1_test = b_length_train_test(train, test)
    s_b1_train = b_length_train_test(train, train)
    results["B1_length"] = baseline_metrics(
        y_test, s_b1_test, train_f1_threshold(y_train, s_b1_train)
    )
    s_b2_test = b_features_train_test(train, test)
    s_b2_train = b_features_train_test(train, train)
    results["B2_features"] = baseline_metrics(
        y_test, s_b2_test, train_f1_threshold(y_train, s_b2_train)
    )
    # B3 — Binoculars-Falcon-EN (upstream thresholds unchanged)
    if args.skip_falcon:
        results["B3_falcon_en"] = {"note": "skipped (--skip-falcon)"}
    else:
        from binoculars_eu.detector import _legacy_profile
        legacy = _legacy_profile(FALCON_OBSERVER, FALCON_PERFORMER)
        results["B3_falcon_en"] = baseline_metrics(
            y_test,
            b3_falcon_scores(test, args.batch_size, args.falcon_precision == "8bit"),
            -legacy.threshold_accuracy,
        )
        results["B3_falcon_en"]["precision"] = args.falcon_precision
    # B4 — evaluated profile (threshold from the published profile)
    from binoculars_eu.profiles import get_profile
    profile = get_profile(args.profile)
    results["B4_profile"] = baseline_metrics(
        y_test, b4_profile_scores(test, scores), -profile.threshold_accuracy
    )

    print("\nBaseline × Metric (point [CI95]):")
    for name, m in results.items():
        if "auc" in m:
            print_row(name, m)
        else:
            print(f"  {name:10s} skipped")

    git_sha, git_dirty = git_state()
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "corpus_sha256": corpus_sha256,
        "seeds": {"split": 42, "bootstrap": SEED_BOOTSTRAP, "b0": SEED_B0, "torch": SEED_TORCH},
        "config": "v01",
        "versions": library_versions(),
        "n_test": int(len(y_test)),
        "baselines": results,
    }
    lang = args.profile
    out = args.output_dir / f"baselines_{lang}_v01.json"
    write_json(out, payload)
    print(f"\nartefact: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
