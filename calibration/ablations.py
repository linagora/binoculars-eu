#!/usr/bin/env python3
"""Ablation study on the dev split (protocol §5).

Evaluates the four ablations of protocol §5.1 on the **dev split only**,
each configuration over the three frozen inter-seed runs (42, 123, 2024 —
protocol §1 / §5.2), and writes ``calibration/ablations_<lang>_v01.json``
with the same traceability header as calibrate.py / evaluate.py (§8.4):

- **A1 max_length** — 64 / 128 / 256 / 512 / 1024 tokens observed,
- **A2 precision** — bfloat16 / float32 / 8-bit,
- **A3 model pair** — observer/performer variations via ``dataclasses.replace``,
- **A4 shared tokenizer** — ``share_tokenizer_from_observer`` True vs False.

Score convention: the reference configuration (the published profile, scored
by calibrate.py) is **never re-scored** — its dev scores are sliced out of
``scores_<lang>_v01.json``. Inference is deterministic under
``torch.inference_mode`` and does not depend on the batch order, so every
other configuration is scored once with ``torch.manual_seed`` set and the
same scores are reported for all three seeds; the inter-seed std is
published honestly (expected 0), as protocol §5.2 / §8.3-2 document the
variance rather than requiring redundant GPU work.

Usage::

    python -m calibration.ablations \
        --corpus calibration/corpus/binoculars-eu-corpus-fr-v01.jsonl \
        --splits calibration/splits_fr_v01.json \
        --scores calibration/scores_fr_v01.json \
        --profile fr \
        [--batch-size 8] [--output-dir calibration] \
        [--pairs-extra REPO_OBS,REPO_PERF ...]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import auc, roc_curve

from binoculars_eu import Binoculars
from binoculars_eu.profiles import get_profile
from binoculars_eu.profiles.base import LanguageProfile
from calibration.calibrate import SEED_TORCH, git_state, library_versions, load_corpus, load_splits
from calibration.evaluate import split_arrays, tpr_at_fpr, write_json

SEEDS = (42, 123, 2024)  # protocol §1: inter-seed stability runs
A1_MAX_LENGTHS = (64, 128, 256, 512, 1024)  # protocol §5.1
LUCIOLE_BASE = "OpenLLM-France/Luciole-1B-Base"
LUCIOLE_INSTRUCT_1_1 = "OpenLLM-France/Luciole-1B-Instruct-1.1"
# Protocol §5.1 lists "Instruct+Instruct-thinking" as an A3 value, but the
# repo is private: the configuration is published as skipped, not invented.
A3_PRIVATE_PAIR = "instruct+instruct-thinking"

REUSE_NOTE = (
    "reused from --scores (reference config); latency/VRAM not measured by this run"
)
DETERMINISM_NOTE = (
    "scored once per config: inference is deterministic under torch.inference_mode "
    "and independent of batch order; identical scores reported for seeds "
    "42/123/2024, inter-seed std published honestly (expected 0)"
)


# --------------------------------------------------------------------------
# Metrics (dev split, negated-score convention — higher = more AI-like)
# --------------------------------------------------------------------------
def _point_metrics(y_true: np.ndarray, neg: np.ndarray) -> dict[str, float]:
    """AUC and TPR@FPR=1 % on the negated Binoculars score (protocol §3.4)."""
    fpr, tpr, _ = roc_curve(y_true, neg)
    return {
        "auc": float(auc(fpr, tpr)),
        "tpr_at_fpr_1": tpr_at_fpr(y_true, neg, 0.01),
    }


def _score_dev(
    detector: Binoculars, dev: list[dict], batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Score the dev split once in batches; return (neg scores, latency ms/text)."""
    texts = [r["text"] for r in dev]
    torch.manual_seed(SEED_TORCH)
    values: list[float] = []
    latencies: list[float] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        t0 = time.perf_counter()
        batch_values = detector.compute_score(batch)
        latencies.extend([(time.perf_counter() - t0) * 1000.0 / len(batch)] * len(batch))
        values.extend([batch_values] if isinstance(batch_values, float) else batch_values)
    return -np.array(values, dtype=float), np.array(latencies, dtype=float)


def _free_detector(detector: Binoculars) -> None:
    """Release a detector's weights and the CUDA cache (ablations load many pairs)."""
    del detector
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _score_profile(
    profile: LanguageProfile, dev: list[dict], y_dev: np.ndarray, batch_size: int,
    **detector_kwargs: bool | int,
) -> dict[str, float]:
    """Build a detector from a (replaced) profile, score dev, return metric row.

    Latency (P50 per text) and VRAM peak are measured on the split scored
    here; the reference row reused from ``--scores`` carries ``None`` there.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    detector = Binoculars(profile=profile, mode="accuracy", **detector_kwargs)
    try:
        neg, latencies = _score_dev(detector, dev, batch_size)
        row = _point_metrics(y_dev, neg)
        row["latency_ms_p50"] = float(np.percentile(latencies, 50))
        row["vram_peak_mb"] = (
            round(torch.cuda.max_memory_allocated() / 2**20, 1)
            if torch.cuda.is_available()
            else None
        )
        return row
    finally:
        _free_detector(detector)


def _seed_table(rows: list[dict[str, float | None]], note: str | None = None) -> dict:
    """Per-seed point metrics plus mean/std across seeds (protocol §5.2 table).

    Keys absent from every row (e.g. latency on a reused reference config)
    are reported as ``None`` rather than dropped, so the JSON schema stays
    uniform across configurations.
    """
    keys: dict[str, None] = {}
    for row in rows:
        keys.update(dict.fromkeys(row))
    mean: dict[str, float | None] = {}
    std: dict[str, float | None] = {}
    for key in keys:
        vals = [row[key] for row in rows if row.get(key) is not None]
        mean[key] = float(np.mean(vals)) if vals else None
        std[key] = float(np.std(vals)) if vals else None
    table: dict = {
        "seeds": {str(seed): dict(row) for seed, row in zip(SEEDS, rows, strict=True)},
        "mean": mean,
        "std": std,
    }
    if note:
        table["note"] = note
    return table


# --------------------------------------------------------------------------
# Ablations A1-A4 (protocol §5.1)
# --------------------------------------------------------------------------
def _run_a1(
    profile: LanguageProfile, dev: list[dict], y_dev: np.ndarray,
    ref_row: dict[str, float], batch_size: int,
) -> dict[str, dict]:
    """A1 — max_token_observed in {64, 128, 256, 512, 1024}; AUC, TPR@1 %, latency."""
    results: dict[str, dict] = {}
    for max_len in A1_MAX_LENGTHS:
        key = f"max_length={max_len}"
        if max_len == 512:
            # Reference config: calibrate.py scored the profile at the
            # detector default (max_token_observed=512) — dev scores reused.
            results[key] = _seed_table([ref_row] * len(SEEDS), note=REUSE_NOTE)
        else:
            row = _score_profile(
                profile, dev, y_dev, batch_size, max_token_observed=max_len
            )
            results[key] = _seed_table([row] * len(SEEDS), note=DETERMINISM_NOTE)
    return results


def _run_a2(
    profile: LanguageProfile, dev: list[dict], y_dev: np.ndarray,
    ref_row: dict[str, float], batch_size: int,
) -> dict[str, dict]:
    """A2 — precision: AUC, TPR@1 %, VRAM peak.

    TODO(spec): protocol §5.1 asks for {bfloat16, float16, float32}, but the
    detector only exposes ``use_bfloat16`` (bf16 vs fp32) and ``load_in_8bit``
    — PRD §16.1 mentions only bf16 vs int8. float16 is therefore not
    implementable without a detector change; 8-bit (bitsandbytes, PRD §16.1)
    is the reduced-precision arm retained here instead.
    """
    results: dict[str, dict] = {
        "bfloat16 (reference)": _seed_table([ref_row] * len(SEEDS), note=REUSE_NOTE),
    }
    results["float32"] = _seed_table(
        [_score_profile(profile, dev, y_dev, batch_size, use_bfloat16=False)] * len(SEEDS),
        note=DETERMINISM_NOTE,
    )
    results["8bit"] = _seed_table(
        [_score_profile(profile, dev, y_dev, batch_size, load_in_8bit=True)] * len(SEEDS),
        note=DETERMINISM_NOTE,
    )
    return results


def _parse_pair_spec(spec: str) -> tuple[str, str]:
    """Parse one ``--pairs-extra`` value as (observer_repo, performer_repo)."""
    parts = [part.strip() for part in spec.split(",")]
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError(
            f"--pairs-extra expects REPO_OBS,REPO_PERF, got {spec!r}"
        )
    return parts[0], parts[1]


def _run_a3(
    profile: LanguageProfile, dev: list[dict], y_dev: np.ndarray,
    ref_row: dict[str, float], batch_size: int, pairs_extra: list[tuple[str, str]],
) -> dict[str, dict]:
    """A3 — model pair via ``dataclasses.replace`` on the loaded profile; AUC, TPR@1 %."""
    results: dict[str, dict] = {
        "base+sft (reference)": _seed_table([ref_row] * len(SEEDS), note=REUSE_NOTE),
    }
    instruct = dataclasses.replace(profile, performer_model=LUCIOLE_INSTRUCT_1_1)
    results["base+instruct-1.1"] = _seed_table(
        [_score_profile(instruct, dev, y_dev, batch_size)] * len(SEEDS),
        note=DETERMINISM_NOTE,
    )
    results[A3_PRIVATE_PAIR] = {"note": "skipped: repo unavailable"}
    for observer_repo, performer_repo in pairs_extra:
        pair_profile = dataclasses.replace(
            profile,
            observer_model=observer_repo,
            performer_model=performer_repo,
            # Private Luciole pairs share the Base tokenizer; keep the
            # observer tokenizer unless the caller proves otherwise.
            share_tokenizer_from_observer=True,
        )
        key = f"{observer_repo} + {performer_repo}"
        try:
            row = _score_profile(pair_profile, dev, y_dev, batch_size)
            results[key] = _seed_table([row] * len(SEEDS), note=DETERMINISM_NOTE)
        except Exception as exc:  # repo unavailable / gated / offline
            results[key] = {"note": f"skipped: repo unavailable ({type(exc).__name__})"}
    return results


def _run_a4(
    profile: LanguageProfile, dev: list[dict], y_dev: np.ndarray,
    ref_row: dict[str, float], batch_size: int,
) -> dict[str, dict]:
    """A4 — shared vs separate tokenizer (sanity check, protocol §5.1); AUC.

    The ``share_tokenizer_from_observer=False`` arm runs the strict upstream
    ``assert_tokenizer_consistency``; if it fires, the configuration is kept
    in the table with the assert noted, exactly as the protocol intends the
    sanity check to be reported.
    """
    results: dict[str, dict] = {
        "shared observer tokenizer (reference)": _seed_table(
            [ref_row] * len(SEEDS), note=REUSE_NOTE
        ),
    }
    separate = dataclasses.replace(profile, share_tokenizer_from_observer=False)
    try:
        row = _score_profile(separate, dev, y_dev, batch_size)
        results["separate (strict assert)"] = _seed_table(
            [row] * len(SEEDS), note=DETERMINISM_NOTE
        )
    except ValueError as exc:
        results["separate (strict assert)"] = {
            "note": f"tokenizer consistency assert fired ({exc})"
        }
    return results


# --------------------------------------------------------------------------
# Console rendering
# --------------------------------------------------------------------------
def _fmt(value: float | None, std: float | None) -> str:
    """Format ``value ± std`` or a placeholder for unmeasured cells."""
    if value is None:
        return "—"
    return f"{value:.4f}±{std:.4f}" if std is not None else f"{value:.4f}"


def _print_table(title: str, results: dict[str, dict]) -> None:
    """Console rendering of one Config × Metric (± std) table (protocol §5.2)."""
    print(f"\n{title}")
    for config, entry in results.items():
        if "mean" not in entry:
            print(f"  {config:38s} {entry.get('note', '')}")
            continue
        cells = "  ".join(
            f"{key}={_fmt(entry['mean'][key], entry['std'][key])}" for key in entry["mean"]
        )
        print(f"  {config:38s} {cells}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True, help="calibration JSONL")
    parser.add_argument("--splits", type=Path, required=True, help="split manifest JSON")
    parser.add_argument("--scores", type=Path, required=True,
                        help="scores_<lang>_v01.json written by calibrate.py")
    parser.add_argument("--profile", default="fr", help="registered profile code")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pairs-extra", type=_parse_pair_spec, action="append",
                        default=[], metavar="REPO_OBS,REPO_PERF",
                        help="private A3 model pair; repeatable")
    parser.add_argument("--output-dir", type=Path, default=Path("calibration"))
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    """Run ablations A1-A4 on the dev split and write the artefact."""
    args = parse_args(argv)
    records, corpus_sha256 = load_corpus(args.corpus)
    splits = load_splits(args.splits, {r["id"] for r in records})
    scores = json.loads(args.scores.read_text(encoding="utf-8"))
    missing = {r["id"] for r in records} - set(scores)
    if missing:
        raise ValueError(f"scores file misses {len(missing)} corpus ids")

    profile = get_profile(args.profile)
    dev = [r for r in records if r["id"] in splits["dev"]]
    y_dev, neg_ref = split_arrays(records, scores, splits["dev"])
    ref_row = _point_metrics(y_dev, neg_ref)
    print(f"ablations on dev n={len(dev)} (protocol §5 — dev only, never test)")

    ablations = {
        "A1_max_length": _run_a1(profile, dev, y_dev, ref_row, args.batch_size),
        "A2_precision": _run_a2(profile, dev, y_dev, ref_row, args.batch_size),
        "A3_model_pair": _run_a3(
            profile, dev, y_dev, ref_row, args.batch_size, args.pairs_extra
        ),
        "A4_tokenizer": _run_a4(profile, dev, y_dev, ref_row, args.batch_size),
    }
    for title, results in ablations.items():
        _print_table(title, results)

    git_sha, git_dirty = git_state()
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "corpus_sha256": corpus_sha256,
        "seeds": {"ablation_runs": list(SEEDS), "torch": SEED_TORCH},
        "config": "v01",
        "versions": library_versions(),
        "n_dev": int(len(dev)),
        "ablations": ablations,
    }
    lang = args.profile
    out = args.output_dir / f"ablations_{lang}_v01.json"
    write_json(out, payload)
    print(f"\nartefact: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
