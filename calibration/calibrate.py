#!/usr/bin/env python3
"""Threshold calibration for a binoculars-eu language profile.

Fits the three profile thresholds on the **train split only** (protocol
§2.2 — dev and test are never read for threshold selection), scores the
whole corpus once, and writes the traceability artefacts required by
protocol §8.4:

- ``calibration/results_<lang>_v01.json`` — run metadata (git sha, corpus
  SHA-256, seeds, versions) + train-split diagnostics,
- ``calibration/scores_<lang>_v01.json`` — ``id -> score`` for reuse by
  ``evaluate.py`` (no double scoring),
- ``calibration/error_analysis_candidates_<lang>_v01.json`` — top-20 false
  positive and top-20 false negative candidates from the **dev** split
  (protocol §7.2), for the annotation notebook.

With ``--write`` it also updates the profile itself:
- ``binoculars_eu/profiles/<lang>/thresholds.json`` — the three thresholds,
- ``binoculars_eu/profiles/<lang>/metadata.json`` — corpus SHA-256,
  calibration date (UTC) and calibration seed (42).

Threshold definitions (all fitted on train, scores oriented so that a LOW
Binoculars score means AI — the detector scale is inverted internally):

- ``accuracy`` : F1-optimal threshold (upstream "optimized for f1-score").
- ``low_fpr``  : strictest threshold whose train FPR stays within one false
  positive over the train human count (upstream spirit "accuse rarely",
  0.01 % FPR, unattainable at 150 train humans → granularity-adjusted).
- ``tpr_at_fpr_1`` : operating point at FPR = 1 % with maximal TPR.

Usage (protocol seeds are frozen, do not parameterise them)::

    python -m calibration.calibrate \
        --corpus calibration/corpus/binoculars-eu-corpus-fr-v01.jsonl \
        --splits calibration/splits_fr_v01.json \
        --profile fr

Add ``--write`` to update ``thresholds.json`` / ``metadata.json`` (without
it the run is a dry-run: artefacts are written, the profile is untouched).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_curve,
)

from binoculars_eu import Binoculars
from binoculars_eu.detector import _resolve_devices

SEED_SPLIT = 42       # protocol §1: StratifiedKFold / split seed
SEED_TORCH = 42       # protocol §1: torch.manual_seed (batch order)
CORPUS_LABELS = {"human", "ai"}
DEFAULT_BATCH_SIZE = 8  # PRD §7.2: conservative batch, never above GPU headroom


# --------------------------------------------------------------------------
# Loading and validation
# --------------------------------------------------------------------------
def load_corpus(path: Path) -> tuple[list[dict], str]:
    """Load the calibration JSONL and return (records, sha256 of the file)."""
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    seen: set[str] = set()
    for record in records:
        missing = {"id", "text", "label"} - record.keys()
        if missing:
            raise ValueError(f"corpus record {record.get('id')!r} misses fields: {missing}")
        if record["label"] not in CORPUS_LABELS:
            raise ValueError(f"record {record['id']!r}: unknown label {record['label']!r}")
        if record["id"] in seen:
            raise ValueError(f"duplicate corpus id: {record['id']!r}")
        seen.add(record["id"])
    return records, digest


def verify_splits_manifest(manifest: dict) -> None:
    """Recompute the manifest SHA-256 (protocol §2.1) and refuse on drift."""
    payload = {k: v for k, v in manifest.items() if k != "sha256"}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if digest != manifest.get("sha256"):
        raise ValueError("splits manifest sha256 mismatch — regenerate with build_splits.py")


def load_splits(path: Path, corpus_ids: set[str]) -> dict[str, set[str]]:
    """Load the split manifest and validate it against the corpus."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    verify_splits_manifest(manifest)
    splits = {
        "train": set(manifest["train_ids"]),
        "dev": set(manifest["dev_ids"]),
        "test": set(manifest["test_ids"]),
    }
    for name, ids in splits.items():
        unknown = ids - corpus_ids
        if unknown:
            raise ValueError(f"split {name!r} references unknown corpus ids: {sorted(unknown)[:3]}")
    overlap = (
        (splits["train"] & splits["dev"])
        | (splits["train"] & splits["test"])
        | (splits["dev"] & splits["test"])
    )
    if overlap:
        raise ValueError(f"split ids overlap across splits: {sorted(overlap)[:3]}")
    covered = splits["train"] | splits["dev"] | splits["test"]
    if covered != corpus_ids:
        raise ValueError(
            f"splits cover {len(covered)}/{len(corpus_ids)} corpus ids — rebuild the manifest"
        )
    return splits


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def score_corpus(
    detector: Binoculars, records: list[dict], batch_size: int
) -> dict[str, float]:
    """Score every record once, in corpus order, deterministic batches."""
    torch.manual_seed(SEED_TORCH)
    scores: dict[str, float] = {}
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        values = detector.compute_score([r["text"] for r in batch])
        if isinstance(values, float):
            values = [values]
        for record, value in zip(batch, values, strict=True):
            scores[record["id"]] = float(value)
    return scores


# --------------------------------------------------------------------------
# Threshold fitting (train split only)
# --------------------------------------------------------------------------
def _train_arrays(
    records: list[dict], scores: dict[str, float], train_ids: set[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true with 1=AI, negative scores) over the train split.

    Negated so that a HIGHER value means more AI-like, as sklearn expects.
    """
    train = [r for r in records if r["id"] in train_ids]
    y = np.array([1 if r["label"] == "ai" else 0 for r in train], dtype=int)
    neg = np.array([-scores[r["id"]] for r in train], dtype=float)
    return y, neg


def fit_accuracy_threshold(y_true: np.ndarray, neg_scores: np.ndarray) -> tuple[float, float]:
    """F1-optimal threshold (mode ``accuracy``). Returns (threshold, best F1)."""
    precision, recall, thresholds = precision_recall_curve(y_true, neg_scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best = int(np.argmax(f1[:-1]))  # last precision/recall pair has no threshold
    return -float(thresholds[best]), float(f1[best])


def fit_low_fpr_threshold(y_true: np.ndarray, neg_scores: np.ndarray) -> tuple[float, float]:
    """Strictest threshold within ~1 false positive on train humans.

    Upstream targets 0.01 % FPR; with ~150 train humans one FP is already
    0.67 %, so the budget is granularity-adjusted (protocol: document as
    such in the report). Falls back to the minimal-FPR operating point.
    """
    fpr, tpr, thresholds = roc_curve(y_true, neg_scores)
    n_human = max(int((y_true == 0).sum()), 1)
    budget = 1.0 / n_human
    ok = np.where(fpr <= budget)[0]
    # Defensive fallback (a train split always has points within budget).
    idx = int(np.argmin(fpr)) if len(ok) == 0 else int(ok[int(np.argmax(tpr[ok]))])
    return -float(thresholds[idx]), float(fpr[idx])


def fit_tpr_at_fpr_1_threshold(
    y_true: np.ndarray, neg_scores: np.ndarray
) -> tuple[float, float]:
    """Operating point at FPR = 1 % with maximal TPR (headline threshold)."""
    fpr, tpr, thresholds = roc_curve(y_true, neg_scores)
    ok = np.where(fpr <= 0.01)[0]
    idx = int(np.argmin(fpr)) if len(ok) == 0 else int(ok[int(np.argmax(tpr[ok]))])
    return -float(thresholds[idx]), float(tpr[idx])


# --------------------------------------------------------------------------
# Error-analysis candidates (dev split only, protocol §7.2)
# --------------------------------------------------------------------------
def extract_error_candidates(
    records: list[dict], scores: dict[str, float], dev_ids: set[str], k: int = 20
) -> dict[str, list[dict]]:
    """Top-k likely false positives (humans scoring lowest) and false
    negatives (AI scoring highest) on the dev split, for manual annotation."""
    dev = [r for r in records if r["id"] in dev_ids]
    humans = sorted((r for r in dev if r["label"] == "human"), key=lambda r: scores[r["id"]])
    ai = sorted((r for r in dev if r["label"] == "ai"), key=lambda r: scores[r["id"]], reverse=True)

    def entry(record: dict) -> dict:
        return {
            "id": record["id"],
            "label": record["label"],
            "source": record.get("source", ""),
            "score": round(scores[record["id"]], 6),
            "length_chars": len(record["text"]),
            "text": record["text"],
        }

    return {
        "false_positives": [entry(r) for r in humans[:k]],
        "false_negatives": [entry(r) for r in ai[:k]],
    }


# --------------------------------------------------------------------------
# Traceability helpers
# --------------------------------------------------------------------------
def git_state() -> tuple[str | None, bool | None]:
    """Return (sha, dirty) or (None, None) outside a git checkout."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip())
        return sha, dirty
    except Exception:
        return None, None


def library_versions() -> dict[str, str]:
    import sklearn
    import transformers

    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
    }


def environment_report(profile_code: str) -> str:
    device_1, device_2 = _resolve_devices()
    return "\n".join([
        f"profile: {profile_code}",
        f"device_map effectif: observer={{'': '{device_1}'}} performer={{'': '{device_2}'}}",
        "dtype: torch.bfloat16 (calibration default)",
        f"library versions: {library_versions()}",
    ])


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def update_profile_files(
    profile_code: str, thresholds: dict[str, float], corpus_sha256: str
) -> None:
    """Write thresholds.json and refresh metadata.json traceability fields."""
    profile_dir = Path("binoculars_eu/profiles") / profile_code
    if not profile_dir.is_dir():
        raise FileNotFoundError(f"profile directory not found: {profile_dir}")
    write_json(profile_dir / "thresholds.json", thresholds)
    metadata_path = profile_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "corpus_sha256": corpus_sha256,
        "calibration_date": datetime.now(UTC).date().isoformat(),
        "calibration_seed": SEED_SPLIT,
        "calibration_note": None,
    })
    write_json(metadata_path, metadata)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True, help="calibration JSONL")
    parser.add_argument("--splits", type=Path, required=True, help="split manifest JSON")
    parser.add_argument("--profile", default="fr", help="registered profile code")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="load both models with bitsandbytes int8 (capacity variants)")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="load both models with bitsandbytes nf4 4-bit (PRD §16.2 fallback)")
    parser.add_argument("--output-dir", type=Path, default=Path("calibration"))
    parser.add_argument(
        "--write", action="store_true",
        help="write thresholds.json and metadata.json (default: dry-run, profile untouched)",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    print("=== calibrate environment ===")
    print(environment_report(args.profile))
    print("=============================\n")

    records, corpus_sha256 = load_corpus(args.corpus)
    splits = load_splits(args.splits, {r["id"] for r in records})
    print(f"corpus: {len(records)} records | sha256={corpus_sha256[:16]}…")
    print("splits: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    detector = Binoculars.for_language(args.profile, mode="low-fpr",
                                       load_in_8bit=args.load_in_8bit,
                                       load_in_4bit=args.load_in_4bit)
    scores = score_corpus(detector, records, args.batch_size)
    print(f"scored {len(scores)} records (batch_size={args.batch_size})\n")

    # --- threshold fitting on train only (protocol §2.2) -------------------
    y_train, neg_train = _train_arrays(records, scores, splits["train"])
    fpr, tpr, _ = roc_curve(y_train, neg_train)
    train_auc = float(auc(fpr, tpr))
    thr_accuracy, best_f1 = fit_accuracy_threshold(y_train, neg_train)
    thr_low_fpr, low_fpr_value = fit_low_fpr_threshold(y_train, neg_train)
    thr_tpr1, tpr_at_1 = fit_tpr_at_fpr_1_threshold(y_train, neg_train)
    thresholds = {
        "accuracy": round(thr_accuracy, 6),
        "low_fpr": round(thr_low_fpr, 6),
        "tpr_at_fpr_1": round(thr_tpr1, 6),
    }
    print("train diagnostics (fit split only — no dev/test metric reported here):")
    print(f"  train AUC           = {train_auc:.4f}")
    print(f"  accuracy threshold  = {thresholds['accuracy']:.6f} (best F1 = {best_f1:.4f})")
    print(f"  low_fpr threshold   = {thresholds['low_fpr']:.6f} (train FPR = {low_fpr_value:.4f})")
    tpr1 = thresholds["tpr_at_fpr_1"]
    print(f"  tpr_at_fpr_1        = {tpr1:.6f} (train TPR@1%FPR = {tpr_at_1:.4f})")

    # --- artefacts ----------------------------------------------------------
    git_sha, git_dirty = git_state()
    results = {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "corpus_sha256": corpus_sha256,
        "seeds": {"split": SEED_SPLIT, "torch": SEED_TORCH},
        "config": "v01",
        "versions": library_versions(),
        "metrics": {
            "train_auc": train_auc,
            "thresholds": thresholds,
            "train_f1_at_accuracy": best_f1,
            "train_fpr_at_low_fpr": low_fpr_value,
            "train_tpr_at_fpr_1": tpr_at_1,
        },
    }
    lang = args.profile
    write_json(args.output_dir / f"results_{lang}_v01.json", results)
    write_json(args.output_dir / f"scores_{lang}_v01.json",
               {rid: round(s, 6) for rid, s in sorted(scores.items())})
    write_json(
        args.output_dir / f"error_analysis_candidates_{lang}_v01.json",
        extract_error_candidates(records, scores, splits["dev"]),
    )
    print(f"\nartefacts written under {args.output_dir}/")

    if args.write:
        update_profile_files(lang, thresholds, corpus_sha256)
        print(f"profile {lang!r} updated: thresholds.json + metadata.json (sha256, date, seed)")
    else:
        print("dry-run: profile untouched "
              "(rerun with --write to update thresholds.json/metadata.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
