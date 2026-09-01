#!/usr/bin/env python3
"""Score the OOD v2 corpus with a detector and report per-family AUC (V0.2 P3.3).

Each OOD v2 family (luciole-8b, gpt-4o, claude, hybrid) is a generalisation
measure: AI texts from a generator never used for calibration, scored against
the frozen test-split humans of the in-distribution corpus. AUC with 95 %
bootstrap CI per family and pooled; never used to fit thresholds (PRD §10.2).

Usage::

    python -m calibration.ood_v2_eval \
        --corpus calibration/corpus/binoculars-eu-corpus-fr-v01.jsonl \
        --splits calibration/splits_fr_v01.json \
        --scores-in calibration/scores_fr-8b_v01.json \
        --ood-corpus calibration/corpus/binoculars-eu-corpus-fr-v02-ood.jsonl \
        --profile fr-8b --load-in-4bit \
        --output calibration/ood_v2_eval_fr8b.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, roc_curve

from binoculars_eu import Binoculars
from calibration.calibrate import load_corpus, load_splits, score_corpus

SEED_BOOTSTRAP = 100  # protocol §3.3
N_BOOT = 1000


def bootstrap_auc(neg: np.ndarray, pos: np.ndarray) -> tuple[float, float]:
    """95 % bootstrap CI of AUC (humans vs one AI family), seed 100."""
    rng = np.random.default_rng(SEED_BOOTSTRAP)
    scores = np.concatenate([neg, pos])
    labels = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
    values = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(scores), len(scores))
        if len(np.unique(labels[idx])) < 2:
            continue
        fpr, tpr, _ = roc_curve(labels[idx], scores[idx])
        values.append(auc(fpr, tpr))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--scores-in", type=Path, required=True,
                        help="frozen in-distribution scores (test humans reused)")
    parser.add_argument("--ood-corpus", type=Path, required=True)
    parser.add_argument("--profile", default="fr-8b")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path,
                        default=Path("calibration/ood_v2_eval_fr8b.json"))
    args = parser.parse_args(argv)

    records, corpus_sha256 = load_corpus(args.corpus)
    splits = load_splits(args.splits, {r["id"] for r in records})
    scores_in = json.loads(args.scores_in.read_text(encoding="utf-8"))
    test_humans = [r for r in records
                   if r["id"] in splits["test"] and r["label"] == "human"]
    neg = -np.array([scores_in[r["id"]] for r in test_humans], dtype=float)
    print(f"test humans reused: n={len(neg)} (not re-scored)", flush=True)

    ood_records, _ = load_corpus(args.ood_corpus)
    print(f"OOD v2 records: {len(ood_records)}", flush=True)
    detector = Binoculars.for_language(args.profile, mode="accuracy",
                                       load_in_8bit=args.load_in_8bit,
                                       load_in_4bit=args.load_in_4bit)
    scored = score_corpus(detector, ood_records, args.batch_size)
    scores_path = args.output.with_name("scores_ood_v2_fr8b.json")
    scores_path.write_text(json.dumps(scored, ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8")

    families: dict[str, dict] = {}
    sources = sorted({r.get("source", "?") for r in ood_records})
    for source in sources:
        pos = -np.array([scored[r["id"]] for r in ood_records
                         if r.get("source") == source], dtype=float)
        fpr, tpr, _ = roc_curve(np.concatenate([np.zeros(len(neg)),
                                                np.ones(len(pos))]),
                                np.concatenate([neg, pos]))
        point = float(auc(fpr, tpr))
        lo, hi = bootstrap_auc(neg, pos)
        families[source] = {"n": len(pos), "auc": round(point, 4),
                            "ci95": [round(lo, 4), round(hi, 4)]}
        print(f"  {source}: n={len(pos)} AUC={point:.3f} [{lo:.3f},{hi:.3f}]",
              flush=True)

    all_pos = -np.array([scored[r["id"]] for r in ood_records], dtype=float)
    fpr, tpr, _ = roc_curve(np.concatenate([np.zeros(len(neg)), np.ones(len(all_pos))]),
                            np.concatenate([neg, all_pos]))
    pooled = float(auc(fpr, tpr))
    report = {
        "study": "ood_v2_per_family_auc",
        "profile": args.profile,
        "load_in_4bit": args.load_in_4bit,
        "seed_bootstrap": SEED_BOOTSTRAP,
        "corpus_sha256": corpus_sha256,
        "n_test_humans": len(neg),
        "scores_file": str(scores_path),
        "families": families,
        "pooled_auc": round(pooled, 4),
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"pooled OOD v2 AUC = {pooled:.3f} | wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
