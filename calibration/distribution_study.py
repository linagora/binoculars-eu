#!/usr/bin/env python3
"""P0.3 (amendment C): score-distribution study 1B bf16 vs 8B int8 on dev.

Scores the dev split with the capacity-variant detector (default profile
``fr-8b``, int8) and compares its score distribution against the frozen V0.1
scores (1B bf16, written by ``calibrate.py``). Two-sample Kolmogorov-Smirnov
tests are run per class (human / ai) and pooled. Decision rule (docs/v02_plan.md
amendment C): if max(D) > 0.1, ``fr-8b`` is a detector with its own thresholds
(recalibrated on train, never a drop-in of the V0.1 thresholds); the V0.2 eval
card must explicit this calibration delta.

Usage::

    python -m calibration.distribution_study \
        --corpus calibration/corpus/binoculars-eu-corpus-fr-v01.jsonl \
        --splits calibration/splits_fr_v01.json \
        --scores-1b calibration/scores_fr_v01.json \
        --profile fr-8b --load-in-8bit \
        --output calibration/distribution_study_fr8b.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ks_2samp

from binoculars_eu import Binoculars
from calibration.calibrate import load_corpus, load_splits, score_corpus

SEED_TORCH = 42  # protocol §1
DECISION_D = 0.10  # amendment C


def ks_report(scores_a: np.ndarray, scores_b: np.ndarray) -> dict[str, float]:
    """One two-sample KS test with D statistic and p-value."""
    result = ks_2samp(scores_a, scores_b)
    return {"D": round(float(result.statistic), 4),
            "p_value": float(f"{result.pvalue:.3g}")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--scores-1b", type=Path, required=True,
                        help="frozen V0.1 scores (1B bf16), never re-scored")
    parser.add_argument("--profile", default="fr-8b")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path,
                        default=Path("calibration/distribution_study_fr8b.json"))
    args = parser.parse_args(argv)

    torch.manual_seed(SEED_TORCH)
    records, corpus_sha256 = load_corpus(args.corpus)
    splits = load_splits(args.splits, {r["id"] for r in records})
    dev_ids = splits["dev_ids"]
    dev = [r for r in records if r["id"] in set(dev_ids)]
    print(f"dev split: n={len(dev)} (corpus sha256={corpus_sha256[:12]}…)", flush=True)

    detector = Binoculars.for_language(args.profile, mode="low-fpr",
                                       load_in_8bit=args.load_in_8bit)
    scores_8b = score_corpus(detector, dev, args.batch_size)

    scores_1b_full = json.loads(args.scores_1b.read_text(encoding="utf-8"))
    scores_1b = {r["id"]: scores_1b_full[r["id"]] for r in dev}

    by_class: dict[str, dict[str, list[float]]] = {
        "human": {"one_b": [], "eight_b": []}, "ai": {"one_b": [], "eight_b": []},
    }
    for r in dev:
        by_class[r["label"]]["one_b"].append(scores_1b[r["id"]])
        by_class[r["label"]]["eight_b"].append(scores_8b[r["id"]])

    report: dict[str, object] = {
        "study": "distribution_1b_vs_8b_dev",
        "profile": args.profile,
        "load_in_8bit": args.load_in_8bit,
        "seed": SEED_TORCH,
        "corpus_sha256": corpus_sha256,
        "n_dev": len(dev),
        "decision_rule": f"max(D) > {DECISION_D} -> own thresholds (no drop-in)",
        "classes": {},
    }
    max_d = 0.0
    for label, pair in by_class.items():
        a = np.array(pair["one_b"], dtype=float)
        b = np.array(pair["eight_b"], dtype=float)
        ks = ks_report(a, b)
        max_d = max(max_d, ks["D"])
        report["classes"][label] = {  # type: ignore[index]
            "n": len(a),
            "one_b": {"mean": round(float(a.mean()), 4), "std": round(float(a.std()), 4)},
            "eight_b": {"mean": round(float(b.mean()), 4), "std": round(float(b.std()), 4)},
            "ks": ks,
        }
        print(f"[{label}] 1B mean={a.mean():.4f} | 8B mean={b.mean():.4f} | D={ks['D']}",
              flush=True)

    pooled_a = np.array([s for pair in by_class.values() for s in pair["one_b"]])
    pooled_b = np.array([s for pair in by_class.values() for s in pair["eight_b"]])
    pooled_ks = ks_report(pooled_a, pooled_b)
    max_d = max(max_d, pooled_ks["D"])
    report["pooled"] = pooled_ks  # type: ignore[index]
    report["decision"] = "own_thresholds" if max_d > DECISION_D else "dropin_possible"
    report["max_D"] = round(max_d, 4)
    print(f"pooled D={pooled_ks['D']} | max(D)={max_d:.4f} -> {report['decision']}", flush=True)

    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
