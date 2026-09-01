#!/usr/bin/env python3
"""R-6bis (amendment A): TPR at the low-fpr threshold on the humanized corpus.

The V0.1 eval card documents that commercial humanization (Undetectable AI)
flips the low-fpr verdict on fully humanized texts. V0.2 acceptance therefore
adds: **TPR@low_fpr on the humanized corpus >= 0.30**, i.e. at least 30 % of
humanized AI texts must still score on the AI side of the profile's low_fpr
threshold (score <= threshold means "AI", per the platform convention).

Scores are either reused from a JSON file {id: score} or computed inline with
the detector. Output: JSON report with per-humanizer breakdown, bootstrap CI
(seed 100, 1000 draws, protocol §3.3) and the pass/fail verdict.

Usage::

    python -m calibration.humanized_eval \
        --corpus calibration/corpus/binoculars-eu-corpus-fr-v02-ood.jsonl \
        --profile fr-8b --load-in-8bit \
        --output calibration/humanized_eval_fr8b_v02.json
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from binoculars_eu import Binoculars
from binoculars_eu.profiles import get_profile
from calibration.calibrate import git_state, score_corpus

SEED_BOOTSTRAP = 100  # protocol §3.3
N_BOOT = 1000
CRITERION = 0.30  # amendment A


def load_records(path: Path) -> list[dict]:
    """Humanized corpus JSONL records (label == 'ai' expected)."""
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def tpr_at(scores: np.ndarray, threshold: float) -> float:
    """Fraction of humanized AI texts on the AI side (score <= threshold)."""
    if len(scores) == 0:
        return 0.0
    return float(np.mean(scores <= threshold))


def bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    """95 % percentile bootstrap CI of the detection rate (seed 100, 1000 draws)."""
    rng = np.random.default_rng(SEED_BOOTSTRAP)
    rates = [float(np.mean(rng.choice(values, size=len(values), replace=True)))
             for _ in range(N_BOOT)]
    return float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True,
                        help="JSONL of humanized AI texts (id, text, label, source)")
    parser.add_argument("--profile", default="fr-8b")
    parser.add_argument("--scores", type=Path, default=None,
                        help="pre-computed {id: score} JSON (else scored inline)")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path,
                        default=Path("calibration/humanized_eval_fr8b_v02.json"))
    args = parser.parse_args(argv)

    profile = get_profile(args.profile)
    threshold = profile.threshold_low_fpr
    records = load_records(args.corpus)
    humanized = [r for r in records if r.get("label") == "ai"]
    print(f"humanized AI texts: {len(humanized)} (threshold low_fpr={threshold})",
          flush=True)

    if args.scores is not None:
        scores_all = json.loads(args.scores.read_text(encoding="utf-8"))
        scores = np.array([scores_all[r["id"]] for r in humanized], dtype=float)
    else:
        detector = Binoculars.for_language(args.profile, mode="low-fpr",
                                           load_in_8bit=args.load_in_8bit)
        scored = score_corpus(detector, humanized, args.batch_size)
        scores = np.array([scored[r["id"]] for r in humanized], dtype=float)

    detected = (scores <= threshold).astype(int)
    overall = tpr_at(scores, threshold)
    lo, hi = bootstrap_ci(detected)

    by_source: dict[str, dict] = {}
    for source in sorted({r.get("source", "?") for r in humanized}):
        idx = [i for i, r in enumerate(humanized) if r.get("source", "?") == source]
        sub = detected[idx]
        rate = tpr_at(scores[idx], threshold)
        sub_lo, sub_hi = bootstrap_ci(sub)
        by_source[source] = {
            "n": len(idx), "tpr_at_low_fpr": round(rate, 4),
            "ci95": [round(sub_lo, 4), round(sub_hi, 4)],
        }
        print(f"  {source}: n={len(idx)} TPR@low_fpr={rate:.3f}", flush=True)

    git_sha, git_dirty = git_state()
    report = {
        "metric": "R-6bis_tpr_at_low_fpr_humanized",
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "profile": args.profile,
        "threshold_low_fpr": threshold,
        "criterion": f">= {CRITERION} (amendment A, docs/v02_plan.md)",
        "n_humanized": len(humanized),
        "tpr_at_low_fpr": round(overall, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "by_source": by_source,
        "verdict": "pass" if overall >= CRITERION else "fail",
        "scores_file": str(args.scores) if args.scores else None,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"R-6bis TPR@low_fpr = {overall:.3f} {report['ci95']} -> {report['verdict']}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
