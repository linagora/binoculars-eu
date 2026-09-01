#!/usr/bin/env python3
"""Pre-generate the R-5/R-6 perturbation texts (V0.2 P2).

The fr-8b scoring pass needs the whole 22 GB L4, so R-5/R-6 cannot be
generated inline while the detector is resident (unlike V0.1, where CPU
scoring coexisted with the vLLM generator). This script runs the SAME
perturbation functions (robustness.r5/r6, seeds 502/503) against the live
generator, ahead of the scoring pass; ``calibration.robustness --r56-file``
then consumes the saved texts, bit-identical to inline generation.

Usage (vLLM 8B running)::

    GENERATOR_API_KEY=dummy python -m calibration.pre_generate_r56 \
        --corpus calibration/corpus/binoculars-eu-corpus-fr-v01.jsonl \
        --splits calibration/splits_fr_v01.json \
        --generator-url http://100.90.203.88:8013 --generator-model luciole-8b-instruct \
        --output calibration/r56_pregen_fr_v02.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibration.calibrate import load_corpus, load_splits
from calibration.robustness import r5_adversarial_prompt, r6_adversarial_rewrite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--generator-url", required=True)
    parser.add_argument("--generator-model", required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("calibration/r56_pregen_fr_v02.jsonl"))
    args = parser.parse_args(argv)

    records, _ = load_corpus(args.corpus)
    splits = load_splits(args.splits, {r["id"] for r in records})
    test = [r for r in records if r["id"] in splits["test"]]
    print(f"test split: n={len(test)}", flush=True)

    r5 = r5_adversarial_prompt(test, args.generator_url, args.generator_model)
    r6 = r6_adversarial_rewrite(test, args.generator_url, args.generator_model)
    assert r5 is not None and r6 is not None
    combined = r5 + r6
    with args.output.open("w", encoding="utf-8") as handle:
        for record in combined:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {args.output} ({len(r5)} R-5 + {len(r6)} R-6 records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
