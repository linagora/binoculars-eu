#!/usr/bin/env python3
"""Build the stratified train/dev/test split manifest for a profile corpus.

Protocol §2: 60/20/20, stratified on 4 axes (label, source, generator,
length bin), frozen seed (§1: 42). The manifest is versioned in the repo
(``calibration/splits_<lang>_v01.json``) and carries its own SHA-256 so any
later drift is detected before calibration or evaluation.

Discipline (protocol §2.2): the test split must be loaded for the first time
by ``evaluate.py``, never in an exploratory notebook.

Usage::

    python -m calibration.build_splits \
        --corpus calibration/corpus/binoculars-eu-corpus-fr-v01.jsonl \
        --output calibration/splits_fr_v01.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from sklearn.model_selection import StratifiedShuffleSplit

SEED = 42  # protocol §1 — frozen


def stratum_key(record: dict) -> str:
    """4-axis stratification key: label | source | generator | length bin."""
    length = record["meta"]["length_tokens"]
    if length < 150:
        length_bin = "L1"
    elif length < 300:
        length_bin = "L2"
    elif length < 500:
        length_bin = "L3"
    else:
        length_bin = "L4"
    generator = record["meta"].get("generator", "human")
    return f"{record['label']}|{record['source']}|{generator}|{length_bin}"


def build_splits(corpus: list[dict], seed: int = SEED) -> tuple[list, list, list, dict]:
    """Split 80/20 then 75/25 → 60/20/20, stratified; return manifest."""
    strata = [stratum_key(r) for r in corpus]
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(corpus, strata))
    trainval = [corpus[i] for i in trainval_idx]
    test = [corpus[i] for i in test_idx]
    strata_tv = [strata[i] for i in trainval_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, dev_idx = next(sss2.split(trainval, strata_tv))
    train = [trainval[i] for i in train_idx]
    dev = [trainval[i] for i in dev_idx]

    manifest = {
        "seed": seed,
        "train_ids": [r["id"] for r in train],
        "dev_ids": [r["id"] for r in dev],
        "test_ids": [r["id"] for r in test],
        "counts": {"train": len(train), "dev": len(dev), "test": len(test)},
    }
    payload = json.dumps(manifest, sort_keys=True).encode()
    manifest["sha256"] = hashlib.sha256(payload).hexdigest()
    return train, dev, test, manifest


def validate(corpus: list[dict], train: list, dev: list, test: list) -> None:
    """Refuse overlapping or incomplete splits before anything is written."""
    ids = [r["id"] for r in corpus]
    if len(set(ids)) != len(ids):
        raise ValueError("corpus contains duplicate ids")
    corpus_ids = set(ids)
    splits = {
        name: {r["id"] for r in records_}
        for name, records_ in (("train", train), ("dev", dev), ("test", test))
    }
    overlap = (
        (splits["train"] & splits["dev"])
        | (splits["train"] & splits["test"])
        | (splits["dev"] & splits["test"])
    )
    if overlap:
        raise ValueError(f"split overlap: {sorted(overlap)[:5]}")
    covered = splits["train"] | splits["dev"] | splits["test"]
    if covered != corpus_ids:
        missing = corpus_ids - covered
        raise ValueError(f"splits do not cover the corpus, missing: {sorted(missing)[:5]}")


def load_corpus(path: Path) -> list[dict]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for record in records:
        for field in ("id", "label", "source", "meta"):
            if field not in record:
                raise ValueError(f"record {record.get('id')!r} misses field {field!r}")
        if "length_tokens" not in record["meta"]:
            raise ValueError(
                f"record {record['id']!r} misses meta.length_tokens — "
                "enrich the corpus metadata first (build_corpus)"
            )
    return records


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True, help="calibration JSONL")
    parser.add_argument("--output", type=Path, required=True, help="manifest JSON output path")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    corpus = load_corpus(args.corpus)
    train, dev, test, manifest = build_splits(corpus)
    validate(corpus, train, dev, test)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    args.output.write_text(payload, encoding="utf-8")
    print(f"corpus: {len(corpus)} records")
    print(f"splits: {manifest['counts']} (seed={manifest['seed']})")
    print(f"manifest sha256: {manifest['sha256']}")
    print(f"written: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
