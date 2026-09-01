#!/usr/bin/env python3
"""Audit the V1.0 calibration corpus before building V1.1 (docs/v03_plan.md P0).

Quantifies every known or suspected defect of
``binoculars-eu-corpus-fr-v01.jsonl`` so the V1.1 perimeter is measured, not
estimated: mojibake per source (both labels), empty titles, degenerate AI
twin prompts (empty topic), twin distribution, exact duplicates, and length
stats per source. Output: JSON report + console summary.

Usage::

    python -m calibration.audit_corpus \
        --corpus calibration/corpus/binoculars-eu-corpus-fr-v01.jsonl \
        --output calibration/audit_corpus_v10.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, cast

JsonDict = dict[str, Any]  # JSON payload; Any is the practical JSON type

MOJIBAKE = re.compile(r"Ã.|â€™|â€œ|â€|Â |ï»¿|�")
# Degenerate twin prompt: "sur : ." or "sur : <only punctuation/space>"
DEGENERATE_TOPIC = re.compile(r"sur\s*:\s*[.…]+\s*Un seul")


def source_prefix(record_id: str) -> str:
    """Source family of a record id (last numeric segment stripped)."""
    return "-".join(record_id.split("-")[:-1])


def load_records(path: Path) -> list[JsonDict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(cast(JsonDict, json.loads(line)))
    return records


def mojibake_report(records: list[JsonDict]) -> JsonDict:
    """Mojibake hits per (label, source prefix), with one sample each."""
    hits: dict[str, list[str]] = {}
    samples: dict[str, str] = {}
    for r in records:
        text = str(r["text"])
        match = MOJIBAKE.search(text)
        if match is None:
            continue
        key = f'{r["label"]}:{source_prefix(str(r["id"]))}'
        hits.setdefault(key, []).append(str(r["id"]))
        ctx = text[max(0, match.start() - 25):match.start() + 25]
        samples[key] = ctx
    return {
        "total_records_with_mojibake": sum(len(v) for v in hits.values()),
        "by_source": {k: {"n": len(v), "ids": v[:5]} for k, v in sorted(hits.items())},
        "samples": samples,
    }


def title_report(humans: list[JsonDict]) -> JsonDict:
    """Human records with an empty or missing title, per source."""
    empty: dict[str, list[str]] = {}
    for r in humans:
        title = str(r.get("meta", {}).get("title", "")).strip()
        if not title:
            key = source_prefix(str(r["id"]))
            empty.setdefault(key, []).append(str(r["id"]))
    return {
        "total_empty_titles": sum(len(v) for v in empty.values()),
        "by_source": {k: {"n": len(v), "ids": v[:5]} for k, v in sorted(empty.items())},
    }


def twin_report(records: list[JsonDict]) -> JsonDict:
    """AI twin distribution and degenerate prompts (empty topic)."""
    twins: Counter[str] = Counter()
    degenerate: list[str] = []
    orphan_twins: list[str] = []
    human_ids = {str(r["id"]) for r in records if r["label"] == "human"}
    for r in records:
        if r["label"] != "ai":
            continue
        twin_of = str(r.get("meta", {}).get("twin_of", ""))
        family = source_prefix(twin_of) if twin_of else "(none)"
        twins[family] += 1
        if DEGENERATE_TOPIC.search(str(r.get("meta", {}).get("prompt", ""))):
            degenerate.append(str(r["id"]))
        if twin_of and twin_of not in human_ids:
            orphan_twins.append(str(r["id"]))
    return {
        "twin_distribution": dict(sorted(twins.items())),
        "degenerate_prompt_ids": degenerate,
        "n_degenerate": len(degenerate),
        "orphan_twin_ids": orphan_twins,
    }


def duplicate_report(records: list[JsonDict]) -> JsonDict:
    """Exact-duplicate texts (SHA-256 of normalised text), per label."""
    seen: dict[str, str] = {}
    dupes: list[JsonDict] = []
    for r in records:
        digest = hashlib.sha256(str(r["text"]).strip().encode()).hexdigest()
        if digest in seen:
            dupes.append({"id": str(r["id"]), "duplicate_of": seen[digest]})
        else:
            seen[digest] = str(r["id"])
    return {"n_exact_duplicates": len(dupes), "duplicates": dupes[:20]}


def length_report(records: list[JsonDict]) -> JsonDict:
    """Word-count stats per (label, source prefix)."""
    buckets: dict[str, list[int]] = {}
    for r in records:
        key = f'{r["label"]}:{source_prefix(str(r["id"]))}'
        buckets.setdefault(key, []).append(len(str(r["text"]).split()))
    stats = {}
    for key, words in sorted(buckets.items()):
        ordered = sorted(words)
        stats[key] = {
            "n": len(words), "min": ordered[0], "p50": ordered[len(ordered) // 2],
            "max": ordered[-1],
        }
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("calibration/audit_corpus_v10.json"))
    args = parser.parse_args(argv)

    records = load_records(args.corpus)
    humans = [r for r in records if r["label"] == "human"]
    report: JsonDict = {
        "corpus": str(args.corpus),
        "n_records": len(records),
        "n_human": len(humans),
        "n_ai": len(records) - len(humans),
        "mojibake": mojibake_report(records),
        "empty_titles": title_report(humans),
        "twins": twin_report(records),
        "exact_duplicates": duplicate_report(records),
        "lengths_words": length_report(records),
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")

    m = report["mojibake"]
    print(f"records: {report['n_records']} (human {report['n_human']}, "
          f"ai {report['n_ai']})")
    print(f"mojibake: {m['total_records_with_mojibake']} records")
    for key, info in cast(JsonDict, m["by_source"]).items():
        print(f"  {key}: {info['n']}")
    print(f"empty titles: {report['empty_titles']['total_empty_titles']}")
    t = report["twins"]
    print(f"degenerate twin prompts: {t['n_degenerate']} {t['degenerate_prompt_ids'][:5]}")
    print(f"orphan twins: {len(t['orphan_twin_ids'])}")
    print(f"exact duplicates: {report['exact_duplicates']['n_exact_duplicates']}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
