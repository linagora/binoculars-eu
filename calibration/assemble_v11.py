#!/usr/bin/env python3
"""Assemble corpus V1.1 from V1.0 plus the P1.1/P1.2 patches (docs/v03_plan.md P1.4).

Applies both patches to the V1.0 corpus and verifies every V1.0 defect is
closed before writing:

- the 60 ``human-presse-fr-*`` records (mojibake + empty titles) are replaced
  by the P1.1 re-fetch patch;
- the 20 degenerate presse AI twins (empty topic) are replaced by the P1.2
  regeneration patch.

Output: ``binoculars-eu-corpus-fr-v1.1.jsonl`` + SHA-256 on stdout, for the
profile metadata and the changelog (calibration/protocol.md addendum).

Usage::

    python -m calibration.assemble_v11 \
        --corpus calibration/corpus/binoculars-eu-corpus-fr-v01.jsonl \
        --presse-patch calibration/corpus/_presse_v11_patch.jsonl \
        --twins-patch calibration/corpus/_twins_presse_v11_patch.jsonl \
        --output calibration/corpus/binoculars-eu-corpus-fr-v1.1.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, cast

from calibration.audit_corpus import DEGENERATE_TOPIC, MOJIBAKE

JsonDict = dict[str, Any]  # JSON payload; Any is the practical JSON type


def load_jsonl(path: Path) -> list[JsonDict]:
    return [cast(JsonDict, json.loads(line)) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--presse-patch", type=Path, required=True)
    parser.add_argument("--twins-patch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    records = load_jsonl(args.corpus)
    patches = {str(r["id"]): r for r in load_jsonl(args.presse_patch)}
    patches.update({str(r["id"]): r for r in load_jsonl(args.twins_patch)})
    print(f"{len(patches)} patch records "
          f"({sum(1 for r in load_jsonl(args.presse_patch))} presse, "
          f"{sum(1 for r in load_jsonl(args.twins_patch))} twins)", flush=True)

    assembled: list[JsonDict] = []
    applied = 0
    for r in records:
        rid = str(r["id"])
        if rid in patches:
            assembled.append(patches.pop(rid))
            applied += 1
        else:
            assembled.append(r)
    if patches:
        sys.exit(f"patch records with unknown ids: {sorted(patches)[:5]}")

    humans = [r for r in assembled if r["label"] == "human"]
    errors: list[str] = []
    if len(assembled) != 500:
        errors.append(f"n={len(assembled)} != 500")
    moji = [str(r["id"]) for r in assembled if MOJIBAKE.search(str(r["text"]))]
    if moji:
        errors.append(f"mojibake residual: {len(moji)} {moji[:5]}")
    empty_titles = [str(r["id"]) for r in humans
                    if str(r["id"]).startswith("human-presse")
                    and not str(r.get("meta", {}).get("title", "")).strip()]
    if empty_titles:
        errors.append(f"empty presse titles: {len(empty_titles)}")
    degenerate = [str(r["id"]) for r in assembled
                  if DEGENERATE_TOPIC.search(str(r.get("meta", {}).get("prompt", "")))]
    if degenerate:
        errors.append(f"degenerate prompts: {len(degenerate)}")
    missing_tokens = [str(r["id"]) for r in assembled
                      if "length_tokens" not in r.get("meta", {})]
    if missing_tokens:
        errors.append(f"missing length_tokens: {len(missing_tokens)} {missing_tokens[:5]}")
    dupes = len(assembled) - len({str(r["id"]) for r in assembled})
    if dupes:
        errors.append(f"duplicate ids: {dupes}")

    if errors:
        for e in errors:
            print(f"GATE FAIL: {e}", flush=True)
        return 1

    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in assembled)
    args.output.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    print(f"applied {applied} patches; n={len(assembled)} "
          f"(human {len(humans)}, ai {len(assembled) - len(humans)})")
    print(f"sha256 {digest}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
