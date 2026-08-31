#!/usr/bin/env python3
"""Assemble §7 error-analysis annotations from a filled kit (protocol §7.2).

Validates ``calibration/error_kit_<lang>_v01.json`` (produced by
``calibration/build_error_kit.py``) and writes the annotation artifact
``docs/error_analysis_annotations_<lang>_v01.json``.

Validation: every candidate has exactly one category from the matching
taxonomy group (FP-x for false positives, FN-x for false negatives); the free
note is mandatory for the ``-autre`` categories.

Usage::

    python -m calibration.assemble_error_annotations --lang fr
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_ANNOTATOR = "agent (annotation automatique, à relire par un humain)"


def assemble(kit_path: Path, out_path: Path, annotator: str) -> None:
    """Validate the filled kit and write the annotation JSON artifact."""
    kit = json.loads(kit_path.read_text(encoding="utf-8"))
    valid_fp = {t["code"] for t in kit["taxonomy_fp"]}
    valid_fn = {t["code"] for t in kit["taxonomy_fn"]}
    candidates = kit["candidates"]

    errors: list[str] = []
    annotations: list[dict] = []
    for entry in candidates:
        category = entry.get("category", "").strip()
        note = entry.get("note", "").strip()
        allowed = valid_fp if entry["kind"] == "fp" else valid_fn
        prefix = "FP-" if entry["kind"] == "fp" else "FN-"
        if not category:
            errors.append(f"{entry['id']}: no category")
        elif category not in allowed:
            errors.append(f"{entry['id']}: category {category!r} not in {prefix} taxonomy")
        elif category.endswith("-autre") and not note:
            errors.append(f"{entry['id']}: category {category} requires a free note")
        annotations.append(
            {
                "id": entry["id"],
                "kind": entry["kind"],
                "label": entry["label"],
                "source": entry["source"],
                "score": entry["score"],
                "category": category,
                "note": note,
            }
        )

    if errors:
        head = "\n  ".join(errors[:20])
        raise SystemExit(f"kit validation failed ({len(errors)} errors):\n  {head}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "artifact": "error_analysis_annotations",
        "lang": kit["lang"],
        "config": "v01",
        "protocol": "§7.2",
        "annotator": annotator,
        "review_status": "pending_human_review",
        "date": datetime.now(tz=UTC).date().isoformat(),
        "threshold_accuracy": kit["threshold_accuracy"],
        "candidates_sha256": hashlib.sha256(
            kit_path.read_bytes()
        ).hexdigest(),
        "n_annotated": len(annotations),
        "annotations": annotations,
    }
    out_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    counts: dict[str, int] = {}
    for a in annotations:
        counts[a["category"]] = counts.get(a["category"], 0) + 1
    top = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"wrote {out_path} — {len(annotations)} annotations: {top}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", default="fr")
    parser.add_argument("--kit", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--annotator", default=DEFAULT_ANNOTATOR)
    parser.add_argument("--reviewed-by", default=None,
                        help="human reviewer endorsing the annotation (sets review_status)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    kit = args.kit or Path(f"calibration/error_kit_{args.lang}_v01.json")
    out = args.out or Path(f"docs/error_analysis_annotations_{args.lang}_v01.json")
    assemble(kit, out, args.annotator)
    if args.reviewed_by:
        artifact = json.loads(out.read_text(encoding="utf-8"))
        artifact["review_status"] = "human_reviewed"
        artifact["reviewed_by"] = args.reviewed_by
        artifact["review_date"] = artifact["date"]
        out.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        print(f"marked as human_reviewed by {args.reviewed_by}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
