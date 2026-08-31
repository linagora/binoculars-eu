#!/usr/bin/env python3
"""Build the R-2 paraphrase kit (protocol §6.1, light manual paraphrase).

Segments every test-split text into sentences and marks a deterministic
~10 % spread for manual light paraphrase. Output is a hand-editable pretty
JSON kit (``calibration/r2_kit_<lang>_v01.json``) in which the annotator fills
ONLY the ``paraphrase`` fields of marked sentences. ``calibration/assemble_r2.py``
then validates the kit and writes the final ``--r2-file`` JSONL consumed by
``calibration/robustness.py`` (which requires every test id to be present).

Usage::

    python -m calibration.build_r2_kit \
        --lang fr \
        --splits calibration/splits_fr_v01.json \
        --corpus calibration/corpus/binoculars-eu-corpus-fr-v01.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Same segmentation rule as calibration/robustness.py (kept in sync deliberately).
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]")
PARAPHRASE_RATE = 0.10  # protocol §6.1: ~10 % of sentences per text


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentence segments, keeping any trailing fragment."""
    segments: list[str] = []
    pos = 0
    for match in SENTENCE_RE.finditer(text):
        segments.append(match.group(0))
        pos = match.end()
    if pos < len(text):
        segments.append(text[pos:])
    return segments


def pick_indices(n: int) -> list[int]:
    """Deterministically pick ~10 % of ``n`` sentence indices, spread across the text."""
    k = max(1, round(PARAPHRASE_RATE * n))
    if n <= 3:
        return [n // 2]
    # Interior spread: indices 1..n-2, never the very first or last sentence.
    return sorted(
        {min(n - 2, max(1, round(1 + j * (n - 2) / max(k - 1, 1)))) for j in range(k)}
    )


def build_kit(lang: str, splits_path: Path, corpus_path: Path, out_path: Path) -> None:
    """Write the hand-editable R-2 kit for the test split of ``lang``."""
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    records: dict[str, dict] = {}
    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["id"]] = record

    texts: list[dict] = []
    total_sentences = 0
    total_marked = 0
    for text_id in splits["test_ids"]:
        segments = split_sentences(records[text_id]["text"])
        marked = set(pick_indices(len(segments)))
        total_sentences += len(segments)
        total_marked += len(marked)
        texts.append(
            {
                "id": text_id,
                "label": records[text_id]["label"],
                "sentences": [
                    {"i": i, "to_paraphrase": i in marked, "text": seg, "paraphrase": ""}
                    for i, seg in enumerate(segments)
                ],
            }
        )

    kit = {
        "kit_version": "1.0",
        "lang": lang,
        "instructions": (
            "Fill ONLY the 'paraphrase' fields where 'to_paraphrase' is true. "
            "Light reformulation: keep meaning, entities and numbers; do not touch "
            "any other field. See calibration/R2_INSTRUCTIONS_FR.md."
        ),
        "texts": texts,
    }
    out_path.write_text(json.dumps(kit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {out_path} — {len(texts)} texts, {total_sentences} sentences, "
        f"{total_marked} marked for paraphrase (~{total_marked / max(len(texts), 1):.1f}/text)"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", default="fr")
    parser.add_argument("--splits", type=Path, default=Path("calibration/splits_fr_v01.json"))
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("calibration/corpus/binoculars-eu-corpus-fr-v01.jsonl"),
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.out or Path(f"calibration/r2_kit_{args.lang}_v01.json")
    build_kit(args.lang, args.splits, args.corpus, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
