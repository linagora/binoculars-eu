#!/usr/bin/env python3
"""Assemble the R-2 ``--r2-file`` from a filled paraphrase kit (protocol §6.1).

Validates ``calibration/r2_kit_<lang>_v01.json`` (produced by
``calibration/build_r2_kit.py``) and writes the final JSONL
(``id`` / ``text``) consumed by ``calibration/robustness.py --r2-file``.
Validation is strict on purpose: the robustness run is only meaningful if the
non-paraphrased 90 % of every text are byte-identical to the corpus.

Checks: exact test-id coverage, segment texts untouched, every marked sentence
has a non-empty paraphrase different from the original.

Usage::

    python -m calibration.assemble_r2 --lang fr
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibration.build_r2_kit import split_sentences


def load_jsonl_ids_texts(corpus_path: Path) -> dict[str, str]:
    """Return ``{id: text}`` from a corpus JSONL."""
    texts: dict[str, str] = {}
    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            texts[record["id"]] = record["text"]
    return texts


def assemble(kit_path: Path, splits_path: Path, corpus_path: Path, out_path: Path) -> None:
    """Validate the filled kit and write the R-2 paraphrase JSONL."""
    kit = json.loads(kit_path.read_text(encoding="utf-8"))
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    corpus_texts = load_jsonl_ids_texts(corpus_path)

    errors: list[str] = []
    kit_ids = [t["id"] for t in kit["texts"]]
    if sorted(kit_ids) != sorted(splits["test_ids"]):
        errors.append(
            f"kit covers {len(kit_ids)} ids, test split has {len(splits['test_ids'])}"
        )

    lines: list[str] = []
    for entry in kit["texts"]:
        text_id = entry["id"]
        original = corpus_texts.get(text_id)
        if original is None:
            errors.append(f"{text_id}: not found in corpus")
            continue
        segments = split_sentences(original)
        sentences = entry["sentences"]
        if [s["text"] for s in sentences] != segments:
            errors.append(f"{text_id}: 'text' fields were modified — expected the corpus segments")
            continue
        paraphrased: list[str] = []
        for seg, sentence in zip(segments, sentences, strict=True):
            if not sentence["to_paraphrase"]:
                paraphrased.append(seg)
                continue
            para = sentence.get("paraphrase", "").strip()
            if not para:
                errors.append(f"{text_id} sentence {sentence['i']}: empty paraphrase")
                continue
            if para == seg.strip():
                errors.append(
                    f"{text_id} sentence {sentence['i']}: paraphrase is identical to the original"
                )
                continue
            paraphrased.append(para)
        if len(paraphrased) == len(segments):
            lines.append(json.dumps({"id": text_id, "text": "".join(paraphrased)},
                                    ensure_ascii=False))

    if errors:
        head = "\n  ".join(errors[:20])
        raise SystemExit(f"kit validation failed ({len(errors)} errors):\n  {head}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path} — {len(lines)} paraphrased texts")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", default="fr")
    parser.add_argument("--kit", type=Path, default=None)
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
    kit = args.kit or Path(f"calibration/r2_kit_{args.lang}_v01.json")
    out = args.out or Path(f"calibration/r2_paraphrases_{args.lang}_v01.jsonl")
    assemble(kit, args.splits, args.corpus, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
