#!/usr/bin/env python3
"""Build the §7 error-analysis annotation kit (protocol §7.2).

Reads ``calibration/error_analysis_candidates_fr_v01.json`` (20 worst FP + 20
worst FN from the dev split, extracted by ``evaluate.py``) and writes a
hand-editable JSON kit (``calibration/error_kit_<lang>_v01.json``). The
annotator picks one taxonomy category per candidate plus an optional free note
(mandatory for the ``-autre`` categories) in
``calibration/error_kit_editor.html``; ``calibration/assemble_error_annotations.py``
then validates the kit and writes ``docs/error_analysis_annotations_<lang>_v01.json``.

Usage::

    python -m calibration.build_error_kit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from binoculars_eu.profiles import get_profile

TAXONOMY_FP: list[dict[str, str]] = [
    {"code": "FP-1",
     "description": "Texte administratif / juridique (registre formel proche du style IA)"},
    {"code": "FP-2", "description": "Texte très court (< 100 tokens)"},
    {"code": "FP-3", "description": "Traduction automatique post-éditée"},
    {"code": "FP-4", "description": "Texte avec beaucoup de code, chiffres, tableaux"},
    {"code": "FP-5", "description": "Texte technique très standardisé (RFC, spéc, mode d'emploi)"},
    {"code": "FP-6", "description": "Texte encyclopédique très neutre (Wikipedia dense)"},
    {"code": "FP-autre", "description": "Autre — à caractériser dans la note libre (obligatoire)"},
]

TAXONOMY_FN: list[dict[str, str]] = [
    {"code": "FN-1", "description": "IA générée avec température ≥ 0.9"},
    {"code": "FN-2", "description": "IA post-éditée par un humain"},
    {"code": "FN-3",
     "description": "IA imitant un style très marqué (dialecte, argot, littéraire)"},
    {"code": "FN-4", "description": "IA hors-distribution de Luciole (Mistral, GPT-4, Claude)"},
    {"code": "FN-5", "description": "IA très courte"},
    {"code": "FN-autre", "description": "Autre — à caractériser dans la note libre (obligatoire)"},
]


def build_kit(lang: str, candidates_path: Path, out_path: Path) -> None:
    """Write the hand-editable error-analysis kit for ``lang``."""
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    profile = get_profile(lang)
    entries: list[dict] = []
    for kind, key in (("fp", "false_positives"), ("fn", "false_negatives")):
        for record in candidates[key]:
            entries.append(
                {
                    "kind": kind,
                    "id": record["id"],
                    "label": record["label"],
                    "source": record["source"],
                    "score": record["score"],
                    "length_chars": record["length_chars"],
                    "text": record["text"],
                    "category": "",
                    "note": "",
                }
            )
    kit = {
        "kit_version": "1.0",
        "task": "error-analysis",
        "lang": lang,
        "threshold_accuracy": profile.threshold_accuracy,
        "instructions": (
            "Pick ONE category per candidate (protocol §7.1 taxonomy); the note is "
            "free and mandatory for the '-autre' categories. See "
            "calibration/ERROR_INSTRUCTIONS_FR.md."
        ),
        "taxonomy_fp": TAXONOMY_FP,
        "taxonomy_fn": TAXONOMY_FN,
        "candidates": entries,
    }
    out_path.write_text(json.dumps(kit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    n_fp = sum(1 for e in entries if e["kind"] == "fp")
    print(f"wrote {out_path} — {len(entries)} candidates ({n_fp} FP, {len(entries) - n_fp} FN), "
          f"threshold_accuracy={profile.threshold_accuracy}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", default="fr")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("calibration/error_analysis_candidates_fr_v01.json"),
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.out or Path(f"calibration/error_kit_{args.lang}_v01.json")
    build_kit(args.lang, args.candidates, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
