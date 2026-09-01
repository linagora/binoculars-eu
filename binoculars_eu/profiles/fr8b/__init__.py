"""French 8B capacity-variant profile (binoculars-eu V0.2) — ``fr-8b``.

Same corpus, splits and protocol as profile ``fr`` (PRD §16.2: capacity
variant, not a new language). Differs by model pair (Luciole-8B, hybrid
Mamba/Attention NemotronH), 8-bit quantization at load time (``--load-in-8bit``
on the calibration scripts), and ``trust_remote_code=True`` per PRD §5.

Thresholds are PLACEHOLDERS until the V0.2 P1 calibration pass
(``calibrate.py --profile fr-8b --load-in-8bit --write``), fitted on the train
split only; the calibration decision (own thresholds vs drop-in) is gated by
the KS distribution study (amendment C, docs/v02_plan.md P0.3).
"""

from __future__ import annotations

import json
from pathlib import Path

from binoculars_eu.profiles import register
from binoculars_eu.profiles.base import LanguageProfile

_HERE = Path(__file__).parent

_THRESHOLDS = json.loads((_HERE / "thresholds.json").read_text(encoding="utf-8"))
_METADATA = json.loads((_HERE / "metadata.json").read_text(encoding="utf-8"))

FRENCH_8B_PROFILE_V02 = register(
    LanguageProfile(
        code="fr-8b",
        display_name="Français (8B int8)",
        observer_model="OpenLLM-France/Luciole-8B-Base",
        performer_model="OpenLLM-France/Luciole-8B-Instruct-1.1",
        threshold_accuracy=_THRESHOLDS["accuracy"],
        threshold_low_fpr=_THRESHOLDS["low_fpr"],
        threshold_tpr_at_fpr_1=_THRESHOLDS["tpr_at_fpr_1"],
        corpus_sha256=_METADATA["corpus_sha256"],
        corpus_url=_METADATA["corpus_url"],
        calibration_date=_METADATA["calibration_date"],
        calibration_seed=_METADATA["calibration_seed"],
        share_tokenizer_from_observer=True,
        trust_remote_code=True,
        label_ai="Probablement généré par IA",
        label_human="Probablement écrit par un humain",
        calibration_note=_METADATA.get("calibration_note"),
    )
)
