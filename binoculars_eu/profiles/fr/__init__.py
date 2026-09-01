"""French language profile for binoculars-eu V0.1 (Luciole-1B).

Contract of a profile folder (PRD §6.7): exactly three files, no scoring code,
no import of the detector (would be circular), and a module-level ``register()``
call so auto-discovery is sufficient.
"""

from __future__ import annotations

import json
from pathlib import Path

from binoculars_eu.profiles import register
from binoculars_eu.profiles.base import LanguageProfile

_HERE = Path(__file__).parent

_THRESHOLDS = json.loads((_HERE / "thresholds.json").read_text(encoding="utf-8"))
_METADATA = json.loads((_HERE / "metadata.json").read_text(encoding="utf-8"))

# Performer locked to Luciole-1B-SFT-1.0 per the mission brief; calibrated
# thresholds committed 2026-08-31 (train split, seed 42), see calibration/.
FRENCH_PROFILE_V01 = register(
    LanguageProfile(
        code="fr",
        display_name="Français",
        observer_model="OpenLLM-France/Luciole-1B-Base",
        performer_model="OpenLLM-France/Luciole-1B-SFT-1.0",
        threshold_accuracy=_THRESHOLDS["accuracy"],
        threshold_low_fpr=_THRESHOLDS["low_fpr"],
        threshold_tpr_at_fpr_1=_THRESHOLDS["tpr_at_fpr_1"],
        corpus_sha256=_METADATA["corpus_sha256"],
        corpus_url=_METADATA["corpus_url"],
        calibration_date=_METADATA["calibration_date"],
        calibration_seed=_METADATA["calibration_seed"],
        share_tokenizer_from_observer=True,
        trust_remote_code=False,
        label_ai="Probablement généré par IA",
        label_human="Probablement écrit par un humain",
        calibration_note=_METADATA.get("calibration_note"),
    )
)
