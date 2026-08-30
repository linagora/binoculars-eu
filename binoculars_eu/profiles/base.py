"""Base dataclass for language-specific Binoculars profiles.

A ``LanguageProfile`` is the single unit of variation in binoculars-eu: it
encapsulates everything language-specific (model pair, calibrated thresholds,
calibration corpus fingerprint, localised verdict labels) while the scoring
engine stays strictly identical across languages.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageProfile:
    """Everything specific to a language.

    A published profile is an immutable, versioned, citable fingerprint:
    modifying an attribute on a frozen instance raises ``FrozenInstanceError``.

    Attributes:
        code: ISO 639-1 language code, e.g. ``"fr"``, ``"en"``.
        display_name: Human-readable name, e.g. ``"Français"``.
        observer_model: HuggingFace repo of the observer model.
        performer_model: HuggingFace repo of the performer model.
        threshold_accuracy: F1-optimised threshold used in mode ``"accuracy"``.
        threshold_low_fpr: Low-FPR threshold used in mode ``"low-fpr"``.
        threshold_tpr_at_fpr_1: Threshold at the FPR = 1 % operating point.
        corpus_sha256: SHA-256 fingerprint of the calibration JSONL.
        corpus_url: URL of the public calibration dataset.
        calibration_date: ISO 8601 calibration date, e.g. ``"2026-09-15"``.
        calibration_seed: Seed of the stratified splits used for calibration.
        share_tokenizer_from_observer: Load a single tokenizer (the observer's)
            and share it between both models, bypassing the strict upstream
            ``assert_tokenizer_consistency``. Required when both tokenizers are
            functionally interchangeable (Luciole Base/Instruct case).
        trust_remote_code: Allow custom modelling code from the model repo.
            ``True`` is required for NemotronH (Luciole-8B, V0.2), ``False``
            otherwise; the authorisation is granted per profile, not globally.
        label_ai: Localised label returned for an AI verdict.
        label_human: Localised label returned for a human verdict.
        calibration_note: Optional provenance or caveat text (external
            thresholds, reproduction tolerance, unpublished corpus, etc.).
    """

    # --- Identity -----------------------------------------------------------
    code: str
    display_name: str

    # --- Model pair ---------------------------------------------------------
    observer_model: str
    performer_model: str

    # --- Calibrated thresholds ----------------------------------------------
    threshold_accuracy: float
    threshold_low_fpr: float
    threshold_tpr_at_fpr_1: float

    # --- Calibration traceability -------------------------------------------
    corpus_sha256: str
    corpus_url: str
    calibration_date: str
    calibration_seed: int

    # --- Loading behaviour --------------------------------------------------
    share_tokenizer_from_observer: bool = True
    trust_remote_code: bool = False

    # --- Localised verdict labels -------------------------------------------
    label_ai: str = "Probablement généré par IA"
    label_human: str = "Probablement écrit par un humain"

    # --- Optional -----------------------------------------------------------
    calibration_note: str | None = None
