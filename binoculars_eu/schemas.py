"""Pydantic schemas for the binoculars-eu HTTP API (PRD §13.2.2).

The constraints are functional, not decorative: ``min_length=50`` blocks in
422 texts too short for a reliable Binoculars score (PRD §18.12), and the
``profile`` pattern stops arbitrary strings from reaching the registry.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

MODE = Literal["accuracy", "low-fpr", "tpr-at-fpr-1"]


class DetectRequest(BaseModel):
    """Request body of ``POST /detect``."""

    text: str = Field(
        ...,
        min_length=50,
        max_length=20_000,
        description="Text to analyse. Below ~50 characters the Binoculars "
        "score is not reliable (PRD §18.12).",
        examples=[
            "Dans le paysage numérique en constante évolution, il est "
            "crucial de tirer parti des synergies."
        ],
    )
    profile: str = Field(
        default="fr",
        min_length=2,
        max_length=8,
        pattern=r"^[a-z]{2}(-[a-z0-9]{2,5})?$",
        description="Language profile code. Default: 'fr'. "
        "See GET /profiles for the available list.",
    )
    mode: MODE = Field(
        default="low-fpr",
        description="Threshold applied. 'low-fpr' minimises false positives: "
        "recommended in educational contexts.",
    )


class DetectResponse(BaseModel):
    """Response body of ``POST /detect``."""

    score: float = Field(..., description="PPL / X-PPL ratio. Low = probably AI.")
    verdict: Literal["ai", "human"]
    label: str = Field(..., description="Localised label from the profile.")
    confidence: Literal["low", "medium", "high"]
    threshold_used: float
    mode: str
    profile: str
    input_tokens: int = Field(..., ge=1)
    elapsed_ms: int = Field(..., ge=0)


class ProfileInfo(BaseModel):
    """Public traceability of a registered language profile."""

    code: str
    display_name: str
    observer_model: str
    performer_model: str
    thresholds: dict[str, float]
    corpus_url: str
    corpus_sha256: str
    calibration_date: str
    calibration_seed: int
    is_default: bool
    calibration_note: str | None = None


class HealthResponse(BaseModel):
    """Liveness and inventory of the API process."""

    status: Literal["ok", "degraded"]
    version: str
    default_profile: str
    profiles_loaded: list[str]
    detectors_cached: int
    device: str
