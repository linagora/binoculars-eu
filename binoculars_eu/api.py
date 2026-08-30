"""FastAPI application exposing the binoculars-eu detector over HTTP.

Routes (PRD §13.2.1): ``POST /detect``, ``GET /profiles``, ``GET /health``;
Swagger UI is auto-generated at ``/docs``, ReDoc at ``/redoc``, the raw
OpenAPI 3.1 spec at ``/openapi.json``.

Detectors are cached in an LRU cache keyed by ``(profile, mode)`` so model
weights (gigabytes) are not reloaded on every request. The cache is keyed by
``(profile, mode)`` and not by ``profile`` alone because ``mode`` mutates
``self.threshold`` on the instance: sharing an instance across two modes
would introduce a race on the threshold (PRD §13.2.3).
"""

from __future__ import annotations

import time
from functools import lru_cache

from fastapi import FastAPI, HTTPException

from binoculars_eu import __version__
from binoculars_eu.detector import Binoculars
from binoculars_eu.profiles import DEFAULT_PROFILE_CODE, get_profile, list_profiles
from binoculars_eu.schemas import (
    DetectRequest,
    DetectResponse,
    HealthResponse,
    ProfileInfo,
)

app = FastAPI(
    title="binoculars-eu",
    version=__version__,
    description="Détection zero-shot multilingue, plateforme européenne open "
    "source. Un profil de langue = une paire de modèles + des seuils calibrés.",
    license_info={"name": "Apache 2.0"},
)


@lru_cache(maxsize=4)
def get_detector(profile_code: str, mode: str) -> Binoculars:
    """Return one detector per (profile, mode) couple, cached in an LRU cache.

    ``maxsize=4`` bounds the VRAM footprint (PRD §6.9); explicit LRU eviction
    policy is scheduled for V2.
    """
    return Binoculars.for_language(profile_code, mode=mode)


@app.post("/detect", response_model=DetectResponse, tags=["detection"])
def detect(req: DetectRequest) -> DetectResponse:
    """Score a text and return a localised verdict."""
    try:
        get_profile(req.profile)  # validate before loading any weights
    except KeyError:
        available = [p.code for p in list_profiles()]
        raise HTTPException(
            status_code=404,
            detail=f"Profil inconnu : {req.profile!r}. Disponibles : {available}",
        ) from None
    t0 = time.perf_counter()
    try:
        detector = get_detector(req.profile, req.mode)
    except Exception as exc:  # OOM, weights unavailable, etc.
        raise HTTPException(
            status_code=503, detail=f"Chargement du profil impossible : {exc}"
        ) from exc
    result = detector.analyze(req.text)
    return DetectResponse(
        **result, elapsed_ms=int((time.perf_counter() - t0) * 1000)
    )


@app.get("/profiles", response_model=list[ProfileInfo], tags=["profiles"])
def profiles() -> list[ProfileInfo]:
    """List registered profiles with their calibration traceability."""
    return [
        ProfileInfo(
            code=p.code,
            display_name=p.display_name,
            observer_model=p.observer_model,
            performer_model=p.performer_model,
            thresholds={
                "accuracy": p.threshold_accuracy,
                "low_fpr": p.threshold_low_fpr,
                "tpr_at_fpr_1": p.threshold_tpr_at_fpr_1,
            },
            corpus_url=p.corpus_url,
            corpus_sha256=p.corpus_sha256,
            calibration_date=p.calibration_date,
            calibration_seed=p.calibration_seed,
            is_default=(p.code == DEFAULT_PROFILE_CODE),
            calibration_note=p.calibration_note,
        )
        for p in list_profiles()
    ]


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness plus profile and cache inventory."""
    import torch

    return HealthResponse(
        status="ok",
        version=__version__,
        default_profile=DEFAULT_PROFILE_CODE,
        profiles_loaded=[p.code for p in list_profiles()],
        detectors_cached=get_detector.cache_info().currsize,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
