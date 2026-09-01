"""HTTP API tests with a stubbed detector — no model weights are loaded.

Covers the PRD §13.2 routes and status codes: 200 nominal, 404 unknown
profile, 422 Pydantic constraints (short text, invalid mode, profile
pattern), 503 detector loading failure, plus cache-reuse semantics.
"""

from functools import lru_cache

import pytest
from fastapi.testclient import TestClient

import binoculars_eu.api as api_module
from binoculars_eu.api import app
from binoculars_eu.detector import AnalyzeResult

LONG_TEXT = (
    "Dans le paysage numérique en constante évolution, il est crucial de "
    "tirer parti des synergies pour naviguer dans un écosystème complexe."
)


class StubDetector:
    """Drop-in replacement for Binoculars with a fixed score."""

    def __init__(self, score: float = 0.7) -> None:
        self.calls = 0
        self._score = score

    def analyze(self, text: str) -> AnalyzeResult:
        self.calls += 1
        verdict: AnalyzeResult = {
            "score": self._score,
            "verdict": "ai" if self._score < 0.85 else "human",
            "confidence": "high",
            "label": ("Probablement généré par IA" if self._score < 0.85
                      else "Probablement écrit par un humain"),
            "threshold_used": 0.85,
            "mode": "low-fpr",
            "profile": "fr",
            "input_tokens": len(text.split()),
        }
        return verdict


@pytest.fixture()
def stubbed(monkeypatch: pytest.MonkeyPatch) -> tuple[StubDetector, list[tuple[str, str]]]:
    """Replace get_detector with an LRU-cached stub; record factory calls."""
    stub = StubDetector()
    factory_calls: list[tuple[str, str]] = []

    @lru_cache(maxsize=4)
    def fake_get_detector(profile_code: str, mode: str) -> StubDetector:
        factory_calls.append((profile_code, mode))
        return stub

    monkeypatch.setattr(api_module, "get_detector", fake_get_detector)
    return stub, factory_calls


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def test_detect_nominal(client: TestClient, stubbed: tuple[StubDetector, list]) -> None:
    stub, _ = stubbed
    resp = client.post("/detect", json={"text": LONG_TEXT})
    assert resp.status_code == 200
    body = resp.json()
    assert body["score"] == 0.7
    assert body["verdict"] == "ai"
    assert body["label"] == "Probablement généré par IA"
    assert body["profile"] == "fr"
    assert body["input_tokens"] > 0
    assert body["elapsed_ms"] >= 0
    assert stub.calls == 1


def test_detect_cache_reuse(client: TestClient, stubbed: tuple[StubDetector, list]) -> None:
    _, factory_calls = stubbed
    for _ in range(2):
        resp = client.post("/detect", json={"text": LONG_TEXT, "mode": "accuracy"})
        assert resp.status_code == 200
    assert factory_calls == [("fr", "accuracy")]  # one factory call for two requests


def test_unknown_profile_404(client: TestClient, stubbed: tuple[StubDetector, list]) -> None:
    resp = client.post("/detect", json={"text": LONG_TEXT, "profile": "de"})
    assert resp.status_code == 404
    assert "fr" in resp.json()["detail"]


def test_short_text_422(client: TestClient, stubbed: tuple[StubDetector, list]) -> None:
    resp = client.post("/detect", json={"text": "trop court"})
    assert resp.status_code == 422


def test_invalid_mode_422(client: TestClient, stubbed: tuple[StubDetector, list]) -> None:
    resp = client.post("/detect", json={"text": LONG_TEXT, "mode": "bogus"})
    assert resp.status_code == 422


def test_profile_pattern_422(client: TestClient, stubbed: tuple[StubDetector, list]) -> None:
    resp = client.post("/detect", json={"text": LONG_TEXT, "profile": "FR!!"})
    assert resp.status_code == 422


def test_detector_load_failure_503(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(profile_code: str, mode: str) -> StubDetector:
        raise RuntimeError("simulated OOM")

    monkeypatch.setattr(api_module, "get_detector", boom)
    resp = client.post("/detect", json={"text": LONG_TEXT})
    assert resp.status_code == 503
    assert "simulated OOM" in resp.json()["detail"]


def test_profiles_route(client: TestClient) -> None:
    resp = client.get("/profiles")
    assert resp.status_code == 200
    body = resp.json()
    assert sorted(p["code"] for p in body) == ["fr", "fr-8b"]
    profile = next(p for p in body if p["code"] == "fr-8b")
    assert profile["is_default"] is True
    assert set(profile["thresholds"]) == {"accuracy", "low_fpr", "tpr_at_fpr_1"}
    assert profile["observer_model"] == "OpenLLM-France/Luciole-8B-Base"
    fr_profile = next(p for p in body if p["code"] == "fr")
    assert fr_profile["is_default"] is False
    assert fr_profile["observer_model"] == "OpenLLM-France/Luciole-1B-Base"


def test_health_route(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["default_profile"] == "fr-8b"
    assert sorted(body["profiles_loaded"]) == ["fr", "fr-8b"]
    assert body["device"] in ("cpu", "cuda:0")
    assert body["detectors_cached"] >= 0


def test_openapi_schema_exposed(client: TestClient) -> None:
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    paths = resp.json()["paths"]
    assert {"/detect", "/profiles", "/health"} <= set(paths)
