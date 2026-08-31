"""Unit test for scripts/nail_test.py (mocked HTTP, no live API)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import nail_test  # noqa: E402


class _StubResponse:
    """Minimal requests.Response stand-in for monkeypatched calls."""

    status_code = 200

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


_SCORES = {"accuracy": 0.8245, "low-fpr": 0.9833}


def _fake_request(method: str, url: str, timeout: float, **kwargs: object) -> _StubResponse:
    """Deterministic stub for GET /health, GET /profiles, POST /detect."""
    if url.endswith("/health"):
        return _StubResponse({"status": "ok", "version": "0.1.0-test"})
    if url.endswith("/profiles"):
        return _StubResponse([
            {
                "code": "fr",
                "observer_model": "observer-x",
                "performer_model": "performer-y",
                "thresholds": {"accuracy": 0.955801, "low_fpr": 0.866667},
            }
        ])
    if url.endswith("/detect"):
        mode = kwargs["json"]["mode"]  # type: ignore[index]
        return _StubResponse({
            "score": _SCORES[mode],
            "verdict": "ai" if mode == "accuracy" else "human",
            "confidence": "high",
            "threshold_used": 0.955801 if mode == "accuracy" else 0.866667,
            "mode": mode,
            "profile": "fr",
            "input_tokens": 512,
            "elapsed_ms": 57,
        })
    raise AssertionError(f"unexpected URL: {url}")


def test_nail_test_produces_three_outputs(tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    """Two dummy texts in, markdown + JSON + CSV out, two measurements each."""
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "alpha.txt").write_text("Texte alpha suffisamment long pour le "
                                      "test de détection, avec assez de mots.",
                                      encoding="utf-8")
    (inputs / "beta.txt").write_text("Texte beta tout aussi long, écrit par un "
                                     "humain pour servir de deuxième témoin ici.",
                                     encoding="utf-8")
    out_md = tmp_path / "table.md"
    out_json = tmp_path / "audit.json"
    out_csv = tmp_path / "audit.csv"

    monkeypatch.setattr(nail_test.requests, "request", _fake_request)

    exit_code = nail_test.main([
        "--inputs-dir", str(inputs),
        "--output-md", str(out_md),
        "--output-json", str(out_json),
        "--output-csv", str(out_csv),
        "--quiet",
    ])

    assert exit_code == 0
    for path in (out_md, out_json, out_csv):
        assert path.exists(), f"missing output: {path}"

    audit = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(audit["measurements"]) == 2
    for measurement in audit["measurements"]:
        assert set(measurement["modes"]) == {"accuracy", "low_fpr"}
    assert audit["measurements"][0]["modes"]["accuracy"]["score"] == 0.8245
    assert audit["test_run"]["thresholds"] == {
        "accuracy": 0.955801, "low_fpr": 0.866667,
    }

    csv_lines = out_csv.read_text(encoding="utf-8").strip().splitlines()
    assert len(csv_lines) == 3  # header + 2 measurements
    assert csv_lines[0].startswith("file,label,category,ground_truth")

    markdown = out_md.read_text(encoding="utf-8")
    assert "| Texte | Origine | Longueur |" in markdown
    assert "0.8245 (marge -0.131)" in markdown
    assert "🤖 IA (high)" in markdown


def test_nail_test_no_inputs_is_exit_2(tmp_path: Path) -> None:
    """An empty inputs directory is a clean input error, not a traceback."""
    empty = tmp_path / "empty"
    empty.mkdir()
    assert nail_test.main(["--inputs-dir", str(empty), "--quiet"]) == 2
