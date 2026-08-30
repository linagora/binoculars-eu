#!/usr/bin/env python3
"""End-to-end sanity check of the binoculars-eu HTTP API.

Contract:
  1. Start the API (uvicorn) in a subprocess on a free local port.
  2. ``GET /profiles`` and verify exactly ``["fr"]``.
  3. ``POST /detect`` on the two input texts and verify the API score matches
     the direct (in-process) Binoculars score within ±0.001.
  4. Warm-call latency: strict < 500 ms (CLI) and < 200 ms GPU target
     (pytest marker ``acceptance_gpu``).

Pytest markers:
  * ``acceptance_cpu`` — numeric and cache checks, must pass on every
    platform (profiles, score match ±0.001, LRU cache evidence).
  * ``acceptance_gpu`` — strict warm-latency target < 200 ms (PRD §14.1,
    L4-class). Runs only when ``torch.cuda.is_available()`` or when
    ``BINOCULARS_EU_HAS_GPU=1`` is set; otherwise skipped.

Usage:
    python scripts/sanity_check_api.py [input.json]   # CLI, PASS/FAIL lines
    pytest scripts/sanity_check_api.py -v             # marked test suite

Input JSON (default: scripts/sanity_check_input.json, overridable with the
BINOCULARS_EU_SANITY_INPUT environment variable)::

    {"texts": ["<text 1, >= 50 chars>", "<text 2>"], "mode": "low-fpr"}

Exit code 0 iff every check passes strictly (CLI) / no test failed (pytest;
a skipped GPU test is not a failure).

Robustness notes:
  * The port is chosen by probing the OS for a free one — a fixed port may
    already be taken by an unrelated service whose ``/health`` would
    false-positive the readiness probe.
  * The readiness probe validates the payload identity (``default_profile``
    field), not just the HTTP status.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

HOST = "127.0.0.1"
HEALTH_TIMEOUT_S = 90
SCORE_TOLERANCE = 0.001
WARM_MAX_S = 0.5        # CLI strict threshold (CPU-inclusive historical target)
GPU_TARGET_S = 0.2      # PRD §14.1 L4-class warm-latency target
SCRIPT_DIR = Path(__file__).parent
DEFAULT_INPUT = SCRIPT_DIR / "sanity_check_input.json"
SERVER_LOG = Path("/tmp/binoculars_eu_sanity_uvicorn.log")


# --------------------------------------------------------------------------
# Environment report (self-descriptive output)
# --------------------------------------------------------------------------
def environment_report() -> str:
    """One-block description of the execution environment."""
    import platform

    import torch
    import transformers

    from binoculars_eu.detector import _resolve_devices

    cuda = torch.cuda.is_available()
    device_line = (
        f"cuda: available={cuda}"
        + (f", {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}" if cuda else "")
    )
    device_1, device_2 = _resolve_devices()
    return "\n".join([
        f"python {platform.python_version()} | torch {torch.__version__} "
        f"| transformers {transformers.__version__}",
        f"platform: {platform.platform()}",
        device_line,
        "detector dtype: torch.bfloat16 (use_bfloat16=True, API default)",
        f"device_map effectif: observer={{'': '{device_1}'}} "
        f"performer={{'': '{device_2}'}} (DEVICE_1/DEVICE_2 or auto)",
        f"gpu acceptance enabled: {gpu_acceptance_enabled()}",
    ])


def gpu_acceptance_enabled() -> bool:
    """GPU marker activation: explicit env var or a live CUDA device."""
    if os.environ.get("BINOCULARS_EU_HAS_GPU") == "1":
        return True
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(
    not gpu_acceptance_enabled(),
    reason="GPU acceptance target: run on a CUDA host or set BINOCULARS_EU_HAS_GPU=1",
)


# --------------------------------------------------------------------------
# Helpers (shared by the CLI entry point and the pytest fixtures)
# --------------------------------------------------------------------------
def load_input(path: Path) -> tuple[list[str], str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    texts = data["texts"]
    if len(texts) != 2 or any(len(t) < 50 for t in texts):
        raise ValueError("input must provide exactly 2 texts of >= 50 chars")
    return texts, data.get("mode", "low-fpr")


def pick_free_port() -> int:
    """Ask the OS for an ephemeral port, then release it for uvicorn."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        return int(s.getsockname()[1])


def start_api(port: int) -> subprocess.Popen[bytes]:
    log = SERVER_LOG.open("wb")
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "binoculars_eu.api:app",
         "--host", HOST, "--port", str(port), "--log-level", "warning"],
        stdout=log, stderr=subprocess.STDOUT,
    )


def wait_for_health(proc: subprocess.Popen[bytes], base: str) -> None:
    """Poll until OUR API answers, validating payload identity.

    A bare 200 is not enough: an unrelated service on the same port may
    expose its own /health and would false-positive the probe.
    """
    deadline = time.monotonic() + HEALTH_TIMEOUT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"uvicorn exited early (see {SERVER_LOG})")
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=2) as r:
                body = json.load(r)
            if isinstance(body, dict) and "default_profile" in body:
                return
        except (urllib.error.URLError, json.JSONDecodeError, OSError):
            pass
        time.sleep(0.5)
    raise TimeoutError(f"API not healthy after {HEALTH_TIMEOUT_S}s")


def get_json(base: str, path: str) -> tuple[int, object]:
    with urllib.request.urlopen(f"{base}{path}", timeout=30) as r:
        return r.status, json.load(r)


def post_detect(base: str, text: str, mode: str) -> tuple[float, dict]:
    payload = json.dumps({"text": text, "mode": mode}).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/detect", data=payload, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            body = json.load(r)
    except urllib.error.HTTPError as exc:  # 404/422/503 → no JSON score
        raise RuntimeError(f"POST /detect -> HTTP {exc.code}: {exc.read()[:200]!r}") from exc
    return time.perf_counter() - t0, body


def check(name: str, ok: bool, detail: str) -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


# --------------------------------------------------------------------------
# Pytest mode: marked acceptance tests
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def api_base():
    port = pick_free_port()
    proc = start_api(port)
    base = f"http://{HOST}:{port}"
    try:
        wait_for_health(proc, base)
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="module")
def payload() -> tuple[list[str], str]:
    path = Path(os.environ.get("BINOCULARS_EU_SANITY_INPUT", DEFAULT_INPUT))
    return load_input(path)


@pytest.fixture(scope="module")
def calls(api_base: str, payload: tuple[list[str], str]) -> dict:
    """The two scoring POSTs (cold then warm) plus one extra warm call."""
    texts, mode = payload
    durations: list[float] = []
    responses: list[dict] = []
    for text in texts:
        elapsed, body = post_detect(api_base, text, mode)
        durations.append(elapsed)
        responses.append(body)
    t3, _ = post_detect(api_base, texts[0], mode)
    return {"durations": durations, "responses": responses, "t3": t3}


@pytest.fixture(scope="module")
def direct_scores(payload: tuple[list[str], str]) -> list[float]:
    from binoculars_eu import Binoculars  # deferred: heavy import

    texts, mode = payload
    detector = Binoculars.for_language("fr", mode=mode)
    return [float(detector.compute_score(t)) for t in texts]


@pytest.fixture(scope="module", autouse=True)
def _environment_header() -> None:
    print("\n=== sanity_check_api environment ===")
    print(environment_report())
    print("====================================\n")


@pytest.mark.acceptance_cpu
def test_profiles_registered(api_base: str) -> None:
    status, profiles = get_json(api_base, "/profiles")
    codes = sorted(p["code"] for p in profiles)
    assert status == 200 and codes == ["fr"], f"GET /profiles -> {status}, codes={codes}"


@pytest.mark.acceptance_cpu
@pytest.mark.parametrize("i", [0, 1])
def test_api_score_matches_direct(i: int, calls: dict, direct_scores: list[float]) -> None:
    delta = abs(calls["responses"][i]["score"] - direct_scores[i])
    assert delta <= SCORE_TOLERANCE, f"text {i + 1}: |delta|={delta}"


@pytest.mark.acceptance_cpu
def test_lru_cache_warm(calls: dict) -> None:
    t1, t2 = calls["durations"][0], calls["durations"][1]
    t3 = calls["t3"]
    assert (t1 - t2) > 1.0, f"no load eliminated: 1st {t1:.2f}s vs 2nd {t2:.2f}s"
    stability = abs(t2 - t3)
    assert stability < 0.3 * max(t2, 1e-9), (
        f"warm calls unstable: 2nd {t2:.2f}s vs 3rd {t3:.2f}s (|d|={stability:.2f}s)"
    )


@pytest.mark.acceptance_gpu
@requires_gpu
def test_warm_latency_gpu_target(calls: dict) -> None:
    t2 = calls["durations"][1]
    assert t2 < GPU_TARGET_S, f"2nd call {t2 * 1000:.0f} ms >= {GPU_TARGET_S * 1000:.0f} ms"


# --------------------------------------------------------------------------
# CLI mode: procedural PASS/FAIL run (same steps, no pytest required)
# --------------------------------------------------------------------------
def main() -> int:
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    texts, mode = load_input(input_path)
    port = pick_free_port()
    base = f"http://{HOST}:{port}"
    results: list[bool] = []

    print("=== sanity_check_api environment ===")
    print(environment_report())
    print("====================================\n")

    # --- 1) start the API in a subprocess ---------------------------------
    proc = start_api(port)
    try:
        wait_for_health(proc, base)
        print(f"[INFO] API healthy on {base} (log: {SERVER_LOG})")

        # --- 2) GET /profiles must be exactly ["fr"] -----------------------
        status, profiles = get_json(base, "/profiles")
        codes = sorted(p["code"] for p in profiles)
        results.append(check(
            "profiles", status == 200 and codes == ["fr"],
            f"GET /profiles -> {status}, codes={codes}",
        ))

        # --- 3) API score vs direct Binoculars score, ±0.001 ---------------
        # The two POSTs below are also the cold (1st) and warm (2nd) API
        # calls: their client-side durations feed check 4.
        from binoculars_eu import Binoculars  # deferred: heavy import

        direct = Binoculars.for_language("fr", mode=mode)
        durations: list[float] = []
        for i, text in enumerate(texts, start=1):
            elapsed, api = post_detect(base, text, mode)
            durations.append(elapsed)
            ref = float(direct.compute_score(text))
            delta = abs(api["score"] - ref)
            results.append(check(
                f"score match (text {i})", delta <= SCORE_TOLERANCE,
                f"api={api['score']:.6f} direct={ref:.6f} |delta|={delta:.6f} "
                f"(tol {SCORE_TOLERANCE})",
            ))

        # --- 4) 2nd /detect call: strict < 500 ms + cache evidence ---------
        t1, t2 = durations[0], durations[1]  # cold (load) vs warm (cached)
        t3, _ = post_detect(base, texts[0], mode)
        strict = t2 < WARM_MAX_S
        evidence = (t1 - t2) > 1.0 and abs(t2 - t3) < 0.3 * max(t2, 1e-9)
        results.append(check(
            "warm latency strict (<500 ms)", strict,
            f"2nd call {t2 * 1000:.0f} ms (1st {t1 * 1000:.0f} ms, "
            f"3rd {t3 * 1000:.0f} ms)",
        ))
        results.append(check(
            "LRU cache evidence", evidence,
            f"load eliminated {(t1 - t2) * 1000:.0f} ms; 2nd/3rd stable "
            f"(|d|={abs(t2 - t3) * 1000:.0f} ms)",
        ))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    ok = all(results)
    print(f"\nRESULT: {'ALL CHECKS PASSED' if ok else 'FAILURES PRESENT'} "
          f"({sum(results)}/{len(results)})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
