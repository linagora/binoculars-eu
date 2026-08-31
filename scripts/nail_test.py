#!/usr/bin/env python3
"""Reproducible editorial benchmark ("test au clou") for binoculars-eu.

Measures a batch of UTF-8 text files against the running API (POST /detect,
modes ``accuracy`` and ``low-fpr``), then emits a markdown table for the
article, a full JSON audit (hashes, thresholds, latencies, timestamps) and a
flat CSV. Thresholds and model identity are discovered dynamically from
``GET /profiles`` / ``GET /health`` so the script replays unchanged on future
versions.

Exit codes: 0 success, 2 input error, 3 API error, 4 output write error.

Usage::

    python scripts/nail_test.py --inputs-dir nail_test_inputs/ \
        --output-md article/section-8-table.md --output-json audit.json \
        --output-csv audit.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests

try:  # rich is optional: pretty progress/summary when available
    from rich.console import Console

    RICH = Console(stderr=True)
    HAS_RICH = True
except ImportError:  # pragma: no cover - environment dependent
    RICH = None
    HAS_RICH = False

MODES = ("accuracy", "low-fpr")
CATEGORIES = {
    "human_reference": "Humain (référence)",
    "ai_pure": "IA brute",
    "ai_humanized": "IA + Undetectable AI",
    "mixed": "Mixte",
    "article_itself": "Article lui-même",
}
GROUND_TRUTHS = {"human", "ai", "unknown"}
MODE_KEY = {"accuracy": "accuracy", "low-fpr": "low_fpr"}  # API mode -> audit key


class ApiError(RuntimeError):
    """HTTP or payload failure while talking to the detector API."""


def log(message: str, quiet: bool = False) -> None:
    """Progress log; stdout stays clean for piping."""
    if quiet:
        return
    if HAS_RICH and RICH is not None:
        RICH.print(message)
    else:
        print(message, file=sys.stderr)


@dataclass
class ModeResult:
    """One POST /detect measurement."""

    score: float
    verdict: str
    confidence: str
    threshold: float
    margin: float
    latency_ms_client: int
    latency_ms_server: int
    tokens_analyzed: int


@dataclass
class Measurement:
    """All data collected for a single input file."""

    file: str
    sha256: str
    label: str
    category: str | None
    ground_truth: str | None
    source: str | None
    note: str | None
    word_count: int
    char_count: int
    modes: dict[str, ModeResult] = field(default_factory=dict)


@dataclass
class TestConfig:
    """Detector identity discovered at runtime (null-tolerant)."""

    api_url: str
    detector_version: str | None
    git_sha: str | None
    profile_version: str | None
    model_id: str | None
    thresholds: dict[str, float]


def http_json(method: str, url: str, timeout: float, **kwargs: Any) -> Any:
    """One HTTP call, raising ApiError with a readable message on failure."""
    try:
        response = requests.request(method, url, timeout=timeout, **kwargs)
    except requests.RequestException as exc:
        raise ApiError(f"{method} {url}: connection error: {exc}") from exc
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise ApiError(f"{method} {url}: HTTP {response.status_code}: {exc}") from exc
    try:
        return response.json()
    except ValueError as exc:
        raise ApiError(f"{method} {url}: invalid JSON payload") from exc


def discover(api_url: str, profile: str, timeout: float) -> TestConfig:
    """Read /health and /profiles; thresholds must come from the API."""
    health = http_json("GET", f"{api_url}/health", timeout)
    profiles = http_json("GET", f"{api_url}/profiles", timeout)
    info = next((p for p in profiles if p.get("code") == profile), None)
    if info is None:
        raise ApiError(f"profile {profile!r} not served by {api_url}/profiles")
    thresholds = info.get("thresholds") or {}
    missing = [m for m in MODES if MODE_KEY[m] not in thresholds]
    if missing:
        raise ApiError(f"profile {profile!r} misses thresholds: {missing}")
    model_id = f"{info.get('observer_model')} + {info.get('performer_model')}"
    return TestConfig(
        api_url=api_url,
        detector_version=health.get("version"),
        git_sha=health.get("git_sha"),
        profile_version=health.get("profile_version"),
        model_id=model_id,
        thresholds={k: float(v) for k, v in thresholds.items()},
    )


def load_meta(inputs: Path, text_file: Path) -> dict[str, Any]:
    """Optional <name>.meta.json sidecar; tolerant of absent or partial meta."""
    meta_path = text_file.with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def measure_file(api_url: str, profile: str, timeout: float,
                 inputs: Path, text_file: Path) -> Measurement:
    """Load one text and run both POST /detect modes against it."""
    text = text_file.read_text(encoding="utf-8").strip()
    meta = load_meta(inputs, text_file)
    category = meta.get("category")
    if category is not None and category not in CATEGORIES:
        category = None
    ground_truth = meta.get("ground_truth")
    if ground_truth is not None and ground_truth not in GROUND_TRUTHS:
        ground_truth = None
    measurement = Measurement(
        file=text_file.name,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        label=str(meta.get("label") or text_file.stem),
        category=category,
        ground_truth=ground_truth,
        source=meta.get("source"),
        note=meta.get("note"),
        word_count=len(text.split()),
        char_count=len(text),
    )
    for mode in MODES:
        started = datetime.now(tz=UTC)
        payload = http_json("POST", f"{api_url}/detect", timeout,
                            json={"text": text, "profile": profile, "mode": mode})
        client_ms = int((datetime.now(tz=UTC) - started).total_seconds() * 1000)
        threshold = float(payload["threshold_used"])
        score = float(payload["score"])
        measurement.modes[MODE_KEY[mode]] = ModeResult(
            score=score,
            verdict=str(payload["verdict"]),
            confidence=str(payload["confidence"]),
            threshold=threshold,
            margin=round(score - threshold, 4),
            latency_ms_client=client_ms,
            latency_ms_server=int(payload.get("elapsed_ms", 0)),
            tokens_analyzed=int(payload.get("input_tokens", 0)),
        )
    return measurement


def verdict_cell(result: ModeResult) -> str:
    """Emoji verdict for the markdown table; near-threshold scores flagged."""
    if abs(result.margin) < 0.01:
        return "⚠️ Limite"
    if result.verdict == "ai":
        return f"🤖 IA ({result.confidence})"
    return f"👤 Humain ({result.confidence})"


def score_cell(result: ModeResult) -> str:
    """Score with 4 decimals and signed margin, e.g. 0.9833 (marge +0.028)."""
    return f"{result.score:.4f} (marge {result.margin:+.3f})"


def render_markdown(measurements: list[Measurement], config: TestConfig,
                    seed: int, profile: str, audit_paths: dict[str, str]) -> str:
    """Article-ready markdown: comparison table + configuration + audit files."""
    lines = [
        "| Texte | Origine | Longueur | Score accuracy | Verdict accuracy "
        "| Score low-fpr | Verdict low-fpr |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in measurements:
        acc, low = m.modes["accuracy"], m.modes["low_fpr"]
        origin = CATEGORIES.get(m.category or "", m.category or "—")
        lines.append(
            f"| {m.label} | {origin} | {m.word_count} mots / {m.char_count} caractères "
            f"| {score_cell(acc)} | {verdict_cell(acc)} "
            f"| {score_cell(low)} | {verdict_cell(low)} |"
        )
    local = datetime.now(tz=ZoneInfo("Europe/Paris")).strftime("%Y-%m-%d %H:%M:%S %Z")
    lines += [
        "",
        "### Configuration du test",
        "",
        f"- Date et heure du test (Europe/Paris) : {local}",
        f"- Détecteur : binoculars-eu {config.detector_version or '?'}"
        f", profil {profile}, git_sha {config.git_sha or 'n/a'}",
        f"- Seuils actifs : accuracy = {config.thresholds['accuracy']}"
        f", low-fpr = {config.thresholds['low_fpr']}",
        f"- Modèles : `{config.model_id}`",
        f"- Graine consignée (metadata, non utilisée côté détecteur) : {seed}",
        "",
        "### Fichiers audit",
        "",
        f"- JSON : `{audit_paths['json']}`",
        f"- CSV : `{audit_paths['csv']}`",
        "",
    ]
    return "\n".join(lines)


def audit_document(measurements: list[Measurement], config: TestConfig,
                   seed: int, profile: str) -> dict[str, Any]:
    """Full audit document: run metadata plus raw per-file measurements."""
    now = datetime.now(tz=UTC)
    return {
        "test_run": {
            "timestamp_utc": now.isoformat().replace("+00:00", "Z"),
            "timestamp_local": now.astimezone(ZoneInfo("Europe/Paris")).isoformat(),
            "seed": seed,
            "api_url": config.api_url,
            "detector": {
                "git_sha": config.git_sha,
                "profile": profile,
                "profile_version": config.profile_version,
                "version": config.detector_version,
                "model_id": config.model_id,
            },
            "thresholds": {
                "accuracy": config.thresholds["accuracy"],
                "low_fpr": config.thresholds["low_fpr"],
            },
        },
        "measurements": [
            {
                "file": m.file,
                "sha256": m.sha256,
                "label": m.label,
                "category": m.category,
                "ground_truth": m.ground_truth,
                "source": m.source,
                "note": m.note,
                "word_count": m.word_count,
                "char_count": m.char_count,
                "modes": {
                    key: vars(result) for key, result in m.modes.items()
                },
            }
            for m in measurements
        ],
    }


def write_csv(measurements: list[Measurement], path: Path) -> None:
    """Flat CSV, one row per text, both modes side by side."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "file", "label", "category", "ground_truth", "word_count",
            "char_count", "sha256", "accuracy_score", "accuracy_verdict",
            "accuracy_margin", "low_fpr_score", "low_fpr_verdict", "low_fpr_margin",
        ])
        for m in measurements:
            acc, low = m.modes["accuracy"], m.modes["low_fpr"]
            writer.writerow([
                m.file, m.label, m.category or "", m.ground_truth or "",
                m.word_count, m.char_count, m.sha256,
                f"{acc.score:.6f}", acc.verdict, f"{acc.margin:.6f}",
                f"{low.score:.6f}", low.verdict, f"{low.margin:.6f}",
            ])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=Path("nail_test_table.md"))
    parser.add_argument("--output-json", type=Path, default=Path("nail_test_audit.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("nail_test_audit.csv"))
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--profile", default="fr")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42,
                        help="recorded in metadata only (detector is deterministic)")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    text_files = sorted(args.inputs_dir.glob("*.txt"))
    if not text_files:
        print(f"error: no .txt files in {args.inputs_dir}", file=sys.stderr)
        return 2
    try:
        log(f"[nail-test] discovering detector at {args.api_url} …", args.quiet)
        config = discover(args.api_url, args.profile, args.timeout)
        measurements: list[Measurement] = []
        for index, text_file in enumerate(text_files, start=1):
            log(f"[nail-test] [{index}/{len(text_files)}] {text_file.name}", args.quiet)
            measurements.append(
                measure_file(args.api_url, args.profile, args.timeout,
                             args.inputs_dir, text_file)
            )
    except ApiError as exc:
        print(f"error: API: {exc}", file=sys.stderr)
        return 3
    except OSError as exc:
        print(f"error: input: {exc}", file=sys.stderr)
        return 2

    audit_paths = {"json": str(args.output_json), "csv": str(args.output_csv)}
    try:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(
            render_markdown(measurements, config, args.seed, args.profile,
                            audit_paths), encoding="utf-8")
        args.output_json.write_text(
            json.dumps(audit_document(measurements, config, args.seed, args.profile),
                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_csv(measurements, args.output_csv)
    except OSError as exc:
        print(f"error: output: {exc}", file=sys.stderr)
        return 4

    summary = (f"[nail-test] {len(measurements)} textes mesurés × {len(MODES)} modes "
               f"→ {args.output_md} | {args.output_json} | {args.output_csv}")
    log(summary, args.quiet)
    if not args.quiet:
        print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
