#!/usr/bin/env python3
"""Build the commercial-humanizer family of the OOD v2 corpus (P3.2, amendment B).

Humanizes the 90 AI source texts of the OOD v2 corpus (30 luciole-8b, 30 GPT-4o,
30 Claude) through the Undetectable AI Humanization API v2, with the same
parameters as the V0.1 nail test (web UI, 2026-08-31): readability University,
purpose General Writing, strength Balanced. Model v2 is used because it is the
multilingual model (v11/v11sr are English-first); the web UI default for French
maps to v2. Rate: 0.1 credit per word, submitted documents are cached by source
id so a re-run never pays twice for the same text.

Costs real money from the operator's Undetectable AI credit balance: run only
with explicit approval (docs/v02_plan.md P3.2). Never commit the API key; it is
read from the UNDETECTABLE_API_KEY environment variable.

Usage::

    UNDETECTABLE_API_KEY=… python -m calibration.humanize_undetectable \
        --corpus calibration/corpus/binoculars-eu-corpus-fr-v02-ood.jsonl \
        --output calibration/corpus/binoculars-eu-corpus-fr-v02-ood-humanized.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

import httpx

JsonDict = dict[str, Any]  # JSON payload; Any is the practical JSON type

API_BASE = "https://humanize.undetectable.ai"
SOURCE_FAMILIES = ("ood2-luciole-8b", "ood2-gpt-4o", "ood2-claude")
READABILITY = "University"
PURPOSE = "General Writing"
STRENGTH = "Balanced"
MODEL = "v2"  # multilingual; v11/v11sr are English-first (API docs)
POLL_SECONDS = 6.0
POLL_TIMEOUT = 300.0
MAX_WORKERS = 3
RETRYABLE = (429, 500, 502, 503)

Cache = dict[str, JsonDict]


def load_cache(path: Path) -> Cache:
    """Progress cache {source_id: {doc_id, status}}; empty dict if absent."""
    if path.exists():
        return cast(Cache, json.loads(path.read_text(encoding="utf-8")))
    return {}


def save_cache(path: Path, cache: Cache) -> None:
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8")


def submit(client: httpx.Client, api_key: str, content: str) -> str:
    """Submit one text; returns the document id."""
    resp = client.post(
        f"{API_BASE}/submit",
        headers={"apikey": api_key},
        json={"content": content, "readability": READABILITY,
              "purpose": PURPOSE, "strength": STRENGTH, "model": MODEL},
        timeout=60.0,
    )
    if resp.status_code == 402:
        raise RuntimeError("Undetectable AI: insufficient credits (402)")
    resp.raise_for_status()
    doc_id = resp.json().get("id")
    if not doc_id:
        raise RuntimeError(f"submit returned no id: {resp.text[:200]}")
    return str(doc_id)


def poll_document(client: httpx.Client, api_key: str, doc_id: str) -> JsonDict:
    """Poll /document until an output exists; returns the document object."""
    deadline = time.monotonic() + POLL_TIMEOUT
    while time.monotonic() < deadline:
        resp = client.post(f"{API_BASE}/document",
                           headers={"apikey": api_key},
                           json={"id": doc_id}, timeout=60.0)
        if resp.status_code in RETRYABLE:
            time.sleep(POLL_SECONDS * 2)
            continue
        resp.raise_for_status()
        doc = cast(JsonDict, resp.json())
        if doc.get("output"):
            return doc
        time.sleep(POLL_SECONDS)
    raise TimeoutError(f"document {doc_id}: no output after {POLL_TIMEOUT:.0f}s")


def humanize_one(client: httpx.Client, api_key: str, record: JsonDict,
                 cache: Cache, cache_path: Path) -> JsonDict:
    """Humanize a single source record, reusing a cached submission if present."""
    source_id = str(record["id"])
    if source_id in cache and cache[source_id].get("status") == "done":
        return cast(JsonDict, cache[source_id]["record"])
    doc_id = str(cache.get(source_id, {}).get("doc_id", "")) or None
    if not doc_id:
        for attempt in range(4):
            try:
                doc_id = submit(client, api_key, str(record["text"]))
                break
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in RETRYABLE and attempt < 3:
                    time.sleep(10.0 * (attempt + 1))
                    continue
                raise
        cache[source_id] = {"doc_id": doc_id, "status": "pending"}
        save_cache(cache_path, cache)
    assert doc_id is not None
    doc = poll_document(client, api_key, doc_id)
    output = str(doc["output"]).strip()
    if len(output) < 50:
        raise RuntimeError(f"document {doc_id}: suspiciously short output")
    family = str(record["source"]).removeprefix("ood2-")
    out: JsonDict = {
        "id": f"ai-ood2-hum-undetectable-{family}-{source_id.rsplit('-', 1)[-1]}",
        "text": output,
        "label": "ai",
        "source": f"humanized-undetectable-{family}",
        "meta": {
            "source_id": source_id,
            "humanizer": "undetectable-ai",
            "model": MODEL,
            "readability": READABILITY,
            "purpose": PURPOSE,
            "strength": STRENGTH,
            "doc_id": doc_id,
            "createdDate": str(doc.get("createdDate", "")),
            "source_chars": len(str(record["text"])),
            "output_chars": len(output),
        },
    }
    cache[source_id] = {"doc_id": doc_id, "status": "done", "record": out}
    save_cache(cache_path, cache)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True,
                        help="OOD v2 corpus JSONL (sources of the humanized texts)")
    parser.add_argument("--output", type=Path, required=True,
                        help="humanized corpus JSONL")
    parser.add_argument("--cache", type=Path,
                        default=Path("calibration/corpus/_humanize_undetectable_cache.json"))
    parser.add_argument("--per-family", type=int, default=0,
                        help="cap texts per family (0 = all); credits are ~1/word")
    args = parser.parse_args(argv)

    api_key = os.environ.get("UNDETECTABLE_API_KEY", "")
    if not api_key:
        sys.exit("UNDETECTABLE_API_KEY is not set")

    sources = [cast(JsonDict, json.loads(line)) for line in args.corpus.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    sources = [r for r in sources if r.get("source") in SOURCE_FAMILIES
               and r.get("label") == "ai"]
    if args.per_family > 0:
        capped = []
        for fam in SOURCE_FAMILIES:
            capped.extend([r for r in sources if r.get("source") == fam][:args.per_family])
        sources = capped
    print(f"{len(sources)} source texts to humanize "
          f"({', '.join(SOURCE_FAMILIES)})", flush=True)

    cache = load_cache(args.cache)
    results: dict[str, JsonDict] = {}
    with httpx.Client() as client, ThreadPoolExecutor(MAX_WORKERS) as pool:
        futures = {pool.submit(humanize_one, client, api_key, r, cache,
                               args.cache): r for r in sources}
        done = 0
        for fut in as_completed(futures):
            record = futures[fut]
            try:
                out = fut.result()
            except Exception as exc:  # noqa: BLE001 - report and continue batch
                print(f"ERROR {record['id']}: {exc}", flush=True)
                continue
            results[str(record["id"])] = out
            done += 1
            meta = cast(JsonDict, out["meta"])
            print(f"[{done}/{len(sources)}] {record['id']} -> {out['id']} "
                  f"({meta['output_chars']} chars)", flush=True)

    ordered = [results[str(r["id"])] for r in sources if str(r["id"]) in results]
    args.output.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered),
        encoding="utf-8")
    print(f"wrote {len(ordered)}/{len(sources)} records to {args.output}")
    if len(ordered) < len(sources):
        print("some sources failed; re-run to retry (cache preserves submissions)",
              flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
