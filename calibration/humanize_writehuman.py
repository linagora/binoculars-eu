#!/usr/bin/env python3
"""Build the WriteHuman branch of the commercial-humanizer family (P3.2, amendment B).

Humanizes the same source subset as calibration.humanize_undetectable (OOD v2 AI
families, --per-family cap) through the WriteHuman API (POST /v1/humanize,
synchronous, Bearer auth). French is passed explicitly via the `language`
parameter (40+ supported languages). Standard plan billing is per input word
against the monthly allowance; a completed source id is cached so a re-run
never bills twice.

Never commit the API key; it is read from the WRITEHUMAN_API_KEY environment
variable. Docs: https://writehuman.ai/api/docs.

Usage::

    WRITEHUMAN_API_KEY=… python -m calibration.humanize_writehuman \
        --corpus calibration/corpus/binoculars-eu-corpus-fr-v02-ood.jsonl \
        --output calibration/corpus/binoculars-eu-corpus-fr-v02-ood-humanized-writehuman.jsonl \
        --per-family 23
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
API_URL = "https://api.writehuman.ai/v1/humanize"
LANGUAGE = "French"
SOURCE_FAMILIES = ("ood2-luciole-8b", "ood2-gpt-4o", "ood2-claude")
MAX_WORKERS = 3
RETRYABLE = (429, 500, 502, 503)

Cache = dict[str, JsonDict]


def load_cache(path: Path) -> Cache:
    """Progress cache {source_id: record}; empty dict if absent."""
    if path.exists():
        return cast(Cache, json.loads(path.read_text(encoding="utf-8")))
    return {}


def save_cache(path: Path, cache: Cache) -> None:
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8")


def humanize(client: httpx.Client, api_key: str, content: str) -> JsonDict:
    """One synchronous humanization call with bounded retries."""
    last_error = ""
    for attempt in range(4):
        resp = client.post(
            API_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"text": content, "language": LANGUAGE},
            timeout=120.0,
        )
        if resp.status_code in RETRYABLE and attempt < 3:
            time.sleep(10.0 * (attempt + 1))
            continue
        if resp.status_code == 402:
            raise RuntimeError("WriteHuman: insufficient word balance (402)")
        resp.raise_for_status()
        return cast(JsonDict, resp.json())
    raise RuntimeError(f"WriteHuman retry budget exhausted: {last_error}")


def humanize_one(client: httpx.Client, api_key: str, record: JsonDict,
                 cache: Cache, cache_path: Path) -> JsonDict:
    """Humanize a single source record, reusing a cached result if present."""
    source_id = str(record["id"])
    if source_id in cache:
        return cache[source_id]
    doc = humanize(client, api_key, str(record["text"]))
    results = cast(list[str], doc.get("results", []))
    if not results or len(results[0].strip()) < 50:
        raise RuntimeError(f"WriteHuman: empty/short result for {source_id}")
    output = results[0].strip()
    family = str(record["source"]).removeprefix("ood2-")
    out: JsonDict = {
        "id": f"ai-ood2-hum-writehuman-{family}-{source_id.rsplit('-', 1)[-1]}",
        "text": output,
        "label": "ai",
        "source": f"humanized-writehuman-{family}",
        "meta": {
            "source_id": source_id,
            "humanizer": "writehuman",
            "language": LANGUAGE,
            "request_id": str(doc.get("id", "")),
            "input_words": doc.get("input_words", 0),
            "words_remaining": doc.get("words_remaining", {}),
            "source_chars": len(str(record["text"])),
            "output_chars": len(output),
        },
    }
    cache[source_id] = out
    save_cache(cache_path, cache)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True,
                        help="OOD v2 corpus JSONL (sources of the humanized texts)")
    parser.add_argument("--output", type=Path, required=True,
                        help="humanized corpus JSONL")
    parser.add_argument("--cache", type=Path,
                        default=Path("calibration/corpus/_humanize_writehuman_cache.json"))
    parser.add_argument("--per-family", type=int, default=0,
                        help="cap texts per family (0 = all); billing per input word")
    args = parser.parse_args(argv)

    api_key = os.environ.get("WRITEHUMAN_API_KEY", "")
    if not api_key:
        sys.exit("WRITEHUMAN_API_KEY is not set")

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
    return 0 if len(ordered) == len(sources) else 1


if __name__ == "__main__":
    raise SystemExit(main())
