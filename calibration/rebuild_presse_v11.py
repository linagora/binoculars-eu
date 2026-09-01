#!/usr/bin/env python3
"""Rebuild the 60 corrupted human-presse-fr records for corpus V1.1 (P1.1).

Every V1.0 presse record carries mojibake (U+FFFD from forced UTF-8 decoding,
now fixed in ``human_sources.get``) and an empty title. This script re-fetches
each original URL, extracts the real title (h1, then og:title) and the
trimmed article text, validates the result (no mojibake, >= MIN_CHARS), and
emits a patch JSONL keeping the original record ids. URLs that fail are
substituted with fresh articles from the same press listings, flagged in meta
(``v11_substituted``) with the original URL kept for traceability.

Usage::

    python -m calibration.rebuild_presse_v11 \
        --corpus calibration/corpus/binoculars-eu-corpus-fr-v01.jsonl \
        --output calibration/corpus/_presse_v11_patch.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any, cast

from calibration.audit_corpus import MOJIBAKE
from calibration.human_sources import MIN_CHARS, fetch_presse, get, html_to_text, trim

JsonDict = dict[str, Any]  # JSON payload; Any is the practical JSON type

OG_TITLE_RE = re.compile(
    r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']', re.I)
OG_TITLE_RE_ALT = re.compile(
    r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:title["\']', re.I)

_tokenizer_cache = None


def _tokenizer() -> Any:
    global _tokenizer_cache
    if _tokenizer_cache is None:
        from transformers import AutoTokenizer

        _tokenizer_cache = AutoTokenizer.from_pretrained("OpenLLM-France/Luciole-1B-Base")
    return _tokenizer_cache


def title_from_url(url: str) -> str:
    """Reconstruct a readable title from the article URL slug.

    Le Monde Informatique URLs carry the title as a slug
    (``lire-conference-<slug>-<id>.html``); their h1 is often breadcrumb
    chrome, so the slug is more reliable than the markup.
    """
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    slug = re.sub(r"-\d+\.html$", "", slug)
    slug = re.sub(r"^lire-(conference-)?", "", slug)
    slug = urllib.parse.unquote(slug).replace("|", " ")
    title = re.sub(r"\s+", " ", slug.replace("-", " ")).strip()
    return title[:1].upper() + title[1:]


def extract_title(page: str, url: str) -> str:
    """Article title: og:title first, then the URL slug (see title_from_url)."""
    m = OG_TITLE_RE.search(page) or OG_TITLE_RE_ALT.search(page)
    if m:
        return m.group(1).strip()
    return title_from_url(url)


def fetch_record(url: str) -> JsonDict | None:
    """Fetch and trim one article; None if unavailable or too short."""
    try:
        page = get(url)
    except Exception:
        return None
    text = trim(html_to_text(page))
    if text is None or len(text) < MIN_CHARS or MOJIBAKE.search(text):
        return None
    return {"title": extract_title(page, url), "url": url, "text": text}


def build_record(record_id: str, fetched: JsonDict,
                 substituted_from: str | None = None) -> JsonDict:
    """V1.1 presse record, same schema as V1.0 with real title and lengths."""
    text = str(fetched["text"])
    meta: JsonDict = {
        "title": fetched["title"],
        "url": fetched["url"],
        "length_words": len(text.split()),
        "length_tokens": len(_tokenizer()(text, add_special_tokens=False).input_ids),
        "generator": "human",
    }
    if substituted_from is not None:
        meta["v11_substituted"] = True
        meta["v11_original_url"] = substituted_from
    return {"id": record_id, "text": text, "label": "human",
            "source": "presse-fr", "meta": meta}


def substitute(harvested: list[JsonDict], used_urls: set[str],
               needed: int) -> list[JsonDict]:
    """Fresh presse articles for URLs that could not be recovered."""
    fresh: list[JsonDict] = []
    for rec in harvested:
        url = str(rec["url"])
        if url in used_urls:
            continue
        text = str(rec["text"])
        if MOJIBAKE.search(text):
            continue
        fresh.append(rec)
        used_urls.add(url)
        if len(fresh) >= needed:
            break
    return fresh


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("calibration/corpus/_presse_v11_patch.jsonl"))
    args = parser.parse_args(argv)

    records = [cast(JsonDict, json.loads(line)) for line in args.corpus.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    presse = [r for r in records if str(r["id"]).startswith("human-presse-fr-")]
    print(f"{len(presse)} presse records to rebuild", flush=True)

    used_urls: set[str] = set()
    out: list[JsonDict] = []
    failed: list[JsonDict] = []
    for i, r in enumerate(presse, 1):
        url = str(r.get("meta", {}).get("url", ""))
        fetched = fetch_record(url) if url else None
        if fetched is not None:
            used_urls.add(url)
            out.append(build_record(str(r["id"]), fetched))
            status = "ok"
        else:
            failed.append(r)
            status = "FETCH_FAILED"
        print(f"[{i}/{len(presse)}] {r['id']} {status}", flush=True)
        time.sleep(0.4)

    if failed:
        print(f"{len(failed)} URLs failed; harvesting substitutes", flush=True)
        harvested = fetch_presse(len(failed) * 2 + 5)
        fresh = substitute(harvested, used_urls, len(failed))
        if len(fresh) < len(failed):
            sys.exit(f"only {len(fresh)} substitutes for {len(failed)} failures")
        for r, f in zip(failed, fresh, strict=True):
            out.append(build_record(str(r["id"]), f,
                                    substituted_from=str(r.get("meta", {}).get("url", ""))))

    out.sort(key=lambda r: str(r["id"]))
    args.output.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out),
        encoding="utf-8")
    n_sub = sum(1 for r in out if r["meta"].get("v11_substituted"))
    print(f"wrote {len(out)} records ({n_sub} substituted) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
