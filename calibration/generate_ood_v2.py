#!/usr/bin/env python3
"""Generate the extended OOD corpus v2 (PRD §14.2 amendment B, docs/v02_plan.md P3.1).

Four families, 30 texts each, topics sampled from the in-distribution human
corpus (comparability), same sampling parameters as Luciole (0.7/0.9):
luciole-8b (local vLLM), gpt-4o and claude (OpenRouter), and deterministic
human/AI sentence-alternation hybrids. Generalisation measure only: never
used to fit thresholds. The commercial-humanizer family (P3.2) is produced
separately from the 90 AI sources of this corpus.

Usage::

    OPENROUTER_API_KEY=… python -m calibration.generate_ood_v2 \
        --corpus calibration/corpus/binoculars-eu-corpus-fr-v01.jsonl \
        --generator-url http://100.90.203.88:8013/v1 \
        --output calibration/corpus/binoculars-eu-corpus-fr-v02-ood.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import httpx

GPT4O_MODEL = "openai/gpt-4o"
CLAUDE_MODEL = "anthropic/claude-sonnet-4.5"
LUCIOLE_MODEL = "luciole-8b-instruct"
PER_FAMILY = 30
SEED_BASE = 800  # OOD v2 family seeds: 800 + family offset (protocol §1 has no OOD seed)
TEMPERATURE, TOP_P, MAX_TOKENS = 0.7, 0.9, 280
SENTENCE_END = (".", "!", "?")


def load_topics(corpus_path: Path) -> list[dict]:
    """Human records with a title, in corpus order (deterministic sampling)."""
    topics = []
    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["label"] == "human" and record["meta"].get("title"):
            topics.append({"id": record["id"], "title": record["meta"]["title"]})
    return topics


def sample_topics(topics: list[dict], offset: int) -> list[dict]:
    """Deterministic stride sampling: 30 topics per family, distinct windows."""
    step = max(1, len(topics) // PER_FAMILY)
    start = (offset * PER_FAMILY) % max(len(topics) - step, 1)
    window = topics[start:] + topics[:start]
    return window[::step][:PER_FAMILY]


def generate_one(client: httpx.Client, base_url: str, model: str, api_key: str,
                 prompt: str, seed: int) -> str:
    """One chat completion with retries; first paragraph line only."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "seed": seed,
    }
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    for attempt in range(4):
        try:
            resp = client.post(f"{base_url}/chat/completions", headers=headers,
                               json=payload)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].split("\n")[0].strip()
        except (httpx.HTTPError, KeyError, IndexError) as exc:
            wait = 5 * (attempt + 1)
            print(f"    attempt {attempt + 1} failed ({type(exc).__name__}) — "
                  f"retry in {wait}s", flush=True)
            time.sleep(wait)
    raise SystemExit(f"generation failed permanently for model {model}")


def sentence_split(text: str) -> list[str]:
    """Minimal sentence split on terminal punctuation (same rule as R-2 kit)."""
    sentences, current = [], []
    for char in text:
        current.append(char)
        if char in SENTENCE_END:
            sentences.append("".join(current).strip())
            current = []
    if current:
        sentences.append("".join(current).strip())
    return [s for s in sentences if s]


def hybrid_text(human_text: str, ai_text: str) -> str:
    """Deterministic human/AI sentence alternation (RAID-style mix)."""
    human_sents = sentence_split(human_text)
    ai_sents = sentence_split(ai_text)
    mixed: list[str] = []
    for pair in zip(human_sents, ai_sents, strict=False):
        mixed.extend(pair)
    return " ".join(mixed)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--generator-url", default="http://100.90.203.88:8013/v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    topics = load_topics(args.corpus)
    by_id = {}
    for line in args.corpus.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            by_id[record["id"]] = record

    families = [
        ("luciole-8b", args.generator_url, LUCIOLE_MODEL, "", 0),
        ("gpt-4o", "https://openrouter.ai/api/v1", GPT4O_MODEL, openrouter_key, 1),
        ("claude", "https://openrouter.ai/api/v1", CLAUDE_MODEL, openrouter_key, 2),
    ]
    records: list[dict] = []
    with httpx.Client(timeout=180) as client:
        for family, base_url, model, key, offset in families:
            sampled = sample_topics(topics, offset)
            print(f"[{family}] {len(sampled)} topics", flush=True)
            for i, topic in enumerate(sampled, start=1):
                prompt = (f"Écris un paragraphe informatif d'environ 120 mots sur : "
                          f"{topic['title']}. Un seul paragraphe, sans titre ni liste.")
                text = generate_one(client, base_url, model, key, prompt,
                                    SEED_BASE + offset)
                records.append({
                    "id": f"ai-ood2-{family}-{i:03d}",
                    "text": text,
                    "label": "ai",
                    "source": f"ood2-{family}",
                    "meta": {
                        "prompt": prompt, "temperature": TEMPERATURE, "top_p": TOP_P,
                        "seed": SEED_BASE + offset, "twin_of": topic["id"],
                        "generator": model, "corpus": "ood-v02",
                        "length_words": len(text.split()),
                    },
                })
                print(f"  {family} {i}/{len(sampled)}: {len(text)} chars", flush=True)

    # Hybrids: deterministic alternation from 30 human/AI twin pairs.
    ai_twins = [r for r in by_id.values()
                if r["label"] == "ai" and r["meta"].get("twin_of") in by_id]
    step = max(1, len(ai_twins) // PER_FAMILY)
    for i, twin in enumerate(ai_twins[::step][:PER_FAMILY], start=1):
        human = by_id[twin["meta"]["twin_of"]]
        records.append({
            "id": f"mixed-ood2-hybrid-{i:03d}",
            "text": hybrid_text(human["text"], twin["text"]),
            "label": "ai",
            "source": "ood2-hybrid",
            "meta": {
                "twin_of": human["id"], "generator": "human-ai-alternation",
                "corpus": "ood-v02", "seed": SEED_BASE + 3,
                "length_words": len(hybrid_text(human["text"], twin["text"]).split()),
            },
        })
    print(f"[hybrid] {PER_FAMILY} deterministic mixes", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"written: {args.output} ({len(records)} records, sha256 {digest[:16]}…)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
