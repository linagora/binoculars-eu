#!/usr/bin/env python3
"""Generate the light OOD corpus: 50 French texts by Mistral Small 24B.

PRD §10.2 / §11.3.1: topics are sampled from the in-distribution human
corpus (comparability), same temperature/top_p as Luciole (0.7/0.9), served
via OpenRouter (key in ``.env`` as ``OPENROUTER_API_KEY``). The OOD corpus
is a *generalisation measure only* — never used to fit thresholds.

Protocol §1 fixes no Mistral seed; we use 700 and document the gap
(TODO protocol-revision).

Usage::

    python -m calibration.generate_ood_mistral \
        --corpus calibration/corpus/binoculars-eu-corpus-fr-v01.jsonl \
        --output calibration/corpus/binoculars-eu-corpus-fr-v01-ood.jsonl
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

MODEL = "mistralai/mistral-small-3.2-24b-instruct"
N_TEXTS = 50
SEED = 700  # protocol §1 has no Mistral seed — TODO(protocol-revision)
TEMPERATURE, TOP_P, MAX_TOKENS = 0.7, 0.9, 280


def load_topics(corpus_path: Path) -> list[dict]:
    topics = []
    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["label"] == "human":
            topics.append({"id": record["id"], "title": record["meta"].get("title", "")})
    return topics


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set (see .env)")

    topics = load_topics(args.corpus)
    step = max(1, len(topics) // N_TEXTS)
    sampled = topics[::step][:N_TEXTS]
    print(f"topics: {len(topics)} human records -> sampled {len(sampled)}")

    records = []
    with httpx.Client(base_url="https://openrouter.ai/api/v1", timeout=180) as client:
        for i, topic in enumerate(sampled, start=1):
            prompt = (f"Écris un paragraphe informatif d'environ 120 mots sur : "
                      f"{topic['title']}. Un seul paragraphe, sans titre ni liste.")
            payload = {
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_tokens": MAX_TOKENS,
                "seed": SEED,
            }
            text = ""
            for attempt in range(4):
                try:
                    resp = client.post(
                        "/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}"},
                        json=payload,
                    )
                    resp.raise_for_status()
                    text = resp.json()["choices"][0]["message"]["content"].split("\n")[0].strip()
                    break
                except (httpx.HTTPError, KeyError, IndexError) as exc:
                    wait = 5 * (attempt + 1)
                    print(f"  {i}: attempt {attempt + 1} failed ({type(exc).__name__}: "
                          f"{str(exc)[:120]}) — retrying in {wait}s", flush=True)
                    time.sleep(wait)
            if not text:
                raise SystemExit(f"generation failed permanently at item {i}")
            records.append({
                "id": f"ai-mistral24b-{i:03d}",
                "text": text,
                "label": "ai",
                "source": "mistral-small-24b-openrouter",
                "meta": {
                    "prompt": prompt,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "seed": SEED,
                    "twin_of": topic["id"],
                    "generator": "mistral-small-24b",
                    "corpus": "ood-v01",
                    "length_words": len(text.split()),
                },
            })
            print(f"  {i}/{len(sampled)} {topic['id']}: {len(text)} chars")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"written: {args.output} ({len(records)} records)")
    print(f"corpus sha256: {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
