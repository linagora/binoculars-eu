#!/usr/bin/env python3
"""Build the FR in-distribution calibration corpus (PRD §10.1).

Human side (250 records): Wikipedia 80, press 60, tech blogs 40, personal
editorial blog 30 (blog.maudet.cloud — LinkedIn strate replacement, owner
agreed), public-domain literature 40.

AI side (250 records): thematic twins of the human texts, generated with
Luciole instruct models at three capacity points, protocol §1 generation
seeds (23B→0, 8B→1, 1B→2), temperature 0.7 / top_p 0.9:

- 100 textes via ``luciole-23b-instruct`` (direct vLLM, default
  http://100.90.203.88:8015/v1) — local nf4 serving, see
  ``scripts/luciole_switch.sh``,
- 75 via ``luciole-8b-instruct`` (direct vLLM, :8013),
- 75 via ``luciole-1b-instruct`` (LiteLLM gateway, key in ``.env``).

The Mistral OOD corpus is generated separately by
``calibration/generate_ood_mistral.py`` (protocol §10.2).

Usage::

    python -m calibration.build_corpus \
        --output calibration/corpus/binoculars-eu-corpus-fr-v01.jsonl

Options ``--vllm-23b``/``--vllm-8b``/``--litellm-url`` override backends;
``--skip-ai`` collects the human side only (idempotent cache in
``calibration/corpus/_human_cache.json``).
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

from calibration import human_sources

# PRD §10.1 composition (LinkedIn strate → blog-maudet, documented).
HUMAN_TARGETS = {
    "wikipedia-fr": 80,
    "presse-fr": 60,
    "linuxfr": 40,
    "blog-maudet": 30,   # blog-pro register
    "litterature": 40,
}

GENERATOR_SHARES = [
    ("luciole-23b-instruct", 100),
    ("luciole-8b-instruct", 75),
    ("luciole-1b-instruct", 75),
]
GENERATION_SEEDS = {"luciole-23b-instruct": 0, "luciole-8b-instruct": 1, "luciole-1b-instruct": 2}
TEMPERATURE, TOP_P, MAX_TOKENS = 0.7, 0.9, 280

PROMPT_STYLES = {
    "wikipedia-fr": "un paragraphe encyclopédique neutre et informatif",
    "litterature": "un paragraphe de prose littéraire descriptive",
    "presse-fr": "le premier paragraphe d'un article de presse informatique",
    "linuxfr": "un billet de blog technique au ton naturel",
    "blog-maudet": "un billet d'opinion personnel au ton naturel et engagé",
}


# --------------------------------------------------------------------------
# Human side
# --------------------------------------------------------------------------
def collect_humans(cache: Path) -> dict[str, list[dict]]:
    """Collect per source, re-fetching only sources still below target."""
    collected: dict[str, list[dict]] = {}
    if cache.exists():
        collected = json.loads(cache.read_text(encoding="utf-8"))
    fetchers = {
        "wikipedia-fr": lambda: human_sources.fetch_wikipedia(HUMAN_TARGETS["wikipedia-fr"]),
        "litterature": lambda: human_sources.fetch_wikisource(HUMAN_TARGETS["litterature"]),
        "presse-fr": lambda: human_sources.fetch_presse(HUMAN_TARGETS["presse-fr"]),
        "linuxfr": lambda: human_sources.fetch_linuxfr(HUMAN_TARGETS["linuxfr"]),
        "blog-maudet": lambda: human_sources.fetch_ghost_blog(
            HUMAN_TARGETS["blog-maudet"]
        ),
    }
    for source, fetch in fetchers.items():
        if len(collected.get(source, [])) < HUMAN_TARGETS[source]:
            collected[source] = fetch()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(collected, ensure_ascii=False, indent=1), encoding="utf-8")
    return collected


def human_records(collected: dict[str, list[dict]], tokenizer) -> list[dict]:
    records = []
    counters: dict[str, int] = {}
    for source, target in HUMAN_TARGETS.items():
        pool = collected.get(source, [])[:target]
        if len(pool) < target:
            print(f"WARNING: source {source!r} delivered {len(pool)}/{target}", file=sys.stderr)
        for item in pool:
            counters[source] = counters.get(source, 0) + 1
            text = item["text"]
            records.append({
                "id": f"human-{source}-{counters[source]:03d}",
                "text": text,
                "label": "human",
                "source": source,
                "meta": {
                    "title": item["title"],
                    "url": item["url"],
                    "length_words": len(text.split()),
                    "length_tokens": len(tokenizer(text, add_special_tokens=False).input_ids),
                    "generator": "human",
                },
            })
    return records


# --------------------------------------------------------------------------
# AI side
# --------------------------------------------------------------------------
def twin_prompt(human: dict) -> str:
    style = PROMPT_STYLES.get(human["source"], "un paragraphe informatif")
    title = human["meta"].get("title", human["source"])
    return (f"Écris {style} d'environ 120 mots sur : {title}. "
            "Un seul paragraphe, sans titre ni liste.")


def chat(base_url: str, api_key: str | None, model: str, prompt: str, seed: int,
         client: httpx.Client) -> str:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "seed": seed,
    }
    resp = client.post(f"{base_url}/chat/completions", json=payload, headers=headers, timeout=180)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return content.split("\n")[0].strip()  # first paragraph only (PRD §10.1)


def generate_ai(
    humans: list[dict], backends: dict[str, str | None], selected: set[str]
) -> list[dict]:
    """Generate the twins for the selected generators (e.g. {"8b"}).

    The GPU serves ONE model profile at a time (scripts/luciole_switch.sh),
    so generation runs per backend; unselected shares are merged from the
    previous corpus file by the caller.
    """
    shares = [(m, s) for m, s in GENERATOR_SHARES if m.split("-")[1] in selected]
    cursor = 0
    assignments: list[tuple[str, dict]] = []
    for model, share in shares:
        assignments.extend((model, h) for h in humans[cursor:cursor + share])
        cursor += share

    records: list[dict] = []
    counters: dict[str, int] = {}
    with httpx.Client() as client:
        for model, human in assignments:
            base = backends.get(model)
            if not base:
                raise SystemExit(f"no backend configured for {model}")
            seed = GENERATION_SEEDS[model]
            short = model.replace("luciole-", "").replace("-instruct", "")
            counters[model] = counters.get(model, 0) + 1
            prompt = twin_prompt(human)
            text = chat(base, _api_key_for(base), model, prompt, seed, client)
            if len(text) < 200:
                text = chat(base, _api_key_for(base), model, prompt, seed + 1000, client)
            records.append({
                "id": f"ai-{short}-{counters[model]:03d}",
                "text": text,
                "label": "ai",
                "source": model,
                "meta": {
                    "prompt": prompt,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "seed": seed,
                    "twin_of": human["id"],
                    "generator": model,
                    "length_words": len(text.split()),
                    "length_tokens": len(_tokenizer()(text, add_special_tokens=False).input_ids),
                },
            })
            time.sleep(0.2)
    return records


def _api_key_for(base: str) -> str | None:
    return os.environ.get("LITELLM_API_KEY") if "4000" in base else None


_tokenizer_cache = None


def _tokenizer():
    global _tokenizer_cache
    if _tokenizer_cache is None:
        from transformers import AutoTokenizer

        _tokenizer_cache = AutoTokenizer.from_pretrained("OpenLLM-France/Luciole-1B-Base")
    return _tokenizer_cache


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-ai", action="store_true", help="human side only")
    parser.add_argument(
        "--generators", default="23b,8b,1b",
        help="comma list of backends to (re)generate: 23b,8b,1b",
    )
    parser.add_argument("--vllm-23b", default="http://100.90.203.88:8015/v1")
    parser.add_argument("--vllm-8b", default="http://100.90.203.88:8013/v1")
    parser.add_argument("--litellm-url", default=None,
                        help="default: LITELLM_BASE_URL from the environment (.env loaded first)")
    return parser.parse_args(argv)


def load_previous_ai(output: Path) -> list[dict]:
    """AI records from a previous corpus file, for merge across runs."""
    if not output.exists():
        return []
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    return [r for r in records if r["label"] == "ai"]


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    dotenv = Path(".env")
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

    collected = collect_humans(args.output.parent / "_human_cache.json")
    humans = human_records(collected, _tokenizer())
    counts = {s: sum(1 for r in humans if r["source"] == s) for s in HUMAN_TARGETS}
    print(f"human records: {len(humans)} ({counts})")

    records = humans
    if not args.skip_ai:
        selected = {g.strip() for g in args.generators.split(",") if g.strip()}
        backends = {
            "luciole-23b-instruct": args.vllm_23b,
            "luciole-8b-instruct": args.vllm_8b,
            "luciole-1b-instruct": args.litellm_url or os.environ.get("LITELLM_BASE_URL") or None,
        }
        ai = generate_ai(humans, backends, selected)
        kept = [r for r in load_previous_ai(args.output)
                if r["source"].split("-")[1] not in selected]
        ai = kept + ai
        counts = {m: sum(1 for r in ai if r["source"] == m) for m, _ in GENERATOR_SHARES}
        print(f"AI records: {len(ai)} ({counts}) — merged with previous runs")
        records = humans + ai

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
