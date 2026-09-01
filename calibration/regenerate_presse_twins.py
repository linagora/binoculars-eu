#!/usr/bin/env python3
"""Regenerate the 20 degenerate presse AI twins for corpus V1.1 (P1.2).

The V1.0 records ``ai-23b-081..100`` were generated with empty twin topics
(the presse source titles were empty), producing the degenerate prompt
"sur : .". After P1.1 the presse records carry real titles; this script
regenerates those 20 twins with the standard ``twin_prompt`` template (the
same one as the 230 wikipedia twins), the original generator
(``luciole-23b-instruct``) and seed (0), against the local vLLM backend.
Output is a patch JSONL preserving the original record ids.

Requires the vLLM 23b profile on the GPU box (scripts/luciole_switch.sh 23b).

Usage::

    python -m calibration.regenerate_presse_twins \
        --corpus calibration/corpus/binoculars-eu-corpus-fr-v01.jsonl \
        --presse-patch calibration/corpus/_presse_v11_patch.jsonl \
        --generator-url http://100.90.203.88:8015/v1 \
        --output calibration/corpus/_twins_presse_v11_patch.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, cast

import httpx

from calibration.audit_corpus import DEGENERATE_TOPIC
from calibration.build_corpus import TEMPERATURE, TOP_P, chat, twin_prompt

JsonDict = dict[str, Any]  # JSON payload; Any is the practical JSON type

_tokenizer_cache = None


def _tokenizer() -> Any:
    global _tokenizer_cache
    if _tokenizer_cache is None:
        from transformers import AutoTokenizer

        _tokenizer_cache = AutoTokenizer.from_pretrained("OpenLLM-France/Luciole-1B-Base")
    return _tokenizer_cache


def load_jsonl(path: Path) -> list[JsonDict]:
    return [cast(JsonDict, json.loads(line)) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True,
                        help="V1.0 corpus (locates the degenerate twin records)")
    parser.add_argument("--presse-patch", type=Path, required=True,
                        help="P1.1 presse patch (humans with real titles)")
    parser.add_argument("--generator-url", default="http://100.90.203.88:8015/v1")
    parser.add_argument("--output", type=Path,
                        default=Path("calibration/corpus/_twins_presse_v11_patch.jsonl"))
    args = parser.parse_args(argv)

    records = load_jsonl(args.corpus)
    degenerate = [r for r in records
                  if r["label"] == "ai"
                  and DEGENERATE_TOPIC.search(str(r.get("meta", {}).get("prompt", "")))]
    if not degenerate:
        raise SystemExit("no degenerate twin records found")
    humans = {str(r["id"]): r for r in load_jsonl(args.presse_patch)}
    print(f"{len(degenerate)} degenerate twins to regenerate", flush=True)

    out: list[JsonDict] = []
    with httpx.Client() as client:
        for i, r in enumerate(degenerate, 1):
            twin_of = str(r["meta"]["twin_of"])
            human = humans.get(twin_of)
            if human is None:
                raise SystemExit(f"patched presse record missing for {twin_of}")
            title = str(human["meta"]["title"])
            if not title.strip():
                raise SystemExit(f"patched presse record still has an empty title: {twin_of}")
            model = str(r["meta"]["generator"])
            seed = int(r["meta"]["seed"])
            prompt = twin_prompt(human)
            text = chat(args.generator_url, None, model, prompt, seed, client)
            if len(text) < 200:
                text = chat(args.generator_url, None, model, prompt, seed + 1000, client)
            out.append({
                "id": str(r["id"]),
                "text": text,
                "label": "ai",
                "source": str(r["source"]),
                "meta": {
                    "prompt": prompt,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "seed": seed,
                    "twin_of": twin_of,
                    "generator": model,
                    "length_words": len(text.split()),
                    "length_tokens": len(_tokenizer()(text, add_special_tokens=False).input_ids),
                },
            })
            print(f"[{i}/{len(degenerate)}] {r['id']} <- {twin_of} ({len(text)} chars)",
                  flush=True)
            time.sleep(0.2)

    args.output.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out),
        encoding="utf-8")
    print(f"wrote {len(out)} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
