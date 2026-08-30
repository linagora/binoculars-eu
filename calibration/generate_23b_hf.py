#!/usr/bin/env python3
"""Generate the 23B twin share with transformers + bitsandbytes nf4 (plan C).

Why this exists: vLLM 0.28 dropped the ``bitsandbytes`` quantizer, and its
``torchao`` frontend expects a serialized-checkpoint config schema — serving
the 23B in 4-bit through vLLM would require an offline-quantized checkpoint
we do not have. This fallback produces the **same distribution** (nf4
weights, bf16 compute) with the pinned eval stack (transformers 4.57.1,
bitsandbytes 0.44.1) at ~15-30 tok/s on the L4, which is acceptable for one
hundred ~220-token paragraphs.

Run on gpu-ubuntu (weights come from the HF cache, token already in place)::

    ~/.venv-binoculars-eu/bin/python -m calibration.generate_23b_hf \
        --corpus calibration/corpus/binoculars-eu-corpus-fr-v01.jsonl

Assigns the first 100 human records (same split of shares as
``build_corpus.py``), reuses its ``twin_prompt``/styles, applies the protocol
§1 generation seed for the 23B (0), temperature 0.7 / top_p 0.9, keeps the
first paragraph, then merges ``ai-23b-*`` records into the corpus file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

from calibration.build_corpus import (
    GENERATION_SEEDS,
    MAX_TOKENS,
    TEMPERATURE,
    TOP_P,
    twin_prompt,
)

MODEL = "OpenLLM-France/Luciole-23B-Instruct-1.1"
LENGTH_TOKENIZER = "OpenLLM-France/Luciole-1B-Base"  # same counter as build_corpus
SHARE_23B = 100
MIN_CHARS = 200


def load_corpus(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


PRESSE_OVERRIDE = (
    "Tu es journaliste à la rédaction. Rédige le chapeau (premier paragraphe) "
    "d'un article de presse sur : {title}. Un seul paragraphe, ton journalistique, "
    "sans titre ni liste."
)


def override_prompt(human: dict) -> str:
    title = human["meta"].get("title", human["source"])
    return PRESSE_OVERRIDE.format(title=title)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--ids", default=None,
                        help="1-based inclusive range to regenerate, e.g. 81-100")
    parser.add_argument("--style-override", action="store_true",
                        help="journalist chapeau prompt (press register fix)")
    args = parser.parse_args(argv)

    from transformers import (  # deferred: heavy import
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    corpus = load_corpus(args.corpus)
    humans = [r for r in corpus if r["label"] == "human"]
    if args.ids:
        start, end = (int(x) for x in args.ids.split("-"))
        twins = humans[start - 1:end]
        base_index = start
    else:
        twins = humans[:SHARE_23B]
        base_index = 1
    assert twins, "no twins selected for regeneration"

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print("loading 23B in nf4 (first run: quantizing ~46 GB bf16 → ~13 GB)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quant, device_map="cuda:0"
    )
    model.eval()
    chat_tok = AutoTokenizer.from_pretrained(args.model)
    len_tok = AutoTokenizer.from_pretrained(LENGTH_TOKENIZER)

    seed = GENERATION_SEEDS["luciole-23b-instruct"]  # protocol §1: 0
    records = []
    produced_ids: set[str] = set()
    with torch.inference_mode():
        for offset, human in enumerate(twins):
            i = base_index + offset  # global 1-based position in the 23B share
            prompt = override_prompt(human) if args.style_override else twin_prompt(human)
            enc = chat_tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to("cuda:0")
            torch.manual_seed(seed + i - 1)  # per-text variation (regen documented)
            out = model.generate(
                **enc,
                max_new_tokens=MAX_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                pad_token_id=chat_tok.eos_token_id,
            )
            text = chat_tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            text = text.split("\n")[0].strip()
            if len(text) < MIN_CHARS:
                print(f"  {i}: short output ({len(text)} chars), skipped", flush=True)
                continue
            records.append({
                "id": f"ai-23b-{i:03d}",
                "text": text,
                "label": "ai",
                "source": "luciole-23b-instruct",
                "meta": {
                    "prompt": prompt,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "seed": seed,
                    "twin_of": human["id"],
                    "generator": "luciole-23b-instruct",
                    "length_words": len(text.split()),
                    "length_tokens": len(len_tok(text, add_special_tokens=False).input_ids),
                },
            })
            produced_ids.add(f"ai-23b-{i:03d}")
            print(f"  {i}/{base_index + len(twins) - 1} {human['id']}: {len(text)} chars",
                  flush=True)

    lo, hi = base_index, base_index + len(twins) - 1
    def in_range(record: dict) -> bool:
        if record["source"] != "luciole-23b-instruct":
            return False
        num = int(record["id"].rsplit("-", 1)[1])
        return lo <= num <= hi

    kept = [r for r in corpus if not in_range(r)]
    merged = kept + records
    with args.corpus.open("w", encoding="utf-8") as fh:
        for record in merged:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(args.corpus.read_bytes()).hexdigest()
    print(f"merged {len(records)} ai-23b records → {len(merged)} total")
    print(f"corpus sha256: {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
