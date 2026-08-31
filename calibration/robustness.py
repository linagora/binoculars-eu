#!/usr/bin/env python3
"""Robustness tests R-1 to R-6 on the held-out test split (protocol §6).

Applies each perturbation to the test split, re-scores the perturbed texts
with the profile (a single pass, one ``Binoculars`` instantiation), and
reports per test: ΔAUC = AUC(perturbed) − AUC(original) with a paired
bootstrap (seed 100, 1000 draws) — except R-4 (mean normalized score, labels
lose their meaning). The ORIGINAL test split is never re-scored: original
scores come from ``scores_<lang>_v01.json`` (``--scores``); only perturbed
texts are scored.

- **R-1 typos** — 5 % of alphabetic chars swapped with a neighbour (seed 500).
- **R-2 light paraphrase** — manual reformulation of 10 % of sentences; pass
  ``--r2-file`` (JSONL with ``id`` / ``text``) or the test is skipped.
- **R-3 truncation** — first 100 tokens only (profile tokenizer).
- **R-4 concat** — twin human/AI sentence alternation (``twin_of`` or random
  pairing seed 501); expected mean normalized score in [0.4, 0.6].
- **R-5 adversarial prompting** — AI texts regenerated with the RAID prefix
  (seed 502); needs ``--generator-url`` / ``--generator-model``.
- **R-6 adversarial rewriting** — human texts rewritten "clarifie et
  professionnalise" (seed 503, same backend).

Writes ``calibration/robustness_<lang>_v01.json`` with the baselines.py
traceability header (timestamp, git sha, corpus SHA-256, seeds, versions).

Usage::

    python -m calibration.robustness \
        --corpus calibration/corpus/binoculars-eu-corpus-fr-v01.jsonl \
        --splits calibration/splits_fr_v01.json \
        --scores calibration/scores_fr_v01.json --profile fr
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import auc, roc_curve

from binoculars_eu import Binoculars
from binoculars_eu.profiles import get_profile
from binoculars_eu.utils import load_profile_tokenizer
from calibration.calibrate import SEED_TORCH, git_state, library_versions, write_json
from calibration.evaluate import SEED_BOOTSTRAP, load_corpus, load_splits

SEED_R1 = 500   # protocol §1: typos
SEED_R4 = 501   # protocol §1: concatenation pairing
SEED_R5 = 502   # protocol §1: adversarial prompting
SEED_R6 = 503   # protocol §1: adversarial rewriting
N_BOOT = 1000   # protocol §3.3
R1_TYPO_RATE = 0.05
R3_KEEP_TOKENS = 100
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]")
FRAGILITY: dict[str, float] = {
    "R-1": -0.15, "R-2": -0.20, "R-3": -0.30, "R-5": -0.25, "R-6": -0.30,
}
R4_EXPECTED_RANGE = (0.4, 0.6)
R5_PROMPT_PREFIX = "écris comme un humain, évite les tournures de LLM"
R6_REWRITE_INSTRUCTION = "clarifie et professionnalise"
NOTE_R2 = "skipped: manual paraphrase not provided (--r2-file)"
NOTE_R56 = "skipped: no --generator-url"


# Perturbations — each returns perturbed records (label and base id kept,
# id suffixed "-r<x>" for traceability), in the same order as the originals
# whenever the originals keep their meaning (R-1/R-2/R-3/R-5/R-6).
def r1_typos(records: list[dict], rate: float = R1_TYPO_RATE) -> list[dict]:
    """R-1: swap ``rate`` of alphabetic characters with their right neighbour."""
    rng = np.random.default_rng(SEED_R1)
    perturbed = []
    for record in records:
        text = record["text"]
        chars = list(text)
        swapable = [i for i, c in enumerate(chars[:-1]) if c.isalpha() and chars[i + 1].isalpha()]
        k = min(len(swapable), max(1, round(rate * sum(c.isalpha() for c in chars))))
        used: set[int] = set()
        for i in rng.choice(swapable, size=min(k, len(swapable)), replace=False):
            i = int(i)
            if i in used or i + 1 in used:
                continue
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
            used.update((i, i + 1))
        perturbed.append({**record, "id": f"{record['id']}-r1", "text": "".join(chars)})
    return perturbed


def r2_paraphrase(records: list[dict], r2_file: Path | None) -> list[dict] | None:
    """R-2: substitute manually paraphrased texts (``--r2-file``), else None."""
    if r2_file is None:
        return None
    paraphrases: dict[str, str] = {}
    for line in r2_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            paraphrases[entry["id"]] = entry["text"]
    missing = [r["id"] for r in records if r["id"] not in paraphrases]
    if missing:
        raise ValueError(f"--r2-file misses {len(missing)} test ids, e.g. {missing[:3]}")
    return [{**r, "id": f"{r['id']}-r2", "text": paraphrases[r["id"]]} for r in records]


def r3_truncate(records: list[dict], profile_code: str) -> list[dict]:
    """R-3: keep only the first 100 tokens (profile tokenizer, no specials)."""
    tokenizer = load_profile_tokenizer(get_profile(profile_code))
    perturbed = []
    for record in records:
        ids = tokenizer.encode(record["text"], add_special_tokens=False)[:R3_KEEP_TOKENS]
        perturbed.append({
            **record, "id": f"{record['id']}-r3",
            "text": tokenizer.decode(ids, skip_special_tokens=True),
        })
    return perturbed


def _sentences(text: str) -> list[str]:
    """Naive French sentence split (regex ``[^.!?]+[.!?]``), space-joined."""
    return [s.strip() for s in SENTENCE_RE.findall(text) if s.strip()]


def r4_concat(records: list[dict]) -> list[dict]:
    """R-4: alternate one human / one AI sentence per pair (``twin_of`` first,
    leftovers paired randomly, seed 501)."""
    humans = [r for r in records if r["label"] == "human"]
    ais = [r for r in records if r["label"] == "ai"]
    human_by_id = {r["id"]: r for r in humans}
    used_h: set[str] = set()
    used_ai: set[str] = set()
    pairs: list[tuple[dict, dict]] = []
    for ai in ais:
        twin = ai.get("twin_of")
        if twin and twin in human_by_id and twin not in used_h:
            pairs.append((human_by_id[twin], ai))
            used_h.add(twin)
            used_ai.add(ai["id"])
    left_h = [r for r in humans if r["id"] not in used_h]
    left_ai = [r for r in ais if r["id"] not in used_ai]
    n_pairs = min(len(left_h), len(left_ai))
    if n_pairs:
        rng = np.random.default_rng(SEED_R4)
        h_order = rng.permutation(len(left_h))[:n_pairs]
        ai_order = rng.permutation(len(left_ai))[:n_pairs]
        pairs.extend(
            (left_h[int(i)], left_ai[int(j)]) for i, j in zip(h_order, ai_order, strict=True)
        )
    perturbed = []
    for human, ai in pairs:
        h_sents = _sentences(human["text"])
        ai_sents = _sentences(ai["text"])
        sentences = [s for pair in zip(h_sents, ai_sents, strict=False) for s in pair]
        sentences += h_sents[len(ai_sents):] + ai_sents[len(h_sents):]
        perturbed.append({
            "id": f"{human['id']}-r4", "text": " ".join(sentences),
            "label": human["label"], "source": human.get("source", ""),
        })
    return perturbed


# R-5 / R-6 — minimal OpenAI-compatible generator client over urllib
def generate_text(url: str, model: str, api_key: str | None, prompt: str) -> str:
    """Call an OpenAI-compatible ``/v1/chat/completions`` endpoint via urllib.

    TODO(spec): protocol §6.1 fixes neither a decoding temperature nor a
    retry policy for the regeneration backend; server defaults are used and
    a single attempt is made. Non-2xx responses raise.
    """
    key = api_key or os.environ.get("GENERATOR_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("no generator API key: set GENERATOR_API_KEY or OPENAI_API_KEY")
    base = url.rstrip("/")
    endpoint = base if base.endswith("/v1/chat/completions") else f"{base}/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    request = urllib.request.Request(
        endpoint, data=payload, method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310
        body = json.loads(response.read().decode("utf-8"))
    return str(body["choices"][0]["message"]["content"])


def _regenerate(records: list[dict], label: str, seed: int, suffix: str,
                url: str | None, model: str, prompt_fn) -> list[dict] | None:
    """R-5/R-6 shared body: regenerate texts of ``label`` in ``seed`` order."""
    if url is None:
        return None
    rng = np.random.default_rng(seed)  # fixed generation order for traceability
    texts: dict[str, str] = {}
    for i in rng.permutation(len(records)):
        record = records[int(i)]
        if record["label"] != label:
            continue
        texts[record["id"]] = generate_text(url, model, None, prompt_fn(record))
    return [{**r, "id": f"{r['id']}-{suffix}", "text": texts[r["id"]]}
            for r in records if r["id"] in texts]


def r5_adversarial_prompt(records: list[dict], url: str | None, model: str) -> list[dict] | None:
    """R-5: regenerate AI texts with the RAID human-style prefix (seed 502)."""
    def prompt(record: dict) -> str:
        n_words = max(len(record["text"].split()), 1)
        return (f"{R5_PROMPT_PREFIX}. Réécris le texte suivant en environ "
                f"{n_words} mots :\n\n{record['text']}")

    return _regenerate(records, "ai", SEED_R5, "r5", url, model, prompt)


def r6_adversarial_rewrite(records: list[dict], url: str | None, model: str) -> list[dict] | None:
    """R-6: rewrite human texts "clarifie et professionnalise" (seed 503)."""
    def prompt(record: dict) -> str:
        return (f"{R6_REWRITE_INSTRUCTION} le texte suivant, en conservant "
                f"approximativement sa longueur :\n\n{record['text']}")

    return _regenerate(records, "human", SEED_R6, "r6", url, model, prompt)


# Scoring and paired ΔAUC bootstrap
def score_records(detector: Binoculars, records: list[dict],
                  batch_size: int) -> np.ndarray:
    """Score records once, deterministic batches (torch seed 42 per protocol §1)."""
    torch.manual_seed(SEED_TORCH)
    scores: list[float] = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        values = detector.compute_score([r["text"] for r in batch])
        scores.extend([values] if isinstance(values, float) else values)
    return np.array(scores, dtype=float)


def delta_auc_paired(y: np.ndarray, neg_orig: np.ndarray, neg_pert: np.ndarray,
                     n_boot: int = N_BOOT, seed: int = SEED_BOOTSTRAP) -> dict:
    """Paired bootstrap CI for ΔAUC = AUC(perturbed) − AUC(original).

    Resamples paired indices so each draw keeps the score pairing per text
    (protocol §6). Draws with a single class are skipped, per protocol §3.3.
    """
    auc_fn = lambda a, b: float(auc(*roc_curve(a, b)[:2]))  # noqa: E731
    point = auc_fn(y, neg_pert) - auc_fn(y, neg_orig)
    rng = np.random.default_rng(seed)
    n = len(y)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            values.append(auc_fn(y[idx], neg_pert[idx]) - auc_fn(y[idx], neg_orig[idx]))
        except ValueError:
            continue  # resample with a single class — skip, per protocol
    values_arr = np.array(values)
    return {
        "point": float(point),
        "ci_low": float(np.percentile(values_arr, 2.5)),
        "ci_high": float(np.percentile(values_arr, 97.5)),
        "n_boot_valid": int(len(values_arr)),
    }


def delta_auc_result(y: np.ndarray, neg_orig: np.ndarray, neg_pert: np.ndarray,
                     test_id: str) -> dict:
    """ΔAUC block for one robustness test, with its fragility flag (§6.1)."""
    threshold = FRAGILITY[test_id]
    delta = delta_auc_paired(y, neg_orig, neg_pert)
    return {
        "delta_auc": delta,
        "fragility_threshold": threshold,
        "fragile": bool(delta["point"] <= threshold),
    }


def normalized_mean_score(neg_scores: np.ndarray, records: list[dict],
                          scores: dict[str, float], train_ids: set[str]) -> float:
    """Mean score on the [0, 1] train-min-max scale (0.5 = undecidable).

    The min-max map is fitted on the train split only (protocol §2.2).
    TODO(spec): protocol §6.2 expects R-4 concatenated scores in [0.4, 0.6],
    which only makes sense on a normalized scale — raw Binoculars scores sit
    around ~0.8 (AI) to ~1.0 (human). Closest intent: the same train-fitted
    min-max map used for the ECE, so 0.5 marks the train midpoint.
    """
    neg_train = -np.array([scores[r["id"]] for r in records if r["id"] in train_ids], dtype=float)
    lo, hi = float(neg_train.min()), float(neg_train.max())
    span = max(hi - lo, 1e-12)
    return float(np.mean(np.clip((neg_scores - lo) / span, 0.0, 1.0)))


# CLI
def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True, help="calibration JSONL")
    parser.add_argument("--splits", type=Path, required=True, help="split manifest JSON")
    parser.add_argument("--scores", type=Path, required=True,
                        help="scores_<lang>_v01.json written by calibrate.py")
    parser.add_argument("--profile", default="fr", help="registered profile code")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--generator-url", default=None,
                        help="OpenAI-compatible endpoint for R-5/R-6 (else skipped)")
    parser.add_argument("--generator-model", default=None,
                        help="model name sent to the R-5/R-6 generator endpoint")
    parser.add_argument("--r2-file", type=Path, default=None,
                        help="JSONL (id, text) of manual paraphrases for R-2")
    parser.add_argument("--output-dir", type=Path, default=Path("calibration"))
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.generator_url and not args.generator_model:
        raise ValueError("--generator-url requires --generator-model")
    records, corpus_sha256 = load_corpus(args.corpus)
    splits = load_splits(args.splits, {r["id"] for r in records})
    scores = json.loads(args.scores.read_text(encoding="utf-8"))
    test = [r for r in records if r["id"] in splits["test"]]
    missing = {r["id"] for r in test} - set(scores)
    if missing:
        raise ValueError(f"scores file misses {len(missing)} test ids, e.g. {sorted(missing)[:3]}")
    y_test = np.array([1 if r["label"] == "ai" else 0 for r in test], dtype=int)
    neg_orig = -np.array([scores[r["id"]] for r in test], dtype=float)
    print(f"robustness on test n={len(test)} (original scores reused, NOT re-scored)")
    results: dict[str, dict] = {}
    perturbations: dict[str, list[dict] | None] = {
        "R-1": r1_typos(test),
        "R-2": r2_paraphrase(test, args.r2_file),
        "R-3": r3_truncate(test, args.profile),
        "R-4": r4_concat(test),
        "R-5": r5_adversarial_prompt(test, args.generator_url, args.generator_model or ""),
        "R-6": r6_adversarial_rewrite(test, args.generator_url, args.generator_model or ""),
    }

    # Single instantiation, shared by every test that needs scoring (§6: one pass).
    detector: Binoculars | None = None
    # R-5/R-6 perturb a single class (50 texts); the paired dAUC runs on the
    # matching original subset, not the full test split.
    id_to_pos = {r["id"]: i for i, r in enumerate(test)}
    for test_id, perturbed in perturbations.items():
        if perturbed is None:
            note = NOTE_R2 if test_id == "R-2" else NOTE_R56
            results[test_id] = {"note": note}
            continue
        if detector is None:
            detector = Binoculars.for_language(args.profile, mode="accuracy")
        neg_pert = -score_records(detector, perturbed, args.batch_size)
        if test_id == "R-4":
            lo, hi = R4_EXPECTED_RANGE
            mean_score = normalized_mean_score(neg_pert, records, scores, splits["train"])
            results[test_id] = {
                "mean_score": mean_score,
                "expected_range": [lo, hi],
                "in_range": bool(lo <= mean_score <= hi),
                "n_concat": len(perturbed),
            }
        else:
            positions = [id_to_pos[p["id"].rsplit("-", 1)[0]] for p in perturbed]
            results[test_id] = delta_auc_result(
                y_test[positions], neg_orig[positions], neg_pert, test_id
            )
            results[test_id]["n_perturbed"] = len(perturbed)

    print("\nRobustness (protocol §6):")
    for test_id, res in results.items():
        if "delta_auc" in res:
            d = res["delta_auc"]
            ci = f"[{d['ci_low']:+.4f},{d['ci_high']:+.4f}]"
            print(f"  {test_id:4s} ΔAUC={d['point']:+.4f} {ci} "
                  f"(fragile if ≤ {res['fragility_threshold']}) fragile={res['fragile']}")
        elif "mean_score" in res:
            print(f"  {test_id:4s} mean_score={res['mean_score']:.4f} "
                  f"(expected {res['expected_range']}) in_range={res['in_range']}")
        else:
            print(f"  {test_id:4s} {res['note']}")

    git_sha, git_dirty = git_state()
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "corpus_sha256": corpus_sha256,
        "seeds": {"split": 42, "bootstrap": SEED_BOOTSTRAP, "torch": SEED_TORCH,
                  "r1": SEED_R1, "r4": SEED_R4, "r5": SEED_R5, "r6": SEED_R6},
        "config": "v01",
        "versions": library_versions(),
        "n_test": int(len(y_test)),
        "generator": {"url": args.generator_url, "model": args.generator_model}
                     if args.generator_url else None,
        "robustness": results,
    }
    out = args.output_dir / f"robustness_{args.profile}_v01.json"
    write_json(out, payload)
    print(f"\nartefact: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
