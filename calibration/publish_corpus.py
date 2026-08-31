#!/usr/bin/env python3
"""Publish the FR V0.1 calibration corpora to Hugging Face Datasets.

Creates (idempotently) ``OpenLLM-France/binoculars-eu-corpus-fr-v01`` and the
``-ood`` variant, uploads the corpus JSONL files with a dataset card carrying
the SHA-256, splits, sources and known limitations. Run once per release from
the repo root:

    python -m calibration.publish_corpus

Requires a Hugging Face token with write access to the ``OpenLLM-France``
organisation (``huggingface-cli login`` or ``HF_TOKEN``).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from huggingface_hub import HfApi

ORG = "OpenLLM-France"
BASE = f"{ORG}/binoculars-eu-corpus-fr-v01"
CORPUS = Path("calibration/corpus/binoculars-eu-corpus-fr-v01.jsonl")
OOD = Path("calibration/corpus/binoculars-eu-corpus-fr-v01-ood.jsonl")
SPLITS = Path("calibration/splits_fr_v01.json")

CARD = """---
license: apache-2.0
language: fr
tags:
  - ai-text-detection
  - binoculars-eu
  - calibration-corpus
---

# {title}

Corpus de calibration du profil `fr` de
[binoculars-eu](https://github.com/linagora/binoculars-eu)
(détection zero-shot de texte généré par IA, méthode Binoculars).

{body}

## Structure

Chaque ligne JSONL : `{{"id", "label", "source", "text", "meta"}}` —
`label` ∈ `human` | `ai` ; sources humaines : wikipedia-fr, presse-fr,
littérature, linuxfr, blog-maudet ; sources IA : Luciole 1B/8B/23B (prompts
jumeaux des textes humains, seeds 0/1/2).

## Traçabilité

- SHA-256 du JSONL : `{sha}`
- Splits stratifiés 60/20/20 (seed 42) : `calibration/splits_fr_v01.json` dans
  le repo GitHub ; le test set n'est évalué qu'une seule fois (protocole §2.2).
- Rapport d'évaluation : `docs/evaluation_report_fr_v01.md` (repo GitHub).

## Limitation connue

{limitation}

## Licence

Apache 2.0, même licence que le projet binoculars-eu.
"""

LIMITATION = (
    "60/500 textes de la source « presse » contiennent des caractères U+FFFD "
    "(décodage charset erroné lors du scraping, en amont du projet). Le "
    "sha256 ci-dessus couvre le corpus tel quel ; un re-scraping propre est "
    "prévu en v1.1."
)


def publish(api: HfApi, repo_id: str, data: Path, title: str, body: str) -> None:
    """Create the dataset repo (idempotent) and upload data + card."""
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    sha = hashlib.sha256(data.read_bytes()).hexdigest()
    card = CARD.format(title=title, body=body, sha=sha, limitation=LIMITATION)
    api.upload_file(str(data), f"{data.name}", repo_id=repo_id, repo_type="dataset")
    api.upload_file(str(SPLITS), "splits_fr_v01.json", repo_id=repo_id,
                    repo_type="dataset")
    api.upload_file(str.encode(card), "README.md", repo_id=repo_id,
                    repo_type="dataset")
    print(f"published {repo_id} (sha256={sha[:12]}…)")


def main() -> int:
    api = HfApi()
    n_in = sum(1 for _ in CORPUS.open(encoding="utf-8"))
    n_ood = sum(1 for _ in OOD.open(encoding="utf-8"))
    publish(
        api, BASE, CORPUS,
        "binoculars-eu corpus FR v0.1 (calibration, n = 500)",
        f"Corpus principal : {n_in} textes français (250 humains / 250 IA), "
        "jumeaux thématiques humain/IA, bins de longueur équilibrés.",
    )
    publish(
        api, f"{BASE}-ood", OOD,
        "binoculars-eu corpus FR v0.1 OOD (Mistral-7B, n = 50)",
        f"Corpus out-of-distribution : {n_ood} textes générés par "
        "Mistral-7B-Instruct (mêmes prompts jumeaux), utilisés pour mesurer la "
        "généralisation hors famille Luciole.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
