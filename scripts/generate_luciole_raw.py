#!/usr/bin/env python3
"""
Génère un texte "Luciole raw" pour le test au clou de l'article
transparence-IA / binoculars-eu.

Usage typique (depuis Athena) :
    python generate_luciole_raw.py

Configuration via variables d'environnement ou flags CLI :
    LUCIOLE_API_URL      URL de l'API vLLM / OpenAI-compatible
                         (défaut : http://gpu.maudet.cloud:8013/v1)
    LUCIOLE_MODEL        nom du modèle côté serveur
                         (défaut : luciole-8b-instruct)
    LUCIOLE_API_KEY      clé API si le serveur en exige une
                         (défaut : "dummy", ignorée par vLLM en local)
    LUCIOLE_SEED         graine (défaut : 42)

Sortie :
    - Impression du texte sur stdout
    - Sauvegarde dans nail_test_inputs/luciole_raw.txt
    - Sauvegarde des métadonnées dans nail_test_inputs/luciole_raw.meta.json
      (paramètres, timestamp, hash SHA-256 du texte, nombre de tokens
      approximatif, latence)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    sys.stderr.write(
        "Le paquet 'openai' est requis. Installe-le avec :\n"
        "    uv pip install openai\n"
        "ou :\n"
        "    pip install openai\n"
    )
    sys.exit(1)


DEFAULT_API_URL = "http://gpu.maudet.cloud:8013/v1"
DEFAULT_MODEL = "luciole-8b-instruct"
DEFAULT_OUTPUT_DIR = Path("nail_test_inputs")

SYSTEM_PROMPT = (
    "Tu es un rédacteur de vulgarisation technique en français. "
    "Tu écris des articles de blog clairs, informatifs et fluides."
)

USER_PROMPT = """Rédige un article de blog de 400 à 500 mots en français sur le sujet suivant : « Les enjeux de la transparence dans l'IA générative : étiquetage, watermarking et responsabilité des plateformes ».

Contraintes :
- Ton neutre et informatif, style article de vulgarisation technique
- 4-5 paragraphes structurés
- Pas de titre H1, commence directement par le premier paragraphe
- Cite au moins un cadre réglementaire (AI Act européen, ou équivalent)
- Termine par une phrase d'ouverture sur les défis à venir
- Français soigné, tournures naturelles, pas de jargon excessif
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Générateur Luciole raw pour test binoculars-eu",
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get("LUCIOLE_API_URL", DEFAULT_API_URL),
        help=f"URL de l'API OpenAI-compatible (défaut : {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("LUCIOLE_MODEL", DEFAULT_MODEL),
        help=f"Nom du modèle côté serveur (défaut : {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("LUCIOLE_API_KEY", "dummy"),
        help="Clé API (défaut : 'dummy', ignorée par vLLM local)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("LUCIOLE_SEED", "42")),
        help="Graine de génération (défaut : 42)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Température (défaut : 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="top_p (défaut : 0.9)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=700,
        help="Nombre max de tokens de sortie (défaut : 700)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Dossier de sortie (défaut : {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--output-name",
        default="luciole_raw",
        help="Préfixe des fichiers de sortie (défaut : luciole_raw)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Ne pas afficher le texte sur stdout (utile pour piping)",
    )
    return parser.parse_args()


def approximate_token_count(text: str) -> int:
    """Estimation rapide sans dépendance à un tokenizer précis.

    Règle de pouce : ~4 caractères par token en français. Suffisant
    pour le champ metadata ; le vrai comptage se fait côté binoculars-eu
    à la mesure du score.
    """
    return max(1, len(text) // 4)


def main() -> int:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_txt = args.output_dir / f"{args.output_name}.txt"
    output_meta = args.output_dir / f"{args.output_name}.meta.json"

    if not args.quiet:
        sys.stderr.write(
            f"[info] Appel de {args.api_url} sur le modèle {args.model}\n"
            f"[info] seed={args.seed}, temperature={args.temperature}, "
            f"top_p={args.top_p}, max_tokens={args.max_tokens}\n"
        )

    client = OpenAI(base_url=args.api_url, api_key=args.api_key)

    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
            ],
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            seed=args.seed,
        )
    except Exception as exc:  # noqa: BLE001 - remonter tel quel pour l'utilisateur
        sys.stderr.write(f"[error] Échec de l'appel API : {exc}\n")
        return 2
    elapsed_s = time.perf_counter() - start

    text = response.choices[0].message.content or ""
    text = text.strip()

    if not text:
        sys.stderr.write("[error] Le modèle a retourné un texte vide.\n")
        return 3

    sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()

    usage: dict[str, Any] = {}
    if response.usage is not None:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "api_url": args.api_url,
        "model": args.model,
        "parameters": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "seed": args.seed,
        },
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": USER_PROMPT,
        "output": {
            "sha256": sha256,
            "char_count": len(text),
            "word_count": len(text.split()),
            "approx_token_count": approximate_token_count(text),
            "usage_from_server": usage,
        },
        "latency_seconds": round(elapsed_s, 3),
    }

    output_txt.write_text(text + "\n", encoding="utf-8")
    output_meta.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if not args.quiet:
        sys.stderr.write(
            f"\n[ok] Texte généré en {elapsed_s:.1f} s "
            f"({metadata['output']['word_count']} mots, "
            f"{metadata['output']['char_count']} caractères, "
            f"~{metadata['output']['approx_token_count']} tokens)\n"
            f"[ok] Texte    -> {output_txt}\n"
            f"[ok] Metadata -> {output_meta}\n"
            f"[ok] SHA-256  : {sha256[:16]}...\n\n"
            "----- TEXTE -----\n"
        )
        print(text)
        sys.stderr.write("----- FIN -----\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
