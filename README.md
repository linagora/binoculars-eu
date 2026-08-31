# binoculars-eu

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![uv](https://img.shields.io/badge/packaging-uv-blueviolet)](https://docs.astral.sh/uv/)

> European open-source platform for zero-shot detection of AI-generated text,
> organised around a single abstraction: the **language profile**
> (`LanguageProfile`).

`binoculars-eu` industrialises the [Binoculars method](https://arxiv.org/abs/2401.12070)
(Hans et al., ICML 2024) — a perplexity / cross-perplexity ratio between two
models of the same family — ported to open European model families. V0.1 ships
a single **French** profile (`fr`, the platform default), built on Luciole-1B.
The English profile (`en`, Falcon-7B) arrives in V1; `es`, `de`, `it`, `pt`,
`pl` are community-driven afterwards.

## Profiles

| Code | Display name | Observer | Performer | Status |
|------|--------------|----------|-----------|--------|
| `fr` | Français | OpenLLM-France/Luciole-1B-Base | OpenLLM-France/Luciole-1B-SFT-1.0 | **default (V0.1)** |
| `en` | English | tiiuae/falcon-7b | tiiuae/falcon-7b-instruct | V1 (opt-in) |

Every call that does not name a profile resolves to `fr`. Additional profiles
are always explicit opt-in (`for_language("en")`, `"profile": "en"`).

## Install

```bash
uv pip install "binoculars-eu[api]"
```

## Quick start (Python, 3 lines)

```python
from binoculars_eu import Binoculars

detector = Binoculars(mode="low-fpr")   # French profile by default
print(detector.predict("Votre texte à analyser…"))
# → "Probablement généré par IA" / "Probablement écrit par un humain"
```

## Quick start (HTTP API)

```bash
uvicorn binoculars_eu.api:app --host 0.0.0.0 --port 8000
```

```bash
curl -s -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "Dans le paysage numérique en constante évolution, il est crucial de tirer parti des synergies.", "mode": "low-fpr"}'
```

Interactive Swagger UI: `http://localhost:8000/docs`, ReDoc: `/redoc`,
raw OpenAPI spec: `/openapi.json`.

## Test manuel

The API starts in under 2 s (model loading is lazy: the first `POST /detect`
on a given `(profile, mode)` pays the ~20-40 s weight-loading cost on GPU,
then hits the LRU cache).

```bash
# 1) Liveness + registry inventory — no model loaded yet
curl -s http://localhost:8000/health | python3 -m json.tool
# → {"status": "ok", "version": "0.1.0", "default_profile": "fr",
#    "profiles_loaded": ["fr"], "detectors_cached": 0, "device": "cpu"}

# 2) Available profiles with their calibration traceability
curl -s http://localhost:8000/profiles | python3 -m json.tool

# 3) Nominal detection — profile omitted, defaults to "fr"; first call loads
#    the model pair, subsequent calls answer from the LRU cache
curl -s -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "Dans le paysage numérique en constante évolution, il est crucial de tirer parti des synergies pour naviguer dans un écosystème complexe.", "mode": "accuracy"}' \
  | python3 -m json.tool
# → {"score": …, "verdict": "ai"|"human", "label": "Probablement …",
#    "confidence": "low"|"medium"|"high", "threshold_used": …, "mode": "accuracy",
#    "profile": "fr", "input_tokens": …, "elapsed_ms": …}

# 4) Unknown profile → 404 with the list of available profiles
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "Texte suffisamment long pour passer la validation du schéma.", "profile": "de"}'
# → 404

# 5) Text too short → 422 (Pydantic constraint min_length=50)
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "trop court"}'
# → 422
```

The same routes are exercisable from the browser via « Try it out » on
`http://localhost:8000/docs`.

## Reproduce the evaluation (Docker)

The V0.1 evaluation of profile `fr` (protocol §18.8) runs in a pinned
container — CUDA 12.4 runtime, Python 3.12, exact dependency pins from
`requirements-eval.txt`:

```bash
docker build -f docker/Dockerfile.eval -t binoculars-eu-eval:v01 .
docker run --rm --gpus all \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  binoculars-eu-eval:v01
```

The image runs one held-out evaluation pass (`calibration.evaluate --config v01`)
and writes `calibration/evaluation_fr_v01.json` plus an append-only line in
`calibration/evaluation_runs_fr.jsonl`. Results and methodology are documented
in [`docs/evaluation_report_fr_v01.md`](docs/evaluation_report_fr_v01.md).

## Roadmap

| Version | Scope | Highlights |
|---------|-------|------------|
| V0.1 | profile `fr` | Luciole-1B pair, calibration corpus, FastAPI + Swagger, HF Space |
| V0.2 | profile `fr` (capacity variant) | Luciole-8B int8, first Binoculars on a Mamba-hybrid, extended OOD benchmark |
| V1 | profiles `fr` + `en` | Falcon-7B legacy thresholds + independent reproduction (±0.01 tolerance) |
| V2 | production hardening | rate limiting, auth, Prometheus metrics, upstream sync CI |
| V3+ | `es`, `de`, `it`, `pt`, `pl` | community profiles, engine maintained as protocol guardian |

One version = **one profile increment or one robustness increment**, never both.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). A profile is merged only with: a
conformant `profiles/<lang>/` folder, a public corpus on HuggingFace Datasets
(`OpenLLM-France/binoculars-eu-corpus-<lang>-v<version>`), its SHA-256, three
calibrated thresholds, an evaluation card, and a green
`tests/test_profile_integrity.py`.

## Credits

- Method: [Hans et al., ICML 2024](https://arxiv.org/abs/2401.12070) —
  upstream repo [ahans30/Binoculars](https://github.com/ahans30/Binoculars)
- French models: [OpenLLM-France / Luciole](https://huggingface.co/OpenLLM-France)
- Author: Michel-Marie Maudet

## License

Apache 2.0 — see [LICENSE](LICENSE).
