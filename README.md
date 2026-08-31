# binoculars-eu

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![uv](https://img.shields.io/badge/packaging-uv-blueviolet)](https://docs.astral.sh/uv/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Profile fr](https://img.shields.io/badge/profile-fr%20(V0.1)-green)](binoculars_eu/profiles/fr/)

> European open-source platform for zero-shot detection of AI-generated text,
> organised around a single abstraction: the **language profile**.

`binoculars-eu` industrialises the [Binoculars method](https://arxiv.org/abs/2401.12070)
(Hans et al., ICML 2024), a perplexity / cross-perplexity ratio between two
models of the same family, ported to open European model families. V0.1 ships
one **French** profile (`fr`, the platform default) built on Luciole-1B, with
calibrated thresholds, a public calibration corpus, a full evaluation
protocol, and an HTTP API.

⚠️ **A verdict is not proof.** binoculars-eu is a decision-support tool, not
an infallible detector and not a plagiarism or misconduct detector. Its known
failure modes are documented below and in the
[eval card](docs/eval-card-fr.md). Do not use a single verdict as sole
evidence in high-stakes decisions (disciplinary, evaluative) without human
review.

## Results at a glance (profile `fr`, hold-out test n = 100)

| Metric | Value [95 % CI] | V0.1 target |
|--------|-----------------|-------------|
| AUC ROC (in-distribution) | **0.959** [0.914, 0.989] | ≥ 0.80 |
| TPR @ FPR = 1 % (headline) | **0.480** [0.356, 0.923] | ≥ 0.45 |
| F1 @ threshold `accuracy` | 0.879 | ≥ 0.75 |
| AUC ROC (OOD Mistral-7B) | **0.944** [0.896, 0.984] | n/a |

Full protocol (baselines, ablations, 6 invariance tests, error analysis):
[`docs/evaluation_report_fr_v01.md`](docs/evaluation_report_fr_v01.md) ·
[`docs/eval-card-fr.md`](docs/eval-card-fr.md).

**Honest caveats** (read before trusting a score):

- **Human texts in encyclopedic registers are misread as AI.** On the dev
  split, accuracy is **0.50** for dense Wikipedia-style text and **0.56** for
  classical literature, vs ≥ 0.93 for AI texts. See
  [`docs/error_analysis_fr_v01.md`](docs/error_analysis_fr_v01.md).
- **Fragile to character noise** (typos, bad OCR): ΔAUC −0.242 (R-1).
  Pre-process noisy input or refuse to score.
- **In-distribution blind spot**: Luciole-generated encyclopedic/tourism/presse
  text sometimes reads as human (19/20 residual false negatives).
- Texts shorter than ~100 tokens: unreliable score.

## How it works

Binoculars scores a text with a **performer** (small, fronto-littéral) and an
**observer** (same family, base): the ratio of their perplexities separates
machine-smooth text from human text, zero-shot, with no training on labeled
data. `binoculars-eu` wraps the pair + calibrated thresholds + tokenizer
policy in a `LanguageProfile`, discovered automatically from
`binoculars_eu/profiles/<lang>/`:

```python
from binoculars_eu import Binoculars

Binoculars()                          # default profile (fr), default mode
Binoculars.for_language("fr", mode="low-fpr")   # explicit profile lookup
Binoculars.from_legacy(                         # upstream-compatible (Falcon-7B, Hans thresholds)
    observer="tiiuae/falcon-7b",
    performer="tiiuae/falcon-7b-instruct",
    mode="accuracy",
)
```

Modes: `accuracy` (F1-optimal threshold), `low-fpr` (false-positive-averse),
`tpr-at-fpr-1` (headline operating point). Scoring is bfloat16, with optional
dual-device placement via `DEVICE_1` / `DEVICE_2`.

## Profiles

| Code | Display name | Observer | Performer | Status |
|------|--------------|----------|-----------|--------|
| `fr` | Français | [Luciole-1B-Base](https://huggingface.co/OpenLLM-France/Luciole-1B-Base) | [Luciole-1B-SFT-1.0](https://huggingface.co/OpenLLM-France/Luciole-1B-SFT-1.0) | **default (V0.1)** |
| `en` | English | tiiuae/falcon-7b | tiiuae/falcon-7b-instruct | V1 (planned) |

Every call that does not name a profile resolves to `fr`. Thresholds are
calibrated per profile and shipped with traceability (corpus SHA-256,
calibration date and seed) in `profiles/<lang>/{thresholds,metadata}.json`.

Calibration corpus (500 texts + 50 OOD, with splits and full provenance) is
versioned in this repository:
[`calibration/corpus/binoculars-eu-corpus-fr-v01.jsonl`](calibration/corpus/binoculars-eu-corpus-fr-v01.jsonl)
and [`-ood`](calibration/corpus/binoculars-eu-corpus-fr-v01-ood.jsonl), with the
SHA-256 pinned in `binoculars_eu/profiles/fr/metadata.json`. Publication on
Hugging Face Datasets (`OpenLLM-France/binoculars-eu-corpus-fr-v01`) is
planned once the hosting organisation approves the project.

## Install

```bash
uv pip install "binoculars-eu[api]"   # adds FastAPI + uvicorn
uv pip install "binoculars-eu[quant]" # optional bitsandbytes int8
```

Requires Python 3.12+, PyTorch ≥ 2.4, transformers ≥ 4.44. GPU optional
(CPU works, ~1B models); ~5 GB VRAM for the `fr` pair.

## Quick start (Python, 3 lines)

```python
from binoculars_eu import Binoculars

detector = Binoculars(mode="low-fpr")   # French profile by default
print(detector.predict("Votre texte à analyser…"))
# → "Probablement généré par IA" / "Probablement écrit par un humain"
```

`detector.analyze(text)` returns the full detail: score, verdict, label,
confidence, threshold used, token count, latency.

## Quick start (HTTP API)

```bash
uvicorn binoculars_eu.api:app --host 0.0.0.0 --port 8000
```

```bash
curl -s -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "Dans le paysage numérique en constante évolution, il est crucial de tirer parti des synergies.", "mode": "low-fpr"}'
```

Interactive Swagger UI: `http://localhost:8000/docs` · ReDoc: `/redoc` ·
OpenAPI spec: `/openapi.json`. Detectors are cached per `(profile, mode)`;
the first call on a pair pays the ~20-40 s weight-loading cost, subsequent
calls answer from the LRU cache.

Production image (multi-stage, CUDA 12.4):

```bash
docker build -t binoculars-eu:0.1.0 .
docker run --rm --gpus all -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  binoculars-eu:0.1.0
```

## Reproduce the evaluation (Docker)

The V0.1 evaluation runs in a pinned container (CUDA 12.4 runtime, Python
3.12, exact pins from `requirements-eval.txt`):

```bash
docker build -f docker/Dockerfile.eval -t binoculars-eu-eval:v01 .
docker run --rm --gpus all \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  binoculars-eu-eval:v01
```

One held-out pass (`calibration.evaluate --config v01`) writes
`calibration/evaluation_fr_v01.json` and an append-only run log. Protocol,
seeds, and artifacts: [`calibration/protocol.md`](calibration/protocol.md).

## Project layout

```
binoculars_eu/            # detector engine (metrics, utils, detector, api, profiles/)
calibration/              # corpus, splits, scoring, thresholds, robustness, error kits
docs/                     # evaluation report, eval card, error analysis
docker/Dockerfile.eval    # pinned evaluation image
tests/                    # profile registry, API, from_legacy compatibility
```

## Roadmap

| Version | Scope | Highlights |
|---------|-------|------------|
| V0.1 | profile `fr` | Luciole-1B pair, calibrated thresholds, corpus, FastAPI + Swagger, eval protocol |
| V0.2 | profile `fr` (capacity) | Luciole-8B int8, Mamba-hybrid observer pair, extended OOD benchmark, corpus v1.1 (encoding cleanup) |
| V1 | profiles `fr` + `en` | Falcon-7B legacy thresholds + independent reproduction (±0.01) |
| V2 | production hardening | rate limiting, auth, Prometheus metrics, upstream sync CI |
| V3+ | `es`, `de`, `it`, `pt`, `pl` | community profiles; engine maintained as protocol guardian |

One version = **one profile increment or one robustness increment**, never both.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). A new profile is merged only with: a
conformant `profiles/<lang>/` folder, a public corpus (versioned under
`calibration/corpus/`, published on Hugging Face Datasets once the
`OpenLLM-France` namespace approves it), its SHA-256, three calibrated
thresholds, an evaluation card, and a green `tests/test_profile.py`.

## Credits

- Method: [Hans et al., ICML 2024](https://arxiv.org/abs/2401.12070),
  upstream repo [ahans30/Binoculars](https://github.com/ahans30/Binoculars)
  (metrics ported under Apache 2.0, credited in `binoculars_eu/metrics.py`)
- French models: [OpenLLM-France / Luciole](https://huggingface.co/OpenLLM-France)
- Author: Michel-Marie Maudet (LINAGORA) <mmaudet@linagora.com>

## License

Apache 2.0, see [LICENSE](LICENSE).
