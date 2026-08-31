# Contributing to binoculars-eu

Thanks for your interest in the project. `binoculars-eu` is a platform: the
project team maintains the **scoring engine** and guards the **evaluation
protocol**, while language communities contribute **profiles**.

## Adding a new language profile

A profile is merged only when it ships, in a single PR:

1. A conformant folder `binoculars_eu/profiles/<lang>/` (PRD §6.7):
   - `__init__.py`: builds the profile and calls `register()` at module level;
   - `thresholds.json`: the three calibrated thresholds (`accuracy`,
     `low_fpr`, `tpr_at_fpr_1`);
   - `metadata.json`: `corpus_sha256`, `calibration_date`,
     `calibration_seed`, optional `calibration_note`.
   Never any scoring code, never an import of the detector.
2. A public corpus on HuggingFace Datasets following the naming pattern
   `OpenLLM-France/binoculars-eu-corpus-<lang>-v<version>[-ood]`.
3. An evaluation card `docs/eval-card-<lang>.md` produced with the full
   evaluation protocol (one complete protocol pass per profile: no metric,
   threshold or confidence interval is transferable across profiles).
4. A green `tests/test_profile_integrity.py`.

**Explicit refusal: no profile without a public corpus.** An unreproducible
threshold is an indefensible threshold.

## Code rules

- Python 3.12+, strict type hints, no unjustified `Any`.
- `ruff check` and `mypy --strict` must pass.
- Google-style docstrings, in English.
- No file over 400 lines; split by responsibility.
- Package manager: [uv](https://docs.astral.sh/uv/).
- Atomic [Conventional Commits](https://www.conventionalcommits.org/), in
  English (e.g. `feat(profile): add French language profile`).

## Development setup

```bash
uv venv --python 3.12
uv pip install -e ".[dev,api]"
ruff check .
mypy --strict .
pytest
```
