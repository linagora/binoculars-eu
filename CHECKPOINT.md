# CHECKPOINT — binoculars-eu

**Date** : 2026-08-30
**Raison** : session interrompue (déconnexion/reconnexion utilisateur). Reprendre à partir de ce fichier.

---

## 1. État livré

- **Phase 1 (squelette) : TERMINÉE et validée** par l'utilisateur — commit `0454adc`.
- **Phase 2 (cœur du détecteur) : TERMINÉE** — vérifiée (ruff + tests logique), **en attente de validation utilisateur** avant phase 3.
- Phase 3 (API FastAPI) : **pas commencée**.

Working dir : `/home/mmaudet/work/binoculars-eu`. Repo git : branche `main`, racine `0454adc` (phase 1).

| Fichier | Lignes | Contenu |
|---------|--------|---------|
| `pyproject.toml` | ~97 | Packaging uv/hatchling, deps §8 PRD, extras `[api]` `[quant]` `[notebooks]` `[dev]`, config ruff (per-file-ignores pour `metrics.py` verbatim, pydocstyle google) + mypy strict + pytest (cov > 70 %) |
| `binoculars_eu/__init__.py` | 19 | Exporte `Binoculars`, `AnalyzeResult`, `LanguageProfile`, `get_profile`, `list_profiles`, `__version__ = "0.1.0"` |
| `binoculars_eu/profiles/base.py` | 73 | Dataclass `LanguageProfile` (frozen), 16 champs conformes PRD §6.2 (syntaxe moderne `str \| None`) |
| `binoculars_eu/profiles/__init__.py` | 69 | Registre : `register()`, `_discover()` via `pkgutil.iter_modules`, `get_profile()`, `list_profiles()`, `DEFAULT_PROFILE_CODE = "fr"` |
| `binoculars_eu/profiles/fr/__init__.py` | 44 | `FRENCH_PROFILE_V01` + auto-enregistrement, TODO divergence performer (voir §3) |
| `binoculars_eu/profiles/fr/thresholds.json` | — | Placeholders : `accuracy 0.9`, `low_fpr 0.85`, `tpr_at_fpr_1 0.85` |
| `binoculars_eu/profiles/fr/metadata.json` | — | `corpus_sha256` à zéros (placeholder), date `2026-08-30`, seed `42`, note explicative |
| `binoculars_eu/metrics.py` | ~85 | `perplexity()` + `entropy()` **verbatim upstream** ahans30/Binoculars ; en-tête créditant BSD 3-Clause (© Hans, Schwarzschild, Goldstein) |
| `binoculars_eu/utils.py` | ~60 | `assert_tokenizer_consistency` (stricte, comportement upstream) + `load_profile_tokenizer(profile)` implémentant `share_tokenizer_from_observer` |
| `binoculars_eu/detector.py` | 282 | Classe `Binoculars` : `__init__(profile=…)` (défaut `fr`), `for_language(code)`, `from_legacy(obs, perf, mode)` ; `change_mode`, `compute_score`, `predict`, `analyze` (→ `AnalyzeResult` TypedDict) ; bf16 ; `trust_remote_code` piloté par le profil ; dual-device `DEVICE_1`/`DEVICE_2` (fallback `"auto"` → cuda sinon cpu) ; `_legacy_profile()` avec seuils Falcon exacts |
| `README.md`, `LICENSE`, `.gitignore`, `CONTRIBUTING.md` | — | Inchangés depuis phase 1 |
| `binoculars-eu-prd.md` | 2613 | **Copie locale du PRD v2.0** — LE contrat |

## 2. Vérifications effectuées

- Environnement : Python 3.12.3, torch 2.13.0+cu130, transformers 5.15.0, uv (pas de GPU utilisé, aucun poids de modèle téléchargé).
- `uvx ruff check .` → **clean** (après correction UP006/UP035/UP045/F401 héritées de la phase 1, jamais lintée faute de ruff installé à l'époque).
- Tests logique réels (import torch OK, **sans chargement de modèles** — instances via `Binoculars.__new__` + stubs) : `change_mode` × 3 modes + `ValueError` ; seuils lus depuis le profil `fr` ; `predict`/`analyze` FR (verdict, label localisé, `input_tokens`, `confidence`) ; batch `predict` ; parité `_legacy_profile` avec les constantes upstream au bit près ; `_resolve_devices` avec/sans env.
- **Non testé** (le dire si demandé) : scoring de bout en bout avec vrais poids (Luciole non téléchargés ici) ; `mypy --strict` (mypy non installé ; torch/transformers non typés — config mypy à ajuster en phase 4).
- Bug trouvé et corrigé en smoke test : `predict` sur une string unique passait par un tableau `np.where` 0-d dont `tolist()` rendait un scalaire → `pred[0]` indexait la chaîne. Corrigé par une branche explicite str/list.

## 3. Décisions & points d'attention

- **Divergence performer FR** : brief « déjà tranché » = `OpenLLM-France/Luciole-1B-SFT-1.0` (dépôt à passer public), PRD §3.1/§6.8 = `Luciole-1B-Instruct-1.1`. Implémenté : **SFT-1.0**, TODO dans `binoculars_eu/profiles/fr/__init__.py`. Vérifier parité tokenizer du dépôt SFT + recalibrer en phase 5.
- `detector.py` utilise `dtype=` (et non `torch_dtype=`) dans `from_pretrained` : valable transformers ≥ 4.56 donc compatible avec le pin PRD (4.57.1) et l'environnement (5.15).
- Heuristique `confidence` (non spécifiée par le PRD, requise par `DetectResponse`) : distance relative au seuil — < 2 % = `low`, < 5 % = `medium`, sinon `high`. Documentée dans `_confidence`.
- `predict` renvoie `str` pour une entrée `str` (le PRD §13.1.1 l'attend), `list[str]` pour un batch. Upstream renvoyait toujours une liste malgré son type `Union`.
- Seuils 0.9/0.85/0.85 et SHA-256 à zéros = **placeholders interdits en prod**, recalibration phase 5 obligatoire.
- 0 question bloquante ouverte. **Phase 5 interdite sans `binoculars-eu-protocol.md`** fourni par l'utilisateur.

## 4. Prochaine étape : Phase 3 — API FastAPI (après validation phase 2)

- `binoculars_eu/schemas.py` : Pydantic `DetectRequest` (`min_length=50`, `max_length=20_000`, pattern `^[a-z]{2}(-[a-z0-9]{2,5})?$` sur `profile`, `Literal` sur `mode`), `DetectResponse`, `ProfileInfo`, `HealthResponse` (§13.2.2).
- `binoculars_eu/api.py` : `POST /detect` (404 profil inconnu, 422 validation, 503 chargement), `GET /profiles`, `GET /health`, Swagger auto `/docs` ; cache LRU `@lru_cache(maxsize=4)` par `(profile, mode)`.
- `Dockerfile` production (uvicorn `--host 0.0.0.0 --port 8000`), section « Test manuel » au README.
- Dépendances : `uv pip install -e ".[api]"` (fastapi/uvicorn/pydantic non installés dans l'env actuel).

Règles permanentes : Python 3.12+, type hints stricts, `ruff check` + `mypy --strict` verts, uv, docstrings Google **en anglais**, Conventional Commits en anglais, aucune dépendance hors PRD §8, mode exécution (contradiction = ≤ 3 lignes + TODO).

## 5. Rappel phases 4 → 5

- **Phase 4** : `tests/test_profile.py`, `tests/test_api.py` (TestClient, **détecteur mocké**), `tests/test_from_legacy.py`, `pytest.ini`, couverture > 70 % partie non-IO. Invariant à tester : `metrics.py` identique à l'upstream (hash du corps, hors en-tête de licence).
- **Phase 5** : calibration — **NE PAS LANCER** sans `binoculars-eu-protocol.md`.

## 6. Comment reprendre

Nouvelle session : « Reprends binoculars-eu à partir de `CHECKPOINT.md` ». Contexte complet ici + `binoculars-eu-prd.md` (contrat). Commits git atomiques par phase sur `main`, en anglais, **toujours avec confirmation explicite de l'utilisateur** (ex. `feat(detector): add Binoculars scoring engine around LanguageProfile`).
