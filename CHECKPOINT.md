# CHECKPOINT — binoculars-eu

**Date** : 2026-08-30
**Raison** : session interrompue (déconnexion/reconnexion utilisateur). Reprendre à partir de ce fichier.

---

## 1. État livré

**Phase 1 (squelette) : TERMINÉE** — vérifiée fonctionnellement, **en attente de validation utilisateur** avant phase 2.

Working dir : `/home/mmaudet/work/binoculars-eu` (pas de repo git initialisé à ce jour).

| Fichier | Lignes | Contenu |
|---------|--------|---------|
| `pyproject.toml` | 87 | Packaging uv/hatchling, deps §8 PRD (`torch>=2.5.1`, `transformers>=4.57.1`, …), extras `[api]` `[quant]` `[notebooks]` `[dev]`, config ruff + mypy strict + pytest (cov > 70 %) |
| `binoculars_eu/__init__.py` | 17 | Exporte `LanguageProfile`, `get_profile`, `list_profiles`, `__version__ = "0.1.0"` (classe `Binoculars` = phase 2) |
| `binoculars_eu/profiles/base.py` | 75 | Dataclass `LanguageProfile` (frozen), 16 champs conformes PRD §6.2 |
| `binoculars_eu/profiles/__init__.py` | 70 | Registre : `register()`, `_discover()` via `pkgutil.iter_modules`, `get_profile()`, `list_profiles()`, `DEFAULT_PROFILE_CODE = "fr"` |
| `binoculars_eu/profiles/fr/__init__.py` | 44 | `FRENCH_PROFILE_V01` + auto-enregistrement, TODO divergence performer (voir §3) |
| `binoculars_eu/profiles/fr/thresholds.json` | — | Placeholders : `accuracy 0.9`, `low_fpr 0.85`, `tpr_at_fpr_1 0.85` |
| `binoculars_eu/profiles/fr/metadata.json` | — | `corpus_sha256` à zéros (placeholder), date `2026-08-30`, seed `42`, note explicative |
| `README.md` | — | Badges, quick start Python + curl, tableau profils, roadmap V0.1→V3+, contributing, crédits |
| `LICENSE` | — | Apache 2.0 complet, copyright 2026 Michel-Marie Maudet |
| `.gitignore` | — | Python standard + `models/`, `calibration/output/`, caches HF, artifacts tests |
| `CONTRIBUTING.md` | — | Checklist d'acceptation d'un profil (PRD §4.6/§12), règles de code |
| `binoculars-eu-prd.md` | 2613 | **Copie locale du PRD v2.0** (source `/tmp`, volatile) — c'est LE contrat |

## 2. Vérifications effectuées

- Python 3.12.3 : `import binoculars_eu` OK sans dépendance lourde.
- `list_profiles()` → `['fr']` ; `get_profile('fr')` seuils lus depuis JSON.
- `get_profile('de')` → `KeyError` propre listant les profils disponibles.
- Aucun fichier > 400 lignes.

## 3. Décisions & points d'attention

- **Divergence performer FR** : le brief « déjà tranché » impose `OpenLLM-France/Luciole-1B-SFT-1.0` (dépôt à passer public à la livraison), alors que le PRD §3.1/§6.8 cite `Luciole-1B-Instruct-1.1`. Implémenté : **SFT-1.0**, avec TODO commenté dans `binoculars_eu/profiles/fr/__init__.py`. Vérifier la parité tokenizer du dépôt SFT et recalibrer en phase 5.
- Seuils 0.9/0.85/0.85 et SHA-256 à zéros = **placeholders interdits en prod**, recalibration phase 5 obligatoire.
- 0 question bloquante ouverte. **Phase 5 interdite sans le document `binoculars-eu-protocol.md`** fourni par l'utilisateur.

## 4. Prochaine étape : Phase 2 — cœur du détecteur (après validation phase 1 par l'utilisateur)

- `binoculars_eu/metrics.py` : `perplexity()` + `entropy()` repris **tels quels** de [ahans30/Binoculars](https://github.com/ahans30/Binoculars), crédit licence en tête de fichier.
- `binoculars_eu/utils.py` : `assert_tokenizer_consistency` assoupli selon `profile.share_tokenizer_from_observer` (tokenizer unique chargé depuis l'observer).
- `binoculars_eu/detector.py` : classe `Binoculars` — constructeurs `__init__(profile=...)`, `for_language(code)`, `from_legacy(obs, perf, mode)` (seuils Falcon `0.9015310749276843` / `0.8536432310785527`, labels anglais) ; méthodes `change_mode`, `compute_score`, `predict`, `analyze` ; bf16 ; `trust_remote_code` piloté par le profil ; dual-device via `DEVICE_1` / `DEVICE_2` (fallback `"auto"`).

Règles permanentes : Python 3.12+, type hints stricts (pas de `Any` non justifié), `ruff check` + `mypy --strict` verts, uv, docstrings Google **en anglais**, Conventional Commits en anglais, aucune dépendance hors PRD §8 sans validation, mode exécution (pas de conseil, spec verrouillée — signaler une contradiction en ≤ 3 lignes + TODO).

## 5. Rappel phases 3 → 5

- **Phase 3** : `api.py` FastAPI (`POST /detect`, `GET /profiles`, `GET /health`, Swagger `/docs`), `schemas.py` Pydantic (constraints §13.2 : `min_length=50`, pattern profil, `Literal` mode), cache LRU par `(profile, mode)`, `Dockerfile` production uvicorn, section « Test manuel » dans README.
- **Phase 4** : `tests/test_profile.py`, `tests/test_api.py` (TestClient, **détecteur mocké** — pas de vrais modèles), `tests/test_from_legacy.py`, `pytest.ini`, couverture > 70 %.
- **Phase 5** : calibration — **NE PAS LANCER** sans `binoculars-eu-protocol.md`.

## 6. Comment reprendre

Nouvelle session : « Reprends binoculars-eu à partir de `CHECKPOINT.md` ». Le contexte complet est ici + `binoculars-eu-prd.md` (contrat). En option : `git init` + commit atomique du squelette (non fait — mutation git à valider par l'utilisateur).
