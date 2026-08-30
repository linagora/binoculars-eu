# CHECKPOINT — binoculars-eu

**Date** : 2026-08-30
**Raison** : session interrompue (déconnexion/reconnexion utilisateur). Reprendre à partir de ce fichier.

---

## 1. État livré

- **Phase 1 (squelette) : TERMINÉE et validée** par l'utilisateur — commit `0454adc`.
- **Phase 2 (cœur du détecteur) : TERMINÉE et validée** par l'utilisateur — commit `0fd1bda` (+ `469eca9` checkpoint).
- **Phase 3 (API FastAPI) : TERMINÉE** — vérifiée live (tous cas passés), **en attente de validation utilisateur** avant phase 4.
- Phase 4 (tests) : **pas commencée**.

Working dir : `/home/mmaudet/work/binoculars-eu`. Repo git : branche `main`, HEAD `469eca9`. `.venv` local (system-site-packages) avec `accelerate` + extra `[api]` installé en editable.

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
| `binoculars_eu/schemas.py` | ~70 | Pydantic v2 : `DetectRequest` (`min_length=50`, `max_length=20_000`, pattern profil, `Literal` mode), `DetectResponse`, `ProfileInfo`, `HealthResponse` (PRD §13.2.2) |
| `binoculars_eu/api.py` | ~110 | FastAPI : `POST /detect` (404/422/503), `GET /profiles`, `GET /health`, Swagger auto `/docs` ; cache LRU `maxsize=4` par `(profile, mode)` (PRD §13.2.3) |
| `Dockerfile` | ~40 | Multi-stage production (uv + torch cu124 + `[api]`), `uvicorn --host 0.0.0.0 --port 8000`, cache HF en volume (PRD §9bis.5/§11.5 ter) |
| `README.md` | ~135 | + section « Test manuel » : 5 cas curl commentés (health, profiles, détection nominale, 404, 422) |
| `README.md`, `LICENSE`, `.gitignore`, `CONTRIBUTING.md` | — | Inchangés depuis phase 1 |
| `binoculars-eu-prd.md` | 2613 | **Copie locale du PRD v2.0** — LE contrat |

## 2. Vérifications effectuées

- Environnement : Python 3.12.3, torch 2.13.0+cu130, transformers 5.15.0, uv (pas de GPU utilisé, aucun poids de modèle téléchargé).
- `uvx ruff check .` → **clean** (après correction UP006/UP035/UP045/F401 héritées de la phase 1, jamais lintée faute de ruff installé à l'époque).
- Tests logique réels (import torch OK, **sans chargement de modèles** — instances via `Binoculars.__new__` + stubs) : `change_mode` × 3 modes + `ValueError` ; seuils lus depuis le profil `fr` ; `predict`/`analyze` FR (verdict, label localisé, `input_tokens`, `confidence`) ; batch `predict` ; parité `_legacy_profile` avec les constantes upstream au bit près ; `_resolve_devices` avec/sans env.
- **E2E réel `for_language("fr")` RÉUSSI** (token HF auth via `hf auth login`, compte `mmaudet`, dépôt SFT accessible) : instanciation 35 s, `NemotronForCausalLM` ×2 (Base + **SFT-1.0**), devices `cpu/cpu`, scoring ~1,5–2,1 s/texte, `change_mode("accuracy")` → seuil 0,9 + `predict` OK. ⚠️ Signal inversé sur les 2 mêmes échantillons ad hoc (IA-ish 1,0476 → « humain » ; humain-ish 0,8350 → « IA » au seuil placeholder 0,85 ; n=2, textes de 35/45 tokens < 100 → bruit structurel PRD §18.12). À quantifier par le test J1 à 20 textes (PRD §11.2) avant tout diagnostic de « méthode qui ne prend pas » ; risque déjà listé PRD §15 (fallback 8B int8).
- **E2E réel effectué** (CPU, `.venv` avec accelerate 1.14, system-site-packages) : instanciation `Binoculars(profile=…)` Luciole-1B-Base + **Luciole-1B-Instruct-1.1** → **OK en 71 s** (téléchargement inclus), scoring ~1,8 s/texte bf16 CPU, labels FR corrects.
- **Non testé** : `mypy --strict` (mypy non installé ; torch/transformers non typés — config mypy à ajuster en phase 4).
- **API testée LIVE** (uvicorn + curl, CPU) : `/health` 200 (`profiles_loaded: ["fr"]`, `device: cpu`), `/profiles` 200 traçabilité complète, `/detect` nominal 200 (score reproductible à l'identique sur 2ᵉ appel : 0.9378530979156494, `elapsed_ms` 3408 → 1240 via cache LRU, `detectors_cached: 1`), profil inconnu → **404**, texte < 50 car. → **422**, `/docs` Swagger 200.
- **Infra d'inférence** : aucun LiteLLM/vLLM joignable depuis ici — local : litellm non installé, rien sur 4000/8000/8001 ; `gpu.maudet.cloud` (204.168.196.226) : ports 4000/8000/8001/11434 en **connection refused** (services arrêtés ou bind localhost / firewall — cohérent avec PRD §7.2 : vLLM TTS arrêté mais `enabled`) ; SSH en échec (**host key verification failed** — à régler dans `known_hosts` par l'utilisateur, ne pas bypasser). Impact : génération du corpus IA via endpoint Luciole-23B (PRD §10.1) et fallback Mistral LINAGORA (§11.3.1) = phase 5 — à vérifier sur la machine GPU avant la calibration.
- Bug trouvé et corrigé en smoke test : `predict` sur une string unique passait par un tableau `np.where` 0-d dont `tolist()` rendait un scalaire → `pred[0]` indexait la chaîne. Corrigé par une branche explicite str/list.

## 3. Décisions & points d'attention

- **Divergence performer FR** : brief « déjà tranché » = `OpenLLM-France/Luciole-1B-SFT-1.0` (dépôt à passer public), PRD §3.1/§6.8 = `Luciole-1B-Instruct-1.1`. Implémenté : **SFT-1.0**, TODO dans `binoculars_eu/profiles/fr/__init__.py`. Vérifier parité tokenizer du dépôt SFT + recalibrer en phase 5.
- `detector.py` utilise `dtype=` (et non `torch_dtype=`) dans `from_pretrained` : valable transformers ≥ 4.56 donc compatible avec le pin PRD (4.57.1) et l'environnement (5.15).
- Heuristique `confidence` (non spécifiée par le PRD, requise par `DetectResponse`) : distance relative au seuil — < 2 % = `low`, < 5 % = `medium`, sinon `high`. Documentée dans `_confidence`.
- `predict` renvoie `str` pour une entrée `str` (le PRD §13.1.1 l'attend), `list[str]` pour un batch. Upstream renvoyait toujours une liste malgré son type `Union`.
- Seuils 0.9/0.85/0.85 et SHA-256 à zéros = **placeholders interdits en prod**, recalibration phase 5 obligatoire.
- 0 question bloquante ouverte. **Phase 5 interdite sans `binoculars-eu-protocol.md`** fourni par l'utilisateur.

## 4. Prochaine étape : Phase 4 — tests (après validation phase 3)

- `tests/test_profile.py` : dataclass `LanguageProfile` + registre (auto-discovery, unicité des codes, défaut `fr`).
- `tests/test_api.py` : `fastapi.testclient.TestClient` — 200 nominal, 404, 422, cache réutilisé, **détecteur mocké** (patch `binoculars_eu.api.get_detector`, aucun vrai modèle).
- `tests/test_from_legacy.py` : parité seuils Falcon (`0.9015310749276843` / `0.8536432310785527`) + labels anglais + `share_tokenizer_from_observer=False`.
- Invariant : `metrics.py` identique à l'upstream (hash du corps hors en-tête de licence).
- `pytest.ini`/config déjà dans `pyproject.toml` (cov > 70 % partie non-IO) ; installer `[dev]` (`uv pip install --python .venv/bin/python -e ".[dev]"`) — pytest non présent dans l'env.
- Config mypy à ajuster pour torch/transformers (libs non typées) avant de revendiquer `mypy --strict`.

Règles permanentes : Python 3.12+, type hints stricts, `ruff check` + `mypy --strict` verts, uv, docstrings Google **en anglais**, Conventional Commits en anglais, aucune dépendance hors PRD §8, mode exécution (contradiction = ≤ 3 lignes + TODO).

## 5. Rappel phase 5

- **Phase 5** : calibration — **NE PAS LANCER** sans `binoculars-eu-protocol.md`. Prérequis infra à vérifier avant : endpoint LiteLLM/vLLM sur `gpu.maudet.cloud` (ports fermés au dernier check), accès SSH (host key à valider par l'utilisateur).

## 6. Comment reprendre

Nouvelle session : « Reprends binoculars-eu à partir de `CHECKPOINT.md` ». Contexte complet ici + `binoculars-eu-prd.md` (contrat). Commits git atomiques par phase sur `main`, en anglais, **toujours avec confirmation explicite de l'utilisateur** (ex. `feat(detector): add Binoculars scoring engine around LanguageProfile`).
