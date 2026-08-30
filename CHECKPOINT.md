# CHECKPOINT — binoculars-eu

**Date** : 2026-08-30
**Raison** : session interrompue (déconnexion/reconnexion utilisateur). Reprendre à partir de ce fichier.

---

## 1. État livré

- **Phase 1 (squelette) : TERMINÉE et validée** par l'utilisateur — commit `0454adc`.
- **Phase 2 (cœur du détecteur) : TERMINÉE et validée** par l'utilisateur — commit `0fd1bda` (+ `469eca9` checkpoint).
- **Phase 3 (API FastAPI) : TERMINÉE et validée** par l'utilisateur — commit `52197f4` (+ `a4650f4`). Sanity check E2E : matching API/direct au bit, LRU prouvé ; latence <500 ms requalifiée en test d'acceptation GPU (`acceptance_gpu`, <200 ms) — script `scripts/sanity_check_api.py` double mode CLI/pytest avec marqueurs `acceptance_cpu`/`acceptance_gpu` + en-tête d'environnement (commit `54079ac`).
- **Phase 4 (tests) : TERMINÉE** — `tests/test_profile.py` (dataclass + registre), `tests/test_api.py` (TestClient, détecteur stubbé : 200/404/422×3/503, réutilisation cache, OpenAPI), `tests/test_from_legacy.py` (seuils Falcon au bit, câblage sans poids, **invariant `metrics.py` verbatim** = hash AST `8234c4…b7cdc0`, utils tokenizer). **39/39 passés, couverture 93,5 % (seuil 70 %), ruff clean.** Config pytest dans `pyproject.toml` (pas de `pytest.ini` séparé — équivalent standard). En attente de commit.
- Phase 5 : **pas commencée** (interdite sans `binoculars-eu-protocol.md`).

Working dir : `/home/mmaudet/work/binoculars-eu`. Repo git : branche `main`, HEAD `54079ac`, remote `origin` = `git@github.com:linagora/binoculars-eu.git` (**privé** — placeholders et SFT non public ; passage en public via `gh repo edit linagora/binoculars-eu --visibility public` quand la V0.1 sera calibrée). `.venv` local (system-site-packages) avec `accelerate` + extras `[api]` `[dev]` installés en editable.

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

- **Dry-run calibrate RÉUSSI** (2026-08-31, `301363b`, sur le L4, profil intact) : scoring 500 textes batch 8 bf16 ; **train AUC = 0,9734** ; seuils candidats (fit train 300 uniquement) : `accuracy = 0,955801` (F1 0,9265), `low_fpr = 0,866667` (train FPR 0,0067 = 1/150, granularité), `tpr_at_fpr_1 = 0,866667` (train TPR@1 % = 0,6400 — coïncidence low_fpr/tpr@1 % = effet de la granularité à 150 humains train, à documenter). Artefacts : `results/scores/error_analysis_candidates_fr_v01.json`. ⚠️ Métriques train = fit (optimistes) ; les métriques publiables exigent `evaluate.py` sur test (une seule fois, §2.2) + baselines + robustesse (§4-§6). **Décision `--write` réservée à l'utilisateur.** Corpus 500/500 (sha256 `ca70aaa4…`) : 100×23B nf4 (20 jumeaux presse régénérés après dégénérescence détectée — prompt journaliste, seed dérivé documenté), 75×8B vLLM NemotronH natif, 75×1B LiteLLM ; zéro doublon. Splits 300/100/100 seed 42 (sha256 `1b1b34f4…`, fusion déterministe des cellules singleton).
- **Phase 5 EN COURS** : corpus humain **250/250** ✔ (`calibration/corpus/binoculars-eu-corpus-fr-v01.jsonl`, sha256 `c6fb9b50…6025`, composition PRD §10.1 exacte avec LinkedIn→blog-maudet documenté ; presse via pagination LMI `toute-l-actualite-page-N`, littérature via sous-pages Wikisource rendues par `action=parse`). OOD Mistral **50/50** ✔ (`…-ood.jsonl`, sha256 `39b42fb8…69b9`, seed 700 faute de seed protocole §1 — TODO révision). Scripts : `calibration/build_corpus.py` (collecte + génération jumelle par backend `--generators 23b,8b,1b` avec merge incrémental), `calibration/human_sources.py`, `calibration/generate_ood_mistral.py`, `calibration/build_splits.py` (livré, lint OK), `calibration/calibrate.py` (**LIVRÉ NON EXÉCUTÉ** — point d'arrêt utilisateur), `scripts/luciole_switch.sh`, `scripts/setup_gpu_env.sh`. **Inférence** : unités `luciole-8b-instruct` (:8013 bf16, bind Tailscale) et `luciole-23b-instruct` (:8015 nf4 bnb) créées sur `gpu-ubuntu`, venv dédié `~/.venv-vllm-serve` (vllm 0.28 + bnb 0.50.2, torch 2.13) ; deux dépendances build corrigées (`python3.12-dev`+`build-essential` pour Python.h, `ninja-build`+`cmake` pour le JIT vLLM) ; NemotronH supporté nativement (poids 8B chargés 15,07 GiB). Séquence de génération prévue : 8B → 23B nf4 → 1B (LiteLLM) → splits seed 42 → arrêt avant `calibrate.py`. Passerelle LiteLLM inactive pendant les bascules, rétablie en fin de séquence.
- Environnement : Python 3.12.3, torch 2.13.0+cu130, transformers 5.15.0, uv (pas de GPU utilisé, aucun poids de modèle téléchargé).
- `uvx ruff check .` → **clean** (après correction UP006/UP035/UP045/F401 héritées de la phase 1, jamais lintée faute de ruff installé à l'époque).
- Tests logique réels (import torch OK, **sans chargement de modèles** — instances via `Binoculars.__new__` + stubs) : `change_mode` × 3 modes + `ValueError` ; seuils lus depuis le profil `fr` ; `predict`/`analyze` FR (verdict, label localisé, `input_tokens`, `confidence`) ; batch `predict` ; parité `_legacy_profile` avec les constantes upstream au bit près ; `_resolve_devices` avec/sans env.
- **E2E réel `for_language("fr")` RÉUSSI** (token HF auth via `hf auth login`, compte `mmaudet`, dépôt SFT accessible) : instanciation 35 s, `NemotronForCausalLM` ×2 (Base + **SFT-1.0**), devices `cpu/cpu`, scoring ~1,5–2,1 s/texte, `change_mode("accuracy")` → seuil 0,9 + `predict` OK. ⚠️ Signal inversé sur les 2 mêmes échantillons ad hoc (IA-ish 1,0476 → « humain » ; humain-ish 0,8350 → « IA » au seuil placeholder 0,85 ; n=2, textes de 35/45 tokens < 100 → bruit structurel PRD §18.12). À quantifier par le test J1 à 20 textes (PRD §11.2) avant tout diagnostic de « méthode qui ne prend pas » ; risque déjà listé PRD §15 (fallback 8B int8).
- **E2E réel effectué** (CPU, `.venv` avec accelerate 1.14, system-site-packages) : instanciation `Binoculars(profile=…)` Luciole-1B-Base + **Luciole-1B-Instruct-1.1** → **OK en 71 s** (téléchargement inclus), scoring ~1,8 s/texte bf16 CPU, labels FR corrects.
- **Non testé** : `mypy --strict` (mypy non installé ; torch/transformers non typés — config mypy à ajuster en phase 4).
- **J1 dry run : SÉPARATION PARFAITE (n=20)** — 10 textes IA réels (Luciole-1B-Instruct-1.1, jumeaux thématiques, temp 0.7) vs 10 humains réels (Wikipédia FR, ~1200 car.) scorés par le vrai profil `fr` : IA **0,719–0,865** (médiane 0,807) vs humains **0,884–1,047** (médiane 0,984), **zéro chevauchement** (écart 0,019), critère PRD §11.2 atteint (**10/10 IA sous le P90 humain**). Les « inversions » observées avant étaient des artefacts de textes ad hoc de 24–45 tokens (< 100 tokens = bruit structurel, PRD §18.12) et d'imitations IA écrites à la main. ⚠️ Reste à quantifier en calibration : le français familier/hors-distribution score dans la zone IA (faux positifs, taxon FP-2) et la marge réelle sur 500 textes en 3 registres. Brut : `/tmp/j1_dryrun.json`.
- **API testée LIVE** (uvicorn + curl, CPU) : `/health` 200 (`profiles_loaded: ["fr"]`, `device: cpu`), `/profiles` 200 traçabilité complète, `/detect` nominal 200 (score reproductible à l'identique sur 2ᵉ appel : 0.9378530979156494, `elapsed_ms` 3408 → 1240 via cache LRU, `detectors_cached: 1`), profil inconnu → **404**, texte < 50 car. → **422**, `/docs` Swagger 200.
- **Infra d'inférence RÉSOLUE** : la machine GPU = alias SSH **`gpu-ubuntu`** (147.135.140.193, OVH L4 Gravelines, user `ubuntu`, clé `~/.ssh/gpu_ubuntu` déjà trustée ; `gpu.maudet.cloud`/204.168.196.226 était un mauvais hostname — ports fermés, clé SSH inconnue). Athena est sur Tailscale (`100.64.110.85`) : LiteLLM joignable en direct sur `http://100.90.203.88:4000` (bind Tailscale). Clé « modèles » LiteLLM validée (liste : `luciole-1b-base`, `luciole-1b-instruct` ; génération testée) — stockée dans `.env` racine (chmod 600, gitignored, clé OpenRouter ajoutée depuis `~/.dsh/.env`). **Env de scoring provisionné** (`scripts/setup_gpu_env.sh`, idempotent) : venv `~/.venv-binoculars-eu`, torch 2.5.1+cu124, pins `requirements-eval.txt` — ⚠️ **2 pins du protocole §8.1 incohérents et corrigés avec TODO** (`tokenizers==0.20.3→0.22.2`, `huggingface_hub==0.27.0→0.36.2`, exigés par transformers 4.57.1) —, poids Base+SFT (~5 Go), **smoke scoring GPU OK** (score 0.9432 sur texte témoin, `cuda:0/cuda:0`). vLLM sur la box : services `luciole-1b-base` (:8010) + `luciole-1b-instruct` (:8011) actifs ; poids 8B/23B déjà en cache HF ; Qwen TTS arrêté ; bascule de modèles autorisée par l'utilisateur ; 23B bf16 impossible sur 24 Go → prévu en nf4 (vLLM bitsandbytes) ou endpoint LINAGORA (en panne, décision utilisateur : monter en local en arrêtant les autres). Corpus humain : **blog.maudet.cloud retenu** (Ghost, registre éditorial/tribunes, source idéale côté FR) en remplacement de la strate LinkedIn (consentement), + Wikipedia/Wikisource/presse libre.
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

## 6. Runbook autonome (nuit du 30 → 31 août)

Ordre décidé avec l'utilisateur avant son absence. Arrêts durs respectés : **pas de `--write`**, **pas de changement de composition du corpus**, publication publique interdite (repo privé OK).

1. **Watcher 23B** (`bash-ha672i1k`) → si ready : génération 100 jumeaux (`python -m calibration.build_corpus --generators 23b`, merge incrémental) → corpus **500/500**.
2. Si 23B torchao échoue → **plan C** : `calibration/generate_23b_hf.py` (transformers + BitsAndBytesConfig nf4, venv `~/.venv-binoculars-eu` de la box, bnb 0.44.1 + transformers 4.57.1 pinés) — distribution équivalente au nf4 vLLM.
3. Si plan C échoue → **stop**, rapport au réveil, décision (endpoint LINAGORA / substitution 8B documentée).
4. Corpus complet → `python -m calibration.build_splits --corpus … --output calibration/splits_fr_v01.json` (seed 42) ; rsync `calibration/` → `gpu-ubuntu:~/projects/binoculars-eu/`.
5. **Dry-run calibrate autorisé** (SANS `--write`) sur la box : `~/.venv-binoculars-eu/bin/python -m calibration.calibrate --corpus … --splits … --profile fr --batch-size 8` → `results_fr_v01.json` + `scores_fr_v01.json` + `error_analysis_candidates_fr_v01.json` ; retrait des résultats sur athena, commit + push, **rapport du matin** (train AUC, seuils candidats) — décision `--write` réservée à l'utilisateur.
6. `bash scripts/luciole_switch.sh 1b` (rétablit la passerelle LiteLLM) — en fin de séquence GPU.
7. Optionnel si tout est vert et de l'énergie reste : écrire `calibration/evaluate.py` (métriques primaires + IC bootstrap §3, baselines B0-B4 §4) — **sans jamais toucher au test set au-delà d'un seul run protocolaire** (§2.2).

## 7. Comment reprendre

Nouvelle session : « Reprends binoculars-eu à partir de `CHECKPOINT.md` ». Contexte complet ici + `binoculars-eu-prd.md` (contrat). Commits git atomiques par phase sur `main`, en anglais, **toujours avec confirmation explicite de l'utilisateur** (ex. `feat(detector): add Binoculars scoring engine around LanguageProfile`).
