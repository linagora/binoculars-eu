# Plan d'exécution V0.2 — profil `fr-8b` (variante de capacité)

Statut : validé par Michel-Marie Maudet le 2026-08-31 (structure + amendements A/B/C + arbitrages).
Référentiels : PRD §4.2, §14.2 (amendé A), §16.2 ; amendements B (OOD humanisé) et C (étude KS).

## Décisions figées

| Point | Décision |
|---|---|
| Critère d'acceptation (amendement A, remplace §14.2/§16.2) | **AUC test ≥ 0.97 ET TPR@low_fpr sur corpus humanisé ≥ 0.30** (R-6bis : limite low_fpr V0.1 face à Undetectable AI, eval card V0.1) |
| Famille humanisée OOD (amendement B) | 30 sources IA (Luciole/GPT-4o/Claude) × 3 humanizers (Undetectable AI, WriteHuman, StealthGPT) = **90 textes** — **réalisé : 2 humanizers × 23 sources = 136 textes** (StealthGPT geo-bloqué UE citant l'article 50(2) ; TextGuard sans API ; 21 sources/famille relançables après rechargement de crédits Undetectable) |
| Étude de distribution (amendement C) | KS 1B bf16 vs 8B int8 sur dev, par classe + pooled ; si max(D) > 0.1 → seuils propres `fr-8b` recalibrés sur train, jamais drop-in |
| Exposition | Nouveau code de profil **`fr-8b`** (pas un flag) |
| Corpus v1.1 (mojibake + littérature) | **Reporté V0.3** |
| Comparaison GPTZero/Originality (§4.2 « si accès ») | **Reportée** |
| Budget humanizers | Abos ponctuels WriteHuman + StealthGPT (~25-30 USD) ; Undetectable = abo annuel existant (accès API à confirmer au lancement de P3.2, repli web documenté) — **réalisé : accès API Undetectable (abo annuel) + WriteHuman (API, abo souscrit 2026-09-01) ; StealthGPT impossible depuis l'UE** |

## Statut d'exécution (2026-09-01)

P0, P1, P2, P3.1, P3.3, P4 (rédaction) : **faits** (HEAD `ef050a3`).
P3.2 : **fait** — corpus humanisé 136 textes, R-6bis **passé** (0.309 [0.228, 0.390]).
Verdict : critères A passés → profil `fr-8b` **cible officielle** (PRD §16.2).
Décision produit 2026-09-01 : `fr-8b` devient le **profil par défaut**
API/CLI (`DEFAULT_PROFILE_CODE`), avec `default_load_in_4bit=True` sur le
profil ; `fr` 1B reste disponible pour les contextes CPU/petit GPU. Détail :
rapport `docs/evaluation_report_fr8b_v02.md` §10.

## P0. Faisabilité + étude de distribution (box, cible 2-3 jours)

Box : L4 22 Go via Tailscale `100.90.203.88`, venv `~/.venv-binoculars-eu`, repo `~/projects/binoculars-eu` (rsync, pas de git).

- **P0.1 Kernels** : `uv pip install mamba-ssm causal-conv1d bitsandbytes` ; gate : imports OK.
- **P0.2 Smoke test 8B int8** : `LanguageProfile` inline (observer `OpenLLM-France/Luciole-8B-Base`,
  performer `Luciole-8B-Instruct-1.1`, `trust_remote_code=True`),
  `Binoculars(profile=..., load_in_8bit=True)` (le moteur supporte déjà `load_in_8bit`,
  voir `detector.py`). Gate G0 : chargement OK, VRAM < 22 Go.
- **P0.3 Étude KS** : nouveau `calibration/distribution_study.py` — score dev (n=100, splits figés)
  avec la paire 8B int8, comparaison aux scores dev 1B dans `calibration/scores_fr_v01.json`,
  `scipy.stats.ks_2samp` par classe + pooled, rapport `calibration/distribution_study_fr8b.json`.

## P1. Profil `fr-8b` + scoring corpus (cible 2-3 jours)

- `binoculars_eu/profiles/fr8b/` : instance enregistrée (seuils placeholder 0.9/0.85,
  `trust_remote_code=True`, auto-découverte registre) + `thresholds.json`/`metadata.json` placeholders.
- Extension CLI : flag `--load-in-8bit` dans `calibrate.py`, `evaluate.py`, `robustness.py`
  (transmis à `for_language`).
- Box (vLLM arrêté : `scripts/luciole_switch.sh stop`) :

```bash
python -m calibration.calibrate --corpus calibration/corpus/binoculars-eu-corpus-fr-v01.jsonl \
    --splits calibration/splits_fr_v01.json --profile fr-8b --load-in-8bit          # dry-run
python -m calibration.calibrate ... --profile fr-8b --load-in-8bit --write          # fige les seuils
```

## P2. Évaluation + robustesse (cible 3-4 jours)

```bash
python -m calibration.evaluate --corpus ... --splits ... --profile fr-8b --config v02 \
    --scores calibration/scores_fr8b_v02.json \
    --ood-corpus calibration/corpus/binoculars-eu-corpus-fr-v01-ood.jsonl
python -m calibration.robustness ... --profile fr-8b --load-in-8bit \
    --r2-file calibration/r2_paraphrases_fr_v01.jsonl \
    --generator-url http://100.90.203.88:8013 --generator-model luciole-8b-instruct
```

- **R-6bis** : nouveau `calibration/humanized_eval.py`, TPR@low_fpr sur le corpus humanisé
  (P3), critère ≥ 0.30.
- Gate G2 : AUC test ≥ 0.97 ; sinon bascule §16.2 (1B+1B reste officiel, `fr-8b` exploratoire).

## P3. Corpus OOD v2 + famille humanisée (cible 1 semaine)

- **P3.1 Sources** (`calibration/generate_ood_v2.py`, pattern `generate_ood_mistral.py`) :
  30 Luciole-8B (vLLM local, prompts jumeaux) + 30 GPT-4o (OpenRouter) + 30 Claude (OpenRouter)
  + 30 hybrides (concat déterministe, seed figé). Sortie `binoculars-eu-corpus-fr-v02-ood.jsonl`.
- **P3.2 Humanisation** : 30 sources IA × 3 humanizers = 90 textes (API si clé, sinon kit web). —
  **Réalisé** : `calibration/humanize_undetectable.py` (API v2, 69 textes, paramètres = test au clou
  V0.1, ~1 crédit/mot, cache de reprise) + `calibration/humanize_writehuman.py` (API, 67 textes,
  2 rejets de modération « violence ») → corpus fusionné
  `binoculars-eu-corpus-fr-v02-ood-humanized.jsonl` (136). R-6bis via `humanized_eval` :
  **0.309 [0.228, 0.390] ≥ 0.30, PASSÉ**.
- **P3.3 Scoring** : 150 textes en une passe `fr-8b` (vLLM arrêté) → `scores_ood_v2_fr8b.json`.

## P4. Acceptation + documentation (cible 2-3 jours)

- Critères : AUC test ≥ 0.97 ; TPR@low_fpr humanisé ≥ 0.30 ; delta KS explicité.
- Livrables : `docs/evaluation_report_fr8b_v02.md`, `docs/eval-card-fr8b.md`,
  README (ligne profil), JSON de robustesse.
- Commits atomiques, ruff + mypy strict + suite de tests verte à chaque étape.

## Risques

Kernels mamba-ssm fragiles (compilation source en repli) ; VRAM juste en int8
(baisser le batch) ; API humanizers indisponibles au run (repli web) ; cohabitation
vLLM/scoring (séquencé : génération P3.1 → stop → scoring P3.3).
