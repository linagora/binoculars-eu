# Rapport d'évaluation — profil `fr` V0.1

**Projet** : binoculars-eu (détection zero-shot de texte généré par IA, profils de langue)
**Profil évalué** : `fr` (Luciole-1B-Base observer / Luciole-1B-SFT-1.0 performer)
**Protocole** : `calibration/protocol.md` v2.0-v0.1 — passage complet §1-§9
**Date de l'évaluation** : 2026-08-31 · **Commits** : `7c58b81` (évaluation), `9fe126a` (baselines), `4284cc2` (ablations), `a04c73e` (robustesse)

---

## 1. Executive summary

La métrique headline **TPR@FPR = 1 % sur le hold-out test est de 0.480** (IC 95 % bootstrap [0.356, 0.923]), au-dessus de la cible V0.1 (≥ 0.45). L'**AUC est de 0.959** [0.914, 0.989] pour une cible ≥ 0.80, et l'**AUC OOD Mistral 24B de 0.944** [0.896, 0.984] pour une cible ≥ 0.65 — le profil généralise donc au-delà du générateur Luciole qui compose le corpus. Les cinq cibles primaires chiffrées du protocole §3.1 sont atteintes (section 3).

Trois résultats structurels encadrent la lecture de ces chiffres :

- **Le détecteur est fragile aux fautes de frappe** (R-1 : ΔAUC −0.242, au-delà du seuil de fragilité −0.15). Il ne doit pas être utilisé seul sur du texte bruité (OCR, saisie mobile) — section 6.
- **L'avantage « francophonie » n'est pas mesurable à l'échelle 1B** : la baseline B3 (Binoculars original Falcon-7B anglais) domine le profil `fr` en valeurs ponctuelles sur toutes les métriques, avec des intervalles de confiance qui se chevauchent (section 4). Le pari de la plateforme — des seuils et une traçabilité par langue — reste valide, mais le gain qualitatif à 1B n'est pas démontré sur ce corpus.
- **La paire Base+Instruct-1.1 surpasse Base+SFT-1.0 sur le dev** (AUC 0.997 vs 0.970) — piste prioritaire pour la V0.2 (section 5).

## 2. Setup

| Volet | Valeur |
|---|---|
| Corpus | `binoculars-eu-corpus-fr-v01`, 500 textes (250 humains / 250 IA), sha256 `ca70aaa4…` |
| Sources humaines | wikipedia-fr (80), presse-fr (60), linuxfr (40), blog-maudet (30, remplacement documenté de la strate LinkedIn), littérature (40) |
| Générateurs IA | luciole-23b-instruct (100, nf4), luciole-8b-instruct (75), luciole-1b-instruct (75) |
| OOD | 50 textes Mistral 24B, sha256 `39b42fb8…`, seed 700 |
| Splits | 60/20/20 stratifié 4 axes, seed 42, sha256 `1b1b34f4…` → train 300 / dev 100 / test 100 |
| Paire de modèles | observer `OpenLLM-France/Luciole-1B-Base`, performer `OpenLLM-France/Luciole-1B-SFT-1.0` |
| Précision | bfloat16 (défaut de calibration) |
| Seuils (fittés sur train uniquement) | `accuracy` 0.955801 · `low_fpr` 0.866667 · `tpr_at_fpr_1` 0.866667 |
| Matériel d'évaluation | OVH L4 24 Go (Gravelines), torch 2.5.1+cu124, transformers 4.57.1 |
| Discipline §2.2 | test lu une seule fois par `evaluate.py` ; chaque run loggé dans `evaluation_runs_fr.jsonl` |

## 3. Métriques primaires

### 3.1 Hold-out test (n = 100) — passe unique

| Métrique | Résultat [IC 95 %] | Cible V0.1 | Atteint |
|---|---|---|---|
| **TPR@FPR = 1 %** (headline) | **0.480** [0.356, 0.923] | ≥ 0.45 | ✔ |
| AUC ROC | 0.959 [0.914, 0.989] | ≥ 0.80 | ✔ |
| TPR@FPR = 5 % | 0.860 [0.447, 0.980] | ≥ 0.65 | ✔ |
| F1 @ seuil `accuracy` | 0.879 [0.800, 0.941] | ≥ 0.75 | ✔ |
| Accuracy @ seuil `accuracy` | 0.870 [0.800, 0.940] | ≥ 0.75 | ✔ |
| AUC OOD Mistral 24B | 0.944 [0.896, 0.984] | ≥ 0.65 | ✔ |
| ECE | 0.1495 | ≤ 0.15 | ✔ (de justesse) |

IC = bootstrap percentile, 1000 rééchantillonnages, seed 100 (§3.3). L'ECE est calculée sur une carte min-max fittée sur train (le score Binoculars n'est pas probabiliste — choix documenté, voir TODO dans `evaluate.py`).

L'IC du TPR@1 % est large de part la construction : à FPR = 1 % sur 50 humains, le rééchantillon bootstrap peut tomber sur très peu de faux positifs admissibles. C'est la variance attendue à n = 100, pas un artefact.

### 3.2 Complément 5-fold (§2.3, corpus entier, seuils figés)

| Métrique | moyenne ± écart-type |
|---|---|
| AUC | 0.971 ± 0.004 |
| TPR@FPR = 1 % | 0.667 ± 0.050 |
| TPR@FPR = 5 % | 0.867 ± 0.050 |
| F1 @ `accuracy` | 0.899 ± 0.030 |
| Accuracy @ `accuracy` | 0.888 ± 0.030 |

Le hold-out (AUC 0.959, TPR@1 % 0.480) est **en dessous** de la moyenne 5-fold pour le TPR@1 % mais dans sa zone de variance — signal de fragilité du split à n = 100 publié comme tel (§2.3).

### 3.3 Matrice de confusion stratifiée (test, seuil `accuracy`)

Par source humaine :

| Source | n | FP (humain→IA) | FN (IA→humain) |
|---|---|---|---|
| wikipedia-fr | 16 | 6 | 0 |
| presse-fr | 12 | 2 | 0 |
| linuxfr | 8 | 0 | 0 |
| blog-maudet | 6 | 0 | 0 |
| littérature | 8 | 2 | 0 |
| luciole-8b-instruct (IA) | 15 | 0 | 2 |
| luciole-1b-instruct (IA) | 15 | 0 | 0 |
| luciole-23b-instruct (IA) | 20 | 0 | 1 |

Les faux positifs se concentrent sur le registre encyclopédique/neutre (taxon FP-6 « Wikipedia dense » attendu) ; les faux négatifs sur le plus grand générateur (8B). Détail complet dans `evaluation_fr_v01.json`.

## 4. Comparaison des baselines (§4)

Toutes les baselines sur le même hold-out, mêmes IC bootstrap. Seuil de décision naturel par baseline (0.5 pour B0, F1-optimal train pour B1/B2, seuil upstream pour B3, seuil calibré pour B4).

| Baseline | AUC [IC 95 %] | TPR@1 % | TPR@5 % | F1 | Accuracy |
|---|---|---|---|---|---|
| B0 Random | 0.472 [0.370, 0.584] | 0.000 | 0.020 | 0.549 | 0.490 |
| B1 Longueur (LR) | 0.024 [0.002, 0.061] | 0.000 | 0.000 | 0.658 | 0.490 |
| B2 Features shallow (LR) | 0.054 [0.011, 0.113] | 0.000 | 0.000 | 0.671 | 0.510 |
| B3 Binoculars-Falcon-EN (8-bit) | **0.990** [0.970, 1.000] | **0.600** | **0.980** | **0.949** | **0.950** |
| **B4 Profil `fr` (nous)** | 0.959 [0.914, 0.989] | 0.480 | 0.860 | 0.879 | 0.870 |

Lecture protocole §4.3 :

- **B4 domine statistiquement B0-B2** — exigence remplie. B1/B2 confirment que la longueur et les features de surface ne séparent pas nos paires humain/IA jumelles (le corpus contrôle la longueur par construction) : la détection Binoculars capture un signal que les heuristiques ne voient pas.
- **B4 ne domine pas B3.** B3 (paire Falcon-7B anglaise, ~7× plus de paramètres, seuils Hans et al. inchangés) mène sur les cinq métriques en valeurs ponctuelles, mais les ICs se chevauchent (AUC [0.970, 1.000] vs [0.914, 0.989]) → **écart non significatif à 95 %**. Conclusion honnête : à l'échelle 1B, la francophonie n'apporte pas de gain mesurable sur ce corpus. Ce résultat alimente la recommandation V0.2 (paire 8B, PRD §15).
- Baselines optionnelles B5-B8 (§4.2) : non évaluées (coût API / hors repo) — documentées comme telles.

## 5. Ablations (§5, dev n = 100, seeds 42/123/2024)

Chaque configuration scorée une fois (l'inférence `inference_mode` est déterministe et indépendante de l'ordre des batches ; l'écart-type inter-seed publié est 0, documenté honnêtement — §5.2/§8.3-2).

### A1 — `max_length`

| Valeur | AUC | TPR@1 % | latence P50 (ms) | VRAM (Mo) |
|---|---|---|---|---|
| 64 | 0.896 | 0.420 | 11 | 5 923 |
| 128 | 0.959 | 0.440 | 36 | 6 791 |
| 256 | 0.968 | 0.740 | 53 | 8 541 |
| **512 (référence)** | **0.970** | **0.720** | ~57 | ~9 800 |
| 1024 | 0.962 | 0.740 | 57 | 10 004 |

Plateau à partir de 256 ; 64 tokens dégrade nettement. Le défaut 512 est un bon compromis qualité/coût.

### A2 — précision

| Valeur | AUC | TPR@1 % | latence P50 (ms) | VRAM (Mo) |
|---|---|---|---|---|
| **bfloat16 (référence)** | **0.970** | **0.720** | ~57 | ~9 800 |
| float32 | 0.961 | 0.700 | 207 | 20 000 |
| 8-bit | 0.960 | 0.660 | 99 | 9 158 |

TODO(spec) : §5.1 demande float16, non exposé par le détecteur (PRD §16.1 ne couvre que bf16/int8) ; 8-bit est le bras « précision réduite » retenu. bf16 optimal ; fp32 coûte 2× la VRAM et ~4× la latence pour un gain nul ; 8-bit divise la VRAM pour −0.011 AUC.

### A3 — paire de modèles

| Paire | AUC | TPR@1 % |
|---|---|---|
| **Base + SFT-1.0 (référence, profil)** | 0.970 | 0.720 |
| Base + Instruct-1.1 | **0.997** | **0.920** |
| Instruct + Instruct-thinking | *skipped : dépôt privé* |

**Résultat majeur de l'ablation** : le performer Instruct-1.1 améliore fortement la séparation. Décision V0.1 (validée) : conserver SFT-1.0 pour la cohérence du profil publié ; la bascule vers Instruct-1.1 est la piste n°1 de la V0.2 (avec recalibration complète). Voir divergence brief/PRD documentée dans le profil.

### A4 — tokenizer partagé

| Configuration | Résultat |
|---|---|
| Tokenizer observer partagé (référence) | AUC 0.970 |
| Tokenizers séparés (assert stricte) | **assert déclenchée** : « Tokenizers are not identical for Luciole-1B-Base and Luciole-1B-SFT-1.0 » |

Les tokenizers Base et SFT-1.0 ne sont **pas identiques** : `share_tokenizer_from_observer=True` est donc nécessaire au fonctionnement, pas un raccourci cosmétique. Sanity check du protocole validé (le cas « séparé » est bien celui qui doit échouer ici).

## 6. Robustesse (§6, test perturbé)

| ID | Perturbation | ΔAUC [IC 95 %] | Seuil fragilité | Verdict |
|---|---|---|---|---|
| R-1 | Fautes de frappe 5 % (seed 500) | **−0.242** [−0.353, −0.143] | ≤ −0.15 | **FRAGILE** |
| R-2 | Paraphrase légère 10 % (103 phrases, manuel) | **−0.005** [−0.029, +0.019] | ≤ −0.20 | robuste |
| R-3 | Troncature 100 tokens | −0.022 [−0.058, +0.011] | ≤ −0.30 | robuste |
| R-4 | Concat humain/IA (seed 501) | score moyen normalisé **0.545** ∈ [0.4, 0.6] | — | ✔ conforme |
| R-5 | Adversarial prompting (seed 502, 8B) | −0.039 [−0.095, +0.006] | ≤ −0.25 | robuste |
| R-6 | Adversarial rewriting (seed 503, 8B) | −0.030 [−0.071, +0.004] | ≤ −0.30 | robuste |

ΔAUC = AUC(test perturbé) − AUC(test original), bootstrap apparié (seed 100, 1000 tirages). Pour R-5/R-6 (mono-classe perturbée), le tableau de scores complet est reconstruit : moitié perturbée + moitié originale (§6 évalue la détection sur l'ensemble du test).

**R-1 dépasse le seuil de fragilité** — obligation de rapport §6.2 : *ne pas faire confiance au détecteur sur du texte avec fautes de frappe/bruit caractère* (OCR basse qualité, saisie mobile). Recommandation : pré-traitement (correction) ou refus de score. Les attaques adversariales déclaratives (R-5) et de réécriture (R-6) suivant RAID sont au contraire bien absorbées. R-4 confirme un comportement honnête sur texte mixte (score proche du point indécidable 0.5).

**R-2 complété (2026-08-31)** : 103 phrases (10 % des phrases du test, sélection déterministe) reformulées manuellement à la légère, le reste byte-identique (`calibration/r2_paraphrases_fr_v01.jsonl`, annotation via `r2_kit_editor.html`). ΔAUC ≈ 0 : la reformulation humaine légère ne dégrade pas la détection — cohérent avec R-3 (insensibilité à la réduction de longueur). La seule perturbation qui fragilise réellement le détecteur reste la corruption au niveau caractère (R-1). R-5/R-6 régénérés à cette occasion (génération stochastique 8B, seeds figées) : valeurs actualisées dans le tableau, conclusions inchangées.

## 7. Analyse d'erreurs (§7)

- Candidats extraits du **dev** (jamais du test) : 20 pires FP + 20 pires FN dans `calibration/error_analysis_candidates_fr_v01.json`.
- Notebook d'annotation : `notebooks/04_error_analysis.ipynb` (widget ipywidgets, une case par catégorie de la taxonomie §7.1 + note libre, export `docs/error_analysis_annotations_fr_v01.json`).
- **Statut au gel de ce rapport** : annotation en attente. Les candidats sont pré-extraits et le FP/FN du test (section 3.3) pointe déjà le taxon dominant attendu **FP-6** (texte encyclopédique neutre, 8/10 erreurs humaines sur wikipedia-fr + littérature). Le rapport `docs/error_analysis_fr_v01.md` (table de contingence Catégorie × Compte, 5 exemples par catégorie majeure, mitigations V0.2) sera complété après annotation ; Cohen's kappa si second annotateur.

## 8. Reproductibilité (§8)

| Élément | Valeur |
|---|---|
| Seeds (§1) | split 42 · bootstrap 100 · torch 42 · génération 23B/8B/1B = 0/1/2 · R-1..R-6 = 500-503 · runs 42/123/2024 |
| Environnement | `requirements-eval.txt` (torch 2.5.1, transformers 4.57.1, …) ; venv épinglé `~/.venv-binoculars-eu` sur la box |
| Corpus sha256 | `ca70aaa41c90321e96dcb51f58f34371ed5905853b6b090b9e05c1886cb9abf3` (vérifié à chaque run) |
| Splits sha256 | `1b1b34f4fc349b93fa6e343411f00f18813d999fa06c62859c738e4bcee2be81` |
| Traçabilité git | chaque artefact porte `git_sha` + `git_dirty` (§8.4) ; runs d'évaluation append-only dans `evaluation_runs_fr.jsonl` |
| Déterminisme intra-seed | dry-run et `--write` de calibrate.py ont produit des seuils bit-identiques ; scoring déterministe sous `inference_mode` (écart-type inter-seed 0, §5) |

Dockerfile de référence §8.2 : livré (`docker/Dockerfile.eval`, §18.8 garantie 4 — build vérifié, smoke test entrypoint OK ; run GPU complet à faire sur la box avant release).

## 9. Limites

Reprises de §11 du protocole, instanciées sur ce run :

- **Biais de corpus** : wikipedia-fr + presse dominent ; pas de SMS, oral transcrit, dialecte, très courts. Les FP du test confirment le biais encyclopédique.
- **Biais diachronique** : corpus 2026 ; la pollution progressive du français par les tournures IA dégradera la séparation humain/IA.
- **Biais démographique** : origine sociolinguistique non annotée — risque de faux positifs sur locuteurs non-natifs.
- **Biais adversarial** : R-1 montre la fragilité au bruit caractère ; les humanizers co-évolutifs ne sont pas couverts.
- **Biais inter-profils** : non mesuré (un seul profil en V0.1) ; prévu en V1 (§11).
- **Biais de générateur** : corpus IA 100 % Luciole ; l'OOD Mistral (0.944) est rassurant mais GPT/Claude/Qwen/DeepSeek non testés.
- **Qualité du corpus (connue, non bloquante)** : 60/500 textes de la source « presse » contiennent des caractères `U+FFFD` (décodage charset erroné lors du scraping, en amont de ce projet) — 36 train / 12 dev / 12 test. Le sha256 publié couvre le corpus tel quel : les métriques ci-dessus et la reproductibilité en sont inchangées, mais la propreté typographique de ces textes n'est pas représentative. Correctif prévu en corpus v1.1 (re-scraping propre → recalibration complète).
- **Échelle** : l'avantage francophonie n'est pas démontré à 1B face au Falcon-7B original (section 4) ; la V0.2 (8B) est le chantier de preuve.

## 10. Annexes

Artefacts (tous dans `calibration/`, traçabilité git complète) :

- `corpus/binoculars-eu-corpus-fr-v01.jsonl` (+ `-ood`), `splits_fr_v01.json`
- `results_fr_v01.json`, `scores_fr_v01.json`, `error_analysis_candidates_fr_v01.json`
- `evaluation_fr_v01.json`, `evaluation_runs_fr.jsonl`
- `baselines_fr_v01.json`, `ablations_fr_v01.json`, `robustness_fr_v01.json`
- Scripts : `calibrate.py`, `evaluate.py`, `baselines.py`, `ablations.py`, `robustness.py`, `build_corpus.py`, `build_splits.py`, `protocol.md`

Figures ROC / distributions de scores / courbes de calibration : non générées à ce stade (tableaux chiffrés ci-dessus ; `evaluation_fr_v01.json` porte les scores par id pour tout re-tracé).

---

*Rapport produit par le pipeline de calibration de binoculars-eu. Protocole : `calibration/protocol.md` v2.0-v0.1. Toute publication se référant à binoculars-eu doit citer ce rapport et le protocole.*
