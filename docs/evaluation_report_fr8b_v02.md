# Rapport d'évaluation — profil `fr-8b` V0.2 (variante de capacité)

Plan d'exécution : `docs/v02_plan.md` (validé 2026-08-31, amendements A/B/C).
Ce rapport suit la structure du rapport V0.1 (`docs/evaluation_report_fr_v01.md`)
pour comparabilité directe.

## 1. Executive summary

La variante de capacité `fr-8b` (Luciole-8B ×2, nf4, hybride Mamba/Attention)
passe le premier critère d'acceptation (amendment A) : **AUC test 0.988
[0.971, 0.998] ≥ 0.97**. Elle améliore chaque métrique du profil `fr` et
ferme la fragilité R-1 (bruit caractère) qui était le principal défaut
documenté de la V0.1. Le second critère (TPR@low_fpr sur corpus humanisé
≥ 0.30, R-6bis) est **en attente** de la famille humanisée du corpus OOD v2
(P3.2, accès aux humanizers commerciaux).

## 2. Setup

- Paire : `OpenLLM-France/Luciole-8B-Base` (observer) +
  `Luciole-8B-Instruct-1.1` (performer), `NemotronHForCausalLM`,
  `trust_remote_code=True`, tokenizer partagé depuis l'observer.
- Quantization : **nf4 4-bit** (bitsandbytes). int8 impossible sur 22 Go :
  ~17 Go de poids + logits vocab 128k + activations > VRAM (constaté en P0.2,
  OOM à la loss). PRD §16.2 documente int4 comme repli explicite.
- Kernels : mamba-ssm 2.2.6, causal-conv1d (compilés nvcc 12.9 contre torch
  2.5.1+cu124 ; mamba-ssm 2.3.x exige triton ≥ 3.5, incompatible).
- Environnement : box L4 22 Go, venv épinglé `requirements-eval.txt`,
  scoring GPU dédié (vLLM arrêté), générations R-5/R-6 et OOD v2 effectuées
  dans une fenêtre vLLM 8B séparée (pré-génération `--r56-file`).
- Corpus/splits : identiques V0.1 (sha256 `ca70aaa4…`, splits seed 42) ;
  le test n'a été scoré qu'une fois (discipline §2.2).

## 3. Calibration et décision de distribution (amendment C)

- Étude KS dev (n=100) : D(humain)=0.48, D(IA)=0.88, pooled=0.45 ;
  **max(D)=0.88 > 0.1 → seuils propres** (`calibration/distribution_study_fr8b.json`).
- `--write` sur train : accuracy 0.810022 (F1 0.9342), low_fpr 0.737481
  (FPR train 0.0067), tpr_at_fpr_1 0.737481 (TPR@1 % 0.86) ; train AUC 0.978.
- Dérive systématique vers le bas des scores 8B (humain 1.02→0.92, IA
  0.84→0.58 en moyenne dev) : **les scores `fr-8b` ne sont pas comparables
  en valeur absolue aux scores `fr`** ; documentation obligatoire (eval card).

## 4. Métriques primaires (hold-out test n = 100, passe unique)

| Métrique | `fr` V0.1 | `fr-8b` V0.2 | IC 95 % |
|---|---:|---:|---|
| AUC ROC | 0.959 | **0.988** | [0.971, 0.998] |
| TPR @ FPR = 1 % | 0.480 | **0.900** | [0.816, 0.980] |
| TPR @ FPR = 5 % | 0.860 | 0.920 | [0.840, 1.000] |
| F1 @ accuracy | 0.879 | 0.923 | [0.867, 0.972] |
| Accuracy @ accuracy | — | 0.920 | [0.870, 0.970] |
| ECE | 0.1495 | 0.1957 | (min-max on train) |

Complément 5-fold (seuils figés) : AUC 0.980 ± 0.003, TPR@1 % 0.845 ± 0.030.
Critère AUC ≥ 0.97 : **PASSÉ**.

## 5. Robustesse (§6, une passe, R-2 manuel + R-5/R-6 pré-générés)

| ID | Perturbation | `fr` V0.1 | `fr-8b` V0.2 | Verdict V0.2 |
|---|---|---:|---:|---|
| R-1 | Fautes 5 % | **−0.242 FRAGILE** | **−0.046** | robuste |
| R-2 | Paraphrase 10 % manuelle | −0.005 | +0.004 | robuste |
| R-3 | Troncature 100 tokens | −0.022 | −0.010 | robuste |
| R-4 | Concat humain/IA | 0.545 | 0.434 ∈ [0.4, 0.6] | conforme |
| R-5 | Adversarial prompting | −0.039 | +0.010 | robuste |
| R-6 | Adversarial rewriting | −0.030 | −0.043 | robuste |

Lecture : la montée 8B + nf4 transforme le seul point fragile V0.1 en test
robuste, sans dégrader aucun autre. L'amélioration R-1 (−0.242 → −0.046) est
le gain principal de la V0.2 côté robustesse.

## 6. OOD

- OOD v1 (Mistral-7B, 50) : AUC **0.975** [0.941, 0.996] (V0.1 : 0.944).
- OOD v2 (120, générateurs jamais calibrés) : Luciole-8B 1.000, GPT-4o 0.969,
  Claude 0.925, hybrides 0.851, pooled 0.936 (`calibration/ood_v2_eval_fr8b.json`).
- La famille « humanisé commercial » (30 sources × 3 humanizers, amendment B)
  est en attente de P3.2 ; elle complétera ce tableau et alimentera R-6bis.

## 7. Analyse d'erreurs

Candidats extraits du dev par `calibrate` pour `fr-8b`
(`calibration/error_analysis_candidates_fr-8b_v01.json`, 20 pires FP + 20
pires FN). Annotation reportée : la V0.2 n'étend pas la taxonomie §7, les
enseignements V0.1 (biais encyclopédique/littéraire, résidu FN
in-distribution) restent la référence, à vérifier empiriquement sur ces
candidats si le profil devient la cible officielle.

## 8. Reproductibilité

| Élément | Valeur |
|---|---|
| Seeds | split 42 · bootstrap 100 · torch 42 · OOD v2 800-803 · R-1..R-6 500-503 |
| Environnement | torch 2.5.1+cu124, transformers 4.57.1, mamba-ssm 2.2.6, causal-conv1d, bitsandbytes nf4 |
| Artefacts | scores/evaluation/robustness `fr-8b` v01, distribution_study_fr8b, ood_v2_eval_fr8b, r56_pregen_fr_v02 |
| Chaîne | `calibrate --write` → `evaluate --config v02` → `robustness --r2-file --r56-file` → `ood_v2_eval` ; R-5/R-6 pré-générés via `pre_generate_r56` |
| Traçabilité | chaque artefact porte git_sha + seuils lus depuis le profil enregistré |

## 9. Limites

1. **R-6bis non évalué** (critère 2 d'acceptation) : en attente de P3.2.
2. **Effet quantization × architecture non démêlé** (PRD §16.2.4) : le nf4
   et l'hybride Mamba varient ensemble ; un run 8B bf16 sur carte plus grande
   isolerait les deux effets (piste d'ablation V0.3 si nécessaire).
3. **Hybrides humain/IA à 0.851** : limite structurelle d'un score global
   sur un texte moitié humain.
4. Corpus V0.1 inchangé (mojibake presse, littérature sous-représentée) :
   fix reporté V0.3 (corpus v1.1 + recalibration).
5. Comparaison GPTZero/Originality (§4.2 « si accès ») : reportée, abonnements
   non souscrits.

## 10. Verdict d'acceptation (amendment A)

- AUC test ≥ 0.97 : **PASSÉ** (0.988 [0.971, 0.998]).
- TPR@low_fpr corpus humanisé ≥ 0.30 : **EN ATTENTE** (P3.2).
- Décision finale de promotion du profil `fr-8b` (cible officielle vs
  exploratoire, PRD §16.2) : après R-6bis.
