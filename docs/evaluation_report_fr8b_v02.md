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
≥ 0.30, R-6bis) est **passé** : 0.309 [0.228, 0.390] sur 136 textes
humanisés commercialement (§6). Les deux critères de l'amendment A sont
satisfaits : le profil `fr-8b` atteint le statut de **cible officielle**
(§10).

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
- Famille « humanisé commercial » (P3.2, amendment B) : **136 textes** issus
  des 3 familles IA sources (luciole-8b, GPT-4o, Claude), humanisés par
  **Undetectable AI** (API, 69 textes, paramètres identiques au test au clou
  V0.1 : University / General Writing / Balanced, modèle v2 multilingue) et
  **WriteHuman** (API, 67 textes ; 2 sources rejetées par la modération
  « violence », faux positifs sur vocabulaire historique : `claude-018`,
  `claude-022`). StealthGPT est geo-bloqué pour les clients UE (page de blocage
  citant l'article 50(2), constaté 2026-09-01) ; TextGuard n'expose pas d'API
  : la famille compte 2 humanizers au lieu des 3 prévus, écart à l'amendment B
  documenté. Corpus : `calibration/corpus/binoculars-eu-corpus-fr-v02-ood-humanized.jsonl`.
- **R-6bis** (`calibration/humanized_eval_fr8b_v02.json`) : TPR@low_fpr global
  **0.309** [0.228, 0.390] → **PASSÉ** (critère ≥ 0.30). Par famille :
  undetectable-luciole-8b 0.913, undetectable-claude 0.435,
  undetectable-gpt-4o 0.435, writehuman-luciole-8b 0.043,
  writehuman-claude 0.000, writehuman-gpt-4o 0.000.
- Lecture : le seuil low_fpr `fr-8b` résiste à Undetectable AI sur les sources
  Luciole et partiellement sur GPT-4o/Claude, mais **WriteHuman humanisé
  contourne quasi totalement le détecteur** (TPR ≈ 0 hors sources Luciole).
  Le passage du critère est donc porté par les familles undetectable et
  luciole ; la limite WriteHuman est structurelle et rejoint la conclusion
  du test au clou V0.1 (§7.3 du rapport V0.1).

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
| Artefacts | scores/evaluation/robustness `fr-8b` v01, distribution_study_fr8b, ood_v2_eval_fr8b, r56_pregen_fr_v02, humanized_eval_fr8b_v02 |
| Chaîne | `calibrate --write` → `evaluate --config v02` → `robustness --r2-file --r56-file` → `ood_v2_eval` ; R-5/R-6 pré-générés via `pre_generate_r56` ; humanisation via `humanize_undetectable` + `humanize_writehuman` (caches de reprise) puis `humanized_eval` |
| Traçabilité | chaque artefact porte git_sha + seuils lus depuis le profil enregistré ; exception connue : le git_sha de `humanized_eval_fr8b_v02.json` (68b36fb) provient du checkout de la box GPU dont le `.git` n'est pas synchronisé ; l'état canonique du code est Athena `ef050a3` + fichiers P3.2 non commités |

## 9. Limites

1. **WriteHuman humanisé quasi non détecté** (TPR@low_fpr ≈ 0 hors sources
   Luciole) : limite structurelle partagée avec toute la famille zero-shot,
   documentée §6. Le critère R-6bis est passé, mais la marge est faible
   (0.309, borne basse de l'IC 0.228) et dépend de la répartition des
   humanizers dans le corpus.
2. **Famille humanisée à 2 humanizers** au lieu de 3 (amendment B) :
   StealthGPT geo-bloqué UE, TextGuard sans API. Étendue possible V0.3.
3. **Effet quantization × architecture non démêlé** (PRD §16.2.4) : le nf4
   et l'hybride Mamba varient ensemble ; un run 8B bf16 sur carte plus grande
   isolerait les deux effets (piste d'ablation V0.3 si nécessaire).
4. **Hybrides humain/IA à 0.851** : limite structurelle d'un score global
   sur un texte moitié humain.
5. Corpus V0.1 inchangé (mojibake presse, littérature sous-représentée) :
   fix reporté V0.3 (corpus v1.1 + recalibration).
6. Comparaison GPTZero/Originality (§4.2 « si accès ») : reportée, abonnements
   non souscrits.
7. Sous-ensemble humanisé limité par le solde de crédits Undetectable AI
   (23 sources/famille au lieu de 30 ; 21 restantes/famille relançables après
   rechargement, caches de reprise en place).

## 10. Verdict d'acceptation (amendment A)

- AUC test ≥ 0.97 : **PASSÉ** (0.988 [0.971, 0.998]).
- TPR@low_fpr corpus humanisé ≥ 0.30 : **PASSÉ** (0.309 [0.228, 0.390]).
- **Décision : le profil `fr-8b` atteint le statut de cible officielle**
  (PRD §16.2, critères d'acceptation V0.2).
- Décision produit (2026-09-01) : `fr-8b` devient le **profil par défaut** de
  l'API et de la CLI (`DEFAULT_PROFILE_CODE = "fr-8b"`). Le chargement par
  défaut suit le profil (`default_load_in_4bit=True`) : un appel nu
  `Binoculars()` charge la paire en nf4 (~12 GiB), jamais en bfloat16. Le
  profil `fr` (1B) reste disponible via `for_language("fr")` pour les
  contextes CPU/petit GPU ; les scores des deux profils ne sont pas
  comparables en valeur absolue (§3).
