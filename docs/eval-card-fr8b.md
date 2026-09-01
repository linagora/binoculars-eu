# Eval Card — profil `fr-8b` V0.2 (variante de capacité)

Profil de capacité du français (PRD §16.2) : paire **Luciole-8B-Base +
Luciole-8B-Instruct-1.1** en nf4 4-bit, architecture hybride
Mamba/Attention (`NemotronHForCausalLM`, `trust_remote_code=True`).
Première application de Binoculars à un modèle non purement Transformer.

## Résumé

Le profil `fr-8b` est la variante « montée d'échelle » du profil `fr` : même
corpus, mêmes splits, même protocole ; seule la paire de modèles change. Il
améliore nettement la séparation (AUC 0,988 vs 0,959) et **referme la
fragilité au bruit caractère** (R-1 : −0,046 vs −0,242 en V0.1).

## Métriques clés (hold-out test n = 100, IC 95 % bootstrap)

| Métrique | `fr` V0.1 | `fr-8b` V0.2 |
|---|---:|---:|
| AUC ROC in-distribution | 0.959 [0.914, 0.989] | **0.988 [0.971, 0.998]** |
| TPR @ FPR = 1 % | 0.480 [0.356, 0.923] | **0.900 [0.816, 0.980]** |
| TPR @ FPR = 5 % | 0.860 | 0.920 |
| F1 @ threshold `accuracy` | 0.879 | 0.923 |
| Accuracy @ threshold `accuracy` | — | 0.920 |
| AUC OOD Mistral-7B (v1) | 0.944 | 0.975 |
| 5-fold AUC | 0.971 ± 0.004 | 0.980 ± 0.003 |

Seuils calibrés (train only, seed 42) : accuracy **0.810022**, low-fpr
**0.737481**, tpr-at-fpr-1 **0.737481** (train AUC 0.978, F1 0.934).

## Delta de calibration vs V0.1 (amendment C)

Étude KS sur le dev : D(humain) = 0.48, D(IA) = 0.88, pooled D = 0.45 ;
max(D) = 0.88 > 0.1 → **seuils propres**, jamais un drop-in des seuils `fr`
(`calibration/distribution_study_fr8b.json`). Le 8B décale les deux classes
vers le bas (humain 1.02 → 0.92, IA 0.84 → 0.58 en moyenne dev) : un score
`fr-8b` n'est **pas** comparable en valeur absolue à un score `fr`.

## Robustesse (§6, test perturbé, une passe)

| Test | ΔAUC [IC 95 %] | Seuil | Verdict |
|---|---|---|---|
| R-1 fautes de frappe 5 % | **−0.046** [−0.096, −0.010] | ≤ −0.15 | **robuste** (V0.1 : fragile) |
| R-2 paraphrase légère 10 % (manuel) | +0.004 [−0.010, +0.019] | ≤ −0.20 | robuste |
| R-3 troncature 100 tokens | −0.010 [−0.036, +0.015] | ≤ −0.30 | robuste |
| R-4 concat humain/IA | score moyen 0.434 ∈ [0.4, 0.6] | — | conforme |
| R-5 adversarial prompting | +0.010 [0.000, +0.026] | ≤ −0.25 | robuste |
| R-6 adversarial rewriting | −0.043 [−0.083, −0.016] | ≤ −0.30 | robuste |

## OOD étendu (corpus v2, 120 textes, vs 50 humains du test)

| Famille | AUC [IC 95 %] |
|---|---:|
| Luciole-8B (in-distribution) | 1.000 |
| GPT-4o | 0.969 [0.928, 0.995] |
| Claude | 0.925 [0.855, 0.976] |
| Hybrides humain/IA | 0.851 [0.752, 0.937] |
| **Pooled** | **0.936** |

## Utilisation recommandée

```python
from binoculars_eu import Binoculars
detector = Binoculars.for_language("fr-8b", mode="low-fpr", load_in_4bit=True)
```

Chargement nf4 (~12 GiB VRAM les deux modèles) ; un L4 22 GiB suffit avec
marge. Ne pas invoquer `fr-8b` sans `load_in_4bit` sur ce type de carte
(int8 ≈ 17 GiB de poids + logits 128k vocab ne passent pas, PRD §16.2).

## Limites principales

- **Humanizers commerciaux** : le test R-6bis (TPR@low-fpr sur corpus
  humanisé, critère ≥ 0.30, amendment A) est **en attente** de la famille
  humanisée du corpus OOD v2. L'eval card V0.1 documente que le mode
  low-fpr V0.1 tombait face à Undetectable AI ; la V0.2 doit prouver le
  contraire ou l'assumer.
- **nf4** : la quantization 4-bit est un compromis VRAM documenté ; le gain
  net reste mesuré (AUC 0.988), mais le signal bruité par la quantization n'est
  pas démêlé de l'effet architecture hybride (PRD §16.2 point 4).
- **Hybrides humain/IA** : AUC 0.851, la famille la plus dure ; un texte
  moitié humain reste intrinsèquement ambigu pour un score global.
- Biais encyclopédique/littéraire hérités du corpus V0.1 (corpus v1.1
  reporté V0.3) ; textes courts (< ~100 tokens) peu fiables.
- Fraîgile restant : aucun test R-1..R-6 ne dégrade le verdict, mais un
  humanizer dédié appliqué à 100 % du texte reste l'attaque de référence.

## Données & traçabilité

- Corpus : `binoculars-eu-corpus-fr-v01` (500, sha256 dans
  `binoculars_eu/profiles/fr8b/metadata.json`) ; OOD v2 :
  `calibration/corpus/binoculars-eu-corpus-fr-v02-ood.jsonl` (120, sha256
  `2dbe78b7…`).
- Artefacts : `scores_fr-8b_v01.json`, `evaluation_fr-8b_v01.json`,
  `robustness_fr-8b_v01.json`, `distribution_study_fr8b.json`,
  `ood_v2_eval_fr8b.json`, `r56_pregen_fr_v02.jsonl`.
- Environnement : torch 2.5.1+cu124, transformers 4.57.1, bitsandbytes nf4,
  kernels mamba-ssm 2.2.6 / causal-conv1d ; L4 22 Go.

## Avertissement

Outil d'aide à la décision, **non une preuve** : un verdict « IA » n'est pas
une détection d'infraction. Ne pas employer pour des décisions à fort enjeu
sans validation humaine. Le profil `fr` (1B) reste disponible pour les
contextes où la latence/la VRAM priment ; `fr-8b` privilégie la qualité.
