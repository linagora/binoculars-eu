# Plan d'exécution V0.3 — corpus v1.1 + recalibration des profils `fr` et `fr-8b`

Statut : proposé le 2026-09-01, en attente de validation par Michel-Marie Maudet.
Référentiels : PRD §1/§4.2, protocole `calibration/protocol.md` §1 (constitution du
corpus), plan V0.2 (`docs/v02_plan.md`) dont les limites 4 (corpus v1.0) et 7
(humanisé partiel) sont le périmètre de cette version.

## Motivation et périmètre (audit v1.0, 2026-09-01)

Constats chiffrés sur `binoculars-eu-corpus-fr-v01.jsonl` (500 = 250 humains +
250 IA) :

| Défaut | Étendue | Conséquence |
|---|---|---|
| Mojibake (décodeur `fetch_presse`) | 60/60 records `human-presse-fr-*` | 24 % des humains bruités ; biais d'encodage appris par les seuils |
| Titres vides (`"title": ""`) | 60 records presse | prompts de jumeaux dégénérés |
| Jumeaux IA à sujet vide (`sur : .`) | 20 records (`meta.twin_of` = presse) | 8 % des IA sans ancree topique ; à régénérer |
| Twins déséquilibrés | wikipedia 230, presse 20, blog/linuxfr/littérature 0 | écart à §1 du protocole ; à trancher (question ouverte ci-dessous) |
| Littérature (wikisource) | 40 records, qualité non auditée | audit P1.3 (doublons, longueurs, homogénéité) |

Décisions figées :

| Point | Décision |
|---|---|
| Fichier corpus | `calibration/corpus/binoculars-eu-corpus-fr-v1.1.jsonl` (n = 500, mêmes ids sauf remplacements documentés) + sha256 + changelog |
| Fix presse | re-fetch des 60 URLs avec décodage correct + extraction de titre réelle (repli : réparation par transcodage latin1/UTF-8 si page indisponible) |
| Twins presse | régénération des 20 jumeaux à prompts dégénérés, seeds conservés, vLLM luciole-8b box |
| Recalibration | les DEUX profils (`fr` 1B et `fr-8b` nf4), splits v1.1 seed 42, train-only, test scoré une fois |
| Gates | `fr` : AUC ≥ 0.80 et TPR@FPR=1 % ≥ 0.45 (cibles V0.1) ; `fr-8b` : AUC ≥ 0.97 et R-6bis ≥ 0.30 (amendement A) |
| Ablation nf4 vs bf16 | **reportée** (carte > 22 Go ou offload CPU non disponible) |
| Complément humanisé 21/famille | intégré dès rechargement des crédits Undetectable (action MMM) ; rescoring dans P2 |
| TextGuard | optionnel, seulement si WebBridge installé |

## P0. Audit formalisé (0,5 j)

- `calibration/audit_corpus.py` : mojibake par source (humain + IA), titres vides,
  distribution des twins, longueurs, doublons exacts, détection d'encodage.
- Sortie `calibration/audit_corpus_v10.json` ; le périmètre v1.1 est chiffré à
  partir de ce rapport, pas d'estimations.
- Gate G0 : rapport complet, sans exception non listée.

## P1. Corpus v1.1 (1-2 j)

- **P1.1** Corriger `fetch_presse` (`human_sources.py`) : décodage via
  `response.apparent_encoding`, titre via `og:title`/`h1`. Re-fetch des 60 URLs
  des meta ; remplacement record par record (id conservé).
- **P1.2** Régénérer les 20 twins presse (`twin_of` presse) avec le vrai titre,
  mêmes seeds et paramètres (0.7/0.9), via vLLM luciole-8b sur la box
  (fenêtre dédiée, puis stop).
- **P1.3** Audit littérature : doublons, longueurs, extraits vides ; re-fetch
  wikisource de remplacement via `WIKISOURCE_TITLES` si besoin.
- **P1.4** Assemblage v1.1 (`build_corpus.py` réutilisé), sha256, changelog dans
  `calibration/protocol.md` (addendum v1.1) et métadonnées.
- Gate G1 : 0 mojibake résiduel sur les 60 presse ; 20/20 twins à sujet non vide.

## P2. Recalibration + chaîne d'évaluation (2-3 j box GPU)

```bash
# splits v1.1 (seed 42, protocole §2.1)
python -m calibration.build_splits --corpus .../binoculars-eu-corpus-fr-v1.1.jsonl \
    --output calibration/splits_fr_v11.json
# scoring des deux profils (vLLM arrêté ; fr 1B en bf16 ~5 Go, fr-8b nf4 ~12 Go)
python -m calibration.calibrate --corpus .../v1.1.jsonl --splits .../v11.json \
    --profile fr --write
python -m calibration.calibrate ... --profile fr-8b --load-in-4bit --write
# évaluation une passe + robustness + OOD + humanisé, pour les deux profils
python -m calibration.evaluate --config v03 ...
python -m calibration.robustness ... --r2-file ... --r56-file ...
python -m calibration.ood_v2_eval ...
python -m calibration.humanized_eval ... --profile fr-8b
```

- R-1/R-3/R-4 : perturbations dérivées des nouveaux splits (rejeu des scripts).
  R-2 : kit manuel existant rescoré. R-5/R-6 : pré-génération vLLM puis scoring
  (séquencé, comme V0.2).
- OOD v1 (Mistral, 50) et OOD v2 (120) : rescoring ; corpus inchangés.
- Mise à jour `thresholds.json` + `metadata.json` (corpus_sha256 v1.1,
  calibration_date) des deux profils.
- Gate G2 : cibles V0.1 (`fr`) et amendment A (`fr-8b`) atteintes ; sinon
  analyse d'erreurs avant toute décision.

## P3. Documentation + livraison (1 j)

- Rapports : `docs/evaluation_report_fr_v11_v03.md`,
  `docs/evaluation_report_fr8b_v11_v03.md` ; eval cards v1.1 des deux profils ;
  README (liens corpus, table profils, chiffres clés).
- Commits atomiques Conventional Commits ; ruff + pytest verts à chaque étape ;
  artefacts git_sha tracés.

## Questions ouvertes — tranchées le 2026-09-01

1. **Twins : alternative minimale retenue** — corriger uniquement les 20 twins
   presse à sujet vide, documenter l'équilibre wikipedia-dominant (230/20/0/0/0)
   comme limite du corpus v1.1. Pas de génération supplémentaire, n = 500
   inchangé. Un rééquilibrage éventuel relève d'une conception corpus v2.0.
2. TextGuard (3e humanizer) : seulement si WebBridge installé par MMM.

## Risques

Re-fetch presse : pages indisponibles/modifiées depuis 2026-08 (repli
transcodage + titre reconstruit depuis l'URL) ; dérive des métriques après
correction du biais d'encodage (les seuils v1.0 « apprenaient » le mojibake :
une baisse de AUC n'est pas un échec, c'est la mesure du biais) ; fenêtre vLLM
sur L4 22 Go (séquencée, comme V0.2).
