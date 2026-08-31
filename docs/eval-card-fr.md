# Eval Card — profil `fr` V0.1

> Fiche synthétique à la manière des HF Model Cards, **une card par profil de langue**.
> À joindre à toute distribution PyPI et à toute annonce de communication.
> Rapport complet : `docs/evaluation_report_fr_v01.md` · Protocole : `calibration/protocol.md` v2.0-v0.1.

## Résumé

Détecteur zero-shot de texte généré par IA pour le **français**, fondé sur la méthode Binoculars (Hans et al., ICML 2024) : un modèle *observer* (Luciole-1B-Base) mesure la perplexité, un modèle *performer* (Luciole-1B-SFT-1.0) la cross-entropie ; leur ratio forme le score de décision. Calibré sur un corpus dédié de 500 textes français (symétrie humain/IA jumelée), évalué selon un protocole figé et reproductible.

## Métriques clés (hold-out test n = 100, IC 95 % bootstrap)

| Métrique | Valeur | Cible V0.1 |
|---|---|---|
| **TPR@FPR = 1 %** (headline) | **0.480** [0.356, 0.923] | ≥ 0.45 ✔ |
| AUC ROC | 0.959 [0.914, 0.989] | ≥ 0.80 ✔ |
| F1 @ seuil `accuracy` | 0.879 [0.800, 0.941] | ≥ 0.75 ✔ |
| AUC OOD Mistral 24B | 0.944 [0.896, 0.984] | ≥ 0.65 ✔ |

Seuils calibrés (train 300, seed 42) : `accuracy` 0.955801 · `low_fpr` 0.866667 · `tpr_at_fpr_1` 0.866667.

## Utilisation recommandée

- Textes français **propres** (≥ ~100 tokens, voir la fragilité aux typos) : verdicts en mode `accuracy` (équilibre) ou `low-fpr` (prudence).
- Score et seuils sont **spécifiques au français** : ne pas réutiliser pour d'autres langues (un profil par langue, aucune mutualisation de seuils).
- Latence : ~57 ms/texte (P50, GPU L4, bf16, batch 8) ; VRAM ~9,8 Go.

## Limites principales

- **Fragile aux fautes de frappe** (ΔAUC −0.242 au test R-1) : ne pas utiliser sur OCR bruité / saisie mobile sans pré-traitement.
- **Avantage « langue native » non démontré à 1B** face au Binoculars Falcon-7B original (écarts non significatifs à 95 %) — chantier V0.2 (paire 8B).
- Corpus IA 100 % Luciole ; GPT/Claude/Qwen non couverts (OOD Mistral seulement).
- Biais de corpus (Wikipédia/presse dominants, pas d'oral/SMS/dialectes) ; risque de faux positifs sur locuteurs non-natifs.
- **Faux positifs fréquents sur humains** : textes encyclopédiques denses (Wikipédia) et littérature classique classés « IA » dans ~la moitié des cas (dev par source : 0.500 / 0.556) ; les textes IA sont détectés ≥ 93 %. Éviter ces contextes ou exiger une revue humaine.
- Textes très courts (< ~100 tokens) : score peu fiable (bruit structurel).
- Qualité du corpus : 60/500 textes (source « presse ») contiennent des caractères `U+FFFD` (scraping décodé en charset erroné) ; affecte la propreté typographique, pas les métriques ni la reproductibilité (sha256 publié couvre le corpus tel quel). Correctif en corpus v1.1.
- **Attaque humanizer commerciale dédiée (hors périmètre R-1..R-6)** : passage d'un texte Luciole-8B raw par Undetectable AI (mode Balanced, 5 USD/mois, 1 clic) fait basculer le verdict de « IA high » à « Humain » sur les deux modes de seuil (score 0.8245 → 0.9833, Δ +0.159, texte source 578 mots → humanisé 730 mots, test du 2026-08-31 · git_sha `7c58b81`). Limite structurelle partagée par les 7 détecteurs commerciaux testés en parallèle par Undetectable AI (GPTZero, Copyleaks, QuillBot, Writer, Sapling, Grammarly, ZeroGPT) ; à notre connaissance, aucun détecteur zero-shot publié en 2026 n'y résiste sans cascade défensive. Roadmap : R-6bis officialisé au protocole V0.2, cascade défensive étudiée en V2.


## Données & traçabilité

- Corpus : `binoculars-eu-corpus-fr-v01` (500) + `-ood` Mistral (50) — publication HF Datasets `OpenLLM-France/` prévue ; sha256 dans `binoculars_eu/profiles/fr/metadata.json`.
- Splits 60/20/20 stratifiés seed 42 ; test évalué **une seule fois** (discipline §2.2).
- Artefacts d'évaluation versionnés dans `calibration/` avec sha git par run.

## Reproductibilité

Seeds figées (§1 du protocole) · `requirements-eval.txt` épinglé · scripts `calibration/*.py` · déterminisme vérifié (écart-type inter-seed 0). Test de reproduction par un tiers : `python -m calibration.evaluate …` documenté dans le rapport §8.

## Avertissement

Outil d'aide à la décision, **non une preuve** : un verdict « IA » n'est pas une détection d'infraction. À utiliser avec discernement, en connaissance des limites ci-dessus. Ne pas employer pour des décisions à fort enjeu (disciplinaires, évaluations) sans validation humaine.
