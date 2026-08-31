# Analyse d'erreurs — profil `fr` V0.1 (protocole §7)

- **Candidats** : 40 pires erreurs du **dev** (jamais du test) — 20 faux positifs (humain classé IA) + 20 faux négatifs (IA classée humain), extraits par `calibration/evaluate.py`.
- **Annotation** : agent, **relue et validée par Michel-Marie Maudet** le 2026-08-31 (`docs/error_analysis_annotations_fr_v01.json`, `review_status: human_reviewed`). Cohen's kappa non calculable (annotateur unique).
- **Seuil** : accuracy 0.955801. Rappel de convention : score *bas* = IA-like ; humain si score > seuil.

## 1. Tables de contingence

### Faux positifs (humain classé IA)

| Code | Description | Compte |
|------|-------------|-------:|
| FP-1 | Administratif / juridique | 0 |
| FP-2 | Texte très court (< 100 tokens) | 0 |
| FP-3 | Traduction post-éditée | 0 |
| FP-4 | Code, chiffres, tableaux | 0 |
| FP-5 | Technique standardisé (RFC, spec, forum) | 2 |
| **FP-6** | **Encyclopédique très neutre (Wikipedia dense)** | **12** |
| FP-autre | Autre (voir notes) | 6 |
| **Total** | | **20** |

### Faux négatifs (IA classée humain)

| Code | Description | Compte |
|------|-------------|-------:|
| FN-1 | Température ≥ 0.9 | 0 |
| FN-2 | Post-éditée par un humain | 0 |
| FN-3 | Style très marqué (dialecte, argot) | 0 |
| FN-4 | Hors-distribution (Mistral, GPT-4, Claude) | 0 |
| FN-5 | IA très courte | 1 |
| **FN-autre** | **Autre — Luciole in-distribution (voir notes)** | **19** |
| **Total** | | **20** |

## 2. Performance par source (dev, seuil accuracy)

Calculée sur l'ensemble du dev (n = 100) à partir des scores figés — pas de nouvelle inférence.

| Source | n | Accuracy | FP | FN |
|--------|--:|---------:|---:|---:|
| wikipedia-fr | 16 | **0.500** | 8 | 0 |
| litterature | 9 | **0.556** | 4 | 0 |
| linuxfr | 8 | 0.875 | 1 | 0 |
| luciole-8b-instruct | 15 | 0.933 | 0 | 1 |
| presse-fr | 12 | 1.000 | 0 | 0 |
| blog-maudet | 5 | 1.000 | 0 | 0 |
| luciole-1b-instruct | 15 | 1.000 | 0 | 0 |
| luciole-23b-instruct | 20 | 1.000 | 0 | 0 |

L'erreur est **entièrement concentrée côté humain**, et source-dépendante : Wikipédia et littérature classique passent pour de l'IA dans ~la moitié des cas ; les textes IA sont détectés ≥ 93 % quelle que soit la taille du générateur (1B/8B/23B).

## 3. Exemples représentatifs (5 par catégorie majeure)

### FP-6 — encyclopédique neutre (12 cas, les 5 plus marqués)

- **human-wikipedia-fr-071** (score 0.887) — biodiversité : registre définitionnel canonique (« désigne… », étymologie, jalons historiques).
- **human-wikipedia-fr-023** (score 0.905) — canal du Midi : intro dense, structure historique linéaire, vocabulaire lisse.
- **human-wikipedia-fr-055** (score 0.909) — page d'homologique « Vulcain » : listes courtes neutres et parallèles, structure normalisée.
- **human-wikipedia-fr-027** (score 0.928) — tour Eiffel : structure canonique (adresse officielle, dates, chiffres de fréquentation, superlatifs).
- **human-wikipedia-fr-072** (score 1.000) — parc national : définition institutionnelle très neutre, historique canonique.

### FP-autre (6 cas) — littérature classique et presse polluée

- **human-litterature-040** (0.900) — Flaubert, *Hérodias* : phrases longues, lexique soutenu, syntaxe mesurée, confondue avec la fluidité IA.
- **human-litterature-022** (0.921) — Zola, *Thérèse Raquin* (incipit) : cadence descriptive, symétries stylistiques.
- **human-litterature-003** (0.936) — Stendhal, *Le Rouge et le Noir* (incipit) : description topographique soignée.
- **human-litterature-017** (0.937) — Molière, *Le Malade imaginaire* (monologue d'Argan) : voix théâtrale, énumérations stylisées.
- **human-presse-fr-014** (0.989) — presse tech (Grok 4.6), livres blancs + artefacts `�` : double signal (registre corporate + pollution d'encoding).
- **human-presse-fr-051** (1.011) — presse tech (Fourthline/Veridas), mêmes artefacts d'encoding.

### FP-5 — technique standardisé (2 cas)

- **human-linuxfr-034** (0.927) — contribution OpenStack Horizon : registre technique normalisé (MFA, TOTP, PR).
- **human-linuxfr-022** (0.981) — fil LinuxFR : métadonnées normalisées, registre de forum technique.

### FN-autre (19 cas, les 5 plus illustratifs) — Luciole in-distribution

- **ai-8b-031** (0.962) — notice Notre-Dame : fluide, ancrée (incendie 2019, UNESCO 1991) ; génération in-distribution indistinguable de Wikipédia FR.
- **ai-23b-078** (0.917) — aviation civile : registre normatif-encyclopédique ; le plus gros modèle produit un style trop naturel pour son propre détecteur.
- **ai-1b-008** (0.947) — Lille : erreurs factuelles subtiles mais prose de guide de voyage convaincante.
- **ai-23b-083** (0.895) — dépêche Alzheimer : registre journalistique science emphatique, imitation fidèle de la dépêche.
- **ai-1b-013** (0.867) — Annecy : erreur factuelle grossière (UNESCO) mais style très naturel ; illustration de l'instabilité factuelle du 1B.

### FN-5 — IA très courte (1 cas)

- **ai-23b-094** (0.926) — dépêche canicule 545 caractères : brièveté + registre dépêche AFP, signal statistique insuffisant.

## 4. Lecture et mitigations V0.2

1. **Biais encyclopédique dominant (confirmé)** : FP-6 = 12/20. Wikipédia FR dense est structurellement proche du style de Luciole. Mitigation : corpus v1.1 avec **contre-exemples humains stylisés** (Wikipédia bruitée/littéraire) pour repousser la frontière ; reporting par source systématique (table §2) — ne pas annoncer un accuracy global sans ce découpage.
2. **Littérature classique** : 4/20 FP hors taxonomie. Pas de profil séparé (un profil = une paire de modèles par langue, pas un seuil par registre) ; mitigation réaliste = données littéraires en calibration v1.1 + mention explicite de la limite « registre littéraire soutenu XIXᵉ » dans l'eval card. Seuil par registre : seulement si la V0.2 expose un mode « registre » documenté.
3. **Mojibake presse** : les artefacts `�` (U+FFFD) sont de l'information **détruite** — `ftfy` ne peut pas les reconstruire (il répare le mojibake réversible type « Ã© », pas les remplacements). La bonne correction reste le **re-scraping en v1.1** (déjà décidé, cf. rapport §9). ftfy est quand même recommandé comme garde-fou en amont de l'inférence.
4. **Taxonomie FN à réviser en V0.2** : FN-4 vise le hors-distribution, mais le résidu réel est **l'in-distribution** (19/20 = Luciole imitant Wikipédia/tourisme/dépêche). Ajouter un code explicite « FN-6 : IA in-distribution, style encyclopédique générique » et documenter que le risque résiduel n'est pas la taille du modèle générateur mais la proximité stylistique avec le corpus de calibration.
5. **Contextes à éviter en V0.1** (pour l'usage de l'API) : textes Wikipedia-like très denses, prose littéraire classique, pages de forum technique normalisées — fort risque de faux positif « IA » ; signal à présenter à l'utilisateur (le champ `confidence` existe déjà, la documentation doit lister ces contextes).

## 5. Honnêteté méthodologique

- Annotation produite par un agent puis **relue et validée** par l'utilisateur (procédure documentée dans le kit §7) ; le champ `review_status` de l'artefact porte la trace.
- Pas de Cohen's kappa (annotateur unique) — un second passage humain indépendant reste possible mais non bloquant pour la V0.1.
- Les candidats sont les pires cas du dev : ils **surexpriment** le taux d'erreur (le dev global est à 0.72 d'accuracy, avec l'échec concentré sur Wikipédia/littérature).
