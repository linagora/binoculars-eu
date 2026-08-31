# Kit R-2 — Paraphrase légère manuelle (protocole §6.1)

Le test de robustesse **R-2** évalue si le détecteur survit à une reformulation
légère : ~10 % des phrases de chaque texte du test set sont paraphrasées à la
main, le reste reste **strictement identique**. C'est la perturbation la plus
réaliste (un humain qui reprend un texte d'IA en modifiant quelques tournures).

## Ce que tu as à faire

**Option recommandée — éditeur web** (aucune installation, sauvegarde auto,
~1 h de travail) :

```bash
cd calibration
python3 -m http.server 8000
# ouvre http://localhost:8000/r2_kit_editor.html
```

Remplis les phrases surlignées, puis « Télécharger le kit rempli » et :

```bash
cp ~/Downloads/r2_kit_fr_v01.filled.json calibration/r2_kit_fr_v01.json
```

**Option brute — édition JSON directe** :

1. Ouvre `calibration/r2_kit_fr_v01.json` (généré par
   `python -m calibration.build_r2_kit`).
2. Pour chaque texte, les objets `"sentences"` avec `"to_paraphrase": true`
   attendent leur reformulation dans le champ `"paraphrase"`.
3. Ne touche à **rien d'autre** : ni aux `"text"`, ni à l'ordre, ni aux champs
   `"i"` / `"to_paraphrase"`. L'assembleur refuse tout écart (les 90 % non
   paraphrasés doivent être byte-identiques au corpus).

Charge de travail : 100 textes, ~1 à 2 phrases marquées par texte
(~100 phrases au total, médiane 6 phrases/texte).

## Règles de reformulation

- **Légère** : synonymes, inversion de propositions, changement de construction.
  Ce n'est pas une réécriture — le texte doit rester très « IA » en surface.
- **Préserve** : le sens, les entités nommées, les nombres, les dates, le niveau
  de langue.
- **Ne fais pas** : ajouter/supprimer des faits, résumer, traduire, corriger
  (même une faute d'origine reste telle quelle — elle fait partie des 90 %).
- Une phrase marquée doit donner un paraphrasé **différent** de l'original
  (l'assembleur le vérifie) et non vide.

## Une fois le kit rempli

```bash
# Validation stricte + écriture du fichier consommé par robustness.py
python -m calibration.assemble_r2 --lang fr
# → calibration/r2_paraphrases_fr_v01.jsonl

# Puis, sur la box GPU, robustesse complète avec R-2 actif
python -m calibration.robustness \
    --corpus calibration/corpus/binoculars-eu-corpus-fr-v01.jsonl \
    --splits calibration/splits_fr_v01.json \
    --scores calibration/scores_fr_v01.json \
    --profile fr \
    --r2-file calibration/r2_paraphrases_fr_v01.jsonl \
    --generator-url http://100.90.203.88:8013 --generator-model luciole-8b-instruct
```

Seuil de fragilité R-2 (protocole §6.1) : ΔAUC ≥ −0,20 = robuste.
