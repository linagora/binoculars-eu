# Kit §7 — Annotation des erreurs (protocole §7.2)

L'analyse d'erreurs qualifie **pourquoi** le détecteur se trompe sur les 40
pires candidats du **dev set** (jamais du test) : 20 faux positifs (humain
classé IA) + 20 faux négatifs (IA classée humain), extraits par
`calibration/evaluate.py` dans `error_analysis_candidates_fr_v01.json`.

## Ce que tu as à faire

**Option recommandée — éditeur web** (sauvegarde auto, ~45 min) :

```bash
cd calibration
python3 -m http.server 8000
# ouvre http://localhost:8000/error_kit_editor.html
```

Pour chaque candidat : lis le texte, choisis **UNE** catégorie de la
taxonomie §7.1 (boutons radio), ajoute une note libre si utile —
**obligatoire** pour « FP-autre » / « FN-autre ». Puis « Télécharger le kit
rempli » :

```bash
cp ~/Downloads/error_kit_fr_v01.filled.json calibration/error_kit_fr_v01.json
python -m calibration.assemble_error_annotations --lang fr
# → docs/error_analysis_annotations_fr_v01.json
```

**Option brute** : édite directement `calibration/error_kit_fr_v01.json`
(champs `"category"` et `"note"` par candidat), puis la commande d'assemblage.

## Taxonomie (protocole §7.1)

**Faux positifs** (humain → classé IA) :

| Code | Description |
|------|-------------|
| FP-1 | Texte administratif / juridique (registre formel proche du style IA) |
| FP-2 | Texte très court (< 100 tokens) |
| FP-3 | Traduction automatique post-éditée |
| FP-4 | Texte avec beaucoup de code, chiffres, tableaux |
| FP-5 | Texte technique très standardisé (RFC, spéc, mode d'emploi) |
| FP-6 | Texte encyclopédique très neutre (Wikipedia dense) |
| FP-autre | Autre — caractérise dans la note (obligatoire) |

**Faux négatifs** (IA → classé humain) :

| Code | Description |
|------|-------------|
| FN-1 | IA générée avec température ≥ 0.9 |
| FN-2 | IA post-éditée par un humain |
| FN-3 | IA imitant un style très marqué (dialecte, argot, littéraire) |
| FN-4 | IA hors-distribution de Luciole (Mistral, GPT-4, Claude) |
| FN-5 | IA très courte |
| FN-autre | Autre — caractérise dans la note (obligatoire) |

> Note méthode : le protocole mentionne des cases à cocher (multi-sélection) ;
> l'éditeur impose **une** catégorie principale par candidat, ce qui rend la
> table de contingence Catégorie × Compte (livrable §7.3) non ambiguë. La note
> libre permet d'enregistrer les facteurs secondaires.

## Après l'annotation

L'agent assemble `docs/error_analysis_annotations_fr_v01.json`, rédige
`docs/error_analysis_fr_v01.md` (table de contingence FP et FN, 5 exemples
par catégorie majeure, mitigations V0.2) et met à jour le rapport §7. Un
second annotateur indépendant permettrait de rapporter un Cohen's kappa
(protocole §7.2).
