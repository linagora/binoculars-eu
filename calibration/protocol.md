# binoculars-eu — Protocole d'évaluation (calibration/protocol.md)

> Version détaillée du §18 du PRD. Ce fichier vit dans le repo à
> `calibration/protocol.md` et fait autorité pour toute publication scientifique
> ou technique se référant à binoculars-eu.

**Version** : 2.0-v0.1
**Standard de référence** : RAID (Dugan et al., ACL 2024), NAACL 2025 Findings 271,
OpenTuringBench (2025)
**Auteur du protocole** : Michel-Marie Maudet (LINAGORA / OpenLLM France)
**Date de gel** : à la première release V0.1
**Portée** : tous les profils de langue de la plateforme ; instancié pour le profil `fr`
en V0.1, pour le profil `en` en V1 (§10)

---

## 0. Portée : un protocole, appliqué intégralement à chaque profil

`binoculars-eu` est une plateforme multi-profils : chaque langue est décrite par un
`LanguageProfile` qui porte sa propre paire observer / performer, ses propres seuils, son
propre corpus et son propre SHA-256 (PRD §6). Ce protocole s'applique **à chaque profil
indépendamment, dans son intégralité, avec les mêmes règles** — seeds figées (§1), splits
stratifiés 60/20/20 sur 4 axes (§2), métrique headline TPR@FPR=1 % avec IC 95 % bootstrap
(§3), les 5 baselines obligatoires (§4), les 4 ablations sur `dev` (§5), les 6 tests de
robustesse (§6), l'analyse d'erreurs en double annotation avec Cohen's kappa (§7), et les
trois vérifications de reproductibilité (§8).

Trois conséquences, non négociables.

**Aucune mutualisation de résultats entre profils.** Un seuil, un AUC, un intervalle de
confiance obtenus sur le profil `fr` n'ont **aucune validité** pour le profil `en`, et
réciproquement. Les distributions de perplexité dépendent de la paire de modèles, de la
langue et du corpus : trois variables qui changent simultanément d'un profil à l'autre.
C'est précisément ce constat qui justifie l'existence de l'abstraction de profil plutôt
qu'un jeu de seuils global.

**Un jeu de livrables par profil.** Le suffixe de langue est porté par tous les artefacts :
`calibration/corpus/binoculars-eu-corpus-<lang>-v01.jsonl`,
`calibration/splits_<lang>_v01.json`, `calibration/results_<lang>_v01.json`,
`docs/evaluation_report_<lang>_v01.md`, `docs/eval-card-<lang>.md`, et les seuils dans
`binoculars_eu/profiles/<lang>/thresholds.json`. Les sections §1 à §9 ci-dessous sont
écrites pour le profil `fr` de la V0.1 ; elles se lisent pour tout autre profil en
substituant le code de langue.

**Pas de profil publié sans passage complet du protocole.** Un profil dont le corpus n'est
pas public, dont le SHA-256 n'est pas vérifiable, ou dont les baselines et la robustesse
n'ont pas été mesurées, n'est pas fusionnable dans la plateforme (PRD §4.6). Cette règle
s'applique aussi aux profils dont des seuils publiés préexistent dans la littérature :
reprendre un seuil externe n'exonère pas de le reproduire (§10).

---

## Résumé pour lecteur pressé

- Corpus 500 textes, split **60/20/20** stratifié sur 4 axes (label, source, générateur, longueur), seed=42.
- Métrique headline : **TPR@FPR=1 %** avec IC 95 % bootstrap (1000 rééchantillonnages).
- **5 baselines** minimales évaluées sur le même hold-out (Random, Longueur, Features shallow, Binoculars-Falcon-EN via `from_legacy`, profil `fr`).
- **4 ablations** sur `dev` (max_length, précision, paire, tokenizer).
- **6 tests de robustesse** (fautes de frappe, paraphrase, troncature, concat, adversarial prompting, adversarial rewriting).
- **Reproductibilité** : seeds figées, versions pinées, Dockerfile, test δ < 1e-4 entre runs identiques.
- **Un passage complet du protocole par profil de langue**, sans mutualisation de résultats (§0). Profil `en` de la V1 : §10.

---

## 1. Seeds figées

Ces seeds sont **immuables** après la première release V0.1. Toute publication future
utilisant un autre jeu doit être documentée comme "V0.1-alt-seeds".

| Usage | Seed |
|-------|------|
| `sklearn.StratifiedKFold` pour split train/dev/test | **42** |
| `numpy.random.default_rng` pour bootstrap | **100** |
| `torch.manual_seed` pour ordre batch | **42** |
| Génération corpus IA avec Luciole-23B | **0** |
| Génération corpus IA avec Luciole-8B | **1** |
| Génération corpus IA avec Luciole-1B | **2** |
| Perturbations robustesse R-1 (typos) | **500** |
| Perturbations robustesse R-4 (concat) | **501** |
| Adversarial prompting R-5 | **502** |
| Adversarial rewriting R-6 | **503** |
| Runs de stabilité inter-seed | **42, 123, 2024** |

---

## 2. Splits stratifiés

### 2.1 Procédure

```python
# calibration/build_splits.py (extrait)
from sklearn.model_selection import StratifiedShuffleSplit
import hashlib, json

SEED = 42

def stratum_key(record: dict) -> str:
    """Clef de stratification à 4 dimensions."""
    length = record["meta"]["length_tokens"]
    if length < 150:
        length_bin = "L1"
    elif length < 300:
        length_bin = "L2"
    elif length < 500:
        length_bin = "L3"
    else:
        length_bin = "L4"
    generator = record["meta"].get("generator", "human")
    source = record["source"]
    label = record["label"]
    return f"{label}|{source}|{generator}|{length_bin}"


def build_splits(corpus, seed=SEED):
    strata = [stratum_key(r) for r in corpus]
    # 80/20 test, puis 75/25 sur les 80 → 60/20/20
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(corpus, strata))
    trainval = [corpus[i] for i in trainval_idx]
    test = [corpus[i] for i in test_idx]
    strata_tv = [strata[i] for i in trainval_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, dev_idx = next(sss2.split(trainval, strata_tv))
    train = [trainval[i] for i in train_idx]
    dev = [trainval[i] for i in dev_idx]

    manifest = {
        "seed": seed,
        "train_ids": [r["id"] for r in train],
        "dev_ids": [r["id"] for r in dev],
        "test_ids": [r["id"] for r in test],
        "counts": {"train": len(train), "dev": len(dev), "test": len(test)},
    }
    # Hash pour reproductibilité
    m_bytes = json.dumps(manifest, sort_keys=True).encode()
    manifest["sha256"] = hashlib.sha256(m_bytes).hexdigest()
    return train, dev, test, manifest
```

### 2.2 Discipline d'usage

| Split | Autorisé | Interdit |
|-------|----------|----------|
| `train` | Calibrer les seuils, chercher les optima ROC, ablations | Lire les erreurs, ajuster manuellement, publier |
| `dev` | Ablations, choix du mode, tuning `max_length` | Publier les métriques primaires |
| `test` | Publier les métriques primaires **une seule fois** | Regarder pendant le développement |

**Règle stricte** : le test set est chargé pour la première fois dans le script
`evaluate.py`, jamais dans un notebook exploratoire. Chaque appel à `evaluate.py`
est loggé (timestamp, hash git) pour tracer les runs.

### 2.3 5-fold en complément

Sur 500 textes, un test de 100 donne des IC larges. On rapporte donc en complément un
5-fold stratifié couvrant les 500 textes : moyenne ± écart-type des métriques primaires
sur les 5 folds.

Le hold-out et le 5-fold doivent être cohérents : si la métrique du test diffère
significativement de la moyenne 5-fold, c'est un signal de fragilité du split — le
publier comme tel.

---

## 3. Métriques

### 3.1 Métriques primaires (avec IC 95 %)

| Métrique | Cible V0.1 | Cible V0.2 |
|----------|------------|------------|
| **AUC ROC** | ≥ 0.80 | ≥ 0.85 |
| **TPR@FPR=1 %** *(headline)* | ≥ 0.45 | ≥ 0.55 |
| **TPR@FPR=5 %** | ≥ 0.65 | ≥ 0.75 |
| **F1** @ threshold `accuracy` | ≥ 0.75 | ≥ 0.80 |
| **Accuracy** @ threshold `accuracy` | ≥ 0.75 | ≥ 0.80 |
| **AUC ROC OOD Mistral 24B (V0.1 seulement)** | ≥ 0.65 | ≥ 0.70 |
| **Expected Calibration Error (ECE)** | ≤ 0.15 | ≤ 0.10 |
| **Cohen's kappa inter-annotateurs (FP+FN)** | ≥ 0.60 | ≥ 0.70 |

> **Positionnement des cibles** : ces valeurs sont **prudentes** pour la première itération.
> binoculars-eu V0.1 vise la reproductibilité et la lisibilité scientifique, pas la
> vitrine de performance. Les contributions externes chercheront à les dépasser.

### 3.2 Métriques secondaires

Toujours rapportées, sans cible chiffrée obligatoire :

- Matrice de confusion stratifiée : par source humaine, par générateur IA, par bin de longueur.
- Distribution des scores : histogramme humain vs IA, courbes de densité.
- Latence par texte : P50 et P99 par bin de longueur.
- Empreinte VRAM P95.

### 3.3 Bootstrap pour IC 95 %

```python
# calibration/bootstrap.py (extrait)
import numpy as np
from sklearn.metrics import roc_auc_score

def bootstrap_metric(y_true, y_score, metric_fn, n_boot=1000, seed=100):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            v = metric_fn(y_true[idx], y_score[idx])
            values.append(v)
        except ValueError:
            continue  # Peut arriver si un rééchantillon n'a qu'une classe
    values = np.array(values)
    return {
        "point": metric_fn(y_true, y_score),
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
        "n_boot_valid": len(values),
    }
```

### 3.4 TPR@FPR=1 % — implémentation

```python
from sklearn.metrics import roc_curve

def tpr_at_fpr(y_true, y_score, target_fpr):
    """TPR au FPR cible. Interpolation linéaire si pas de point exact."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # Interpolation
    return float(np.interp(target_fpr, fpr, tpr))
```

---

## 4. Baselines (à évaluer sur test set)

### 4.1 Liste obligatoire

| ID | Baseline | Implémentation |
|----|----------|----------------|
| B0 | Random | `numpy.random.rand(n)` |
| B1 | Longueur | Régression logistique sur longueur en tokens |
| B2 | Features shallow | LR sur : longueur, taux de virgules, TTR lexical, ratio ponctuation, moyenne longueur mot |
| B3 | Binoculars-Falcon-EN original | `Binoculars.from_legacy("tiiuae/falcon-7b", "tiiuae/falcon-7b-instruct")` — seuils Hans et al. inchangés, appliqué aux textes du profil évalué |
| B4 | **Profil de langue évalué (nous)** | `Binoculars.for_language("fr")` en V0.1 ; cible du projet |

### 4.2 Baselines optionnelles V0.2

Si accès disponible et budget compatible :

| ID | Baseline | Contrainte |
|----|----------|------------|
| B5 | GPTZero API | Coût par appel, ~$0.05 / texte |
| B6 | Originality.ai API | Coût par appel |
| B7 | Fast-DetectGPT (Bao et al., ICLR 2024) sur Luciole-1B-Base | Impl. hors-repo |
| B8 | Profil `fr` en 8B+8B int8 (V0.2) | Requiert bitsandbytes + trust_remote_code |

Toute baseline optionnelle absente est explicitement documentée dans le rapport.

### 4.3 Convention de reporting

Table exhaustive **Baseline × Métrique**, avec IC 95 %. Notre méthode doit dominer
statistiquement B0-B2, et dominer significativement B3 (sinon la francophonie n'apporte
rien mesurable).

**Cas particulier du profil `en` (§10)** : pour ce profil, B3 et B4 partagent la même paire
de modèles et ne diffèrent que par l'origine des seuils (publiés vs reproduits par nous).
La comparaison ne mesure alors plus un gain de langue mais un **écart de reproduction** —
et c'est exactement l'objet de §10.4.

---

## 5. Ablations (à évaluer sur dev set)

### 5.1 Table d'ablation

| ID | Variable | Valeurs | Métriques rapportées |
|----|----------|---------|---------------------|
| A1 | `max_length` | 64, 128, 256, 512, 1024 | AUC, TPR@FPR=1 %, latence |
| A2 | Précision | bfloat16, float16, float32 | AUC, TPR@FPR=1 %, VRAM |
| A3 | Paire de modèles | Base+Instruct-1.1, Instruct+Instruct-thinking (privé), Base+SFT (privé) | AUC, TPR@FPR=1 % |
| A4 | Tokenizer partagé | Tokenizer du Base, Tokenizer du Instruct | AUC (sanity check) |

### 5.2 Règles

- Toutes les ablations sont évaluées sur **`dev` uniquement**, jamais sur test.
- Chaque configuration est run 3 fois avec seeds différentes (42, 123, 2024) pour
  mesurer la variance inter-seed.
- Les résultats sont publiés comme table Config × Métrique(±écart-type).

---

## 6. Robustesse (à évaluer sur test set perturbé)

### 6.1 Six tests obligatoires

Chaque perturbation prend le test set, applique la transformation, ré-évalue.
On rapporte le **Δ AUC = AUC(test perturbé) − AUC(test original)**.

| ID | Nom | Perturbation | Seed | Sensibilité attendue | Fragilité si |
|----|-----|--------------|------|---------------------|--------------|
| R-1 | Fautes de frappe | 5 % caractères inversés aléatoirement | 500 | Faible | Δ AUC ≤ −0.15 |
| R-2 | Paraphrase légère | Reformulation manuelle de 10 % des phrases | — (manuel) | Moyenne | Δ AUC ≤ −0.20 |
| R-3 | Troncature | Ne garder que les 100 premiers tokens | — | Forte | Δ AUC ≤ −0.30 |
| R-4 | Concaténation phrase-à-phrase | Alterne 1 phrase humaine / 1 phrase IA | 501 | Très forte | score autour de 0.5 pas atteint |
| R-5 | Adversarial prompting | IA générée avec "écris comme un humain, évite les tournures de LLM" | 502 | Moyenne à forte | Δ AUC ≤ −0.25 |
| R-6 | Adversarial rewriting | Humain réécrit par LLM "clarifie et professionnalise" | 503 | Forte | Δ AUC ≤ −0.30 |

R-5 et R-6 suivent explicitement le protocole [RAID (Dugan et al., 2024)](https://arxiv.org/abs/2405.07940)
et [NAACL 2025 Findings 271](https://aclanthology.org/2025.findings-naacl.271/).

### 6.2 Interprétation

Le seuil "fragilité si" documente une limite acceptable. Un test qui dépasse la limite
n'invalide pas la V0.1 mais **doit** être documenté explicitement dans le rapport, avec
recommandation de ne pas utiliser binoculars-eu dans le contexte concerné.

Un détecteur robuste aurait typiquement :

- R-1, R-2 : Δ AUC ∈ [−0.10, −0.02]
- R-3 : Δ AUC ∈ [−0.25, −0.10]
- R-4 : score moyen sur textes concat ∈ [0.4, 0.6]
- R-5, R-6 : Δ AUC ∈ [−0.20, −0.05] pour être considéré comme utilisable en prod.

---

## 7. Analyse d'erreurs

### 7.1 Taxonomie a priori

**Faux positifs** (humain classé IA) :

| Code | Description |
|------|-------------|
| FP-1 | Texte administratif / juridique (registre formel proche du style IA) |
| FP-2 | Texte très court (< 100 tokens) |
| FP-3 | Traduction automatique post-éditée |
| FP-4 | Texte avec beaucoup de code, chiffres, tableaux |
| FP-5 | Texte technique très standardisé (RFC, spéc, mode d'emploi) |
| FP-6 | Texte encyclopédique très neutre (Wikipedia dense) |
| FP-autre | À caractériser à la main |

**Faux négatifs** (IA classée humain) :

| Code | Description |
|------|-------------|
| FN-1 | IA générée avec température ≥ 0.9 |
| FN-2 | IA post-éditée par un humain |
| FN-3 | IA imitant un style très marqué (dialecte, argot, littéraire) |
| FN-4 | IA hors-distribution de Luciole (Mistral, GPT-4, Claude) |
| FN-5 | IA très courte |
| FN-autre | À caractériser à la main |

### 7.2 Procédure

1. Extraire les 20 pires FP (score le plus bas parmi les humains) et 20 pires FN
   (score le plus haut parmi les IA) du dev set (jamais du test set).
2. Notebook `notebooks/04_error_analysis.ipynb` présente chaque texte avec :
   - Texte formaté, longueur, source.
   - Score obtenu, seuil, distance au seuil.
   - Widget `ipywidgets` de cases à cocher (une par catégorie de la taxonomie).
   - Zone de note libre.
3. Annotation manuelle par au moins un humain (Michel-Marie), idéalement deux pour
   mesurer un accord inter-annotateurs (Cohen's kappa).
4. Publication d'un tableau Catégorie × Compte pour FP et pour FN.

### 7.3 Livrable

- `docs/error_analysis_fr_v01.md` : rapport structuré avec :
  - Table de contingence Catégorie × Compte pour FP et FN.
  - 5 exemples représentatifs de chaque catégorie majeure (avec permission de partage).
  - Recommandations de mitigation (données à ajouter en V0.2, contextes à éviter).

---

## 8. Reproductibilité

### 8.1 Environnement gelé

`requirements-eval.txt` (à commiter, distinct de `requirements.txt` pour usage) :

```
torch==2.5.1
transformers==4.57.1
tokenizers==0.20.3
accelerate==1.2.1
bitsandbytes==0.44.1
datasets==3.2.0
scikit-learn==1.6.0
numpy==2.1.3
pandas==2.2.3
matplotlib==3.10.0
scipy==1.14.1
huggingface_hub==0.27.0
ipywidgets==8.1.5
```

### 8.2 Dockerfile de référence

```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y \
    python3.12 python3.12-venv python3-pip git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-eval.txt .
RUN python3.12 -m pip install --no-cache-dir -r requirements-eval.txt

COPY . .

CMD ["python3.12", "-m", "calibration.evaluate", "--config", "v01"]
```

### 8.3 Vérifications de reproductibilité

Trois checks obligatoires, exécutés à chaque release :

1. **Determinisme intra-seed** : deux runs consécutifs avec la même seed produisent
   des métriques primaires à `δ < 1e-4`.
2. **Stabilité inter-seed** : 3 runs avec seeds 42, 123, 2024 → variance inter-seed
   publiée comme écart-type dans le rapport.
3. **Hash SHA-256 du corpus** : publié dans `calibration/protocol.md` et vérifié à
   chaque `evaluate.py`.

### 8.4 Traçabilité git

Chaque run de `evaluate.py` produit un JSON qui contient :

```json
{
  "timestamp": "2026-09-15T14:23:11Z",
  "git_sha": "abc123...",
  "git_dirty": false,
  "corpus_sha256": "def456...",
  "seeds": {"split": 42, "bootstrap": 100, "torch": 42},
  "config": "v01",
  "versions": {"torch": "2.5.1", "transformers": "4.57.1"},
  "metrics": { ... }
}
```

---

## 9. Rapport final

### 9.1 Structure obligatoire

`docs/evaluation_report_fr_v01.md` (un rapport par profil, suffixé par le code de langue) :

1. **Executive summary** : métrique headline (TPR@FPR=1 %) + IC + un paragraphe.
2. **Setup** : corpus (taille, splits, sources, hash), modèles (versions, précision), matériel (GPU, VRAM).
3. **Métriques primaires** : table (hold-out) + table (5-fold).
4. **Comparaison baselines** : table `Métrique × Baseline` avec IC.
5. **Ablations** : 4 tables A1-A4 sur dev.
6. **Robustesse** : table R-1 à R-6 avec Δ AUC + IC.
7. **Analyse d'erreurs** : tables de contingence FP et FN + exemples.
8. **Reproductibilité** : seeds, hashes, versions, temps d'exécution.
9. **Limites** : biais connus (corpus, diachronique, démographique, adversarial).
10. **Annexes** : figures ROC, distributions de scores, courbes de calibration.

### 9.2 Eval Card

Fiche synthétique `docs/eval-card-<lang>.md` à la manière des HF Model Cards — **une card
par profil de langue**, jamais une card commune à la plateforme. À joindre à toute
distribution PyPI et à toute annonce de communication. Voir `binoculars-eu-eval-card.md`
pour le gabarit, instancié sur le profil `fr` V0.1 ; `docs/eval-card-en.md` en V1.

---

## 10. Protocole spécifique au profil EN (V1)

Le profil `en` (Falcon-7B + Falcon-7B-Instruct) est le seul cas où des seuils publiés
préexistent à notre calibration : ceux de Hans et al., ICML 2024. Les reprendre sans les
reproduire créerait un précédent inacceptable pour les profils communautaires à venir.
D'où ce protocole complémentaire, obligatoire avant publication du profil.

### 10.1 Script `calibration/reproduce_en_profile.py`

Script autonome, exécutable par un tiers, en quatre étapes séquentielles :

1. `--build-corpus` : construit le corpus EN (§10.2) et écrit son SHA-256.
2. `--build-splits` : splits stratifiés 60/20/20, seed **42** (même seed que FR, §1).
3. `--calibrate` : calibre les trois seuils (`accuracy`, `low_fpr`, `tpr_at_fpr_1`) sur
   `train` uniquement, via `Binoculars.from_legacy("tiiuae/falcon-7b",
   "tiiuae/falcon-7b-instruct")` pour le scoring — les seuils upstream ne sont **jamais
   lus** pendant cette étape, afin d'éviter tout ancrage.
4. `--compare` : confronte les seuils obtenus aux valeurs publiées (§10.4) et écrit
   `profiles/en/thresholds.json` + `profiles/en/metadata.json`.

Sortie : `calibration/results_en_v01.json`, même schéma JSON que le profil FR (§8.4), avec
`git_sha`, `corpus_sha256`, seeds, versions et métriques.

### 10.2 Corpus EN à construire — 500 textes

| Volet | Nombre | Source | Note |
|-------|--------|--------|------|
| Humain | 250 | Wikipedia EN, articles **antérieurs à 2022** | Antériorité stricte pour écarter toute contamination par du texte généré |
| IA | 250 | **Falcon-7B-Instruct**, mêmes prompts que le volet humain | Titre + premier paragraphe, température et top_p identiques au corpus FR |

Même **symétrie thématique** que le corpus FR : chaque texte humain a un jumeau IA sur le
même sujet (champ `twin_of` dans le JSONL). Même format de stockage, mêmes bins de
longueur pour la stratification. Le générateur étant unique, l'axe de stratification
« générateur » de §2 est constant pour ce profil : il est remplacé par un axe
« catégorie thématique Wikipedia » pour conserver quatre axes de stratification.

### 10.3 Même protocole que le profil FR

Aucune allégeance particulière à la littérature ne dispense d'un passage complet :
splits stratifiés étanches (§2), bootstrap 1000 rééchantillonnages pour tous les IC 95 %
(§3.3), les 5 baselines B0-B4 (§4.1), les 4 ablations sur `dev` (§5), les 6 tests de
robustesse R-1 à R-6 (§6), l'analyse d'erreurs en double annotation avec Cohen's kappa
(§7 — le second annotateur doit être anglophone compétent), et les trois vérifications de
reproductibilité (§8.3).

### 10.4 Comparaison des seuils reproduits vs Hans et al.

| Mode | Valeur publiée (Hans et al.) | Valeur reproduite | Tolérance |
|------|------------------------------|-------------------|-----------|
| `accuracy` | `0.9015310749276843` | à mesurer | **±0.01** |
| `low_fpr` | `0.8536432310785527` | à mesurer | **±0.01** |
| `tpr_at_fpr_1` | non publiée | à mesurer | — (valeur propre à binoculars-eu) |

Règle de décision, appliquée séparément à chaque mode :

- **Écart ≤ 0.01 → reproduction validée.** Le profil `en` conserve la valeur publiée (pour
  rester comparable à la littérature) et `calibration_note` mentionne l'écart mesuré.
- **Écart > 0.01 → bascule sur notre valeur.** L'écart devient un résultat publié en soi,
  avec analyse des causes candidates : composition du corpus, version de `transformers`,
  précision de calcul (bf16 vs int8 si contrainte VRAM, PRD §16.1), matériel.

Dans les deux cas, les valeurs publiée **et** reproduite figurent dans
`profiles/en/thresholds.json`, de sorte que `GET /profiles` expose la provenance sans
ambiguïté. Le seuil `tpr_at_fpr_1` est toujours le nôtre : il n'existe pas chez Hans et al.

### 10.5 Publication

- Corpus : **`OpenLLM-France/binoculars-eu-corpus-en-v01`** sur HF Datasets, SHA-256
  inscrit dans `profiles/en/metadata.json` et vérifié par `tests/test_profile_integrity.py`.
- Rapport : `docs/evaluation_report_en_v01.md`, structure de §9.1.
- Eval card : `docs/eval-card-en.md`, même gabarit que la card FR.
- Reproductibilité par un tiers : `python -m calibration.reproduce_en_profile --all` doit
  aboutir aux mêmes chiffres à `δ < 1e-4` (§8.3, check 1).

---

## 11. Limites structurelles du protocole

**À rapporter systématiquement dans toute publication** :

- **Biais de corpus** : Wikipedia + presse dominent → sur-représentés. Pas de SMS, oral
  transcrit, dialecte, textes courts (< 100 mots).
- **Biais diachronique** : le corpus est de 2026. Un texte humain de 2028 pourrait
  être classé IA à cause de la pollution progressive du français par les tournures IA.
- **Biais démographique** : nous n'annotons pas l'origine sociolinguistique. Risque de
  classer systématiquement IA des locuteurs non-natifs.
- **Biais adversarial** : R-1 à R-6 ne couvrent pas les humanizers co-évolutifs
  spécifiquement entraînés contre binoculars-eu.
- **Biais inter-profils** : le protocole évalue chaque profil isolément (§0) et ne mesure
  pas la dégradation lorsqu'un texte est scoré par le profil d'une autre langue — cas
  courant en production dès qu'un champ `profile` est mal renseigné. Mesure prévue en V1,
  quand deux profils coexisteront.
- **Biais de générateur** : notre corpus IA est à 100 % Luciole. La V0.2 étend avec
  Mistral/GPT/Claude, mais reste incomplet (pas de Qwen, DeepSeek, LLaMA, etc.).

Ces limites justifient une **révision du protocole tous les 6-12 mois** avec
extension du corpus.

---

## 12. Historique du protocole

| Version | Date | Changements |
|---------|------|-------------|
| 1.0-v0.1 | à figer | Protocole initial pour V0.1 (mono-langue) |
| 2.0-v0.1 | à figer | Portée multi-profils (§0), livrables suffixés par code de langue, protocole spécifique au profil EN (§10) |
| 2.0-v0.2 | prévu | Ajout corpus OOD étendu, baseline GPTZero, extension 8B+8B int8 du profil `fr` |
| 2.1-v1 | prévu | Exécution effective de §10 sur le profil `en` + mesure inter-profils (texte EN scoré par le profil `fr` et réciproquement) |
| 3.0 | prévu | Révision post-publication : ajouts issus des retours communauté et des profils `es`, `de`, `it`, `pt`, `pl` |
