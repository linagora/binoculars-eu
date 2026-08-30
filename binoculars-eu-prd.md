# binoculars-eu — Spec technique POC

> Détection zero-shot multilingue, plateforme européenne open source
> Méthode Binoculars (Hans et al., ICML 2024), portée sur des paires de modèles
> ouverts européens. Profil FR (famille Luciole, OpenLLM France / LINAGORA) livré
> en V0.1 ; profil EN en V1 ; profils `es`, `de`, `it`, `pt`, `pl` ensuite.

**Version PRD** : 2.0
**Statut** : Prêt pour exécution du POC (profil FR V0.1)
**Auteur** : Michel-Marie Maudet
**Licence cible** : Apache 2.0 (cohérence avec Luciole)

---

## 1. Objectif et résumé exécutif

`binoculars-eu` est une **plateforme européenne open source de détection zero-shot de
texte généré par IA**, organisée autour d'une abstraction unique : le **profil de langue**
(`LanguageProfile`). Un profil encapsule tout ce qui est spécifique à une langue — la paire
observer / performer, les seuils calibrés, le corpus de calibration et son empreinte
SHA-256, les libellés de verdict localisés — de sorte que le moteur de scoring reste
strictement identique d'une langue à l'autre.

La méthode sous-jacente est celle de Binoculars (Hans et al., ICML 2024) : un ratio
perplexité / cross-perplexité entre deux modèles de la même famille, qui sépare texte
humain et texte généré sans aucun entraînement supervisé. `binoculars-eu` n'invente pas
la méthode ; il en industrialise le portage à d'autres langues que l'anglais, avec le
niveau de rigueur d'évaluation attendu d'une publication (§18).

**Ce que livre la V0.1** : un seul profil, `fr`, construit sur deux modèles Luciole
(`Luciole-1B-Base` comme observer, `Luciole-1B-Instruct-1.1` comme performer, bf16),
calibré sur 500 textes français in-distribution plus 50 textes hors-distribution générés
par Mistral Small 24B. Ce profil est le **profil par défaut de la plateforme** : tout appel
qui ne précise pas de langue reçoit `fr`. Les profils suivants (EN en V1, ES en V3,
communautaires ensuite) sont **opt-in explicite**, jamais implicites.

**Différenciation** : c'est aujourd'hui le seul projet de détection IA au monde à disposer
simultanément (a) d'une famille de modèles ouverts et cohérents en français, (b) du corpus
d'entraînement complet publié, (c) de trois tailles pour explorer les compromis
performance / accessibilité — et désormais (d) d'une architecture explicitement conçue
pour que d'autres communautés linguistiques européennes ajoutent leur propre profil sans
toucher au moteur.

**Cible V0.1** : POC démontrant AUC ROC ≥ 0.80 sur corpus in-distribution FR (cible
prudente pour la première itération), publié sous forme de package Python
`pip install binoculars-eu` avec API Python en trois lignes, **API HTTP FastAPI documentée
en Swagger**, exécutable sur CPU (MacBook), Apple Silicon (MPS), ou GPU CUDA
(24 Go suffisent).

**Compatibilité amont** : le repo reste un fork synchronisable avec
[`ahans30/Binoculars`](https://github.com/ahans30/Binoculars). Le constructeur historique
reste accessible via `Binoculars.from_legacy(observer, performer, mode)`, qui reproduit
exactement le comportement upstream (y compris les seuils Falcon `0.9015` / `0.8536`).

---

## 2. Rappel de la méthode Binoculars

### 2.1 Formule

\[
\text{score}(s) = \frac{\text{PPL}_{\text{performer}}(s)}
{\text{X-PPL}_{\text{observer} \to \text{performer}}(s)}
\]

- **PPL** = perplexité du performer sur le texte `s` (cross-entropy moyennée).
- **X-PPL** = cross-entropy entre `softmax(observer_logits)` et `performer_logits`.

Un texte **humain** produit un ratio proche de 1 (le performer et l'observer sont surpris
différemment). Un texte **IA** produit un ratio significativement inférieur à 1 (perplexité
basse et cross-perplexité corrélée).

Décision : `score < threshold` → IA.

### 2.2 Contraintes structurelles

Le code référence [ahans30/Binoculars](https://github.com/ahans30/Binoculars) impose que
**observer et performer partagent exactement le même tokenizer** (assertion stricte dans
`utils.assert_tokenizer_consistency`).

Les seuils originaux (`0.9015` accuracy, `0.8536` low-fpr) sont calibrés sur Falcon-7B /
Falcon-7B-Instruct en bfloat16 et **ne sont pas transférables** à Luciole. Toute la
calibration doit être refaite.

---

## 3. Découvertes techniques préalables

### 3.1 Vérification des configs Luciole (faite via HF Hub API)

| Attribut | Luciole-1B-Base | Luciole-1B-Instruct-1.1 |
|----------|-----------------|--------------------------|
| Architecture | `NemotronForCausalLM` | `NemotronForCausalLM` ✅ identique |
| `hidden_size` | 2048 | 2048 ✅ |
| `num_hidden_layers` | 24 | 24 ✅ |
| `num_attention_heads` | 32 | 32 ✅ |
| `head_dim` | 64 | 64 ✅ |
| `vocab_size` | 128000 | 128000 ✅ |
| `torch_dtype` | bfloat16 | bfloat16 ✅ |
| `transformers` min | 4.51.3 | 4.57.1 |
| Classe tokenizer | `LlamaTokenizerFast` | `LlamaTokenizerFast` ✅ |
| `bos_token_id` | 0 | 2 ⚠️ |
| `eos_token_id` | 1 | 261 ⚠️ |
| Tokens spéciaux | 4 (0-3) | 12 (0-3 + 260-267) ⚠️ |

### 3.2 Vérification exhaustive de la compatibilité tokenizer

Vérification fichier par fichier (SHA256 comparés, contenu octet par octet) :

| Élément | Base | Instruct-1.1 | Statut |
|---------|------|--------------|--------|
| `vocab_size` | 128 000 | 128 000 | ✅ identique |
| Merges BPE | 126 679 | 126 679 | ✅ identiques octet par octet |
| `normalizer`, `pre_tokenizer`, `post_processor`, `decoder` | idem | idem | ✅ identiques |
| `padding`, `truncation`, `version` | idem | idem | ✅ identiques |
| Tokens partagés (contenu + ID) | 127 992 sur 128 000 | 127 992 sur 128 000 | ✅ interchangeables |
| IDs 260-267 | `<unused0>` ... `<unused7>` | `<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`, `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>` | ⚠️ 8 slots réservés remplis |
| IDs 268+ | tous identiques | tous identiques | ✅ |
| `eos_token` | `</s>` (ID 1) | `<|im_end|>` (ID 261) | ⚠️ divergent |
| `cls_token` / `sep_token` | déclarés | retirés | ⚠️ divergent |

**Vérification fonctionnelle sur 7 échantillons** (français accentué, ligatures œ, code
Python, SQL, emoji, guillemets typographiques) : les IDs produits sont **identiques à 100 %**
dans les deux sens. Les deux tokenizers sont donc interchangeables sur tout texte courant.

Seule exception : les marqueurs de chat. Sur le Base, `<|im_start|>` se retokenise en 7
tokens `[1668, 412, 1397, 383, 3478, 412, 350]` au lieu de `[260]`. Comme aucun texte
utilisateur ne contient ces marqueurs (ils sont produits par le template de chat, pas
par l'utilisateur), c'est sans impact pour binoculars-eu.

### 3.3 Trois points d'attention pour l'implémentation

**Point 1 — Architecture Nemotron (pas Llama).**
Les 1B Luciole sont des modèles NVIDIA Nemotron (`NemotronForCausalLM`) adaptés par
LINAGORA/OpenLLM-France et entraînés sur Jean Zay. Le code Binoculars original charge
Falcon via `AutoModelForCausalLM` — la même API fonctionne pour Nemotron sans modification,
mais impose `transformers >= 4.57.1` (contrainte du Instruct-1.1).

À noter pour la V0.2 : le **8B est `NemotronHForCausalLM`** (variante hybride Mamba /
Attention / MLP, 52 couches selon le motif `M-M-M-M*-M-M-M-M-M*-...`). Elle nécessite
`trust_remote_code=True` et les kernels `mamba-ssm` / `causal-conv1d`. Le 1B est du
Transformer pur, ce qui est un point de contrôle méthodologique important : la V0.1
démontre Binoculars sur une architecture standard, la V0.2 étend à l'hybride Mamba.

**Point 2 — Assertion `assert_tokenizer_consistency` de Binoculars.**
Le code original charge deux `AutoTokenizer` séparés et vérifie leur égalité stricte —
ça échoue sur les 8 tokens IDs 260-267 et sur `eos_token`. Solution retenue : **charger
un seul tokenizer** (celui du Base) et le partager entre les deux modèles. Court-circuite
l'assertion sans avoir à la patcher, et économise ~500 Mo de RAM.

```python
tokenizer = AutoTokenizer.from_pretrained("OpenLLM-France/Luciole-1B-Base")
# Ce même tokenizer sert au Base ET au Instruct — 127 992 tokens partagés, garanti.
```

**Point 3 — Les 8 slots `<unused>` remplacés côté Instruct sont invisibles pour Binoculars.**
Binoculars ne décode jamais de tokens, il calcule seulement perplexité et cross-entropy
sur les IDs d'entrée. Aucun texte français naturel ne produit les IDs 260-267, donc la
divergence est fonctionnellement inexistante pour notre cas d'usage. Un test dédié
(`tests/test_tokenizer_compat.py`) verrouille cette invariante.

---

## 4. Objectifs, roadmap versionnée et non-objectifs

### 4.0 Principe directeur

Une version = **un incrément de profil ou un incrément de robustesse**, jamais les deux.
Cette discipline évite l'écueil classique du multilingue : ajouter des langues plus vite
qu'on ne sait les évaluer. Chaque nouveau profil ne sort qu'accompagné de son propre
corpus, de sa propre calibration, et de sa propre evaluation card produite selon §18.

**FR est et reste le profil par défaut.** `Binoculars()` sans argument, `POST /detect`
sans champ `profile`, et `binoculars-eu` en CLI résolvent tous vers `fr`. Les profils
suivants sont opt-in explicite (`for_language("en")`, `"profile": "en"`).

### 4.1 V0.1 — profil `fr` seul (3 semaines)

- Architecture `LanguageProfile` + registre auto-discovery en place dès la première
  version, avec un seul profil enregistré (`fr`).
- Profil `fr` : **Luciole-1B-Base + Luciole-1B-Instruct-1.1, bfloat16**, tokenizer
  partagé depuis l'observer.
- Calibration sur corpus in-distribution 500 textes FR (250 humains + 250 IA Luciole).
- **Corpus OOD léger intégré dès V0.1** : 50 textes supplémentaires générés par
  Mistral Small 24B via OpenRouter, pour démontrer la généralisation hors-Luciole dès
  la première publication.
- AUC ROC ≥ 0.80 sur corpus in-distribution (cible prudente première itération).
- Détecteur installable via `pip`, API Python à trois lignes.
- **API HTTP FastAPI** (`POST /detect`, `GET /profiles`, `GET /health`) avec Swagger UI
  auto-générée à `/docs` (§13.2), cache LRU des détecteurs par `(profile, mode)`.
- Démo publique déployée en **Hugging Face Space**, image Docker de production.
- Publication packaged + notebook de reproduction + article de blog.
- Nœud LangGraph d'exemple pour intégration dans pipeline agentique.

### 4.2 V0.2 — profil `fr` en 8B+8B int8 (5 semaines cumulées)

- Même profil `fr`, **variante de capacité** : `Luciole-8B-Base` + `Luciole-8B-Instruct-1.1`
  en int8 (`bitsandbytes`), architecture `NemotronHForCausalLM` — **hybride Mamba /
  Attention / MLP**, donc `trust_remote_code=True` et kernels `mamba-ssm` /
  `causal-conv1d` (voir §16 pour le détail des défis).
- Première application publique de Binoculars à un modèle non purement Transformer.
- Benchmark hors-distribution étendu (GPT-4o, Claude, hybrides).
- Comparaison mesurée avec GPTZero et Originality (si accès).
- Documentation méthodologique complète pour reproductibilité scientifique.

### 4.3 V1 — profil `fr` + profil `en`

Première version **réellement multi-profils**, et donc première validation de
l'architecture par l'usage.

- Nouveau profil `en` : **Falcon-7B (observer) + Falcon-7B-Instruct (performer)**, la
  paire canonique du papier original.
- Seuils **repris de Hans et al., ICML 2024** (`0.9015310749276843` accuracy,
  `0.8536432310785527` low-fpr), injectés via `Binoculars.from_legacy` puis figés dans
  `profiles/en/thresholds.json` avec `calibration_note` documentant l'origine externe.
- **Reproduction indépendante obligatoire** : le script
  `calibration/reproduce_en_profile.py` recalibre les seuils sur un corpus EN construit
  par nous (500 textes : 250 Wikipedia EN pré-2022 + 250 Falcon-7B-Instruct sur les mêmes
  prompts) selon le protocole §18. Tolérance de validation : **±0.01** entre seuil
  reproduit et seuil publié. Protocole détaillé dans `binoculars-eu-protocol.md`.
- Publication du corpus EN sur HF Datasets :
  `OpenLLM-France/binoculars-eu-corpus-en-v01`.
- Une evaluation card par profil : `docs/eval-card-fr.md`, `docs/eval-card-en.md`.
- FR reste le défaut ; `en` s'obtient par `Binoculars.for_language("en")`.

### 4.4 V2 — hardening production

- Pas de nouveau profil. Rate-limiting, authentification par clé, métriques Prometheus,
  observabilité des temps de chargement modèle, tests de charge sur l'API.
- Politique de cache mémoire bornée (éviction LRU explicite, budget VRAM par profil).
- Durcissement de `trust_remote_code` : liste blanche par profil, refus par défaut.
- CI de synchronisation upstream : rebase automatisé sur `ahans30/Binoculars` + tests de
  non-régression sur `from_legacy`.

### 4.5 V3 — profil `es`

- Paire espagnole ouverte, candidat principal : la famille **Salamandra** (BSC-LT), sous
  réserve de vérification de l'existence d'une paire Base / Instruct au tokenizer
  compatible. **Modèle à identifier définitivement pendant la V2.**
- Corpus `OpenLLM-France/binoculars-eu-corpus-es-v01`, même protocole §18.

### 4.6 V4+ — contributions communautaires

- Profils `de`, `it`, `pt`, `pl` portés par les communautés linguistiques concernées.
- Le rôle du projet devient celui de **mainteneur du moteur et gardien du protocole** :
  un profil n'est fusionné que s'il fournit corpus public + SHA-256 + seuils + evaluation
  card conformes à §18. `CONTRIBUTING.md` documente cette checklist d'acceptation.

### 4.7 Non-objectifs

- **Pas un humanizer** — la détection est un objectif distinct, la réécriture reste
  couverte par le skill `avoid-ai-writing-multilingual` (SKILL-FR.md).
- **Pas un service de contournement** — le projet est explicitement pour les récepteurs
  (éducation, presse, modération, agents comme pipeline de contrôle qualité).
- **Pas d'entraînement supervisé** — la méthode reste zero-shot, seuls les seuils sont
  calibrés.
- **Pas de détection cross-lingue automatique** — la plateforme ne devine pas la langue
  du texte en V0.1 ; le profil est choisi par l'appelant. L'auto-détection de langue est
  une piste V2+, et elle sera explicitement signalée comme heuristique.
- **Pas de garantie multi-modèles hors profil** — chaque profil est calibré sur une
  famille de générateurs ; la généralisation est mesurée en OOD mais pas garantie.
- **Pas de mutualisation des seuils entre profils** — un seuil calibré en FR n'a aucune
  validité en EN, et réciproquement. C'est précisément la raison d'être des profils.

---

## 5. Personas et cas d'usage

| Persona | Cas d'usage | Fréquence attendue |
|---------|-------------|-------------------|
| Enseignant du supérieur FR | Vérification de copies étudiantes | Hebdomadaire |
| Journaliste / fact-checker | Vérification de communiqués, tweets viraux | Ponctuelle |
| Modérateur de plateforme (Twake, forums) | Détection contenu massif généré | Continue (API) |
| Chercheur en NLP | Reproductibilité, benchmarks | Recherche |
| Auteur d'agent (toi, LINAGORA) | Boucle draft → detect → humanize | Pipeline continu |
| Auditeur qualité éditoriale | Audit corpus, dataset, documentation | Trimestriel |

---

## 6. Architecture technique

### 6.1 Principe : le profil de langue est la seule chose qui varie

Le code upstream (`refs/detector.py`, 105 lignes) mélange trois choses dans une seule
classe : le **moteur de scoring** (tokenisation, forward des deux modèles, `perplexity()`
/ `entropy()`, ratio), les **paramètres de la paire de modèles** (noms HF, dtype,
`max_token_observed`), et les **seuils de décision**, ces derniers étant deux constantes
de module (`BINOCULARS_ACCURACY_THRESHOLD`, `BINOCULARS_FPR_THRESHOLD`) valables
uniquement pour Falcon-7B en bf16.

`binoculars-eu` sépare ces trois responsabilités. Le moteur ne change plus jamais ; les
deux autres sont regroupées dans un objet immuable, le `LanguageProfile`.

### 6.2 La classe `LanguageProfile`

```python
# binoculars_eu/profiles/base.py
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class LanguageProfile:
    """Tout ce qui est spécifique à une langue. Immuable par construction :
    un profil publié est une empreinte figée, versionnée, citable."""

    # --- Identité ---------------------------------------------------------
    code: str                      # code ISO 639-1, ex. "fr", "en", "es"
    display_name: str              # ex. "Français", "English"

    # --- Paire de modèles -------------------------------------------------
    observer_model: str            # repo HF de l'observer, ex. "OpenLLM-France/Luciole-1B-Base"
    performer_model: str           # repo HF du performer

    # --- Seuils calibrés --------------------------------------------------
    threshold_accuracy: float      # seuil optimisé F1 (mode "accuracy")
    threshold_low_fpr: float       # seuil optimisé FPR bas (mode "low-fpr")
    threshold_tpr_at_fpr_1: float  # seuil au point FPR = 1 % (métrique headline, §18.2)

    # --- Traçabilité de la calibration ------------------------------------
    corpus_sha256: str             # empreinte du JSONL de calibration
    corpus_url: str                # dataset HF public
    calibration_date: str          # ISO 8601, ex. "2026-09-15"
    calibration_seed: int          # graine des splits stratifiés (§18.1)

    # --- Comportement de chargement ---------------------------------------
    share_tokenizer_from_observer: bool = True   # cf. §3.3 point 2
    trust_remote_code: bool = False              # True requis pour NemotronH (8B, V0.2)

    # --- Localisation des verdicts ----------------------------------------
    label_ai: str = "Probablement généré par IA"
    label_human: str = "Probablement écrit par un humain"

    # --- Optionnel --------------------------------------------------------
    calibration_note: Optional[str] = None       # provenance externe, réserves, etc.
```

Trois champs méritent une justification explicite.

**`threshold_tpr_at_fpr_1`** existe séparément des deux modes upstream parce que la
métrique headline du projet est TPR@FPR=1 % (§18.2). Le seuil correspondant est un objet
publiable de plein droit, pas un sous-produit du mode `low-fpr`.

**`share_tokenizer_from_observer`** est le point d'assouplissement de
`assert_tokenizer_consistency` (`refs/utils.py`, 10 lignes), qui compare les deux `vocab`
et lève une `ValueError` à la moindre divergence. Sur Luciole, 8 slots `<unused0..7>` sont
remplis côté Instruct par des marqueurs de chat : l'assertion upstream échoue alors que
les tokenizers sont fonctionnellement interchangeables sur 127 992 tokens (§3.2). Quand
ce drapeau est `True`, on charge **un seul** tokenizer, celui de l'observer, et on le
partage — l'assertion devient inutile plutôt que d'être patchée, et on économise ~500 Mo
de RAM. Quand il est `False`, l'assertion stricte upstream est appliquée telle quelle.

**`label_ai` / `label_human`** remplacent les chaînes en dur
`"Most likely AI-generated"` / `"Most likely human-generated"` de `predict()`. Un
détecteur destiné à des enseignants français ne doit pas rendre son verdict en anglais.

### 6.3 Registre et auto-discovery

Chaque profil est un **package** sous `binoculars_eu/profiles/`, qui s'auto-enregistre à
l'import. Le registre découvre les profils avec `pkgutil.iter_modules`, sans liste
codée en dur : ajouter une langue = ajouter un dossier.

```python
# binoculars_eu/profiles/__init__.py
import importlib
import pkgutil
from typing import Dict, List
from .base import LanguageProfile

_REGISTRY: Dict[str, LanguageProfile] = {}
DEFAULT_PROFILE_CODE = "fr"        # FR est le défaut de la plateforme (§4.0)


def register(profile: LanguageProfile) -> LanguageProfile:
    """Appelé par chaque profils/<lang>/__init__.py au chargement du module."""
    if profile.code in _REGISTRY:
        raise ValueError(f"Profil déjà enregistré : {profile.code}")
    _REGISTRY[profile.code] = profile
    return profile


def _discover() -> None:
    """Importe tous les sous-packages de profiles/, ce qui déclenche leur register()."""
    for module_info in pkgutil.iter_modules(__path__):
        if module_info.ispkg:                      # base.py n'est pas un package
            importlib.import_module(f"{__name__}.{module_info.name}")


def get_profile(code: str) -> LanguageProfile:
    if not _REGISTRY:
        _discover()
    if code not in _REGISTRY:
        raise KeyError(
            f"Profil inconnu : {code!r}. Disponibles : {sorted(_REGISTRY)}"
        )
    return _REGISTRY[code]


def list_profiles() -> List[LanguageProfile]:
    if not _REGISTRY:
        _discover()
    return [_REGISTRY[c] for c in sorted(_REGISTRY)]
```

Un profil concret, côté FR :

```python
# binoculars_eu/profiles/fr/__init__.py
import json
from pathlib import Path
from ..base import LanguageProfile
from .. import register

_HERE = Path(__file__).parent
_TH = json.loads((_HERE / "thresholds.json").read_text(encoding="utf-8"))
_META = json.loads((_HERE / "metadata.json").read_text(encoding="utf-8"))

FR_PROFILE = register(LanguageProfile(
    code="fr",
    display_name="Français",
    observer_model="OpenLLM-France/Luciole-1B-Base",
    performer_model="OpenLLM-France/Luciole-1B-Instruct-1.1",
    threshold_accuracy=_TH["accuracy"],
    threshold_low_fpr=_TH["low_fpr"],
    threshold_tpr_at_fpr_1=_TH["tpr_at_fpr_1"],
    corpus_sha256=_META["corpus_sha256"],
    corpus_url="https://huggingface.co/datasets/OpenLLM-France/binoculars-eu-corpus-fr-v01",
    calibration_date=_META["calibration_date"],
    calibration_seed=_META["calibration_seed"],
    share_tokenizer_from_observer=True,
    trust_remote_code=False,
    label_ai="Probablement généré par IA",
    label_human="Probablement écrit par un humain",
))
```

Séparer `thresholds.json` (chiffres calibrés) de `metadata.json` (traçabilité) n'est pas
cosmétique : le premier est régénéré par `calibration/calibrate.py`, le second par le
pipeline d'évaluation. Deux producteurs, deux fichiers, pas de conflit de merge.

### 6.4 Les trois méthodes d'instanciation

| Méthode | Usage | Seuils utilisés |
|---------|-------|-----------------|
| `Binoculars.for_language("fr")` | Cas normal, 99 % des appels | Ceux du profil enregistré |
| `Binoculars(profile=MON_PROFIL)` | Profil custom non enregistré (R&D, calibration en cours) | Ceux du `LanguageProfile` fourni |
| `Binoculars.from_legacy(obs, perf, mode)` | Compatibilité upstream stricte | Constantes Falcon de Hans et al. |

```python
class Binoculars:
    def __init__(self, profile: LanguageProfile | None = None,
                 mode: str = "low-fpr", max_token_observed: int = 512,
                 use_bfloat16: bool = True) -> None:
        self.profile = profile or get_profile(DEFAULT_PROFILE_CODE)   # défaut = fr
        ...

    @classmethod
    def for_language(cls, code: str, **kwargs) -> "Binoculars":
        """Résolution par le registre. Lève KeyError si le profil n'existe pas."""
        return cls(profile=get_profile(code), **kwargs)

    @classmethod
    def from_legacy(cls, observer_name_or_path: str = "tiiuae/falcon-7b",
                    performer_name_or_path: str = "tiiuae/falcon-7b-instruct",
                    mode: str = "low-fpr", **kwargs) -> "Binoculars":
        """Reproduit exactement la signature et le comportement de
        ahans30/Binoculars, seuils Falcon compris. Aucun profil requis."""
        legacy = LanguageProfile(
            code="legacy",
            display_name="Upstream (Hans et al., ICML 2024)",
            observer_model=observer_name_or_path,
            performer_model=performer_name_or_path,
            threshold_accuracy=0.9015310749276843,   # f1-optimisé, upstream
            threshold_low_fpr=0.8536432310785527,    # FPR bas, upstream
            threshold_tpr_at_fpr_1=0.8536432310785527,
            corpus_sha256="", corpus_url="https://arxiv.org/abs/2401.12070",
            calibration_date="2024-01-22", calibration_seed=-1,
            share_tokenizer_from_observer=False,     # assertion stricte upstream
            trust_remote_code=True,
            label_ai="Most likely AI-generated",
            label_human="Most likely human-generated",
            calibration_note="Seuils non recalibrés : valeurs publiées par Hans et al. "
                             "pour Falcon-7B / Falcon-7B-Instruct en bfloat16.",
        )
        return cls(profile=legacy, mode=mode, **kwargs)
```

`mode` reste un paramètre d'appel et non un champ de profil, exactement comme upstream :
`change_mode("accuracy" | "low-fpr" | "tpr-at-fpr-1")` réaffecte `self.threshold` depuis
le profil courant. Le troisième mode est l'ajout de `binoculars-eu`.

### 6.5 Trois différences assumées vs upstream

Le fork reste **synchronisable** avec `ahans30/Binoculars` : `metrics.py`
(`perplexity()`, `entropy()`) est repris **inchangé**, et c'est là que se trouve toute la
substance mathématique de la méthode. Les divergences sont volontairement confinées à
trois points, documentés dans `docs/upstream-diff.md` :

| # | Différence | Fichier upstream touché | Réversible ? |
|---|------------|--------------------------|--------------|
| (a) | **Architecture profils** : seuils et paires de modèles sortis du code vers des `LanguageProfile` déclaratifs ; `from_legacy` conserve le chemin d'origine | `detector.py` | Oui — `from_legacy` est le chemin de compatibilité |
| (b) | **Tokenizer partagé optionnel** via `share_tokenizer_from_observer` ; l'assertion stricte reste disponible et reste le défaut pour `from_legacy` | `utils.py` | Oui — drapeau à `False` = comportement upstream |
| (c) | **Labels localisés** `label_ai` / `label_human` au lieu de chaînes anglaises en dur dans `predict()` | `detector.py` | Oui — le profil `legacy` restitue les chaînes originales |

`metrics.py` n'est **pas** modifié, ce qui rend les rebases upstream indolores sur la
partie critique. Une CI de synchronisation (V2, §4.4) vérifie que le fichier reste
identique à l'amont et que `from_legacy` produit les mêmes scores.

### 6.6 Flux d'exécution : du profil au verdict

```
┌────────────────────────────────────────────────────────────────────────────┐
│                            binoculars-eu                                   │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  RÉSOLUTION DU PROFIL                                                      │
│  ────────────────────                                                      │
│   for_language("fr")     Binoculars(profile=P)     from_legacy(obs, perf)   │
│          │                        │                         │              │
│          ▼                        │                         │              │
│   ┌──────────────────────┐        │                         │              │
│   │  Registre            │        │                         │              │
│   │  pkgutil.iter_modules│        │                         │              │
│   │  profiles/fr  ◀ défaut│       │                         │              │
│   │  profiles/en  (V1)   │        │                         │              │
│   └──────────┬───────────┘        │                         │              │
│              └────────────────────┼─────────────────────────┘              │
│                                   ▼                                        │
│                    ┌──────────────────────────────┐                        │
│                    │      LanguageProfile         │                        │
│                    │  (frozen dataclass)          │                        │
│                    │  observer / performer        │                        │
│                    │  thresholds ×3               │                        │
│                    │  corpus_sha256 / url         │                        │
│                    │  share_tokenizer / labels    │                        │
│                    └──────────────┬───────────────┘                        │
│                                   │                                        │
├───────────────────────────────────┼────────────────────────────────────────┤
│  MOTEUR (identique pour tous les profils)                                  │
│  ───────────────────────────────────────                                   │
│   Input text                      │                                        │
│      │                            │ profile.observer_model                 │
│      ▼                            ▼                                        │
│   ┌─────────────────┐      ┌──────────────────────┐                        │
│   │  Tokenizer      │─────▶│  Batch encoding      │                        │
│   │  (observer, si  │      │  max_tokens=512      │                        │
│   │   share=True)   │      └──────────┬───────────┘                        │
│   └─────────────────┘                 │                                    │
│         ┌─────────────────────────────┴──────────────┐                     │
│         ▼                                            ▼                     │
│   ┌─────────────────┐                       ┌────────────────┐             │
│   │  Observer       │                       │  Performer     │             │
│   │  profile.       │                       │  profile.      │             │
│   │  observer_model │                       │ performer_model│             │
│   └────────┬────────┘                       └────────┬───────┘             │
│            │ logits                                  │ logits             │
│            └──────────────┬───────────────────────┬──┘                     │
│                           ▼                       ▼                        │
│                    ┌──────────────┐        ┌──────────────┐                │
│                    │  X-Perplex.  │        │  Perplexity  │                │
│                    │  entropy()   │        │  perplexity()│                │
│                    │  (inchangé)  │        │  (inchangé)  │                │
│                    └──────┬───────┘        └───────┬──────┘                │
│                           └───────────┬────────────┘                       │
│                                       ▼                                    │
│                              ┌────────────────┐                            │
│                              │  score = PPL   │                            │
│                              │       / X-PPL  │                            │
│                              └───────┬────────┘                            │
│                                      ▼                                     │
│                    score < profile.threshold_<mode> ?                      │
│                                      │                                     │
│                                      ▼                                     │
│                    profile.label_ai  |  profile.label_human                │
└────────────────────────────────────────────────────────────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
          API Python (§13.1)                    API HTTP FastAPI (§13.2)
                                          cache LRU par (profile, mode) → /detect
```

### 6.7 Arborescence des profils

```
binoculars_eu/profiles/
├── __init__.py          # registre : register(), get_profile(), list_profiles(), _discover()
├── base.py              # dataclass LanguageProfile (frozen)
├── fr/                  # V0.1 — profil par défaut
│   ├── __init__.py      # construit FR_PROFILE et appelle register()
│   ├── thresholds.json  # {"accuracy": ..., "low_fpr": ..., "tpr_at_fpr_1": ...}
│   └── metadata.json    # corpus_sha256, calibration_date, calibration_seed, notes
└── en/                  # V1 — opt-in
    ├── __init__.py
    ├── thresholds.json  # seuils Hans et al. + seuils reproduits localement
    └── metadata.json    # calibration_note : provenance externe + écart de reproduction
```

Contrat d'un dossier de profil : exactement trois fichiers, aucun code de scoring, aucun
import du détecteur (sinon import circulaire), et un appel à `register()` au niveau module
pour que l'auto-discovery suffise.

### 6.8 Configuration cible V0.1 — profil `fr`

| Élément | Valeur |
|---------|--------|
| `code` / `display_name` | `fr` / `Français` |
| Observer model | `OpenLLM-France/Luciole-1B-Base` |
| Performer model | `OpenLLM-France/Luciole-1B-Instruct-1.1` |
| Architecture | `NemotronForCausalLM` (Transformer pur, 24 couches, 2048 dim) |
| `share_tokenizer_from_observer` | `True` — tokenizer de Luciole-1B-Base (127 992 tokens communs vérifiés) |
| Précision | bfloat16 |
| Max tokens observés | 512 |
| Modes | `low-fpr` (FPR ~1%), `accuracy` (F1 max), `tpr-at-fpr-1` (headline §18.2) |
| `trust_remote_code` | `False` (contrairement au 8B de la V0.2) |
| Corpus de calibration | `OpenLLM-France/binoculars-eu-corpus-fr-v01` (+ `-ood`) |
| Statut | **profil par défaut de la plateforme** |

### 6.9 Empreinte mémoire (mesurée par calcul)

| Composant | VRAM |
|-----------|------|
| Luciole-1B-Base bfloat16 | ~2.1 Go |
| Luciole-1B-Instruct-1.1 bfloat16 | ~2.1 Go |
| Activations batch=1, seq=512 | ~0.4 Go |
| Activations batch=8, seq=512 | ~2.5 Go |
| **Total à batch=1** | **~4.6 Go** |
| **Total à batch=8** | **~6.7 Go** |

**Compatible avec la contrainte OVH L4 actuelle** : 7.7 Go libres → batch max ≈ 8.

**Conséquence multi-profils** : chaque profil chargé occupe sa propre empreinte. Un
serveur qui expose `fr` et `en` (Falcon-7B ×2 en bf16 ≈ 30 Go) ne tient pas sur un seul
L4. C'est la raison du **cache LRU borné par `(profile, mode)`** côté API (§13.2) et de
la politique d'éviction explicite prévue en V2 : on ne garde en mémoire que les profils
réellement sollicités.

---

## 7. Environnement cible : OVH L4 Gravelines

### 7.1 État constaté (post-nettoyage J-1)

| Ressource | Avant nettoyage | Après nettoyage |
|-----------|-----------------|-----------------|
| GPU | NVIDIA L4 (Ada Lovelace) 24 Go | idem |
| VRAM libre | 7.7 Go | **22.5 Go** (0 MiB alloué) |
| Conso GPU | 27 W | 16.7 W |
| Driver | 595.71.05, CUDA 13.2 | idem |
| CPU | AMD EPYC 9454, 22 vCPU | idem |
| RAM | 86 Go (6.8 Go utilisés) | idem |
| Disque | 50 Go libres (88 % occupé) | **273 Go libres (30 % occupé)** |
| OS | Ubuntu 24.04.4 LTS | idem |

Gain de 223 Go de disque (datasets 199 G, Docker 20.3 G, pip 5.5 G, journaux 1.1 G) et
libération complète VRAM (services TTS `voice-stt` conservés en service, `voice-tts` et
`voice-tts-talker` arrêtés mais gardés `enabled` pour redémarrage ultérieur).

### 7.2 Cohabitation future avec vLLM Qwen3-TTS

Contrainte future : quand le service TTS sera relancé, il consommera ~14.8 Go VRAM. Le POC
1B+1B (~5 Go max en bfloat16) rentre confortablement dans les 7-9 Go restants. Deux niveaux
de protection restent recommandés pour toute exécution en cohabitation :

1. **Isolation VRAM** au niveau PyTorch :
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,expandable_segments:True"
   ```

2. **Isolation processus** via `systemd-run` avec cgroups mémoire :
   ```bash
   systemd-run --user --scope --property=MemoryMax=8G --property=MemorySwapMax=0 \
     python -m binoculars_eu.calibrate
   ```

3. **Batch conservateur** : batch=4 par défaut, jamais > 8, avec surveillance
   `nvidia-smi --query-gpu=memory.free` avant chaque batch.

Pour le POC V0.1, tant que le service TTS reste arrêté, la contrainte disparaît de fait :
 on peut monter à batch=16 sans risque.

### 7.3 Prérequis disque

Espace nécessaire pour l'installation complète :

| Actif | Taille | Cumul |
|-------|--------|-------|
| Environnement Python + PyTorch CUDA | ~5 Go | 5 |
| Luciole-1B-Base (bf16) | ~2.5 Go | 7.5 |
| Luciole-1B-Instruct-1.1 (bf16) | ~2.5 Go | 10 |
| Cache HF (configs, tokenizers) | ~0.5 Go | 10.5 |
| Corpus généré (500 textes JSON) | ~5 Mo | 10.5 |
| Notebooks + résultats + logs | ~200 Mo | 10.7 |
| **Total minimum** | | **~11 Go** |

Marge : 273 Go libres après nettoyage → aucune contrainte, y compris pour la V0.2 en 8B+8B
int8 (voir §16.1) qui demanderait ~30 Go supplémentaires en cache modèles.

---

## 8. Prérequis d'installation détaillés

### 8.1 Système (Ubuntu 24.04)

```bash
# Vérifications préalables
nvidia-smi                    # Doit afficher le L4
cat /etc/os-release           # Ubuntu 24.04
df -h /                       # Vérifier 15 Go libres min
free -h                       # Vérifier >= 16 Go RAM libre

# Paquets système
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev git build-essential
```

### 8.2 Environnement Python

```bash
# Création venv isolé
cd ~
python3.12 -m venv .venv-binoculars-eu
source .venv-binoculars-eu/bin/activate

# Dépendances de base
pip install --upgrade pip setuptools wheel

# PyTorch CUDA (wheels pré-compilées, pas besoin de nvcc)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Stack ML
pip install "transformers>=4.57.1"    # requis pour Nemotron + Luciole-Instruct-1.1
pip install accelerate
pip install datasets                  # pour la construction corpus
pip install scikit-learn              # pour ROC / calibration
pip install pandas numpy matplotlib
pip install jupyter                   # optionnel, pour notebooks

# API HTTP + documentation Swagger (§13.2)
pip install fastapi uvicorn pydantic

# Optionnel : client HTTP pour les exemples et les tests d'API
pip install requests httpx

# HuggingFace CLI + auth (si comptes privés)
pip install huggingface_hub
huggingface-cli login  # Luciole est public, l'auth est optionnelle
```

### 8.3 Configuration HuggingFace

```bash
# Emplacement du cache HF — le forcer vers un disque avec de la place
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface/hub"

# Ajouter à ~/.bashrc pour persistance
echo 'export HF_HOME="$HOME/.cache/huggingface"' >> ~/.bashrc
echo 'export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"' >> ~/.bashrc
```

### 8.4 Modèles à télécharger

```bash
# Pré-téléchargement pour tester la connectivité et détecter les problèmes tôt
huggingface-cli download OpenLLM-France/Luciole-1B-Base
huggingface-cli download OpenLLM-France/Luciole-1B-Instruct-1.1

# Vérification
du -sh ~/.cache/huggingface/hub/models--OpenLLM-France--Luciole-1B-*
# Attendu : ~2.5 Go chacun
```

### 8.5 Repo binoculars-eu

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/ahans30/Binoculars.git binoculars-eu
cd binoculars-eu

# Le fork reste synchronisable avec l'upstream (§6.5). On conserve le remote
# d'origine sous le nom 'upstream' pour pouvoir rebaser plus tard.
git remote rename origin upstream
git checkout -b eu-profiles

# Installation en mode éditable, avec l'extra [api] pour FastAPI/uvicorn/pydantic
pip install -e ".[api]"
```

L'extra `[api]` déclaré dans `pyproject.toml` regroupe `fastapi`, `uvicorn` et
`pydantic` : le cœur du package reste installable sans dépendance web pour les
usages purement Python (§13.1).

---

## 9. Script agent de vérification préalable

Ce script est conçu pour être **exécuté par un agent avant tout travail de développement**.
Il vérifie l'ensemble des prérequis et produit un rapport JSON structuré.

**Fichier** : `verify_environment.py` (à placer à la racine du repo)

```python
#!/usr/bin/env python3
"""
verify_environment.py — Vérification exhaustive des prérequis binoculars-eu.

Usage:
    python verify_environment.py [--verbose] [--json report.json]

Retour :
    0 si tout OK, 1 si un ou plusieurs checks critiques échouent.
    Rapport JSON structuré consommable par un agent.
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


CHECKS = []


def check(name, critical=True):
    """Décorateur pour enregistrer un check."""
    def deco(fn):
        CHECKS.append((name, critical, fn))
        return fn
    return deco


@check("Python version >= 3.10")
def check_python():
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 10
    return ok, f"{v.major}.{v.minor}.{v.micro}"


@check("OS is Linux or macOS")
def check_os():
    sys_name = platform.system()
    ok = sys_name in ("Linux", "Darwin")
    return ok, sys_name


@check("nvidia-smi available (skip on macOS)")
def check_nvidia_smi():
    if platform.system() == "Darwin":
        return True, "N/A (macOS, will use MPS)"
    path = shutil.which("nvidia-smi")
    return path is not None, path or "not found"


@check("GPU has >= 6 GB free VRAM (skip on macOS)")
def check_vram():
    if platform.system() == "Darwin":
        return True, "N/A (macOS)"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        free_mib = int(result.stdout.strip().split("\n")[0])
        free_gb = free_mib / 1024
        return free_gb >= 6.0, f"{free_gb:.1f} GB free"
    except Exception as e:
        return False, str(e)


@check("Disk has >= 15 GB free")
def check_disk():
    st = shutil.disk_usage(Path.home())
    free_gb = st.free / (1024**3)
    return free_gb >= 15.0, f"{free_gb:.1f} GB free"


@check("RAM has >= 8 GB free")
def check_ram():
    if platform.system() == "Linux":
        with open("/proc/meminfo") as f:
            meminfo = dict(
                line.split(":") for line in f.read().strip().split("\n")
            )
            free_kb = int(meminfo["MemAvailable"].strip().split()[0])
            free_gb = free_kb / (1024**2)
            return free_gb >= 8.0, f"{free_gb:.1f} GB available"
    return True, "check skipped (non-Linux)"


@check("torch installed")
def check_torch():
    try:
        import torch
        return True, torch.__version__
    except ImportError:
        return False, "not installed"


@check("torch CUDA/MPS available")
def check_torch_device():
    try:
        import torch
        if torch.cuda.is_available():
            return True, f"CUDA {torch.version.cuda}, device={torch.cuda.get_device_name(0)}"
        if torch.backends.mps.is_available():
            return True, "MPS (Apple Silicon)"
        return False, "CPU only — will be very slow"
    except Exception as e:
        return False, str(e)


@check("transformers >= 4.57.1")
def check_transformers():
    try:
        import transformers
        v = transformers.__version__
        parts = [int(p) for p in v.split(".")[:3]]
        ok = parts >= [4, 57, 1]
        return ok, v
    except Exception as e:
        return False, str(e)


@check("Nemotron architecture supported in transformers")
def check_nemotron_support():
    try:
        from transformers.models.nemotron import NemotronForCausalLM  # noqa
        return True, "NemotronForCausalLM available"
    except ImportError as e:
        return False, f"Nemotron not found: {e}"


@check("HuggingFace Hub reachable")
def check_hf_hub():
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.model_info("OpenLLM-France/Luciole-1B-Base")
        return True, f"Reached Luciole-1B-Base (sha={info.sha[:8]})"
    except Exception as e:
        return False, str(e)


@check("Luciole-1B-Base tokenizer loadable")
def check_tokenizer_base():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("OpenLLM-France/Luciole-1B-Base")
        return True, f"vocab_size={tok.vocab_size}, class={tok.__class__.__name__}"
    except Exception as e:
        return False, str(e)


@check("Luciole-1B-Instruct-1.1 tokenizer loadable")
def check_tokenizer_instruct():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("OpenLLM-France/Luciole-1B-Instruct-1.1")
        return True, f"vocab_size={tok.vocab_size}, class={tok.__class__.__name__}"
    except Exception as e:
        return False, str(e)


@check("Vocabularies match on 127 992 shared tokens (excl. IDs 260-267)", critical=True)
def check_vocab_compat():
    """Vérifie que les 127 992 tokens hors 260-267 sont identiques et que
    260-267 sont bien réservés <unused*> côté Base."""
    try:
        from transformers import AutoTokenizer
        tok_b = AutoTokenizer.from_pretrained("OpenLLM-France/Luciole-1B-Base")
        tok_i = AutoTokenizer.from_pretrained("OpenLLM-France/Luciole-1B-Instruct-1.1")

        # Vérifier les 127 992 tokens partagés (hors slots réservés)
        shared_ids = [tid for tid in range(128000) if not (260 <= tid <= 267)]
        # Sampler 200 IDs répartis pour rester rapide (< 1s)
        step = max(1, len(shared_ids) // 200)
        sample_ids = shared_ids[::step]
        mismatches = []
        for tid in sample_ids:
            t_b = tok_b.convert_ids_to_tokens(tid)
            t_i = tok_i.convert_ids_to_tokens(tid)
            if t_b != t_i:
                mismatches.append((tid, t_b, t_i))
        if mismatches:
            return False, f"{len(mismatches)} mismatch(es), first: {mismatches[0]}"

        # Vérifier que 260-267 sont bien réservés côté Base
        expected_unused = {i: f"<unused{i - 260}>" for i in range(260, 268)}
        for tid, expected in expected_unused.items():
            actual = tok_b.convert_ids_to_tokens(tid)
            if actual != expected:
                return False, f"Base ID {tid} = {actual!r}, expected {expected!r}"

        return True, f"{len(sample_ids)} shared tokens identical, IDs 260-267 reserved on Base"
    except Exception as e:
        return False, str(e)


@check("Sample encoding produces identical token IDs (7 test cases)", critical=True)
def check_encoding_parity():
    """Reprend les 7 échantillons validés à la vérification préalable :
    français accentué, ligatures œ, code Python, SQL, emoji, guillemets typo."""
    try:
        from transformers import AutoTokenizer
        tok_b = AutoTokenizer.from_pretrained("OpenLLM-France/Luciole-1B-Base")
        tok_i = AutoTokenizer.from_pretrained("OpenLLM-France/Luciole-1B-Instruct-1.1")
        sample_texts = [
            # Français accentué + ligature
            "Bonjour, comment allez-vous ce matin ?",
            "L'écosystème numérique européen évolue rapidement.",
            "Un vœu de sœur, cœur de bœuf, un œil complice.",
            # Code Python
            "def hello(name: str) -> None:\n    print(f'Bonjour {name}')",
            # SQL
            "SELECT id, nom FROM utilisateurs WHERE créé_le > '2026-01-01';",
            # Emoji + guillemets typographiques
            "« Salut ! 🎉 Ça va ? — Oui, et toi ? »",
            # Mixte technique
            "Le modèle Luciole-1B-Base fait 128k tokens de vocab.",
        ]
        for text in sample_texts:
            ids_b = tok_b(text, add_special_tokens=False)["input_ids"]
            ids_i = tok_i(text, add_special_tokens=False)["input_ids"]
            if ids_b != ids_i:
                return False, f"Mismatch on '{text[:30]}...': {ids_b[:10]}... vs {ids_i[:10]}..."
        return True, f"{len(sample_texts)} sample texts encode identically"
    except Exception as e:
        return False, str(e)


@check("Chat markers ARE expected to differ (documented invariant)", critical=False)
def check_chat_markers_diverge():
    """Vérifie que les marqueurs de chat divergent comme attendu.
    C'est une divergence documentée et sans impact pour binoculars-eu :
    aucun texte utilisateur ne les contient."""
    try:
        from transformers import AutoTokenizer
        tok_b = AutoTokenizer.from_pretrained("OpenLLM-France/Luciole-1B-Base")
        tok_i = AutoTokenizer.from_pretrained("OpenLLM-France/Luciole-1B-Instruct-1.1")
        marker = "<|im_start|>"
        ids_b = tok_b(marker, add_special_tokens=False)["input_ids"]
        ids_i = tok_i(marker, add_special_tokens=False)["input_ids"]
        # Expected: Base retokenizes to 7 tokens, Instruct to 1
        ok = len(ids_b) == 7 and ids_i == [260]
        return ok, f"Base={ids_b} (len={len(ids_b)}), Instruct={ids_i}"
    except Exception as e:
        return False, str(e)


@check("Enough disk to download models (~5 GB)")
def check_disk_for_models():
    st = shutil.disk_usage(Path.home() / ".cache")
    free_gb = st.free / (1024**3)
    return free_gb >= 6.0, f"{free_gb:.1f} GB free in ~/.cache path"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", type=str, default=None,
                        help="Write JSON report to this path")
    args = parser.parse_args()

    report = {
        "environment": {
            "os": platform.system(),
            "os_release": platform.release(),
            "python": sys.version.split()[0],
            "cwd": str(Path.cwd()),
        },
        "checks": [],
        "summary": {"passed": 0, "failed": 0, "critical_failed": 0},
    }

    for name, critical, fn in CHECKS:
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, f"exception: {e}"

        status = "PASS" if ok else ("FAIL-CRITICAL" if critical else "WARN")
        report["checks"].append({
            "name": name,
            "critical": critical,
            "passed": ok,
            "detail": detail,
        })
        if ok:
            report["summary"]["passed"] += 1
        else:
            report["summary"]["failed"] += 1
            if critical:
                report["summary"]["critical_failed"] += 1

        if args.verbose or not ok:
            print(f"[{status:14s}] {name}: {detail}")

    print()
    print(f"Summary: {report['summary']['passed']} passed, "
          f"{report['summary']['failed']} failed "
          f"({report['summary']['critical_failed']} critical).")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"Report written to {args.json}")

    sys.exit(0 if report["summary"]["critical_failed"] == 0 else 1)


if __name__ == "__main__":
    main()
```

### 9.1 Usage par un agent

```bash
# Un agent (Claude Code, Codex, LangGraph, etc.) peut lancer :
python verify_environment.py --json /tmp/env_report.json

# Puis parser /tmp/env_report.json :
# - summary.critical_failed == 0 → prêt pour le POC
# - sinon → agir sur les checks failed avant de continuer
```

L'agent peut prendre des décisions conditionnelles sur les échecs :

| Check échoué | Action agent |
|--------------|--------------|
| `Python version` | Installer Python 3.12 via apt |
| `torch installed` | `pip install torch --index-url ...` |
| `transformers >= 4.57.1` | `pip install -U transformers` |
| `Disk has >= 15 GB free` | Lancer `du -h --max-depth=2 ~` puis nettoyer HF cache d'anciens projets |
| `GPU has >= 6 GB free VRAM` | Alerter l'utilisateur (peut-être arrêter un service temporairement) |
| `Vocabularies match` | **Stop critique** — investigation manuelle nécessaire |

---

## 9 bis. Script de démarrage rapide de l'API

Une fois §8 exécuté et `verify_environment.py` au vert, l'API HTTP est démarrable en une
commande. Cette section est le chemin le plus court entre un environnement prêt et un
premier verdict obtenu par HTTP.

### 9bis.1 Lancement

```bash
source ~/.venv-binoculars-eu/bin/activate
cd ~/projects/binoculars-eu

# Développement : rechargement à chaud, un seul worker (les modèles sont lourds)
uvicorn binoculars_eu.api:app --host 0.0.0.0 --port 8000 --reload

# Production : pas de --reload, un worker par GPU disponible, timeout large au
# démarrage car le premier /detect déclenche le chargement des deux modèles
uvicorn binoculars_eu.api:app --host 0.0.0.0 --port 8000 \
  --workers 1 --timeout-keep-alive 75
```

Le chargement des modèles est **paresseux** : le process démarre en < 2 s, et c'est le
premier `POST /detect` sur un `(profile, mode)` donné qui paie le coût de chargement
(~20-40 s pour la paire Luciole 1B sur L4, cache HF chaud). Les appels suivants tapent le
cache LRU (§13.2).

**Préchauffage recommandé** juste après le démarrage, pour ne pas faire porter la latence
de chargement au premier utilisateur réel :

```bash
curl -s -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"text":"Texte de préchauffage suffisamment long pour être tokenisé normalement.","profile":"fr"}' \
  > /dev/null && echo "profil fr préchauffé"
```

### 9bis.2 Vérification en trois appels curl

```bash
# 1) L'API répond et le registre de profils est chargé
curl -s http://localhost:8000/health | python -m json.tool
# {"status": "ok", "version": "0.1.0", "default_profile": "fr",
#  "profiles_loaded": ["fr"], "detectors_cached": 0, "device": "cuda:0"}

# 2) Quels profils sont disponibles ? (FR seul en V0.1, FR + EN en V1)
curl -s http://localhost:8000/profiles | python -m json.tool

# 3) Détection — le champ "profile" est optionnel, il vaut "fr" par défaut
curl -s -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{
        "text": "Dans le paysage numérique en constante évolution, il est crucial de tirer parti des synergies pour naviguer dans un écosystème complexe.",
        "mode": "accuracy"
      }' | python -m json.tool
```

Réponse attendue sur ce texte (score bas → verdict IA) :

```json
{
  "score": 0.7213,
  "verdict": "ai",
  "label": "Probablement généré par IA",
  "confidence": "high",
  "threshold_used": 0.8402,
  "mode": "accuracy",
  "profile": "fr",
  "input_tokens": 34,
  "elapsed_ms": 118
}
```

### 9bis.3 Exemples complémentaires

```bash
# Texte humain manifeste — score plus élevé, verdict "human"
curl -s -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"text":"Je suis passé chez le boucher hier, il pleuvait des cordes, jai oublié mon parapluie chez ma sœur samedi."}'

# Mode low-fpr : moins de faux positifs, TPR plus bas — à privilégier en éducation
curl -s -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"text":"...","mode":"low-fpr"}'

# Profil explicite (V1) — opt-in, jamais implicite
curl -s -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"text":"In the ever-evolving digital landscape...","profile":"en"}'

# Profil inconnu → 404 avec la liste des profils disponibles
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"text":"Texte suffisamment long pour passer la validation Pydantic.","profile":"de"}'
# 404

# Texte trop court → 422 (contrainte Pydantic min_length=50)
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" -d '{"text":"trop court"}'
# 422
```

### 9bis.4 Swagger UI

La documentation interactive est générée automatiquement par FastAPI :

| URL | Contenu |
|-----|---------|
| `http://localhost:8000/docs` | Swagger UI — formulaires « Try it out » sur les trois routes |
| `http://localhost:8000/redoc` | ReDoc — lecture linéaire de la spec |
| `http://localhost:8000/openapi.json` | Spec OpenAPI 3.1 brute, exploitable pour générer des clients |

C'est le livrable de démonstration : un enseignant ou un journaliste peut tester le
détecteur depuis `/docs` sans écrire une ligne de code, et un intégrateur récupère un
client typé depuis `openapi.json`. Le déploiement public de cette même application en
Hugging Face Space est décrit en §11.5 bis.

### 9bis.5 Docker

```bash
docker build -t binoculars-eu:0.1.0 .
docker run --gpus all -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e BINOCULARS_EU_DEFAULT_PROFILE=fr \
  binoculars-eu:0.1.0
```

Monter le cache HF de l'hôte évite de re-télécharger ~5 Go de poids à chaque
reconstruction d'image — et permet un démarrage hors-ligne.

---


## 10. Corpus de calibration

### 10.1 Composition finale

**500 textes total, 250 humains + 250 IA, symétriques par thème.**

Volet humain (250) :

| Source | Nombre | Registre | Provenance |
|--------|--------|----------|------------|
| Wikipedia FR (articles pré-2022) | 80 | Encyclopédique | Dump officiel Wikimedia |
| Presse française (LeMonde, LeMagIT, LesEchos) | 60 | Journalistique | Archives ouvertes |
| Blogs techniques FR | 40 | Éditorial technique | LinuxFr, blog.maudet.cloud, blogs libres |
| Publications LinkedIn de professionnels vérifiés | 30 | Communication pro | Consentement à demander |
| Littéraires domaine public | 40 | Littéraire | Gallica, Wikisource |

Volet IA (250) :

| Générateur | Nombre | Prompt strategy |
|-----------|--------|-----------------|
| Luciole-23B-Instruct-1.1 (endpoint) | 100 | Titre + premier paragraphe des textes humains |
| Luciole-8B-Instruct-1.1 | 75 | Mêmes prompts |
| Luciole-1B-Instruct-1.1 (local) | 75 | Mêmes prompts |

**Symétrie thématique** : chaque texte humain a un jumeau IA sur le même sujet.

### 10.2 Corpus OOD léger V0.1 (50 textes)

Ajouté dès V0.1 pour démontrer la généralisation hors-Luciole. **Traité séparément** du
corpus in-distribution : il ne sert **pas** à la calibration des seuils, mais uniquement
à la mesure de généralisation.

| Source | Nombre | Fournisseur | Note |
|--------|--------|-------------|------|
| Mistral Small 24B Instruct | 50 | OpenRouter (`mistralai/mistral-small-3.2-24b-instruct`) | Fallback prévu : endpoint Mistral-24B LINAGORA quand disponible |

**Prompt strategy** : les 50 sujets sont repris du corpus humain V0.1 (échantillonnés
selon les 5 sources : Wikipedia, presse, blog tech, LinkedIn, littérature) pour garantir
la comparabilité avec le corpus in-distribution.

**Métriques évaluées sur ce corpus OOD** :

- AUC ROC sur le sous-ensemble Mistral vs 50 textes humains aléatoirement tirés du corpus
  in-distribution (test set).
- Δ AUC entre performance in-distribution (Luciole) et OOD (Mistral) : plus Δ est faible,
  meilleure est la généralisation.
- Ce résultat est publié dans une section dédiée du rapport d'évaluation, distincte des
  métriques primaires in-distribution.

**Justification** : éviter la critique légitime "détecteur qui ne détecte que sa propre
famille de modèles". La V0.1 aura ainsi une première mesure défendable de généralisation,
même partielle.

### 10.3 Corpus OOD complet (V0.2)

100 textes supplémentaires (au-delà des 50 Mistral déjà en V0.1) :

- 50 humains modernes non vus (posts LinkedIn 2026, articles récents).
- 25 générés par GPT-4o (via OpenRouter ou OpenAI direct).
- 25 générés par Claude Sonnet (via OpenRouter ou Anthropic direct).
- 25 hybrides (rédaction humaine + réécriture par un LLM).

### 10.4 Format de stockage

```jsonl
{"id": "human-wiki-001", "text": "...", "label": "human", "source": "wikipedia-fr",
 "meta": {"title": "...", "url": "...", "length_words": 234}}
{"id": "ai-luc23b-001", "text": "...", "label": "ai", "source": "luciole-23b-instruct",
 "meta": {"prompt": "...", "temperature": 0.7, "twin_of": "human-wiki-001"}}
{"id": "ai-mistral24b-001", "text": "...", "label": "ai", "source": "mistral-small-24b-openrouter",
 "meta": {"prompt": "...", "temperature": 0.7, "twin_of": "human-wiki-042", "corpus": "ood-v01"}}
```

Deux fichiers finaux :

Pattern général de nommage, valable pour tous les profils :
`binoculars-eu-corpus-<lang>-v<version>[-ood]`.

Deux fichiers finaux pour le profil FR V0.1 :

| Fichier local | Dataset HF public | Contenu |
|---------------|-------------------|---------|
| `calibration/corpus/binoculars-eu-corpus-fr-v01.jsonl` | `OpenLLM-France/binoculars-eu-corpus-fr-v01` | 500 textes in-distribution FR |
| `calibration/corpus/binoculars-eu-corpus-fr-v01-ood.jsonl` | `OpenLLM-France/binoculars-eu-corpus-fr-v01-ood` | 50 textes Mistral OOD FR |

Le SHA-256 de chaque fichier est inscrit dans `profiles/fr/metadata.json`
(champ `corpus_sha256`) et vérifié par `tests/test_profile_integrity.py`.
Les corpus des profils suivants suivront le même pattern :
`binoculars-eu-corpus-en-v01` (V1), `binoculars-eu-corpus-es-v01` (V3).

---

## 11. Plan d'exécution

### 11.1 Préparation infra OVH (J0, avant le POC)

```bash
# 1. Nettoyage disque préventif
df -h /
du -sh /home/*/.cache/huggingface 2>/dev/null
du -sh /home/*/.local/share 2>/dev/null
sudo journalctl --disk-usage
# Cible : libérer 15 Go min

# 2. Setup Python env (§8.2)
# 3. Configuration variables env (§8.3)
# 4. Pré-téléchargement modèles (§8.4)
# 5. Clone repo + branche (§8.5)
```

### 11.2 J1 — Fork technique + preuve de concept

**Objectifs** :
- Substituer les deux modèles Falcon par les deux Luciole dans `detector.py`.
- Adapter `assert_tokenizer_consistency` pour accepter la divergence tokens ajoutés.
- Test sur 20 textes (10 humains, 10 IA évidents).
- Vérifier que les scores humain > scores IA de façon monotone (au moins qualitativement).

**Livrable** : notebook `notebooks/01_first_scoring.ipynb` avec 20 exemples et scores.

**Critère de continuation** : si les 10 textes IA ont un score < 90ᵉ percentile des humains,
la méthode "prend" sur Luciole → passer à J2. Sinon, investiguer (tokenizer, précision, batch).

### 11.3 J2 — Construction du corpus (in-distribution + OOD léger)

**J2a in-distribution** (existant) :

- 250 textes humains collectés (scripts d'extraction Wikipedia + archives presse).
- 250 textes IA générés (250 prompts identiques envoyés aux 3 tailles Luciole).
- Sauvegarde en JSONL versionné.

**J2b OOD Mistral** (nouveau, même jour) :

- 50 textes IA générés via OpenRouter, modèle `mistralai/mistral-small-3.2-24b-instruct`.
- Prompts échantillonnés parmi les 250 sujets humains, mêmes températures et top_p que
  Luciole pour comparabilité.
- Fallback documenté vers endpoint Mistral-24B LINAGORA quand ce dernier sera de nouveau
  disponible : re-générer les 50 textes et publier une V0.1.1 si l'équivalence n'est
  pas parfaite.
- Clé OpenRouter gérée comme **credential custom** (voir §11.3.1).

**Livrables** :

- `calibration/corpus/binoculars-eu-corpus-fr-v01.jsonl` (500 lignes in-distribution).
- `calibration/corpus/binoculars-eu-corpus-fr-v01-ood.jsonl` (50 lignes OOD Mistral).

### 11.3.1 Gestion de la clé OpenRouter

La génération Mistral s'appuie sur OpenRouter. Deux modes supportés :

1. **Développement local** : clé stockée dans variable d'environnement
   `OPENROUTER_API_KEY` (fichier `.env` gitignoré).

2. **Reproductibilité par un tiers** : le script `calibration/generate_ood_mistral.py`
   accepte `--api-key` en argument ou lit `OPENROUTER_API_KEY`. Le corpus généré est
   ensuite publié sur HF Datasets, donc **un tiers qui reproduit binoculars-eu n'a pas
   besoin de régénérer le corpus** : il peut le télécharger et vérifier son SHA-256
   contre celui publié dans `calibration/protocol.md`.

Budget estimé : Mistral Small 24B sur OpenRouter coûte ~$0.15/M tokens output. 50 textes
× 500 tokens ≈ 25k tokens = **~$0.004**. Négligeable.

### 11.4 J3 — Calibration & courbe ROC

- Score de tous les 500 textes via `binoculars_eu.detector.compute_score`.
- Courbe ROC, choix des **trois** seuils (`accuracy`, `low-fpr`, `tpr-at-fpr-1`).
- Analyse des erreurs : top 20 faux positifs et top 20 faux négatifs annotés à la main.
- Écriture des valeurs finales dans **`binoculars_eu/profiles/fr/thresholds.json`** et de
  la traçabilité (SHA-256 du corpus, date, graine) dans
  **`binoculars_eu/profiles/fr/metadata.json`**. Aucun seuil n'est écrit en dur dans le
  code (§12, invariant 2).

**Livrable** : `calibration/results.json` + notebook d'analyse + profil `fr` peuplé.

### 11.5 J4-J5 — Packaging & API Python

- `pyproject.toml` complet pour `pip install binoculars-eu`, avec extras `[api]`,
  `[quant]`, `[dev]`.
- API publique en 3 lignes, profil FR par défaut :
  ```python
  from binoculars_eu import Binoculars
  detector = Binoculars(mode="accuracy")     # profil "fr" implicite
  detector.predict("Votre texte à analyser.")
  ```
- Vérification des trois méthodes d'instanciation (§13.1) : `for_language("fr")`,
  `Binoculars(profile=…)`, `from_legacy(…)` — cette dernière testée en non-régression
  contre les seuils Falcon upstream (`tests/test_from_legacy.py`).
- Tests du registre : auto-discovery, unicité des codes, défaut = `fr`
  (`tests/test_profiles.py`).
- README avec install, usage, benchmark headline, tableau des profils disponibles.
- Nœud LangGraph d'exemple pour ta pipeline twaky / dsh-mail-agent.
- Release sur PyPI (nom `binoculars-eu` réservé si dispo, sinon `binoculars_eu`).

**Livrable** : V0.1 installable, documentée, publiée.

### 11.5 bis J4 bis — Développement de l'API FastAPI (1 h)

Une heure de travail effectif, à intercaler dans J4 dès que l'API Python est stable.
L'essentiel du travail est déjà fait : l'API HTTP n'est qu'une façade sur
`detector.analyze()`.

| Étape | Contenu | Durée |
|-------|---------|-------|
| 1 | `binoculars_eu/schemas.py` — modèles Pydantic avec constraints (`min_length=50`, `pattern` sur `profile`, `Literal` sur `mode`) | 15 min |
| 2 | `binoculars_eu/api.py` — `POST /detect`, `GET /profiles`, `GET /health` + cache `@lru_cache(maxsize=4)` sur `(profile, mode)` | 25 min |
| 3 | `tests/test_api.py` — `TestClient` : 200 nominal, 404 profil inconnu, 422 texte trop court, cache réutilisé entre deux appels | 15 min |
| 4 | Relecture de la Swagger UI à `/docs` : descriptions, `examples`, tags | 5 min |

**Critère de sortie** : les trois appels curl de §9bis.2 passent, et `/docs` affiche les
trois routes avec des exemples exécutables via « Try it out ».

**Livrable** : `binoculars_eu/api.py`, `binoculars_eu/schemas.py`, `tests/test_api.py`,
`docs/api-http.md`.

### 11.5 ter J5 bis — Déploiement Hugging Face Space (fin de semaine 1)

Dernière tâche de la semaine 1 : rendre le détecteur essayable par quelqu'un qui
n'installe rien.

1. **Dockerfile de production** — multi-stage, `uvicorn` en entrypoint, cache HF monté en
   volume (§9bis.5). Test local `docker run --gpus all -p 8000:8000`.
2. **Space `spaces/hf-space/`** — front Gradio (`app.py`) devant la même application
   FastAPI, en-tête YAML de configuration dans le `README.md` du Space, hardware CPU
   basic (ou T4 si disponible ; noter la latence attendue dans les deux cas).
3. **Préchauffage au démarrage** du Space pour ne pas infliger 40 s de chargement au
   premier visiteur (§9bis.1).
4. **Avertissement en page d'accueil** : rappel des limites (§18.12) et de la règle « un
   verdict n'est pas une preuve ». Non négociable avant toute mise en ligne publique.
5. **Lien croisé** : Space → repo GitHub → datasets HF (`binoculars-eu-corpus-fr-v01`),
   pour que la chaîne de reproductibilité soit navigable en trois clics.

**Livrable** : Space public fonctionnel, image Docker publiée, `docs/api-http.md`
complété par la section déploiement.

### 11.6 Semaine 2 — V0.2 et benchmarks OOD

- Génération corpus OOD (25 textes Mistral + 25 GPT + 25 hybrides).
- Mesure de généralisation : AUC hors-Luciole.
- Comparaison chiffrée avec GPTZero, Originality, ZeroGPT sur le même corpus.
- Documentation méthodologique complète.

### 11.7 Semaine 3 — Communication

- Article de blog complet (blog.maudet.cloud + linagora.ai), incluant la section
  « pourquoi une plateforme de profils plutôt qu'un détecteur français ».
- Mise en avant de la démo Space déjà déployée en fin de semaine 1 (§11.5 ter) et de la
  Swagger UI publique.
- Appel à contributions pour les profils communautaires (§4.6), avec pointeur vers
  `CONTRIBUTING.md` et `docs/architecture-profiles.md`.
- Annonce coordonnée : LinkedIn, Reddit r/LocalLLaMA, contact presse tech FR
  (LeMonde Informatique, LeMagIT, GoodTech).
- Préparation d'une intervention pour la prochaine édition POSAIS.

---

## 12. Structure du repo cible

```
binoculars-eu/
├── README.md                          # Vitrine + install + usage 3 lignes + tableau des profils
├── pyproject.toml                     # Packaging PyPI (extras : [api], [quant], [dev])
├── LICENSE                            # Apache 2.0
├── CHANGELOG.md
├── CONTRIBUTING.md                    # Checklist d'acceptation d'un nouveau profil (§4.6)
├── Dockerfile                         # Image de production de l'API (§9bis.5)
├── docker-compose.yml                 # API + volume cache HF, pour dev local
├── .dockerignore
├── verify_environment.py              # Script agent §9
├── binoculars_eu/
│   ├── __init__.py                    # Expose Binoculars, LanguageProfile, get_profile, list_profiles
│   ├── detector.py                    # Moteur + for_language() / from_legacy() (§6.4)
│   ├── metrics.py                      # perplexity() + entropy() — REPRIS INCHANGÉ de l'upstream
│   ├── utils.py                        # assert_tokenizer_consistency + chemin tokenizer partagé (§6.5b)
│   ├── api.py                          # Application FastAPI : /detect, /profiles, /health (§13.2)
│   ├── schemas.py                      # Modèles Pydantic (DetectRequest, DetectResponse, ProfileInfo…)
│   └── profiles/
│       ├── __init__.py                 # Registre auto-discovery (pkgutil.iter_modules)
│       ├── base.py                     # dataclass LanguageProfile (frozen)
│       ├── fr/                         # V0.1 — profil par défaut
│       │   ├── __init__.py             # FR_PROFILE + register()
│       │   ├── thresholds.json
│       │   └── metadata.json
│       └── en/                         # V1 — opt-in (Falcon-7B + Falcon-7B-Instruct)
│           ├── __init__.py
│           ├── thresholds.json
│           └── metadata.json
├── calibration/
│   ├── build_corpus.py                 # Génération corpus in-distribution (500 textes FR)
│   ├── generate_ood_mistral.py         # Corpus OOD Mistral 24B via OpenRouter (§11.3.1)
│   ├── build_splits.py                 # Splits stratifiés (§18.1)
│   ├── calibrate.py                    # ROC + optimisation des 3 seuils → profiles/<lang>/thresholds.json
│   ├── evaluate.py                     # Évaluation complète §18, sortie JSON standardisée
│   ├── baselines.py                    # Baselines de comparaison (§18.4)
│   ├── reproduce_en_profile.py         # V1 : reproduction indépendante des seuils EN (§16.1)
│   ├── protocol.md                     # Version détaillée de §18 + seeds + hashes
│   ├── splits_v01.json
│   ├── results_v01.json
│   ├── corpus/
│   │   ├── binoculars-eu-corpus-fr-v01.jsonl        # 500 textes in-distribution FR
│   │   ├── binoculars-eu-corpus-fr-v01-ood.jsonl    # 50 textes Mistral OOD FR
│   │   └── binoculars-eu-corpus-en-v01.jsonl        # V1 : 500 textes EN
│   └── notebooks/
│       ├── 01_first_scoring.ipynb
│       ├── 02_corpus_analysis.ipynb
│       ├── 03_calibration.ipynb
│       └── 04_error_analysis.ipynb
├── benchmarks/
│   ├── ood_evaluation.py               # Test hors-distribution
│   ├── robustness.py                   # Tests d'invariance R-1 à R-6 (§18.7)
│   └── comparison_gptzero.py           # Comparaison si accès
├── examples/
│   ├── quickstart.py
│   ├── api_client.py                   # Client HTTP minimal (requests) + exemples curl
│   ├── langgraph_node.py               # Nœud pour pipelines agentiques (§13.3)
│   └── batch_scoring.py
├── spaces/
│   └── hf-space/                       # Démo Hugging Face Space (§11.5 bis)
│       ├── app.py                      # Front Gradio appelant l'app FastAPI
│       ├── requirements.txt
│       └── README.md                   # En-tête YAML de configuration du Space
├── docs/
│   ├── methodology.md                  # Explication accessible
│   ├── architecture-profiles.md        # LanguageProfile, registre, ajout d'un profil
│   ├── upstream-diff.md                # Les 3 différences vs ahans30/Binoculars (§6.5)
│   ├── api-http.md                     # Référence de l'API HTTP + déploiement
│   ├── limitations.md                  # Faux positifs connus, biais
│   ├── tokenizer_note.md               # Divergence Base/Instruct Luciole
│   ├── eval-card-fr.md                 # Evaluation card du profil FR (V0.1)
│   ├── eval-card-en.md                 # Evaluation card du profil EN (V1)
│   └── contributing.md
└── tests/
    ├── test_detector.py
    ├── test_profiles.py               # Registre : auto-discovery, unicité des codes, défaut = fr
    ├── test_profile_integrity.py      # thresholds.json/metadata.json valides, SHA-256 du corpus
    ├── test_from_legacy.py            # Non-régression upstream : seuils Falcon + labels anglais
    ├── test_api.py                     # TestClient FastAPI : 200 / 404 / 422, cache LRU
    ├── test_tokenizer_compat.py        # Test crucial : tokenizer parity
    └── test_thresholds.py
```

Trois invariants structurels garantis par les tests :

1. **`metrics.py` est identique à l'upstream** — vérifié par hash dans `test_detector.py`,
   ce qui rend les rebases sur `ahans30/Binoculars` sûrs (§6.5).
2. **Aucun seuil n'est écrit en dur dans `binoculars_eu/`** hors `from_legacy` — les
   seuils vivent exclusivement dans `profiles/<lang>/thresholds.json`.
3. **`profiles/<lang>/` ne contient jamais de code de scoring** — un profil est
   déclaratif ; il ne peut pas importer `detector.py` (import circulaire).

---

## 13. API cible détaillée

Trois surfaces d'API, du plus simple au plus intégré : Python (§13.1), HTTP FastAPI
(§13.2), nœud LangGraph (§13.3). Toutes trois partagent la même résolution de profil et
la même sémantique de verdict.

### 13.1 API Python

#### 13.1.1 Usage simple — le profil FR est le défaut

```python
from binoculars_eu import Binoculars

detector = Binoculars(mode="accuracy")        # profil "fr" par défaut (§4.0)

verdict = detector.predict(
    "Dans le paysage numérique en constante évolution, il est crucial de "
    "tirer parti des synergies pour naviguer dans un écosystème complexe."
)
# → "Probablement généré par IA"

verdict = detector.predict(
    "Je suis passé chez le boucher hier, il pleuvait des cordes, j'ai oublié "
    "mon parapluie chez ma sœur samedi."
)
# → "Probablement écrit par un humain"
```

Les libellés viennent de `profile.label_ai` / `profile.label_human` : le verdict est rendu
dans la langue du profil, pas en anglais comme upstream (§6.5c).

#### 13.1.2 Méthode 1 — `for_language()` : résolution par le registre

```python
from binoculars_eu import Binoculars, list_profiles

# Inventaire de ce qui est disponible dans la version installée
for p in list_profiles():
    print(p.code, p.display_name, p.observer_model, p.calibration_date)
# fr Français OpenLLM-France/Luciole-1B-Base 2026-09-15
# en English tiiuae/falcon-7b 2024-01-22          (à partir de la V1)

detector_fr = Binoculars.for_language("fr", mode="low-fpr")
detector_en = Binoculars.for_language("en", mode="accuracy")   # opt-in explicite, V1

Binoculars.for_language("de")
# KeyError: Profil inconnu : 'de'. Disponibles : ['en', 'fr']
```

C'est le chemin recommandé : le code appelant ne connaît que des codes ISO, jamais des
noms de modèles ni des seuils.

#### 13.1.3 Méthode 2 — `Binoculars(profile=…)` : profil custom

Utile pendant une calibration en cours, ou pour un profil privé non destiné au registre
public (corpus interne, modèles maison).

```python
from binoculars_eu import Binoculars, LanguageProfile

CUSTOM_PROFILE = LanguageProfile(
    code="fr-legal",
    display_name="Français juridique (interne)",
    observer_model="OpenLLM-France/Luciole-1B-Base",
    performer_model="OpenLLM-France/Luciole-1B-Instruct-1.1",
    threshold_accuracy=0.8711,
    threshold_low_fpr=0.8305,
    threshold_tpr_at_fpr_1=0.8299,
    corpus_sha256="3f1c…",
    corpus_url="s3://interne/corpus-juridique-v1.jsonl",
    calibration_date="2026-10-02",
    calibration_seed=42,
    share_tokenizer_from_observer=True,
    trust_remote_code=False,
    calibration_note="Corpus interne 300 textes, non publié. Ne pas citer.",
)

detector = Binoculars(profile=CUSTOM_PROFILE, mode="low-fpr")
```

Un profil custom n'est **pas** enregistré : il n'apparaît ni dans `list_profiles()` ni
dans `GET /profiles`. C'est volontaire — le registre ne documente que des profils
publiables et reproductibles.

#### 13.1.4 Méthode 3 — `from_legacy()` : compatibilité upstream

```python
from binoculars_eu import Binoculars

# Signature et comportement identiques à ahans30/Binoculars
detector = Binoculars.from_legacy(
    "tiiuae/falcon-7b",
    "tiiuae/falcon-7b-instruct",
    mode="low-fpr",
)

detector.predict("In the ever-evolving digital landscape, it is crucial to leverage…")
# → "Most likely AI-generated"          (labels anglais restitués)
detector.threshold
# → 0.8536432310785527                  (seuil Falcon de Hans et al.)
```

Trois usages : (a) reproduire les chiffres du papier original, (b) servir de baseline
externe dans §18.4, (c) construire le profil `en` de la V1 sans recalibrer d'abord
(§16.1). Note : `share_tokenizer_from_observer=False` sur ce chemin, donc l'assertion
stricte `assert_tokenizer_consistency` s'applique comme upstream.

#### 13.1.5 Usage avancé et introspection

```python
detector = Binoculars.for_language(
    "fr",
    mode="low-fpr",           # accuracy | low-fpr | tpr-at-fpr-1
    device="cuda:0",          # cuda:0, mps, cpu, auto
    batch_size=4,
    max_token_observed=512,
)

result = detector.analyze("Votre texte...")
# result = {
#     "score": 0.72,
#     "verdict": "ai",
#     "label": "Probablement généré par IA",
#     "confidence": "high",              # low, medium, high
#     "threshold_used": 0.8305,
#     "mode": "low-fpr",
#     "profile": "fr",
#     "input_tokens": 87,
# }

results = detector.analyze_batch(["texte 1", "texte 2", "texte 3"])

# Traçabilité : d'où viennent les seuils appliqués ?
p = detector.profile
print(p.corpus_url, p.corpus_sha256, p.calibration_date, p.calibration_seed)
print(p.calibration_note)     # None pour fr V0.1, renseigné pour en V1
```

`detector.profile` est un `frozen dataclass` : toute tentative de modifier un seuil à
chaud lève `FrozenInstanceError`. Un résultat produit est donc toujours rattachable à une
calibration publiée.

### 13.2 API HTTP FastAPI

Application dans `binoculars_eu/api.py`, schémas dans `binoculars_eu/schemas.py`.
Documentation Swagger UI auto-générée à `/docs`, ReDoc à `/redoc`, spec OpenAPI 3.1 à
`/openapi.json`. Démarrage et exemples curl complets en §9bis.

#### 13.2.1 Routes

| Méthode | Route | Rôle | Codes |
|---------|-------|------|-------|
| `POST` | `/detect` | Scoring d'un texte, avec `profile` optionnel (défaut `"fr"`) | 200, 404, 422, 503 |
| `GET` | `/profiles` | Liste des profils enregistrés et de leur traçabilité | 200 |
| `GET` | `/health` | Liveness + inventaire des profils et du cache | 200 |
| `GET` | `/docs` | Swagger UI (auto-généré) | 200 |

#### 13.2.2 Modèles Pydantic

```python
# binoculars_eu/schemas.py
from typing import Literal, Optional
from pydantic import BaseModel, Field

class DetectRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=50,
        max_length=20_000,
        description="Texte à analyser. En dessous de ~50 caractères le score "
                    "Binoculars n'est pas fiable (§18.12).",
        examples=["Dans le paysage numérique en constante évolution, il est "
                  "crucial de tirer parti des synergies."],
    )
    profile: str = Field(
        default="fr",
        min_length=2,
        max_length=8,
        pattern=r"^[a-z]{2}(-[a-z0-9]{2,5})?$",
        description="Code du profil de langue. Défaut : 'fr'. "
                    "Voir GET /profiles pour la liste disponible.",
    )
    mode: Literal["accuracy", "low-fpr", "tpr-at-fpr-1"] = Field(
        default="low-fpr",
        description="Seuil appliqué. 'low-fpr' minimise les faux positifs : "
                    "recommandé en contexte éducatif.",
    )

class DetectResponse(BaseModel):
    score: float = Field(..., description="Ratio PPL / X-PPL. Bas = probablement IA.")
    verdict: Literal["ai", "human"]
    label: str = Field(..., description="Libellé localisé issu du profil.")
    confidence: Literal["low", "medium", "high"]
    threshold_used: float
    mode: str
    profile: str
    input_tokens: int = Field(..., ge=1)
    elapsed_ms: int = Field(..., ge=0)

class ProfileInfo(BaseModel):
    code: str
    display_name: str
    observer_model: str
    performer_model: str
    thresholds: dict[str, float]
    corpus_url: str
    corpus_sha256: str
    calibration_date: str
    calibration_seed: int
    is_default: bool
    calibration_note: Optional[str] = None

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    default_profile: str
    profiles_loaded: list[str]
    detectors_cached: int
    device: str
```

Les `constraints` ne sont pas décoratifs : `min_length=50` bloque en 422 les textes trop
courts pour lesquels le score est structurellement bruité, et le `pattern` du champ
`profile` empêche qu'une chaîne arbitraire atteigne le registre.

#### 13.2.3 Application et cache LRU des détecteurs

```python
# binoculars_eu/api.py
import time
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from . import __version__
from .detector import Binoculars
from .profiles import get_profile, list_profiles, DEFAULT_PROFILE_CODE
from .schemas import DetectRequest, DetectResponse, ProfileInfo, HealthResponse

app = FastAPI(
    title="binoculars-eu",
    version=__version__,
    description="Détection zero-shot multilingue, plateforme européenne open source. "
                "Un profil de langue = une paire de modèles + des seuils calibrés.",
    license_info={"name": "Apache 2.0"},
)

@lru_cache(maxsize=4)
def get_detector(profile_code: str, mode: str) -> Binoculars:
    """Un détecteur par couple (profile, mode). Le cache évite de recharger
    plusieurs Go de poids à chaque requête. maxsize=4 borne l'empreinte VRAM
    (§6.9) ; l'éviction LRU est explicitée en V2."""
    return Binoculars.for_language(profile_code, mode=mode)

@app.post("/detect", response_model=DetectResponse, tags=["detection"])
def detect(req: DetectRequest) -> DetectResponse:
    try:
        get_profile(req.profile)                       # valide avant de charger
    except KeyError:
        available = [p.code for p in list_profiles()]
        raise HTTPException(
            status_code=404,
            detail=f"Profil inconnu : {req.profile!r}. Disponibles : {available}",
        )
    t0 = time.perf_counter()
    try:
        detector = get_detector(req.profile, req.mode)
    except Exception as exc:                            # OOM, poids indisponibles…
        raise HTTPException(status_code=503,
                            detail=f"Chargement du profil impossible : {exc}")
    result = detector.analyze(req.text)
    return DetectResponse(
        **result, elapsed_ms=int((time.perf_counter() - t0) * 1000)
    )

@app.get("/profiles", response_model=list[ProfileInfo], tags=["profiles"])
def profiles() -> list[ProfileInfo]:
    return [
        ProfileInfo(
            code=p.code, display_name=p.display_name,
            observer_model=p.observer_model, performer_model=p.performer_model,
            thresholds={"accuracy": p.threshold_accuracy,
                        "low_fpr": p.threshold_low_fpr,
                        "tpr_at_fpr_1": p.threshold_tpr_at_fpr_1},
            corpus_url=p.corpus_url, corpus_sha256=p.corpus_sha256,
            calibration_date=p.calibration_date, calibration_seed=p.calibration_seed,
            is_default=(p.code == DEFAULT_PROFILE_CODE),
            calibration_note=p.calibration_note,
        )
        for p in list_profiles()
    ]

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    import torch
    return HealthResponse(
        status="ok",
        version=__version__,
        default_profile=DEFAULT_PROFILE_CODE,
        profiles_loaded=[p.code for p in list_profiles()],
        detectors_cached=get_detector.cache_info().currsize,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
```

Le cache est indexé par `(profile, mode)` et non par `profile` seul parce que `mode`
change `self.threshold` sur l'instance : partager une instance entre deux modes
introduirait une condition de course sur le seuil. Coût : deux modes du même profil
peuvent occuper deux fois la VRAM. Alternative envisagée puis écartée — passer le seuil
en argument de `predict()` — car elle divergerait de l'API upstream (`change_mode`).

#### 13.2.4 Exemples curl et JSON

```bash
# POST /detect — profil implicite (fr)
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"text":"Dans le paysage numérique en constante évolution, il est crucial de tirer parti des synergies.","mode":"accuracy"}'
```

```json
{
  "score": 0.7213,
  "verdict": "ai",
  "label": "Probablement généré par IA",
  "confidence": "high",
  "threshold_used": 0.8402,
  "mode": "accuracy",
  "profile": "fr",
  "input_tokens": 34,
  "elapsed_ms": 118
}
```

```bash
# GET /profiles
curl -s http://localhost:8000/profiles
```

```json
[
  {
    "code": "fr",
    "display_name": "Français",
    "observer_model": "OpenLLM-France/Luciole-1B-Base",
    "performer_model": "OpenLLM-France/Luciole-1B-Instruct-1.1",
    "thresholds": {"accuracy": 0.8402, "low_fpr": 0.8305, "tpr_at_fpr_1": 0.8299},
    "corpus_url": "https://huggingface.co/datasets/OpenLLM-France/binoculars-eu-corpus-fr-v01",
    "corpus_sha256": "e3b0c44298fc1c149afbf4c8996fb924…",
    "calibration_date": "2026-09-15",
    "calibration_seed": 42,
    "is_default": true,
    "calibration_note": null
  }
]
```

```json
// GET /health
{
  "status": "ok",
  "version": "0.1.0",
  "default_profile": "fr",
  "profiles_loaded": ["fr"],
  "detectors_cached": 1,
  "device": "cuda:0"
}
```

```json
// POST /detect avec un profil inexistant → 404
{"detail": "Profil inconnu : 'de'. Disponibles : ['fr']"}
```

#### 13.2.5 Déploiement

| Cible | Contenu | Version |
|-------|---------|---------|
| **Hugging Face Space** | `spaces/hf-space/` — front Gradio devant la même app FastAPI, hardware CPU basic ou T4 selon disponibilité, sélecteur de profil désactivé tant qu'un seul profil existe | V0.1 (§11.5 bis) |
| **Docker** | `Dockerfile` multi-stage, `uvicorn` en entrypoint, cache HF monté en volume (§9bis.5) | V0.1 |
| **Endpoint LINAGORA** | même image, derrière rate-limiting et clé d'API | V2 (§4.4) |

Le Space est une **démo**, pas un SLA : pas de rate-limiting en V0.1, un seul worker, et
un avertissement explicite en page d'accueil sur les limites du détecteur (§18.12) — un
verdict ne doit jamais servir seul de preuve dans une procédure disciplinaire.

### 13.3 Nœud LangGraph

```python
# examples/langgraph_node.py
from langgraph.graph import StateGraph
from binoculars_eu import Binoculars

# Un détecteur par langue, instancié une fois au chargement du module.
# Les modèles restent en mémoire pour toute la durée de vie du graphe.
_DETECTORS = {
    "fr": Binoculars.for_language("fr", mode="accuracy"),
    # "en": Binoculars.for_language("en", mode="accuracy"),   # à partir de la V1
}

def detection_node(state):
    """Attend state["draft"] ; state["lang"] est optionnel (défaut : fr)."""
    lang = state.get("lang", "fr")
    detector = _DETECTORS.get(lang) or _DETECTORS["fr"]     # repli sur le défaut
    result = detector.analyze(state["draft"])
    return {
        **state,
        "ai_score": result["score"],
        "ai_verdict": result["verdict"],
        "detector_profile": result["profile"],       # traçabilité dans l'état du graphe
        "threshold_used": result["threshold_used"],
        "needs_humanization": result["verdict"] == "ai",
        "iterations": state.get("iterations", 0) + 1,
    }

# À intégrer dans une boucle generator → detector → humanizer → detector.
# Garde-fou recommandé : arrêter à 3 itérations pour éviter les boucles infinies
# quand le humanizer n'arrive pas à passer sous le seuil.
```

Variante HTTP, quand le graphe tourne ailleurs que sur la machine à GPU :

```python
import requests

def detection_node_http(state, base_url="http://binoculars-eu.internal:8000"):
    r = requests.post(f"{base_url}/detect", timeout=30, json={
        "text": state["draft"],
        "profile": state.get("lang", "fr"),
        "mode": "accuracy",
    })
    r.raise_for_status()
    result = r.json()
    return {**state,
            "ai_score": result["score"],
            "detector_profile": result["profile"],
            "needs_humanization": result["verdict"] == "ai"}
```

---

## 14. Métriques cibles et critères d'acceptation

### 14.1 V0.1 (POC) — profil `fr`

Ces cibles portent sur le **profil FR** de la V0.1 et sur lui seul. Chaque profil ajouté
ultérieurement définit ses propres cibles dans sa propre evaluation card ; aucune valeur
de ce tableau n'est transférable telle quelle à un autre profil.

| Métrique | Cible | Bloquant si |
|----------|-------|-------------|
| AUC ROC in-distribution (Luciole) | ≥ 0.80 | < 0.70 |
| TPR@FPR=1 % in-distribution (headline) | ≥ 0.45 | < 0.30 |
| TPR@FPR=5 % in-distribution | ≥ 0.65 | < 0.50 |
| F1 au seuil `accuracy` | ≥ 0.75 | < 0.65 |
| FPR au seuil `low-fpr` | ≤ 3% | > 5% |
| **AUC ROC OOD (Mistral Small 24B, 50 textes)** | ≥ 0.65 | < 0.55 |
| **Δ AUC (in-distribution − OOD)** | ≤ 0.25 | > 0.35 |
| Cohen's kappa inter-annotateurs (analyse d'erreurs) | ≥ 0.60 | < 0.40 |
| Latence par texte (L4, batch=1) | < 200 ms | > 1 s |
| Latence par texte (MBP M4 MPS) | < 800 ms | > 3 s |
| Empreinte VRAM à batch=1 | < 5 Go | > 8 Go |
| Test de reproductibilité (2 runs) | delta score < 1e-4 | delta > 1e-2 |
| **API HTTP** : `GET /health` répond 200 avec `profiles_loaded == ["fr"]` | oui | non |
| **API HTTP** : profil inconnu → 404, texte < 50 caractères → 422 | oui | non |
| **API HTTP** : surcoût du cache LRU à chaud (2ᵉ appel identique) | < 20 ms vs appel Python direct | > 200 ms |
| **Compatibilité upstream** : `from_legacy` reproduit les seuils Falcon au bit près | oui | non |

> **Note sur la prudence des cibles** : ce sont des cibles de **première itération**. Elles
> sont conçues pour être atteintes sans forçage, ni sur-optimisation d'hyperparamètres.
> Les contributions externes et les V0.2+ chercheront à les dépasser. Voir §18.2 pour
> la justification détaillée du choix TPR@FPR=1 % comme métrique headline.

### 14.2 V0.2

| Métrique | Cible |
|----------|-------|
| AUC ROC in-distribution (8B+8B int8) | ≥ V0.1 + 0.05 |
| AUC ROC OOD étendu (GPT-4o, Claude, hybrides) | ≥ 0.70 |
| Delta vs GPTZero sur corpus FR | positif |
| Latence P99 en Space HF (démo web) | < 3 s |

---

## 15. Risques et mitigations

| Risque | Probabilité | Impact | Mitigation |
|--------|-------------|--------|------------|
| Assertion tokenizer échoue | Moyenne | Bloquant | Patch documenté (§3.2) : forcer tokenizer Base |
| Signal Binoculars trop faible à 1B | Moyenne | Bloquant V0.1 | Fallback : tester paire 8B en int8 avec `bitsandbytes` |
| Disque OVH sature | Faible | Bloquant | Nettoyage préalable + monitoring `df` durant calibration |
| Service vLLM Qwen3-TTS impacté | Faible | Fort | Isolation cgroups (§7.2), batch conservateur |
| Généralisation OOD médiocre | Élevée | Modéré | Assumer : "détecteur spécialisé Luciole/FR", documenter la limite comme feature |
| Réception "outil de contournement" | Moyenne | Communication | Positionnement clair "outil pour récepteurs" (§4.3), note d'éthique |
| Divergence transformers > 4.57 casse Nemotron | Faible | Bloquant | Pinner `transformers==4.57.x` dans pyproject.toml |
| Corpus humain non représentatif du FR moderne | Élevée | Modéré | V0.2 avec corpus 2026 |
| Sur-ingénierie de l'architecture profils pour un seul profil livré | Moyenne | Modéré | Périmètre volontairement minimal : 1 dataclass + 1 registre + 3 fichiers par profil. Le coût réel est de quelques heures ; le coût d'une extraction rétroactive des seuils serait bien supérieur |
| Reproduction des seuils EN hors tolérance ±0.01 (V1) | Moyenne | Modéré | L'écart devient un résultat publié, pas un échec : bascule sur nos seuils + documentation de la cause (corpus, version `transformers`, précision) |
| VRAM insuffisante pour servir deux profils simultanément | Élevée (dès V1) | Modéré | Cache LRU borné à 4 détecteurs (§13.2.3), éviction explicite en V2, quantization int8 pour Falcon |
| Dérive vs upstream après plusieurs rebases | Moyenne | Modéré | `metrics.py` gardé identique et vérifié par hash, `from_legacy` testé en non-régression, CI de synchronisation en V2 (§4.4) |
| `trust_remote_code` accordé trop largement | Faible | Fort | Champ **par profil**, `False` par défaut ; liste blanche durcie en V2 |

---

## 16. Ouvertures : feuille de route multi-profils

La V0.1 livre une plateforme dont un seul profil est peuplé. Tout l'intérêt de
l'architecture §6 se vérifie — ou s'infirme — au moment d'ajouter le deuxième. Cette
section décrit dans l'ordre : le profil EN (V1), l'extension de capacité du profil FR
(V0.2), les profils suivants, puis les pistes transverses.

### 16.1 Profil EN en V1

Le profil `en` est le **test de l'architecture par un cas où nous ne maîtrisons pas la
calibration**. Les seuils du papier original existent, sont largement cités, et ont été
obtenus sur une paire que nous n'avons pas calibrée nous-mêmes. Deux tentations à écarter :
les ignorer et tout recalibrer (on perdrait la comparabilité avec la littérature), ou les
reprendre sans vérification (on publierait des chiffres non reproduits).

La réponse retenue est **les deux, dans cet ordre**.

**Étape 1 — bootstrap via `from_legacy`.** La paire est celle du papier : Falcon-7B
(observer) + Falcon-7B-Instruct (performer), bfloat16, tokenizer identique — l'assertion
stricte `assert_tokenizer_consistency` passe sans assouplissement, donc
`share_tokenizer_from_observer=False` pour ce profil.

```python
from binoculars_eu import Binoculars

# Chemin de compatibilité : seuils Hans et al. tels quels
detector = Binoculars.from_legacy("tiiuae/falcon-7b",
                                  "tiiuae/falcon-7b-instruct",
                                  mode="low-fpr")
```

Les valeurs injectées dans `profiles/en/thresholds.json` sont exactement celles des
constantes upstream de `detector.py` :

| Mode | Constante upstream | Valeur | Origine |
|------|--------------------|--------|---------|
| `accuracy` | `BINOCULARS_ACCURACY_THRESHOLD` | `0.9015310749276843` | Hans et al., optimisé F1 |
| `low-fpr` | `BINOCULARS_FPR_THRESHOLD` | `0.8536432310785527` | Hans et al., FPR ~0.01 % |
| `tpr-at-fpr-1` | — | à produire par notre reproduction | binoculars-eu |

Le troisième seuil **n'existe pas** chez Hans et al. : notre métrique headline étant
TPR@FPR=1 % (§18.2), ce seuil ne peut venir que de notre propre calibration. C'est déjà
une raison suffisante de faire l'étape 2.

`calibration_note` du profil EN documente cette provenance mixte, de sorte qu'un
utilisateur qui inspecte `GET /profiles` voit immédiatement quels chiffres sont repris et
quels chiffres sont à nous.

**Étape 2 — reproduction indépendante : `calibration/reproduce_en_profile.py`.**
Le script construit notre propre corpus EN et recalibre les trois seuils selon le
protocole §18, sans jamais lire les valeurs upstream avant la fin.

- Corpus : **500 textes** — 250 Wikipedia EN pré-2022 (pour écarter toute contamination
  par du texte généré) + 250 Falcon-7B-Instruct sur les **mêmes prompts** (titre +
  premier paragraphe), conservant la symétrie thématique du corpus FR (§10).
- Même protocole que FR : splits stratifiés étanches, bootstrap CI, les 5 baselines,
  les 4 ablations, les 6 tests de robustesse, kappa inter-annotateurs.
- **Critère de validation : ±0.01** entre chaque seuil reproduit et le seuil publié par
  Hans et al.
  - Écart ≤ 0.01 → reproduction validée. On conserve les valeurs upstream dans le profil
    (comparabilité avec la littérature) et on publie l'écart mesuré.
  - Écart > 0.01 → **les seuils du profil `en` basculent sur nos valeurs**, et l'écart
    est documenté comme un résultat en soi (différence de corpus, de version
    `transformers`, ou de matériel).
- Publication du corpus : `OpenLLM-France/binoculars-eu-corpus-en-v01` sur HF Datasets,
  SHA-256 inscrit dans `profiles/en/metadata.json`.
- Livrable documentaire : `docs/eval-card-en.md`, produite selon le même gabarit que
  l'evaluation card FR — une card par profil, sans exception.

Le protocole complet de cette reproduction est spécifié dans
`binoculars-eu-protocol.md`, section « Protocole spécifique au profil EN (V1) ».

**Ce que la V1 démontre, au-delà de l'anglais** : que le moteur est réellement
indépendant de la langue (mêmes `metrics.py`, mêmes chemins de code, seuls les profils
diffèrent), que `from_legacy` fonctionne comme rampe d'accès pour tout profil dont des
seuils publiés préexistent, et que notre protocole d'évaluation sait reproduire un
résultat externe — condition de crédibilité pour accepter ensuite des profils
communautaires.

**Contrainte matérielle à anticiper** : Falcon-7B ×2 en bf16 ≈ 30 Go de VRAM, hors de
portée du L4 (24 Go). Trois options, à trancher au démarrage de la V1 : int8
(`bitsandbytes`, comme la V0.2), machine plus large en location ponctuelle pour la seule
phase de calibration, ou exécution CPU lente mais suffisante sur 500 textes. Le profil
EN publié doit préciser la précision utilisée, car elle change les seuils.

### 16.2 Extension de capacité du profil FR en V0.2 — paire 8B+8B en int8

Une fois le POC V0.1 validé sur 1B+1B, la piste principale de gain de signal **sur le
profil FR** est la paire Luciole-8B-Base + Luciole-8B-Instruct-1.1 en int8. Il s'agit
d'une variante de capacité du même profil, pas d'un nouveau profil.

**Faisabilité VRAM** :

| Précision | VRAM par modèle | Total paire | Faisabilité L4 (24 Go) |
|-----------|-----------------|-------------|-------------------------|
| bfloat16 | ~15 Go | ~30 Go | ❌ dépasse même seul |
| **int8** (`bitsandbytes`) | ~8 Go | ~16 Go | ✅ tient avec marge |
| int4 | ~4.5 Go | ~9 Go | ✅ largement, mais signal dégradé |

La piste FP8 native serait plus propre que int8, mais LINAGORA n'a publié que la version
FP8 du **Instruct** (`Luciole-8B-Instruct-1.1-FP8`), pas du Base. La paire est déséquilibrée →
à écarter tant que le Base FP8 n'est pas publié (potentiellement à demander à l'équipe
OpenLLM-France).

**Défis techniques V0.2** :

1. **Architecture `NemotronHForCausalLM`** — hybride Mamba (~42 couches) / Attention
   (~4 couches) / MLP (~6 couches) selon le motif
   `M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-`. Binoculars ne dépend que
   des logits de sortie, donc la formule reste identique — mais c'est la **première fois**
   que la méthode est appliquée à un modèle non purement Transformer. À valider
   empiriquement (question de recherche à documenter).

2. **`trust_remote_code=True` requis** — le code de modélisation `modeling_nemotron_h.py`
   est packagé dans le repo HF et exécuté depuis la machine. C'est précisément pourquoi
   `trust_remote_code` est un **champ de `LanguageProfile`** et non un réglage global :
   l'autorisation est accordée profil par profil, et vaut `False` par défaut.

3. **Dépendances `mamba-ssm` et `causal-conv1d`** — kernels CUDA compilés, parfois
   fragiles sur des drivers récents. À vérifier avant tout téléchargement modèle.

4. **Double effet quantization × architecture** — impossible de démêler si un signal
   plus faible vient de int8 ou de l'hybride Mamba. D'où l'ordre méthodologique :
   d'abord baseline propre en 1B+1B bf16 (Transformer pur), puis extension en
   8B+8B int8 en la comparant à cette baseline.

**Critères d'acceptation V0.2** :

- AUC ROC in-distribution ≥ AUC V0.1 + 0.05 (gain net attendu par la montée d'échelle).
- Si dégradation observée → conserver 1B+1B comme cible officielle du profil `fr`,
  documenter 8B+8B comme piste exploratoire.

### 16.3 Profil ES en V3, puis profils communautaires

**V3 — `es`.** Candidat principal : la famille **Salamandra** (BSC-LT), sous réserve de
vérifier trois points avant tout engagement : (a) existence d'une paire Base / Instruct
publiée, (b) tokenizer commun aux deux — ou divergence limitée à des tokens spéciaux, cas
que `share_tokenizer_from_observer` couvre, (c) licence compatible Apache 2.0.
Alternative à évaluer si Salamandra ne convient pas : toute paire ouverte
hispanophone équivalente. **Le choix définitif est un livrable de la V2.**

**V4+ — `de`, `it`, `pt`, `pl`.** Portés par les communautés linguistiques concernées, le
projet fournissant le moteur, le protocole et la revue. Un profil n'est fusionné que s'il
apporte : dossier `profiles/<lang>/` conforme (§6.7), corpus public sur HF Datasets avec
SHA-256, les trois seuils calibrés selon §18, une evaluation card, et une PR passant
`test_profile_integrity.py`. `CONTRIBUTING.md` porte cette checklist.

Un refus explicite : **pas de profil sans corpus public**. Un seuil non reproductible est
un seuil non défendable, et la valeur du projet réside entièrement dans sa
reproductibilité.

### 16.4 Autres pistes transverses

1. **Auto-détection de langue** — router automatiquement vers le bon profil (fastText ou
   `langdetect`) plutôt que d'exiger le champ `profile`. À traiter avec précaution : une
   erreur de détection de langue produit un verdict calibré sur la mauvaise distribution.
   Doit rester opt-in et signalé dans la réponse.
2. **Extension multi-tailles orchestrée** — 1B pour l'edge (MacBook, mobile), 8B pour le
   serveur, avec fallback automatique selon les ressources détectées.
3. **Détection ciblée par générateur** — un score plus haut si le texte ressemble
   spécifiquement à Luciole vs Mistral vs GPT (via un classifieur secondaire léger).
4. **Détecteur adversarial** — mesure de robustesse face aux humanizers (dont notre
   propre `avoid-ai-writing-multilingual`).
5. **Comparaison inter-profils** — étudier si un texte français scoré par le profil `en`
   (et réciproquement) produit une dégradation mesurable et publiable. C'est la
   justification empirique de l'existence des profils.
6. **Intégration Twake / Matrix** — modération automatique de messages générés.
7. **Publication scientifique** — atelier NLP FR (JEP-TALN 2027, ou workshop
   AI-detection à COLING/EMNLP), avec l'angle « une plateforme multilingue de détection
   zero-shot et son protocole de reproduction ».

---

## 18. Protocole d'évaluation (niveau publication scientifique)

Cette section formalise le protocole d'évaluation. Elle est indépendante du plan
calendaire (§11) : le protocole ne fait pas que rapporter les métriques — il **définit
les conditions dans lesquelles les chiffres publiés sont défendables**.

Référence méthodologique : le protocole reprend les standards du domaine AI-detection
établis par [RAID (Dugan et al., ACL 2024)](https://arxiv.org/abs/2405.07940),
[OpenTuringBench (2025)](https://arxiv.org/abs/2504.11369) et
[le benchmark NAACL 2025 sur les détecteurs pratiques](https://aclanthology.org/2025.findings-naacl.271/).
Il ajoute deux éléments absents de ces publications : (a) des intervalles de confiance
bootstrap sur toutes les métriques, (b) une vérification de reproductibilité par graines
aléatoires multiples.

### 18.1 Splits stratifiés et étanchéité

Sans split étanche, calibrer les seuils sur les 500 textes puis rapporter les métriques
sur les mêmes 500 textes = overfitting garanti. Le protocole impose :

| Split | Proportion | Rôle | Interdit |
|-------|------------|------|----------|
| `train` | 60 % (300 textes) | Calibration des seuils par ROC | Regardé pour reporting |
| `dev` | 20 % (100 textes) | Sélection du mode (`accuracy` vs `low-fpr`) | Utilisé pour publier |
| `test` | 20 % (100 textes) | **Métriques publiées** | Jamais vu pendant calibration |

**Stratification obligatoire sur 4 axes** (chaque split contient tous les strates au
prorata) :

1. **Label** : 50/50 humain/IA dans chaque split.
2. **Source** (humain) : Wikipedia, presse, blog tech, LinkedIn, littérature — chaque
   source représentée dans les 3 splits.
3. **Générateur** (IA) : Luciole-1B, 8B, 23B — chaque taille représentée dans les 3
   splits.
4. **Longueur** : bins `[0-150, 150-300, 300-500, 500+]` tokens.

**Implémentation** : `sklearn.model_selection.StratifiedKFold` avec `seed=42` (figé), le
split est publié comme fichier JSON versionné dans le repo
(`calibration/splits_v01.json`) pour reproductibilité totale.

**K-fold pour la V0.1** : étant donnée la taille modeste (500 textes), un test set de
100 textes donne des IC larges. On rapporte donc en complément un **5-fold
stratifié** sur l'ensemble des 500 : moyenne ± écart-type des métriques sur les 5 folds.
Les deux chiffres (test hold-out et 5-fold) doivent être cohérents.

### 18.2 Métriques

**Métriques primaires** (rapportées systematiquement, avec IC 95 % bootstrap) :

| Métrique | Définition | Cible V0.1 |
|----------|------------|------------|
| **AUC ROC** | Area under ROC curve | ≥ 0.80 |
| **TPR@FPR=1 %** | True positive rate au seuil où FPR = 1 % | ≥ 0.45 |
| **TPR@FPR=5 %** | True positive rate au seuil où FPR = 5 % | ≥ 0.65 |
| **F1 @ threshold `accuracy`** | F1 au seuil optimal accuracy | ≥ 0.75 |
| **Accuracy @ threshold `accuracy`** | Accuracy au seuil optimal | ≥ 0.75 |

> **Positionnement des cibles** : ces seuils sont volontairement **prudents** pour la
> première itération. binoculars-eu V0.1 est un banc d'essai reproductible, pas une
> vitrine de performance. Les contributions extérieures et les itérations V0.2+ chercheront
> à les améliorer. Ce positionnement "prudent + reproductible" est un choix stratégique
> : mieux vaut annoncer 0.45 mesuré avec IC 95 % que 0.60 sans intervalle publié.

Justification du choix **TPR@FPR=1 %** comme métrique headline : c'est le standard adopté
par RAID et NAACL 2025 pour évaluer les détecteurs en usage réel, où les faux positifs
(accuser un humain d'être une IA) sont beaucoup plus coûteux que les faux négatifs.

**Métriques secondaires** (rapportées pour analyse) :

- Matrice de confusion stratifiée (par source humaine, par générateur IA, par bin de
  longueur).
- Distribution des scores : histogramme humain vs IA, courbe de décision.
- **Expected Calibration Error (ECE)** : à quel point les scores reflètent-ils
  correctement une probabilité.
- Latence P50 et P99 par texte (par bin de longueur).

### 18.3 Intervalles de confiance bootstrap

**Toutes** les métriques primaires sont rapportées avec IC 95 %. Procédure :

1. Rééchantillonner le test set avec remise, 1000 fois (seed figée).
2. Recalculer chaque métrique sur chaque rééchantillon.
3. Rapporter la métrique observée sur le test set réel + IC 95 % (percentiles 2.5 et 97.5
   des 1000 rééchantillons).

Format de reporting :

```
AUC ROC        = 0.872  (IC 95 %: [0.821, 0.918])
TPR@FPR=1 %    = 0.612  (IC 95 %: [0.489, 0.720])
F1             = 0.833  (IC 95 %: [0.782, 0.876])
```

Sans IC, un chiffre nu sur 100 textes est trompeur. **C'est un des deux éléments qui
place ce protocole au-dessus des standards actuels du domaine** (ni RAID, ni NAACL 2025,
ni OpenTuringBench ne rapportent d'IC bootstrap systématiques).

### 18.4 Baselines de comparaison

Un AUC de 0.85 ne veut rien dire seul — il faut savoir à quoi le comparer. Cinq
baselines minimales, toutes évaluées sur **le même test set** (100 textes hold-out) :

| Baseline | Type | Rôle | Implémentation |
|----------|------|------|----------------|
| **Random** | Contrôle | AUC attendu ≈ 0.50 | `numpy.random.rand()` |
| **Longueur** | Naive feature | Test heuristique triviale | `LogisticRegression` sur longueur en tokens |
| **Features shallow** | Naive features | Baseline lexical simple | LR sur : longueur, taux virgules, TTR, ratio ponct. |
| **Binoculars Falcon EN original** | Cross-lingual | Confirme la nécessité d'un profil FR dédié | `Binoculars.from_legacy("tiiuae/falcon-7b", "tiiuae/falcon-7b-instruct")` — seuils Hans et al. inchangés (§13.1.4) |
| **Profil `fr` binoculars-eu (nous)** | Cible | Notre méthode | `Binoculars.for_language("fr")`, §6 du PRD |

**Baselines optionnelles V0.2** (si accès) :

- GPTZero API (payant, limite quotas).
- Originality.ai API.
- Fast-DetectGPT (Bao et al., ICLR 2024) sur Luciole-1B-Base.

Toute baseline optionnelle absente est explicitement documentée comme telle.

### 18.5 Tests d'ablation

Quatre ablations obligatoires pour comprendre ce qui contribue au signal. Chacune est
évaluée sur le **dev set** (jamais le test set) pour éviter le p-hacking.

| Variable | Valeurs testées | Question répondue |
|----------|-----------------|-------------------|
| `max_length` | 64, 128, 256, 512, 1024 | Combien de tokens sont vraiment nécessaires ? |
| Précision | bfloat16, float16, float32 | La quantization bf16 altère-t-elle le signal ? |
| Paire de modèles | Base+Instruct, Instruct+Instruct, Base+SFT | Le choix de la paire est-il critique ? |
| Tokenizer partagé | Base only, Instruct only | Le choix du tokenizer partagé impacte-t-il le score ? |

Résultat attendu : une **table d'ablation** publiée (Métrique × Configuration).

### 18.6 Analyse d'erreurs structurée

Après calibration, taxonomie a priori des erreurs :

**Taxonomie des faux positifs** (humain classé IA) :

- FP-1 : Texte administratif ou juridique (registre formel proche du style IA).
- FP-2 : Texte très court (< 100 tokens — signal trop faible).
- FP-3 : Traduction automatique post-éditée (biais connu).
- FP-4 : Texte avec beaucoup de code, chiffres, tableaux.
- FP-5 : Texte technique très standardisé (RFC, spéc, mode d'emploi).
- FP-autre : à caractériser.

**Taxonomie des faux négatifs** (IA classée humain) :

- FN-1 : IA générée avec température élevée (t > 0.9).
- FN-2 : IA post-éditée par un humain (même légèrement).
- FN-3 : IA imite un style très marqué (dialecte, argot, littéraire).
- FN-4 : IA hors-distribution de Luciole (Mistral, GPT-4, Claude, Gemini).
- FN-autre : à caractériser.

**Livrable** : un **notebook d'annotation interactif**
(`notebooks/04_error_analysis.ipynb`) présente les 20 pires FP et 20 pires FN avec
:

- Le texte formaté (paragraphes visibles).
- Le score et le seuil.
- Un widget de cases à cocher pour la catégorie de la taxonomie.
- Une zone de notes libres.

**Double annotation obligatoire pour la V0.1** : les 20 FP et 20 FN sont annotés
**par deux annotateurs indépendants** (Michel-Marie + un second annotateur à identifier
dans l'équipe LINAGORA / OpenLLM France). L'accord inter-annotateurs est mesuré par
**Cohen's kappa** et publié dans le rapport d'évaluation.

- κ ≥ 0.80 : accord excellent, la taxonomie est bien opérationnalisée.
- 0.60 ≤ κ < 0.80 : accord substantiel, les désaccords sont résolus par discussion.
- κ < 0.60 : accord faible → réviser la taxonomie avant publication.

En cas de désaccord entre annotateurs, l'annotation finale retenue est le consensus post-
discussion, avec traces des positions initiales conservées dans
`docs/error_analysis_v01_annotations.json`.

L'annotation manuelle produit un tableau de contingence Catégorie × Compte (une par
annotateur + consensus), publié dans le rapport d'évaluation.

### 18.7 Tests de robustesse et invariances

Six tests adversariaux minimaux, chacun mesure le Δ AUC entre corpus original et corpus
perturbé. Une baisse d'AUC ≥ 0.15 sur un test signale une fragilité à documenter.

| Test | Perturbation | Sensibilité attendue |
|------|--------------|---------------------|
| **R-1** : Fautes de frappe | 5 % de caractères aléatoirement inversés | Faible — détecteur doit résister |
| **R-2** : Paraphrase légère | Reformulation manuelle mineure (10 % des phrases) | Moyenne — texte reste très IA en surface |
| **R-3** : Troncature | Ne garder que les 100 premiers tokens | Forte — signal réduit |
| **R-4** : Concaténation | Alternance phrase humaine / phrase IA | Très forte — le score doit tomber vers 0.5 |
| **R-5** : Adversarial prompting | IA générée avec instruction "écris comme un humain" | Moyenne à forte |
| **R-6** : Adversarial rewriting | Humain réécrit par LLM "clarifie et professionnalise" | Forte — test critique (RAID 2024) |

Les tests R-5 et R-6 reprennent explicitement les protocoles [RAID](https://arxiv.org/abs/2405.07940)
et [NAACL 2025 findings](https://aclanthology.org/2025.findings-naacl.271/).

**Livrable** : `benchmarks/robustness.py` qui génère les 6 variantes du test set et
produit un tableau détecteur × test avec Δ AUC + IC 95 %.

### 18.8 Reproductibilité

Critère non-négociable pour publication : le protocole entier doit être reproductible
par un tiers avec un simple `git clone && make evaluate`.

**Garanties** :

1. **Seeds figées et publiées** dans `calibration/protocol.md` : split (42),
   bootstrap (100), corpus généré par IA (0, 1, 2 pour chaque taille).
2. **Corpus versionné** sur HuggingFace Datasets avec hash SHA-256 publié.
3. **Pins de versions** : `requirements.txt` avec versions exactes,
   `transformers==4.57.1`, `torch==2.5.1`, `bitsandbytes==0.44.0`.
4. **Container Docker** : `Dockerfile` fournissant l'environnement exact.
5. **Test de reproductibilité** : deux runs consécutifs avec la même seed doivent
   produire des métriques à δ < 1e-4.
6. **Test de robustesse par seed** : 3 runs avec seeds différentes (42, 123, 2024)
   — la variance inter-seed est publiée comme mesure de stabilité.

### 18.9 Livrables du protocole

| Fichier | Contenu | Statut |
|---------|---------|--------|
| `calibration/protocol.md` | Version détaillée de §18 + seeds + hashes | à créer |
| `calibration/splits_v01.json` | Splits train/dev/test figés | généré par `build_splits.py` |
| `calibration/evaluate.py` | Script d'évaluation complet, sortie JSON standardisée | à créer |
| `calibration/build_splits.py` | Création des splits stratifiés | à créer |
| `calibration/baselines.py` | Implémentations des baselines (§18.4) | à créer |
| `benchmarks/robustness.py` | Tests d'invariance R-1 à R-6 | à créer |
| `notebooks/04_error_analysis.ipynb` | Annotation interactive des erreurs | à créer |
| `docs/eval-card-fr.md` | Fiche d'évaluation du profil FR, à la HF Model Card (une card par profil) | à créer |
| `calibration/results_v01.json` | Résultats finaux (métriques + IC) | généré |
| `binoculars_eu/profiles/fr/thresholds.json` | Les trois seuils calibrés du profil FR | généré par `calibrate.py` |
| `binoculars_eu/profiles/fr/metadata.json` | SHA-256 du corpus, date, graine de calibration | généré par `evaluate.py` |

### 18.10 Format de rapport final

Rapport d'évaluation publié dans le repo (`docs/evaluation_report_fr_v01.md` — un rapport
par profil, le suffixe portant le code de langue) avec la structure :

1. **Résumé exec** : métrique headline (TPR@FPR=1 %) + IC.
2. **Table de métriques** : primaires × (test hold-out, 5-fold, dev).
3. **Table de baselines** : notre méthode vs 5 baselines minimales.
4. **Table d'ablation** : 4 ablations obligatoires.
5. **Table de robustesse** : 6 tests d'invariance avec Δ AUC.
6. **Analyse d'erreurs** : contingence par catégorie + exemples représentatifs.
7. **Reproductibilité** : seeds, versions, hashes, temps de calcul par étape.
8. **Limites** : ce que le protocole ne mesure PAS (biais démographiques, biais
   diachroniques, biais de domaine hors corpus).

### 18.11 Intégration au planning

Ce protocole ajoute un jour au plan (§11) :

- J2 (existant) : construction corpus → ajouter création splits stratifiés.
- **J3a nouveau : Implémentation baselines + évaluation baselines seules**.
  Cette étape doit précéder la calibration Binoculars, pour ancrer la comparaison.
- J3 (existant) : calibration Binoculars → sur `train` uniquement.
- **J3b nouveau : évaluation complète sur test + robustesse + ablations**.
- **J3c nouveau : analyse d'erreurs annotée manuellement** (temps : 3-4 h).
- J4-J5 (existant) : packaging inchangé.
- J4 bis / J5 bis (nouveaux, §11.5 bis et §11.5 ter) : API FastAPI et Space HF. Sans
  impact sur le protocole — l'API consomme des seuils déjà calibrés, elle n'en produit
  aucun.

Semaine 2 : la V0.2 hérite du même protocole, appliqué en plus au corpus OOD.

**Portée multi-profils** : ce protocole s'applique **intégralement et indépendamment à
chaque profil de langue**. Aucun résultat, aucun seuil, aucun intervalle de confiance
n'est transférable d'un profil à l'autre — un profil sans son propre passage complet du
protocole n'est pas publiable (§4.6). La version détaillée, y compris le protocole
spécifique au profil EN de la V1, vit dans `binoculars-eu-protocol.md`.

### 18.12 Ce que le protocole ne peut PAS résoudre

Honneteté scientifique : documenter les limites structurelles du protocole lui-même.

- **Biais de corpus** : si notre corpus humain est majoritairement Wikipedia + presse,
  la méthode sera meilleure sur ces registres et faible sur d'autres (SMS, oral
  transcrit, dialecte).
- **Biais diachronique** : le corpus 2026 vieillira. Un texte humain de 2028 pourrait
  être classé IA parce qu'il aura absorbé des tournures IA (pollution progressive).
- **Biais démographique** : nous n'annotons pas l'origine sociolinguistique des humains.
  Une population de locuteurs non-natifs pourrait être systématiquement classée IA.
- **Biais adversarial** : les tests R-1 à R-6 ne couvrent pas les attaques
  co-évolutives (humanizers dédiés à tromper binoculars-eu spécifiquement).
- **Biais inter-profils** : le protocole évalue chaque profil isolément. Il ne dit rien
  du comportement d'un profil sur du texte dans une autre langue (un texte anglais scoré
  par le profil `fr`), cas qui se produira en production dès que le champ `profile` sera
  mal renseigné. Mesure prévue en V1, quand deux profils coexisteront (§16.4, piste 5).

Ces limites sont à rapporter dans `docs/limitations.md` avec pistes de mitigation V2.

---

## 17. Annexes

### 17.1 Références

- Papier Binoculars original : [Hans et al., 2024](https://arxiv.org/abs/2401.12070)
- Code source Binoculars (upstream du fork) : [ahans30/Binoculars](https://github.com/ahans30/Binoculars)
  — `detector.py` (seuils Falcon en constantes de module), `metrics.py` (`perplexity()`,
  `entropy()`, repris inchangés), `utils.py` (`assert_tokenizer_consistency`, assoupli via
  `share_tokenizer_from_observer`)
- Documents jumeaux de cette spécification : `binoculars-eu-protocol.md` (protocole
  d'évaluation, un passage complet par profil) et `binoculars-eu-eval-card.md`
  (evaluation card du profil FR V0.1)
- Modèles Luciole : [OpenLLM-France sur Hugging Face](https://huggingface.co/OpenLLM-France)
- Dataset d'entraînement Luciole : [Luciole-Training-Dataset](https://huggingface.co/datasets/OpenLLM-France/Luciole-Training-Dataset)
- Papier de comparaison humain vs détecteurs : [Russell et al., 2024](https://github.com/jenna-russell/human_detectors)
- Skill FR pour humanisation : [avoid-ai-writing-multilingual](https://github.com/conorbronsdon/avoid-ai-writing) (SKILL-FR.md)

### 17.2 Contacts et écosystème

- Consortium OpenLLM France (Jean-Pierre Lorre, LINAGORA R&D)
- Programme France 2030 / BPI France (contexte de financement Luciole)
- Communauté POSAIS pour communication et validation externe
