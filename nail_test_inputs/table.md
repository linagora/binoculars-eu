| Texte | Origine | Longueur | Score accuracy | Verdict accuracy | Score low-fpr | Verdict low-fpr |
|---|---|---|---|---|---|---|
| Luciole 8B + Undetectable AI | IA + Undetectable AI | 730 mots / 4834 caractères | 0.9833 (marge +0.028) | 👤 Humain (medium) | 0.9833 (marge +0.117) | 👤 Humain (high) |
| Luciole 8B raw (seed 43) | IA brute | 578 mots / 3940 caractères | 0.8245 (marge -0.131) | 🤖 IA (high) | 0.8245 (marge -0.042) | 🤖 IA (medium) |

### Configuration du test

- Date et heure du test (Europe/Paris) : 2026-08-31 23:10:35 CEST
- Détecteur : binoculars-eu 0.1.0, profil fr, git_sha n/a
- Seuils actifs : accuracy = 0.955801, low-fpr = 0.866667
- Modèles : `OpenLLM-France/Luciole-1B-Base + OpenLLM-France/Luciole-1B-SFT-1.0`
- Graine consignée (metadata, non utilisée côté détecteur) : 42

### Fichiers audit

- JSON : `nail_test_inputs/audit.json`
- CSV : `nail_test_inputs/audit.csv`
