# Données (non versionnées)

Ce dossier est volontairement vide dans le repo public : le corpus RAG et les datasets de
fine-tuning contiennent ou dérivent d'œuvres protégées par le droit d'auteur (scénario du
film, articles de presse et de blogs). Fournissez votre propre corpus :

| Dossier | Contenu attendu | Produit par |
| --- | --- | --- |
| `data/raw/*.md` | Documents d'analyse en français, un fichier markdown par source (chunkés par titres) | à fournir |
| `data/raw/script-pulp-fiction-original-en.md` | Script anglais, **jamais indexé**, source de `parse_screenplay.py` | à fournir |
| `data/interim/dialogues.jsonl`, `dialogues_fr.jsonl` | Dialogues parsés puis traduits | `src/preprocessing/` |
| `data/processed/pairs_*.jsonl` | Paires Q/R (script + synthétiques) | `src/datasets/` |
| `datasets/{train,val}.jsonl` | Dataset de fine-tuning | `src/datasets/split.py` |

Voir le pipeline dans le [README](../README.md).
