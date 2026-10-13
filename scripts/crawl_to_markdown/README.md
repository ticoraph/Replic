# scripts

Scripts utilitaires indépendants du frontend et du backend, avec leurs propres dépendances Python.

## crawl_to_markdown.py

Crawle une page web avec [crawl4ai](https://github.com/unclecode/crawl4ai) et sauvegarde son contenu principal (nettoyé du menu/nav) au format Markdown dans `data/processed/`.

### Setup

```bash
python -m venv scripts/venv && source scripts/venv/bin/activate
pip install -r scripts/requirements.txt
crawl4ai-setup
```

### Usage

```bash
python scripts/crawl_to_markdown.py <url>
```

Le fichier généré est nommé d'après le dernier segment du chemin de l'URL (slug), par exemple :

```bash
python scripts/crawl_to_markdown.py "https://fr.storyanddrama.com/analyse-des-personnages-de-pulp-fiction-de-quentin-tarantino-personnages-et-roles-thematiques/"
# -> data/processed/analyse-des-personnages-de-pulp-fiction-de-quentin-tarantino-personnages-et-roles-thematiques.md
```
