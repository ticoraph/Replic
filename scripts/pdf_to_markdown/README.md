# scripts

Scripts utilitaires indépendants du frontend et du backend, avec leurs propres dépendances Python.

## pdf_to_markdown.py

Convertit un fichier PDF en Markdown avec [pymupdf4llm](https://github.com/pymupdf/RAG) et sauvegarde le résultat dans `data/raw/`, avec le même nom que le fichier source.

### Setup

```bash
python -m venv scripts/venv && source scripts/venv/bin/activate
pip install -r scripts/pdf_to_markdown/requirements.txt
```

### Usage

```bash
python scripts/pdf_to_markdown/pdf_to_markdown.py data/raw/4.2-pulp_fiction.pdf
# -> data/raw/4.2-pulp_fiction.md
```
