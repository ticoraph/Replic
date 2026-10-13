#!/usr/bin/env python3
"""Convert a PDF file to Markdown and save it alongside the source in data/raw/.

Setup:
    python -m venv scripts/venv && source scripts/venv/bin/activate
    pip install -r scripts/pdf_to_markdown/requirements.txt

Usage:
    python scripts/pdf_to_markdown/pdf_to_markdown.py <path/to/file.pdf>
"""
import sys
from pathlib import Path

import pymupdf4llm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "data" / "raw"


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/pdf_to_markdown/pdf_to_markdown.py <path/to/file.pdf>", file=sys.stderr)
        sys.exit(1)

    pdf_path = Path(sys.argv[1]).resolve()
    if not pdf_path.is_file():
        print(f"File not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    markdown = pymupdf4llm.to_markdown(str(pdf_path))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{pdf_path.stem}.md"
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Saved markdown to {output_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
