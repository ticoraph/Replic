#!/usr/bin/env python3
"""Crawl a web page with crawl4ai and save its content as Markdown in data/processed/.

Setup:
    python -m venv scripts/venv && source scripts/venv/bin/activate
    pip install -r scripts/requirements.txt
    crawl4ai-setup

Usage:
    python scripts/crawl_to_markdown.py <url>
"""
import asyncio
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "data" / "raw"


def slug_from_url(url: str) -> str:
    parsed = urlparse(url)
    segment = parsed.path.strip("/").rsplit("/", 1)[-1] or parsed.netloc
    slug = re.sub(r"[^a-z0-9]+", "-", segment.lower()).strip("-")
    return slug or "page"


async def crawl(url: str) -> str:
    config = CrawlerRunConfig(
        markdown_generator=DefaultMarkdownGenerator(content_filter=PruningContentFilter())
    )
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url, config=config)
    if not result.success:
        raise RuntimeError(f"Crawl failed for {url}: {result.error_message}")
    return result.markdown.fit_markdown


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/crawl_to_markdown.py <url>", file=sys.stderr)
        sys.exit(1)

    url = sys.argv[1]
    markdown = asyncio.run(crawl(url))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{slug_from_url(url)}.md"
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Saved markdown to {output_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
