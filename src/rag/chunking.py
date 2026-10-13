"""Chunking du corpus pour l'indexation RAG — tout en français.

- Analyses FR (data/raw/*.md) : split par headers Markdown, puis re-découpe ~800 tokens
  (approx. mots × 1.3) avec overlap.
- Scénario : chunks construits depuis les dialogues traduits en VF
  (data/interim/dialogues_fr.jsonl, produit par translate_dialogues.py), une scène =
  un chunk « SCÈNE\nPERSONNAGE : réplique… » (re-découpé si trop long). Le script
  original EN n'est plus indexé.

Métadonnées par chunk : {source, title, characters_mentioned, lang, kind}
"""

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.characters import CHARACTERS  # noqa: E402
from utils.config import DATA_INTERIM, DATA_RAW  # noqa: E402

TARGET_WORDS = 600  # ~800 tokens
OVERLAP_WORDS = 110  # ~150 tokens

# noms/alias détectables dans le texte (nom complet, prénom, alias screenplay)
_CHAR_PATTERNS = {
    key: re.compile(
        "|".join(
            re.escape(n)
            for n in {c.display_name, c.display_name.split()[0], *c.script_aliases}
        ),
        re.IGNORECASE,
    )
    for key, c in CHARACTERS.items()
}


@dataclass
class Chunk:
    content: str
    metadata: dict


def detect_characters(text: str) -> list[str]:
    return [key for key, pat in _CHAR_PATTERNS.items() if pat.search(text)]


def _window(words: list[str]) -> list[str]:
    """Découpe une liste de mots en fenêtres TARGET_WORDS avec overlap."""
    if len(words) <= TARGET_WORDS:
        return [" ".join(words)]
    out, start = [], 0
    while start < len(words):
        out.append(" ".join(words[start : start + TARGET_WORDS]))
        if start + TARGET_WORDS >= len(words):
            break
        start += TARGET_WORDS - OVERLAP_WORDS
    return out


def chunk_markdown(text: str, source: str, lang: str) -> list[Chunk]:
    """Split par headers markdown, re-découpe les sections trop longues."""
    sections = re.split(r"(?m)^(#{1,3} .+)$", text)
    chunks: list[Chunk] = []
    title = source
    buf = sections[0]
    pairs = [(sections[i], sections[i + 1]) for i in range(1, len(sections) - 1, 2)]

    def emit(section_title: str, body: str) -> None:
        words = body.split()
        if len(words) < 30:
            return
        for piece in _window(words):
            chunks.append(
                Chunk(
                    content=f"{section_title}\n\n{piece}" if section_title else piece,
                    metadata={
                        "source": source,
                        "title": section_title or title,
                        "characters_mentioned": detect_characters(piece),
                        "lang": lang,
                        "kind": "analysis",
                    },
                )
            )

    emit("", buf)
    for header, body in pairs:
        emit(header.lstrip("# ").strip(), body)
    return chunks


def chunk_screenplay_fr() -> list[Chunk]:
    """Chunks du scénario VF : une scène = un chunk dialogué (re-découpé si trop long)."""
    path = DATA_INTERIM / "dialogues_fr.jsonl"
    if not path.exists():
        print(f"({path.name} absent — lancer translate_dialogues.py ; scénario non indexé)")
        return []

    scenes: dict[int, list[dict]] = {}
    for line_ in path.read_text(encoding="utf-8").splitlines():
        d = json.loads(line_)
        scenes.setdefault(d["scene_id"], []).append(d)

    chunks: list[Chunk] = []
    for lines in scenes.values():
        heading = lines[0]["scene"]
        body = heading + "\n" + "\n".join(
            f"{d['speaker'].title()} : {d['text']}" for d in lines
        )
        for piece in _window(body.split()):
            chunks.append(
                Chunk(
                    content=piece,
                    metadata={
                        "source": "dialogues_fr.jsonl",
                        "title": heading,
                        "characters_mentioned": detect_characters(piece),
                        "lang": "fr",
                        "kind": "screenplay",
                    },
                )
            )
    return chunks


def chunk_corpus() -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(DATA_RAW.glob("*.md")):
        if "script-pulp-fiction" in path.name:
            continue  # le script EN n'est plus indexé : remplacé par les dialogues VF
        text = path.read_text(encoding="utf-8", errors="replace").replace("�", "—")
        chunks.extend(chunk_markdown(text, path.name, lang="fr"))
    chunks.extend(chunk_screenplay_fr())
    return chunks


if __name__ == "__main__":
    cs = chunk_corpus()
    from collections import Counter

    kinds = Counter(c.metadata["kind"] for c in cs)
    print(f"{len(cs)} chunks : {dict(kinds)}")
    print("Exemple :", cs[10].metadata, "|", cs[10].content[:120])
