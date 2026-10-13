"""Retrieval top-k dans pgvector (cosine), avec filtre optionnel par personnage.

Usage CLI (test) : uv run --group rag python src/rag/retrieve.py "Pourquoi Jules cite Ézéchiel ?"
"""

import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config import settings  # noqa: E402


@dataclass
class RetrievedChunk:
    content: str
    metadata: dict
    score: float


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)


def embed_query(query: str):
    return _model().encode(f"query: {query}", normalize_embeddings=True)


def retrieve(
    query: str, top_k: int = 5, character: str | None = None
) -> list[RetrievedChunk]:
    emb = embed_query(query)
    sql = """SELECT content, metadata, 1 - (embedding <=> %s) AS score
             FROM chunks"""
    params: list = [emb]
    if character:
        sql += " WHERE metadata->'characters_mentioned' ? %s"
        params.append(character)
    sql += " ORDER BY embedding <=> %s LIMIT %s"
    params += [emb, top_k]

    with psycopg.connect(settings.database_url) as conn:
        register_vector(conn)
        rows = conn.execute(sql, params).fetchall()
    return [
        RetrievedChunk(content=r[0], metadata=r[1] if isinstance(r[1], dict) else json.loads(r[1]), score=float(r[2]))
        for r in rows
    ]


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "Pourquoi Jules cite-t-il Ézéchiel 25:17 ?"
    for c in retrieve(q, top_k=3):
        print(f"[{c.score:.3f}] {c.metadata['source']} — {c.metadata['title'][:60]}")
        print("   ", c.content[:150].replace("\n", " "), "\n")
