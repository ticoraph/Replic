"""Indexation du corpus dans pgvector (idempotente : hash de contenu, upsert).

Embeddings : intfloat/multilingual-e5-base (768 dims, CPU suffisant).
Préfixe e5 obligatoire : "passage: " à l'indexation, "query: " au retrieval.

Usage : uv run --group rag python src/rag/index.py
"""

import hashlib
import json
import sys
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.chunking import chunk_corpus  # noqa: E402
from utils.config import settings  # noqa: E402

BATCH = 32


def main() -> None:
    chunks = chunk_corpus()
    print(f"{len(chunks)} chunks à indexer")
    model = SentenceTransformer(settings.embedding_model)

    with psycopg.connect(settings.database_url) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            existing = {
                r[0] for r in cur.execute("SELECT content_hash FROM chunks").fetchall()
            }
            wanted = {hashlib.sha256(c.content.encode()).hexdigest(): c for c in chunks}

            # Purge des chunks obsolètes (ex. anciens chunks EN du script original)
            stale = existing - set(wanted)
            if stale:
                cur.execute(
                    "DELETE FROM chunks WHERE content_hash = ANY(%s)", (list(stale),)
                )
                conn.commit()
                print(f"{len(stale)} chunks obsolètes supprimés")

            todo = [(c, h) for h, c in wanted.items() if h not in existing]
            print(f"{len(todo)} nouveaux ({len(wanted) - len(todo)} déjà indexés)")

            for i in range(0, len(todo), BATCH):
                batch = todo[i : i + BATCH]
                embeddings = model.encode(
                    [f"passage: {c.content}" for c, _ in batch], normalize_embeddings=True
                )
                for (chunk, h), emb in zip(batch, embeddings):
                    cur.execute(
                        """INSERT INTO chunks (content, content_hash, metadata, embedding)
                           VALUES (%s, %s, %s, %s)
                           ON CONFLICT (content_hash) DO NOTHING""",
                        (chunk.content, h, json.dumps(chunk.metadata), emb),
                    )
                conn.commit()
                print(f"  {min(i + BATCH, len(todo))}/{len(todo)}")

    print("Indexation terminée.")


if __name__ == "__main__":
    main()
