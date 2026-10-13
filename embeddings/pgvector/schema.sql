-- Schéma pgvector pour le RAG PulpFiction (multilingual-e5-base → 768 dims)
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    content      TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    metadata     JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding    VECTOR(768) NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_metadata_gin
    ON chunks USING gin (metadata);
