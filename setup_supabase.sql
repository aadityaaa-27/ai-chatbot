-- ============================================================
-- Run this entire script in Supabase → SQL Editor → Run
-- ============================================================

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Company data table (stores text chunks + embeddings)
CREATE TABLE IF NOT EXISTS company_data (
    id         BIGSERIAL    PRIMARY KEY,
    content    TEXT         NOT NULL,
    metadata   JSONB        DEFAULT '{}',
    embedding  vector(768),
    created_at TIMESTAMPTZ  DEFAULT NOW()
);

-- 3. Fast approximate nearest-neighbour index
CREATE INDEX IF NOT EXISTS company_data_emb_idx
ON company_data
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- 4. Semantic search RPC function
CREATE OR REPLACE FUNCTION search_company_data(
    query_embedding  vector(768),
    match_threshold  FLOAT  DEFAULT 0.3,
    match_count      INT    DEFAULT 5
)
RETURNS TABLE (
    id         BIGINT,
    content    TEXT,
    metadata   JSONB,
    similarity FLOAT
)
LANGUAGE SQL STABLE AS $$
    SELECT
        id,
        content,
        metadata,
        1 - (embedding <=> query_embedding) AS similarity
    FROM company_data
    WHERE 1 - (embedding <=> query_embedding) > match_threshold
    ORDER BY embedding <=> query_embedding
    LIMIT match_count;
$$;
