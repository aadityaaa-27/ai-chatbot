-- Persistent memory tables for AI Chatbot
-- Run this once in your Supabase SQL editor (Database → SQL Editor → New query).
-- Required for chat history and user facts to survive Render deploys.

CREATE TABLE IF NOT EXISTS chat_sessions (
    id         TEXT         PRIMARY KEY,
    user_id    INTEGER      NOT NULL REFERENCES app_users(id) ON DELETE CASCADE,
    title      TEXT         NOT NULL DEFAULT 'New Chat',
    msg_count  INTEGER      NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id);

CREATE TABLE IF NOT EXISTS chat_messages (
    id         BIGSERIAL    PRIMARY KEY,
    session_id TEXT         NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    user_id    INTEGER      NOT NULL REFERENCES app_users(id) ON DELETE CASCADE,
    role       TEXT         NOT NULL,
    content    TEXT         NOT NULL,
    created_at TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_user_id    ON chat_messages(user_id);

CREATE TABLE IF NOT EXISTS user_facts (
    user_id    INTEGER      NOT NULL REFERENCES app_users(id) ON DELETE CASCADE,
    key        TEXT         NOT NULL,
    value      TEXT         NOT NULL,
    updated_at TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, key)
);
