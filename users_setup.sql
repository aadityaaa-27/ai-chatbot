-- ============================================================
-- Run this ONCE in Supabase → SQL Editor
-- Creates the app_users table for authentication
-- ============================================================

CREATE TABLE IF NOT EXISTS app_users (
    id            SERIAL PRIMARY KEY,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'hr'
                  CHECK (role IN ('admin', 'hr', 'payroll', 'manager')),
    department    TEXT,                        -- required for 'manager' role
    full_name     TEXT NOT NULL DEFAULT '',
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    last_login    TIMESTAMPTZ
);

-- Optional: disable public access (only service-key can read/write)
-- ALTER TABLE app_users ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY "service key only" ON app_users USING (false);

-- Confirm creation
SELECT 'app_users table ready' AS status;
