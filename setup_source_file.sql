-- ============================================================
-- Run this in Supabase → SQL Editor → Run
-- ONE-TIME SETUP for dataset tracking
-- ============================================================

-- 1. Add source_file column to employees table
ALTER TABLE employees
ADD COLUMN IF NOT EXISTS source_file TEXT DEFAULT 'original';

-- 2. Tag any existing rows
UPDATE employees
SET source_file = 'original'
WHERE source_file IS NULL;

-- 3. Create a write RPC for delete/update operations
--    (the existing run_employee_query only allows SELECT)
CREATE OR REPLACE FUNCTION run_employee_write(query_sql TEXT)
RETURNS JSON
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    EXECUTE query_sql;
    RETURN '{"ok": true}'::JSON;
EXCEPTION WHEN OTHERS THEN
    RETURN json_build_object('error', SQLERRM);
END;
$$;
