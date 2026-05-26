-- ================================================================
-- Run this in Supabase → SQL Editor → Run
-- (Run setup_supabase.sql first if you haven't already)
-- ================================================================

-- 1. Employees table (IBM HR Analytics — 1,470 records, 31 columns)
CREATE TABLE IF NOT EXISTS employees (
    id                          BIGSERIAL PRIMARY KEY,
    age                         INTEGER,
    attrition                   TEXT,
    business_travel             TEXT,
    daily_rate                  INTEGER,
    department                  TEXT,
    distance_from_home          INTEGER,
    education                   INTEGER,   -- 1=Below College 2=College 3=Bachelor 4=Master 5=Doctor
    education_field             TEXT,
    employee_number             INTEGER,
    environment_satisfaction    INTEGER,   -- 1=Low 2=Medium 3=High 4=Very High
    gender                      TEXT,
    hourly_rate                 INTEGER,
    job_involvement             INTEGER,
    job_level                   INTEGER,
    job_role                    TEXT,
    job_satisfaction            INTEGER,   -- 1=Low 2=Medium 3=High 4=Very High
    marital_status              TEXT,
    monthly_income              INTEGER,
    num_companies_worked        INTEGER,
    overtime                    TEXT,      -- Yes / No
    percent_salary_hike         INTEGER,
    performance_rating          INTEGER,   -- 3=Excellent 4=Outstanding
    relationship_satisfaction   INTEGER,
    stock_option_level          INTEGER,
    total_working_years         INTEGER,
    training_times_last_year    INTEGER,
    work_life_balance           INTEGER,   -- 1=Bad 2=Good 3=Better 4=Best
    years_at_company            INTEGER,
    years_in_current_role       INTEGER,
    years_since_last_promotion  INTEGER,
    years_with_curr_manager     INTEGER
);

-- 2. Indexes for fast filtering
CREATE INDEX IF NOT EXISTS emp_dept_idx  ON employees (department);
CREATE INDEX IF NOT EXISTS emp_role_idx  ON employees (job_role);
CREATE INDEX IF NOT EXISTS emp_gender_idx ON employees (gender);

-- 3. Safe dynamic SQL executor (SELECT only)
CREATE OR REPLACE FUNCTION run_employee_query(query_sql TEXT)
RETURNS JSON
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    result JSON;
    clean  TEXT := UPPER(TRIM(query_sql));
BEGIN
    -- Only allow SELECT
    IF NOT (clean LIKE 'SELECT%') THEN
        RAISE EXCEPTION 'Only SELECT queries are allowed';
    END IF;
    -- Block destructive keywords
    IF clean ~ '(DROP|DELETE|INSERT|UPDATE|TRUNCATE|ALTER|CREATE)' THEN
        RAISE EXCEPTION 'Destructive operation blocked';
    END IF;
    EXECUTE format(
        'SELECT COALESCE(json_agg(row_to_json(t)), %L::json) FROM (%s) t',
        '[]', query_sql
    ) INTO result;
    RETURN result;
END;
$$;
