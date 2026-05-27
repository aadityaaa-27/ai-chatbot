"""
SQL Engine — Text-to-SQL for employee analytics.
Converts natural language → PostgreSQL → executes on Supabase → returns results.
Supports per-dataset filtering via source_file column.
"""
import json
import re

from google import genai

from rag_engine import _secret

# ── Schema description fed to Gemini ─────────────────────────────────────────
SCHEMA = """
PostgreSQL table: employees

Columns:
  source_file                TEXT      name of the uploaded file this row came from
  age                        INTEGER   employee age in years
  attrition                  TEXT      'Yes' or 'No' (left company?)
  business_travel            TEXT      'Non-Travel' | 'Travel_Rarely' | 'Travel_Frequently'
  daily_rate                 INTEGER
  department                 TEXT      department name
  distance_from_home         INTEGER   miles
  education                  INTEGER   1=Below College 2=College 3=Bachelor 4=Master 5=Doctor
  education_field            TEXT      field of study
  environment_satisfaction   INTEGER   1=Low 2=Medium 3=High 4=Very High
  gender                     TEXT      'Male' or 'Female'
  holidays                   INTEGER   paid holidays per year
  hourly_rate                INTEGER
  job_involvement            INTEGER   1=Low 2=Medium 3=High 4=Very High
  job_level                  INTEGER   1-5
  job_role                   TEXT      job title / designation
  job_satisfaction           INTEGER   1=Low 2=Medium 3=High 4=Very High
  marital_status             TEXT      'Single' | 'Married' | 'Divorced'
  monthly_income             INTEGER   monthly salary
  num_companies_worked       INTEGER
  overtime                   TEXT      'Yes' or 'No'
  percent_salary_hike        INTEGER
  performance_rating         INTEGER   3=Excellent 4=Outstanding
  total_working_years        INTEGER
  work_life_balance          INTEGER   1=Bad 2=Good 3=Better 4=Best
  years_at_company           INTEGER
  years_in_current_role      INTEGER
  years_since_last_promotion INTEGER
  years_with_curr_manager    INTEGER
"""

# Keywords that suggest an employee-data question
_KEYWORDS = {
    "employee", "employees", "staff", "worker", "workers", "headcount",
    "age", "salary", "income", "department", "attrition", "gender",
    "job", "role", "promotion", "tenure", "satisfaction", "performance",
    "overtime", "training", "workforce", "hire", "hired", "turnover",
    "education", "experience", "years", "monthly", "average", "total",
    "count", "how", "many", "oldest", "youngest", "highest", "lowest",
    "holidays", "holiday", "leave", "days", "paid", "finance", "marketing",
    "legal", "operations", "technology", "support", "department", "team",
    "dept", "sal", "salary", "top", "bottom", "best", "worst",
    "which", "what", "show", "list", "give", "tell", "find", "data", "dataset",
}

# Question keywords → schema column names (for NULL pre-checks)
_COLUMN_HINTS: dict = {
    "marital_status":      ["married", "single", "divorced", "marital", "marriage"],
    "gender":              ["gender", "male", "female", "sex"],
    "attrition":           ["attrition", "left company", "resigned", "turnover", "churned", "quit"],
    "overtime":            ["overtime"],
    "department":          ["department", "dept", "division"],
    "job_role":            ["designation", "position", "job role", "job title"],
    "education_field":     ["education field", "field of study"],
    "education":           ["education", "degree", "qualification"],
    "business_travel":     ["business travel", "travel frequently", "travel rarely"],
    "monthly_income":      ["salary", "income", "ctc", "compensation", "earning", "pay"],
    "performance_rating":  ["performance", "performer", "rating"],
    "job_satisfaction":    ["satisfaction", "job satisfaction"],
    "work_life_balance":   ["work life balance", "work-life"],
    "environment_satisfaction": ["environment satisfaction", "workplace satisfaction"],
    "total_working_years": ["total experience", "working years"],
    "years_at_company":    ["tenure", "years at company"],
    "num_companies_worked":["companies worked", "previous companies"],
    "distance_from_home":  ["distance from home", "commute"],
    "job_involvement":     ["job involvement", "involvement"],
}


def is_employee_question(text: str) -> bool:
    words = re.findall(r"\w+", text.lower())
    return any(w in _KEYWORDS for w in words)


def _sf_filter(source_file: str) -> str:
    """Return a SQL WHERE/AND clause fragment for source_file filtering."""
    if source_file and source_file.lower() != "all":
        safe = source_file.replace("'", "''")   # basic SQL escape
        return f"source_file = '{safe}'"
    return ""


class SQLEngine:
    def __init__(self):
        self._sb     = None
        self._client = None
        self._ready  = False
        url = _secret("SUPABASE_URL")
        key = _secret("SUPABASE_KEY")
        if not url or not key:
            return
        try:
            from supabase import create_client
            self._sb     = create_client(url, key)
            self._client = genai.Client(api_key=_secret("GEMINI_API_KEY"))
            self._ready  = True
            # Tag untagged rows (only works after setup_source_file.sql has been run)
            try:
                self._run_write(
                    "UPDATE employees SET source_file = 'original' "
                    "WHERE source_file IS NULL"
                )
            except Exception:
                pass
        except Exception as e:
            print(f"[SQL] init failed: {e}")

    @property
    def ready(self) -> bool:
        return self._ready

    # ── Text → SQL ────────────────────────────────────────────────────────────

    @staticmethod
    def _inject_filter(sql: str, source_file: str) -> str:
        """
        Programmatically inject a source_file WHERE clause into a SQL query.
        Never relies on Gemini to add it — 100% reliable.
        """
        sf = _sf_filter(source_file)
        if not sf:
            return sql

        up = sql.upper()

        # Find the right insertion point
        for keyword in (" WHERE ", " GROUP BY ", " ORDER BY ", " HAVING ", " LIMIT "):
            idx = up.find(keyword)
            if idx != -1:
                if keyword == " WHERE ":
                    # Wrap existing WHERE condition and AND our filter
                    insert_at = idx + len(" WHERE ")
                    return sql[:insert_at] + f"({sf}) AND (" + sql[insert_at:] + ")"
                else:
                    return sql[:idx] + f" WHERE {sf}" + sql[idx:]

        # No clause found — append at end
        return sql.rstrip(";") + f" WHERE {sf}"

    def _to_sql(self, question: str) -> str:
        """Generate SQL from natural language. source_file filter is injected separately."""
        prompt = f"""You are an expert PostgreSQL analyst.

{SCHEMA}

Convert this question into a single valid PostgreSQL SELECT query:
"{question}"

Rules:
- Use ROUND(value::numeric, 1) for any decimal averages
- Use COUNT(*) for totals / headcounts
- Add clear column aliases (e.g. AS "Average Age", AS "Total Employees")
- For breakdowns use GROUP BY with ORDER BY COUNT(*) DESC
- CRITICAL: In GROUP BY and ORDER BY always use positional references
  (GROUP BY 1, 2 or GROUP BY the actual column/expression) — NEVER
  reference an alias you defined in the same SELECT clause, because
  PostgreSQL does not allow that
- LIMIT 20 for list-type results
- Return ONLY the raw SQL — no markdown, no explanation
- Do NOT add any WHERE clause for source_file
"""
        raw = self._client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        ).text.strip()
        raw = re.sub(r"^```sql\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
        return raw.strip()

    # ── Execute on Supabase ───────────────────────────────────────────────────

    def _run_write(self, sql: str):
        """Execute a non-SELECT statement via the run_employee_write RPC."""
        res = self._sb.rpc("run_employee_write", {"query_sql": sql}).execute()
        return res.data

    def _run(self, sql: str) -> list:
        res = self._sb.rpc("run_employee_query", {"query_sql": sql}).execute()
        data = res.data
        if isinstance(data, str):
            data = json.loads(data)
        return data or []

    # ── Public API ────────────────────────────────────────────────────────────

    def answer(self, question: str, source_file: str = "") -> dict:
        """Return {"sql": ..., "data": [...]} or {"error": ...}."""
        if not self._ready:
            return {"error": "SQL engine not ready"}
        try:
            sql  = self._to_sql(question)
            sql  = self._inject_filter(sql, source_file)   # deterministic filter
            data = self._run(sql)
            return {"sql": sql, "data": data}
        except Exception as e:
            return {"error": str(e)}

    # ── NULL column detection ─────────────────────────────────────────────────

    def _detect_question_columns(self, question: str) -> list:
        """Return schema column names that the question is likely asking about."""
        q = question.lower()
        found = []
        for col, hints in _COLUMN_HINTS.items():
            if any(hint in q for hint in hints):
                found.append(col)
        return found

    def _null_columns(self, columns: list, source_file: str) -> list:
        """
        Return the subset of `columns` that have ZERO non-NULL values in the
        selected dataset.  Silently skips any column that causes a query error.
        """
        sf  = _sf_filter(source_file)
        bad = []
        for col in columns:
            try:
                if sf:
                    sql = (
                        f"SELECT COUNT(*) AS n FROM employees "
                        f"WHERE ({sf}) AND {col} IS NOT NULL"
                    )
                else:
                    sql = f"SELECT COUNT(*) AS n FROM employees WHERE {col} IS NOT NULL"
                rows  = self._run(sql)
                count = int(rows[0]["n"]) if rows else 0
                if count == 0:
                    bad.append(col)
            except Exception:
                pass   # column may not exist — don't block the main query
        return bad

    def query(self, question: str, source_file: str = "") -> tuple:
        """Return (context_string, raw_rows) — raw_rows usable for charting."""
        if not self._ready or not is_employee_question(question):
            return "", []

        # ── Pre-check: are any relevant columns all-NULL in this dataset? ────
        cols_in_q = self._detect_question_columns(question)
        if cols_in_q:
            missing = self._null_columns(cols_in_q, source_file)
            if missing:
                col_display = "', '".join(missing)
                ctx = (
                    f"DATA AVAILABILITY CHECK:\n"
                    f"The column(s) '{col_display}' contain NO data "
                    f"(all values are NULL / missing) in the selected dataset "
                    f"'{source_file or 'all datasets'}'.\n\n"
                    f"REQUIRED RESPONSE: Inform the user clearly that this "
                    f"information is not present in the uploaded file — the "
                    f"column was either empty or not included. "
                    f"Do NOT provide any numbers or estimates for this field."
                )
                return ctx, []

        result = self.answer(question, source_file=source_file)
        if result.get("error") or not result.get("data"):
            return "", []
        rows = result["data"]
        sql  = result["sql"]
        if not rows:
            return "", []
        headers = list(rows[0].keys())
        lines   = [" | ".join(headers)]
        lines  += [" | ".join(str(r.get(h, "")) for h in headers) for r in rows[:20]]
        table   = "\n".join(lines)
        ctx = (
            f"LIVE EMPLOYEE DATABASE RESULTS (answer based on this):\n"
            f"Query: {sql}\n\n{table}"
        )
        return ctx, rows

    def retag_dataset(self, old_name: str, new_name: str) -> bool:
        """Rename an existing dataset (change source_file value)."""
        if not self._ready:
            return False
        try:
            old_safe = old_name.replace("'", "''")
            new_safe = new_name.replace("'", "''")
            self._run_write(
                f"UPDATE employees SET source_file = '{new_safe}' "
                f"WHERE source_file = '{old_safe}'"
            )
            return True
        except Exception as e:
            print(f"[SQL] retag failed: {e}")
            return False

    def has_source_file_column(self) -> bool:
        """Check whether the source_file column exists (setup_source_file.sql been run)."""
        try:
            self._run("SELECT source_file FROM employees LIMIT 1")
            return True
        except Exception:
            return False

    def get_source_files(self) -> list:
        """Return list of dicts: [{source_file, count}] for all datasets."""
        if not self._ready:
            return []
        try:
            return self._run(
                "SELECT COALESCE(source_file, 'original') as source_file, "
                "COUNT(*) as count "
                "FROM employees "
                "GROUP BY COALESCE(source_file, 'original') "
                "ORDER BY source_file"
            )
        except Exception:
            # source_file column not set up yet — fall back to total count
            try:
                r = self._sb.table("employees").select("*", count="exact").execute()
                count = r.count or 0
                if count > 0:
                    return [{"source_file": "original", "count": count}]
            except Exception:
                pass
            return []

    def delete_source_file(self, source_file: str) -> bool:
        """Delete all rows belonging to a specific uploaded file/dataset."""
        if not self._ready or not source_file:
            return False
        try:
            self._sb.table("employees").delete().eq("source_file", source_file).execute()
            return True
        except Exception as e:
            print(f"[SQL] delete_source_file failed: {e}")
            return False

    def get_analytics_data(self, source_file: str = "") -> dict:
        """Pre-built analytics queries for the dashboard tab."""
        if not self._ready:
            return {}
        sf = _sf_filter(source_file)
        w  = f"WHERE {sf}" if sf else ""   # e.g. "WHERE source_file = 'file.csv'"

        queries = {
            "dept_headcount": (
                f"SELECT department, COUNT(*) as employees "
                f"FROM employees {w} GROUP BY department ORDER BY employees DESC"
            ),
            "dept_attrition": (
                f"SELECT department, "
                f"ROUND(100.0*SUM(CASE WHEN attrition='Yes' THEN 1 ELSE 0 END)/COUNT(*)::numeric,1) as attrition_pct "
                f"FROM employees {w} GROUP BY department ORDER BY attrition_pct DESC"
            ),
            "dept_salary": (
                f"SELECT department, ROUND(AVG(monthly_income)::numeric,0) as avg_salary "
                f"FROM employees {w} GROUP BY department ORDER BY avg_salary DESC"
            ),
            "age_groups": (
                f"SELECT CASE "
                f"WHEN age < 25 THEN 'Under 25' "
                f"WHEN age BETWEEN 25 AND 34 THEN '25-34' "
                f"WHEN age BETWEEN 35 AND 44 THEN '35-44' "
                f"WHEN age BETWEEN 45 AND 54 THEN '45-54' "
                f"ELSE '55+' END as age_group, COUNT(*) as employees "
                f"FROM employees {w} GROUP BY age_group ORDER BY MIN(age)"
            ),
            "gender": (
                f"SELECT gender, COUNT(*) as employees "
                f"FROM employees {w} GROUP BY gender"
            ),
            "satisfaction": (
                f"SELECT job_satisfaction, COUNT(*) as employees "
                f"FROM employees {w} GROUP BY job_satisfaction ORDER BY job_satisfaction"
            ),
            "overtime": (
                f"SELECT overtime, COUNT(*) as employees "
                f"FROM employees {w} GROUP BY overtime"
            ),
            "education": (
                f"SELECT education, COUNT(*) as employees "
                f"FROM employees {w} GROUP BY education ORDER BY education"
            ),
            "gender_by_dept": (
                f"SELECT department, gender, COUNT(*) as employees "
                f"FROM employees {w} "
                f"GROUP BY department, gender ORDER BY department, gender"
            ),
            # ── Cross-dimensional (used by chart AI for deeper answers) ────────
            "age_salary": (
                f"SELECT CASE "
                f"WHEN age < 25 THEN 'Under 25' "
                f"WHEN age BETWEEN 25 AND 34 THEN '25-34' "
                f"WHEN age BETWEEN 35 AND 44 THEN '35-44' "
                f"WHEN age BETWEEN 45 AND 54 THEN '45-54' "
                f"ELSE '55+' END as age_group, "
                f"ROUND(AVG(monthly_income)::numeric,0) as avg_salary, "
                f"COUNT(*) as employees "
                f"FROM employees {w} GROUP BY 1 ORDER BY avg_salary DESC"
            ),
            "gender_salary": (
                f"SELECT gender, "
                f"ROUND(AVG(monthly_income)::numeric,0) as avg_salary, "
                f"COUNT(*) as employees "
                f"FROM employees {w} GROUP BY gender ORDER BY avg_salary DESC"
            ),
            "role_salary": (
                f"SELECT job_role, "
                f"ROUND(AVG(monthly_income)::numeric,0) as avg_salary, "
                f"COUNT(*) as employees "
                f"FROM employees {w} GROUP BY job_role "
                f"ORDER BY avg_salary DESC LIMIT 15"
            ),
            "dept_avg_age": (
                f"SELECT department, "
                f"ROUND(AVG(age)::numeric,1) as avg_age, "
                f"ROUND(AVG(monthly_income)::numeric,0) as avg_salary "
                f"FROM employees {w} GROUP BY department ORDER BY avg_salary DESC"
            ),
            "attrition_age": (
                f"SELECT CASE "
                f"WHEN age < 25 THEN 'Under 25' "
                f"WHEN age BETWEEN 25 AND 34 THEN '25-34' "
                f"WHEN age BETWEEN 35 AND 44 THEN '35-44' "
                f"WHEN age BETWEEN 45 AND 54 THEN '45-54' "
                f"ELSE '55+' END as age_group, "
                f"ROUND(100.0*SUM(CASE WHEN attrition='Yes' THEN 1 ELSE 0 END)"
                f"/COUNT(*)::numeric,1) as attrition_pct "
                f"FROM employees {w} GROUP BY 1 ORDER BY attrition_pct DESC"
            ),
        }
        results = {}
        for key, sql in queries.items():
            try:
                results[key] = self._run(sql)
            except Exception as e:
                print(f"[Analytics] {key} failed: {e}")
        return results

    def employee_count(self, source_file: str = "") -> int:
        if not self._ready:
            return 0
        try:
            sf = _sf_filter(source_file)
            if sf:
                rows = self._run(f"SELECT COUNT(*) as n FROM employees WHERE {sf}")
                return rows[0]["n"] if rows else 0
            r = self._sb.table("employees").select("*", count="exact").execute()
            return r.count or 0
        except Exception:
            return 0
