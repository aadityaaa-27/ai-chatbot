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
            # Ensure source_file column exists (safe to run multiple times)
            self._run(
                "ALTER TABLE employees "
                "ADD COLUMN IF NOT EXISTS source_file TEXT DEFAULT 'original'"
            )
            # Reload PostgREST schema cache so REST inserts see the new column
            try:
                self._run("NOTIFY pgrst, 'reload schema'")
            except Exception:
                pass
        except Exception as e:
            print(f"[SQL] init failed: {e}")

    @property
    def ready(self) -> bool:
        return self._ready

    # ── Text → SQL ────────────────────────────────────────────────────────────

    def _to_sql(self, question: str, source_file: str = "") -> str:
        sf = _sf_filter(source_file)
        filter_note = (
            f"\nIMPORTANT: Always add this filter to EVERY query: "
            f"WHERE {sf}\n(If the query already has a WHERE clause, use AND {sf} instead.)"
            if sf else ""
        )
        prompt = f"""You are an expert PostgreSQL analyst.

{SCHEMA}
{filter_note}

Convert this question into a single valid PostgreSQL SELECT query:
"{question}"

Rules:
- Use ROUND(value::numeric, 1) for any decimal averages
- Use COUNT(*) for totals / headcounts
- Add clear column aliases (e.g. AS "Average Age", AS "Total Employees")
- For breakdowns use GROUP BY with ORDER BY COUNT(*) DESC
- LIMIT 20 for list-type results
- Return ONLY the raw SQL — no markdown, no explanation
"""
        raw = self._client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        ).text.strip()
        raw = re.sub(r"^```sql\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
        return raw.strip()

    # ── Execute on Supabase ───────────────────────────────────────────────────

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
            sql  = self._to_sql(question, source_file=source_file)
            data = self._run(sql)
            return {"sql": sql, "data": data}
        except Exception as e:
            return {"error": str(e)}

    def query(self, question: str, source_file: str = "") -> tuple:
        """Return (context_string, raw_rows) — raw_rows usable for charting."""
        if not self._ready or not is_employee_question(question):
            return "", []
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

    def get_source_files(self) -> list:
        """Return list of dicts: [{source_file, count}] for all uploaded datasets."""
        if not self._ready:
            return []
        try:
            return self._run(
                "SELECT source_file, COUNT(*) as count "
                "FROM employees "
                "WHERE source_file IS NOT NULL "
                "GROUP BY source_file "
                "ORDER BY source_file"
            )
        except Exception:
            return []

    def delete_source_file(self, source_file: str) -> bool:
        """Delete all rows belonging to a specific uploaded file/dataset."""
        if not self._ready or not source_file:
            return False
        try:
            safe = source_file.replace("'", "''")
            self._run(f"DELETE FROM employees WHERE source_file = '{safe}'")
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
