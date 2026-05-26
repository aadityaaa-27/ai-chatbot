"""
SQL Engine — Text-to-SQL for employee analytics.
Converts natural language → PostgreSQL → executes on Supabase → returns results.
"""
import json
import re

from google import genai

from rag_engine import _secret

# ── Schema description fed to Gemini ─────────────────────────────────────────
SCHEMA = """
PostgreSQL table: employees  (~2,940 employee records across 9 departments)

Columns:
  age                        INTEGER   employee age in years
  attrition                  TEXT      'Yes' or 'No' (left company?)
  business_travel            TEXT      'Non-Travel' | 'Travel_Rarely' | 'Travel_Frequently'
  daily_rate                 INTEGER
  department                 TEXT      'Sales' | 'Research & Development' | 'Human Resources' |
                                       'Finance' | 'Marketing' | 'Information Technology' |
                                       'Operations' | 'Legal' | 'Customer Support'
  distance_from_home         INTEGER   miles
  education                  INTEGER   1=Below College 2=College 3=Bachelor 4=Master 5=Doctor
  education_field            TEXT      'Life Sciences' | 'Medical' | 'Marketing' | 'Technical Degree' | 'Human Resources' | 'Other'
  employee_number            INTEGER
  environment_satisfaction   INTEGER   1=Low 2=Medium 3=High 4=Very High
  gender                     TEXT      'Male' | 'Female'
  holidays                   INTEGER   paid holidays per year (all employees: 20)
  hourly_rate                INTEGER
  job_involvement            INTEGER   1=Low 2=Medium 3=High 4=Very High
  job_level                  INTEGER   1-5
  job_role                   TEXT      e.g. 'Sales Executive' | 'Research Scientist' | 'Manager' |
                                       'Financial Analyst' | 'Software Engineer' | 'Marketing Manager' |
                                       'Operations Manager' | 'Legal Counsel' | 'Customer Service Representative' | etc.
  job_satisfaction           INTEGER   1=Low 2=Medium 3=High 4=Very High
  marital_status             TEXT      'Single' | 'Married' | 'Divorced'
  monthly_income             INTEGER   in USD
  num_companies_worked       INTEGER
  overtime                   TEXT      'Yes' | 'No'
  percent_salary_hike        INTEGER
  performance_rating         INTEGER   3=Excellent 4=Outstanding
  relationship_satisfaction  INTEGER   1-4
  stock_option_level         INTEGER   0-3
  total_working_years        INTEGER
  training_times_last_year   INTEGER
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
    "count", "how many", "oldest", "youngest", "highest", "lowest",
}


def is_employee_question(text: str) -> bool:
    words = re.findall(r"\w+", text.lower())
    return any(w in _KEYWORDS for w in words)


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
        except Exception as e:
            print(f"[SQL] init failed: {e}")

    @property
    def ready(self) -> bool:
        return self._ready

    # ── Text → SQL ────────────────────────────────────────────────────────────

    def _to_sql(self, question: str) -> str:
        prompt = f"""You are an expert PostgreSQL analyst.

{SCHEMA}

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
        # Strip markdown fences if Gemini added them
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

    def answer(self, question: str) -> dict:
        """Return {"sql": ..., "data": [...]} or {"error": ...}."""
        if not self._ready:
            return {"error": "SQL engine not ready (check Supabase credentials)"}
        try:
            sql  = self._to_sql(question)
            data = self._run(sql)
            return {"sql": sql, "data": data}
        except Exception as e:
            return {"error": str(e)}

    def format_context(self, question: str) -> str:
        """Return a formatted string to inject into Gemini's system prompt."""
        if not self._ready:
            return ""
        if not is_employee_question(question):
            return ""
        result = self.answer(question)
        if result.get("error") or not result.get("data"):
            return ""
        rows    = result["data"]
        sql     = result["sql"]
        # Build a readable table
        if not rows:
            return ""
        headers = list(rows[0].keys())
        lines   = [" | ".join(headers)]
        lines  += [" | ".join(str(r.get(h, "")) for h in headers) for r in rows[:20]]
        table   = "\n".join(lines)
        return (
            f"LIVE EMPLOYEE DATABASE RESULTS (answer based on this):\n"
            f"Query: {sql}\n\n"
            f"{table}"
        )

    def employee_count(self) -> int:
        if not self._ready:
            return 0
        try:
            r = self._sb.table("employees").select("*", count="exact").execute()
            return r.count or 0
        except Exception:
            return 0
