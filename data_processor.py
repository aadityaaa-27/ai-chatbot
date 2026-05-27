"""
Data Processor — AI-powered Excel/CSV → Supabase pipeline.

HR uploads any messy spreadsheet → Gemini maps columns → data is cleaned
and inserted into the employees table automatically.
"""
import json
import re
import pandas as pd
from google import genai
from rag_engine import _secret

# ── Standard schema the DB expects ───────────────────────────────────────────
STANDARD_COLUMNS = {
    "age":                       ("INTEGER",  "Employee age in years"),
    "attrition":                 ("TEXT",     "'Yes' or 'No' — did employee leave?"),
    "business_travel":           ("TEXT",     "'Non-Travel' | 'Travel_Rarely' | 'Travel_Frequently'"),
    "daily_rate":                ("INTEGER",  "Daily pay rate"),
    "department":                ("TEXT",     "Department name (e.g. Sales, Finance, HR)"),
    "distance_from_home":        ("INTEGER",  "Miles from home"),
    "education":                 ("INTEGER",  "1=Below College 2=College 3=Bachelor 4=Master 5=Doctor"),
    "education_field":           ("TEXT",     "Field of study (e.g. Life Sciences, Marketing)"),
    "environment_satisfaction":  ("INTEGER",  "1–4 scale"),
    "gender":                    ("TEXT",     "'Male' or 'Female'"),
    "holidays":                  ("INTEGER",  "Paid holidays per year (default 20)"),
    "hourly_rate":               ("INTEGER",  "Hourly pay rate"),
    "job_involvement":           ("INTEGER",  "1–4 scale"),
    "job_level":                 ("INTEGER",  "1–5"),
    "job_role":                  ("TEXT",     "Job title / designation"),
    "job_satisfaction":          ("INTEGER",  "1–4 scale"),
    "marital_status":            ("TEXT",     "'Single' | 'Married' | 'Divorced'"),
    "monthly_income":            ("INTEGER",  "Monthly salary in USD/INR (numbers only)"),
    "num_companies_worked":      ("INTEGER",  "Number of previous companies"),
    "overtime":                  ("TEXT",     "'Yes' or 'No'"),
    "percent_salary_hike":       ("INTEGER",  "Salary hike percentage"),
    "performance_rating":        ("INTEGER",  "3=Excellent 4=Outstanding"),
    "total_working_years":       ("INTEGER",  "Total years of work experience"),
    "work_life_balance":         ("INTEGER",  "1–4 scale"),
    "years_at_company":          ("INTEGER",  "Years at current company"),
    "years_in_current_role":     ("INTEGER",  "Years in current role"),
    "years_since_last_promotion":("INTEGER",  "Years since last promotion"),
    "years_with_curr_manager":   ("INTEGER",  "Years with current manager"),
}


class DataProcessor:
    def __init__(self):
        self._client = genai.Client(api_key=_secret("GEMINI_API_KEY"))

    # ── Step 1: Read file ─────────────────────────────────────────────────────

    def read_file(self, uploaded_file) -> tuple:
        """
        Read an Excel or CSV file uploaded via st.file_uploader.
        Returns (DataFrame, sheet_names_list_or_None).
        """
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            return df, None
        elif name.endswith((".xlsx", ".xls")):
            xl = pd.ExcelFile(uploaded_file)
            sheets = xl.sheet_names
            # Default to first sheet; caller can re-call with sheet_name param
            df = pd.read_excel(uploaded_file, sheet_name=sheets[0])
            return df, sheets
        else:
            raise ValueError("Only .csv, .xlsx, and .xls files are supported.")

    def read_sheet(self, uploaded_file, sheet_name: str) -> pd.DataFrame:
        return pd.read_excel(uploaded_file, sheet_name=sheet_name)

    # ── Step 2: AI column mapping ─────────────────────────────────────────────

    def analyze_columns(self, df: pd.DataFrame) -> dict:
        """
        Use Gemini to map uploaded column names → standard schema fields.
        Returns dict with 'column_map', 'extra_columns', 'notes'.
        """
        columns = df.columns.tolist()
        # Send first 3 rows as context
        sample_rows = df.head(3).fillna("").astype(str).to_dict(orient="records")

        std_schema_txt = "\n".join(
            f"  {col} ({dtype}): {desc}"
            for col, (dtype, desc) in STANDARD_COLUMNS.items()
        )

        prompt = f"""You are an expert HR data analyst. Your job is to map uploaded spreadsheet columns to our standard database schema.

UPLOADED COLUMNS: {json.dumps(columns)}

SAMPLE DATA (first 3 rows):
{json.dumps(sample_rows, default=str)[:3000]}

STANDARD SCHEMA:
{std_schema_txt}

Your task:
1. For each uploaded column, find the best matching standard field.
2. Be smart about synonyms:
   - "Emp Age", "Age (Years)", "Employee Age" → age
   - "Dept", "Dept.", "Division", "Team" → department
   - "Designation", "Title", "Position", "Role", "Job Title" → job_role
   - "CTC", "Salary", "Monthly Salary", "Pay", "Gross Salary", "Monthly CTC" → monthly_income
   - "Left", "Resigned", "Exit Status", "Churned" → attrition
   - "Sex", "M/F" → gender
   - "Yrs Experience", "Total Exp", "Work Experience" → total_working_years
   - "Education Level", "Qualification", "Degree" → education
3. If a column clearly doesn't match any standard field, use JSON null (not the string "null").
4. Note any data quality issues (currency symbols, inconsistent formats, etc.)

Return ONLY valid JSON (no markdown, no code fences):
{{
  "column_map": {{
    "EXACT_UPLOADED_COLUMN_NAME": "standard_field_name_or_json_null"
  }},
  "extra_columns": ["columns that have no standard mapping"],
  "notes": "short description of data quality and what was found"
}}

IMPORTANT: For unmapped columns use JSON null like this: "Col Name": null
NOT like this: "Col Name": "null"
"""
        r = self._client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        text = r.text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return json.loads(text)

    # ── Step 3: Clean & transform ─────────────────────────────────────────────

    def clean_and_transform(self, df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
        """
        Apply column mapping, clean values, and return a ready-to-insert DataFrame.
        """
        # Filter out null / "null" / "none" — Gemini sometimes returns the string "null"
        _bad = {"null", "none", "n/a", "", "undefined"}
        col_map = {
            k: v for k, v in mapping.get("column_map", {}).items()
            if v is not None and str(v).strip().lower() not in _bad
        }

        # If two uploaded columns map to the same standard field, keep only the first
        seen_targets: dict = {}
        deduped: dict = {}
        for src, tgt in col_map.items():
            if tgt not in seen_targets:
                seen_targets[tgt] = src
                deduped[src] = tgt
        col_map = deduped

        # Only rename columns that actually exist in the uploaded DataFrame
        rename_map = {k: v for k, v in col_map.items() if k in df.columns}
        df = df.rename(columns=rename_map)

        # Keep only mapped standard columns (deduplicated, preserving order)
        keep = list(dict.fromkeys(v for v in rename_map.values() if v in df.columns))
        df   = df[keep].copy()

        # ── Clean each column ─────────────────────────────────────────────────

        def clean_money(s):
            s = str(s).strip()
            s = re.sub(r"[₹$€£,\s]", "", s)      # remove currency symbols
            s = re.sub(r"(?i)k$", "000", s)        # "50k" → "50000"
            s = re.sub(r"[^\d.]", "", s)            # remove remaining non-numeric
            return s

        if "monthly_income" in df.columns:
            df["monthly_income"] = (df["monthly_income"].apply(clean_money))
            df["monthly_income"] = pd.to_numeric(df["monthly_income"], errors="coerce").fillna(0).astype(int)

        if "daily_rate" in df.columns:
            df["daily_rate"] = (df["daily_rate"].apply(clean_money))
            df["daily_rate"] = pd.to_numeric(df["daily_rate"], errors="coerce").fillna(0).astype(int)

        if "hourly_rate" in df.columns:
            df["hourly_rate"] = (df["hourly_rate"].apply(clean_money))
            df["hourly_rate"] = pd.to_numeric(df["hourly_rate"], errors="coerce").fillna(0).astype(int)

        if "age" in df.columns:
            df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(30).clip(18, 80).astype(int)

        if "gender" in df.columns:
            _g = {"m": "Male", "f": "Female", "male": "Male", "female": "Female",
                  "1": "Male", "0": "Female", "man": "Male", "woman": "Female"}
            df["gender"] = (df["gender"].astype(str).str.strip().str.lower()
                            .map(_g).fillna("Male"))

        if "attrition" in df.columns:
            _a = {"yes": "Yes", "no": "No", "true": "Yes", "false": "No",
                  "1": "Yes", "0": "No", "left": "Yes", "stayed": "No",
                  "resigned": "Yes", "active": "No", "churned": "Yes"}
            df["attrition"] = (df["attrition"].astype(str).str.strip().str.lower()
                               .map(_a).fillna("No"))

        if "overtime" in df.columns:
            _o = {"yes": "Yes", "no": "No", "true": "Yes", "false": "No",
                  "1": "Yes", "0": "No"}
            df["overtime"] = (df["overtime"].astype(str).str.strip().str.lower()
                              .map(_o).fillna("No"))

        if "business_travel" in df.columns:
            _bt = {
                "non-travel": "Non-Travel", "non travel": "Non-Travel", "never": "Non-Travel",
                "travel_rarely": "Travel_Rarely", "rarely": "Travel_Rarely",
                "travel_frequently": "Travel_Frequently", "frequently": "Travel_Frequently",
            }
            df["business_travel"] = (df["business_travel"].astype(str).str.strip().str.lower()
                                     .map(_bt).fillna("Non-Travel"))

        if "marital_status" in df.columns:
            _ms = {"single": "Single", "married": "Married", "divorced": "Divorced",
                   "separated": "Divorced", "widowed": "Single"}
            df["marital_status"] = (df["marital_status"].astype(str).str.strip().str.lower()
                                    .map(_ms).fillna("Single"))

        if "department" in df.columns:
            df["department"] = df["department"].astype(str).str.strip().str.title()

        if "job_role" in df.columns:
            df["job_role"] = df["job_role"].astype(str).str.strip().str.title()

        if "education_field" in df.columns:
            df["education_field"] = df["education_field"].astype(str).str.strip().str.title()

        # Integer columns
        for col in ["education", "job_level", "job_satisfaction", "environment_satisfaction",
                    "job_involvement", "work_life_balance", "performance_rating",
                    "percent_salary_hike", "total_working_years", "num_companies_worked",
                    "years_at_company", "years_in_current_role", "years_since_last_promotion",
                    "years_with_curr_manager", "distance_from_home", "stock_option_level",
                    "training_times_last_year", "relationship_satisfaction"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        # ── Set defaults for missing required columns ─────────────────────────
        defaults = {
            "holidays":        20,
            "attrition":       "No",
            "overtime":        "No",
            "business_travel": "Non-Travel",
            "education":       3,
            "job_level":       1,
            "performance_rating": 3,
        }
        for col, val in defaults.items():
            if col not in df.columns:
                df[col] = val

        # Drop rows missing critical fields
        for req in ["department", "job_role", "monthly_income"]:
            if req in df.columns:
                df = df[df[req].notna() & (df[req].astype(str).str.strip() != "")]

        df = df.reset_index(drop=True)
        return df

    # ── Step 4: Insert to Supabase ────────────────────────────────────────────

    def insert_to_db(self, df: pd.DataFrame, sb_client,
                     mode: str = "add_new",
                     source_file: str = "upload") -> dict:
        """
        Insert the processed DataFrame into the employees table.
        mode:
          'add_new'          — just insert (new dataset alongside existing ones)
          'replace_dataset'  — delete rows with same source_file, then insert
          'replace_all'      — delete ALL rows, then insert
        source_file: filename tag stored in each row so datasets stay separate.
        """
        # Tag every row with the source_file name
        safe_name = source_file.replace("'", "''")
        df = df.copy()
        df["source_file"] = source_file

        if mode == "replace_all":
            try:
                sb_client.rpc(
                    "run_employee_query",
                    {"query_sql": "DELETE FROM employees"}
                ).execute()
            except Exception as e:
                print(f"[DataProcessor] replace_all clear failed: {e}")
        elif mode == "replace_dataset":
            try:
                sb_client.rpc(
                    "run_employee_query",
                    {"query_sql": f"DELETE FROM employees WHERE source_file = '{safe_name}'"}
                ).execute()
            except Exception as e:
                print(f"[DataProcessor] replace_dataset clear failed: {e}")

        records = df.to_dict(orient="records")

        # Remove NaN / None values from each record
        clean = []
        for row in records:
            clean.append({
                k: (None if (v != v or str(v).lower() == "nan") else v)
                for k, v in row.items()
            })

        batch_size = 100
        inserted, failed = 0, 0
        errors = []

        for i in range(0, len(clean), batch_size):
            batch = clean[i: i + batch_size]
            try:
                sb_client.table("employees").insert(batch).execute()
                inserted += len(batch)
            except Exception as e:
                failed += len(batch)
                errors.append(str(e)[:120])

        return {
            "inserted": inserted,
            "failed":   failed,
            "total":    len(records),
            "errors":   errors[:3],   # return first 3 error messages only
        }

    # ── Full pipeline (convenience) ───────────────────────────────────────────

    def run(self, df: pd.DataFrame, sb_client,
            mode: str = "add_new", source_file: str = "upload") -> dict:
        """analyze → clean → insert. Returns result dict."""
        mapping  = self.analyze_columns(df)
        clean_df = self.clean_and_transform(df, mapping)
        result   = self.insert_to_db(clean_df, sb_client,
                                     mode=mode, source_file=source_file)
        result["mapping"] = mapping
        result["columns_mapped"] = len(
            [v for v in mapping.get("column_map", {}).values() if v]
        )
        return result
