"""
Load IBM HR Analytics Employee Data into Supabase.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATASET  : IBM HR Analytics Employee Attrition
KAGGLE   : https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
FILENAME : WA_Fn-UseC_-HR-Employee-Attrition.csv
ROWS     : 1,470 employees  |  35 columns
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HOW TO GET THE FILE:
  1. Open the Kaggle URL above
  2. Click "Download" (top right)
  3. Unzip → WA_Fn-UseC_-HR-Employee-Attrition.csv
  4. Put the CSV in this folder (ai_chatbot/)

USAGE:
  python employee_loader.py --csv WA_Fn-UseC_-HR-Employee-Attrition.csv
"""

import argparse
import os
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

BATCH = 100   # rows per insert

# Rename CSV columns → DB snake_case columns
COL_MAP = {
    "Age":                        "age",
    "Attrition":                  "attrition",
    "BusinessTravel":             "business_travel",
    "DailyRate":                  "daily_rate",
    "Department":                 "department",
    "DistanceFromHome":           "distance_from_home",
    "Education":                  "education",
    "EducationField":             "education_field",
    "EmployeeNumber":             "employee_number",
    "EnvironmentSatisfaction":    "environment_satisfaction",
    "Gender":                     "gender",
    "HourlyRate":                 "hourly_rate",
    "JobInvolvement":             "job_involvement",
    "JobLevel":                   "job_level",
    "JobRole":                    "job_role",
    "JobSatisfaction":            "job_satisfaction",
    "MaritalStatus":              "marital_status",
    "MonthlyIncome":              "monthly_income",
    "NumCompaniesWorked":         "num_companies_worked",
    "OverTime":                   "overtime",
    "PercentSalaryHike":          "percent_salary_hike",
    "PerformanceRating":          "performance_rating",
    "RelationshipSatisfaction":   "relationship_satisfaction",
    "StockOptionLevel":           "stock_option_level",
    "TotalWorkingYears":          "total_working_years",
    "TrainingTimesLastYear":      "training_times_last_year",
    "WorkLifeBalance":            "work_life_balance",
    "YearsAtCompany":             "years_at_company",
    "YearsInCurrentRole":         "years_in_current_role",
    "YearsSinceLastPromotion":    "years_since_last_promotion",
    "YearsWithCurrManager":       "years_with_curr_manager",
}


def load(csv_path: str):
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        print("❌  SUPABASE_URL and SUPABASE_KEY must be set in .env")
        return

    sb = create_client(url, key)

    # ── Read & clean CSV ──────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    df = df.rename(columns=COL_MAP)

    # Keep only the columns our table has
    keep = [c for c in COL_MAP.values() if c in df.columns]
    df   = df[keep]

    # Convert numpy int64 → native int (Supabase JSON needs plain ints)
    for col in df.select_dtypes("int64").columns:
        df[col] = df[col].astype(int)

    print(f"✅  CSV ready: {len(df)} employees × {len(df.columns)} columns")

    # ── Insert in batches ─────────────────────────────────────────────────────
    records = df.to_dict(orient="records")
    ok = fail = 0

    for i in range(0, len(records), BATCH):
        chunk = records[i : i + BATCH]
        try:
            sb.table("employees").insert(chunk).execute()
            ok += len(chunk)
            print(f"  → {min(i + BATCH, len(records))}/{len(records)} rows inserted …")
        except Exception as e:
            fail += len(chunk)
            print(f"  ⚠️  Batch {i // BATCH + 1} failed — {e}")

    print(f"\n🎉  Done!  Inserted: {ok}  |  Failed: {fail}")
    if ok > 0:
        print("    Chatbot can now answer employee questions automatically.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to HR CSV file")
    load(ap.parse_args().csv)
