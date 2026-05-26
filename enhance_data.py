"""
enhance_data.py — Double the employee dataset with 6 new departments + holidays column
Run: python enhance_data.py --csv WA_Fn-UseC_-HR-Employee-Attrition.csv
"""
import argparse
import os
import random
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

BATCH    = 100
HOLIDAYS = 20

# New departments + realistic job roles
NEW_DEPTS = {
    "Finance": [
        "Financial Analyst", "Senior Accountant", "Finance Manager",
        "Budget Analyst", "Treasury Analyst",
    ],
    "Marketing": [
        "Marketing Analyst", "Brand Manager", "Digital Marketing Specialist",
        "Content Strategist", "Marketing Manager",
    ],
    "Information Technology": [
        "Software Engineer", "IT Support Specialist", "Data Analyst",
        "Network Engineer", "IT Manager",
    ],
    "Operations": [
        "Operations Analyst", "Process Improvement Specialist", "Operations Manager",
        "Logistics Coordinator", "Supply Chain Analyst",
    ],
    "Legal": [
        "Legal Counsel", "Compliance Officer", "Contracts Manager",
        "Paralegal", "Legal Analyst",
    ],
    "Customer Support": [
        "Customer Service Representative", "Support Team Lead",
        "Technical Support Specialist", "Customer Success Manager", "Support Manager",
    ],
}

COL_MAP = {
    "Age": "age", "Attrition": "attrition", "BusinessTravel": "business_travel",
    "DailyRate": "daily_rate", "Department": "department",
    "DistanceFromHome": "distance_from_home", "Education": "education",
    "EducationField": "education_field", "EmployeeNumber": "employee_number",
    "EnvironmentSatisfaction": "environment_satisfaction", "Gender": "gender",
    "HourlyRate": "hourly_rate", "JobInvolvement": "job_involvement",
    "JobLevel": "job_level", "JobRole": "job_role",
    "JobSatisfaction": "job_satisfaction", "MaritalStatus": "marital_status",
    "MonthlyIncome": "monthly_income", "NumCompaniesWorked": "num_companies_worked",
    "OverTime": "overtime", "PercentSalaryHike": "percent_salary_hike",
    "PerformanceRating": "performance_rating",
    "RelationshipSatisfaction": "relationship_satisfaction",
    "StockOptionLevel": "stock_option_level", "TotalWorkingYears": "total_working_years",
    "TrainingTimesLastYear": "training_times_last_year",
    "WorkLifeBalance": "work_life_balance", "YearsAtCompany": "years_at_company",
    "YearsInCurrentRole": "years_in_current_role",
    "YearsSinceLastPromotion": "years_since_last_promotion",
    "YearsWithCurrManager": "years_with_curr_manager",
}


def enhance(csv_path: str):
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        print("❌  SUPABASE_URL and SUPABASE_KEY must be set in .env")
        return

    sb = create_client(url, key)

    # ── Read original CSV ─────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    df = df.rename(columns=COL_MAP)
    keep = [c for c in COL_MAP.values() if c in df.columns]
    df   = df[keep]
    for col in df.select_dtypes("int64").columns:
        df[col] = df[col].astype(int)

    print(f"✅  Original CSV: {len(df)} employees loaded")

    # ── Generate new employees for new departments ────────────────────────────
    depts      = list(NEW_DEPTS.keys())
    max_emp    = int(df["employee_number"].max())
    new_rows   = []

    for i in range(len(df)):                          # one new row per original row
        base = df.iloc[i].to_dict()
        dept = depts[i % len(depts)]                  # cycle through 6 departments
        role = random.choice(NEW_DEPTS[dept])

        new_rows.append({
            "age":                       max(18, min(60, base["age"] + random.randint(-3, 3))),
            "attrition":                 random.choices(["Yes", "No"], weights=[16, 84])[0],
            "business_travel":           random.choices(
                                             ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
                                             weights=[30, 54, 16])[0],
            "daily_rate":                max(100, int(base["daily_rate"] * random.uniform(0.85, 1.15))),
            "department":                dept,
            "distance_from_home":        max(1, base["distance_from_home"] + random.randint(-5, 5)),
            "education":                 base["education"],
            "education_field":           random.choice([
                                             "Life Sciences", "Medical", "Marketing",
                                             "Technical Degree", "Human Resources", "Other"]),
            "employee_number":           max_emp + i + 1,
            "environment_satisfaction":  random.randint(1, 4),
            "gender":                    random.choice(["Male", "Female"]),
            "hourly_rate":               max(30, int(base["hourly_rate"] * random.uniform(0.85, 1.15))),
            "job_involvement":           random.randint(1, 4),
            "job_level":                 base["job_level"],
            "job_role":                  role,
            "job_satisfaction":          random.randint(1, 4),
            "marital_status":            random.choices(
                                             ["Single", "Married", "Divorced"],
                                             weights=[32, 46, 22])[0],
            "monthly_income":            max(2000, int(base["monthly_income"] * random.uniform(0.85, 1.15))),
            "num_companies_worked":      base["num_companies_worked"],
            "overtime":                  random.choices(["Yes", "No"], weights=[28, 72])[0],
            "percent_salary_hike":       random.randint(11, 25),
            "performance_rating":        random.choices([3, 4], weights=[84, 16])[0],
            "relationship_satisfaction": random.randint(1, 4),
            "stock_option_level":        random.randint(0, 3),
            "total_working_years":       max(0, base["total_working_years"] + random.randint(-2, 2)),
            "training_times_last_year":  random.randint(0, 6),
            "work_life_balance":         random.randint(1, 4),
            "years_at_company":          max(0, base["years_at_company"] + random.randint(-2, 2)),
            "years_in_current_role":     max(0, base["years_in_current_role"] + random.randint(-1, 1)),
            "years_since_last_promotion":max(0, base["years_since_last_promotion"]),
            "years_with_curr_manager":   max(0, base["years_with_curr_manager"] + random.randint(-1, 1)),
            "holidays":                  HOLIDAYS,
        })

    # ── Insert in batches ─────────────────────────────────────────────────────
    print(f"📤  Inserting {len(new_rows)} new employees across {len(depts)} departments…")
    ok = fail = 0

    for i in range(0, len(new_rows), BATCH):
        chunk = new_rows[i:i + BATCH]
        try:
            sb.table("employees").insert(chunk).execute()
            ok   += len(chunk)
            done  = min(i + BATCH, len(new_rows))
            print(f"  → {done}/{len(new_rows)} rows inserted…")
        except Exception as e:
            fail += len(chunk)
            print(f"  ⚠️  Batch {i // BATCH + 1} failed — {e}")

    print(f"\n🎉  Done!  New rows inserted: {ok}  |  Failed: {fail}")
    print(f"    Total employees in DB: ~{1470 + ok:,}")
    print("    New departments added: " + ", ".join(depts))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to original HR CSV")
    enhance(ap.parse_args().csv)
