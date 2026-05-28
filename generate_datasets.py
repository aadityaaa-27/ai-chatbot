"""
Generate 4 realistic HR CSV datasets.
Run: python generate_datasets.py
Outputs:
  IT_Department_2024.csv        (200 employees)
  Sales_Team_2024.csv           (150 employees)
  Operations_2024.csv           (250 employees)
  Finance_Payroll_2024.csv      (100 employees — RESTRICTED / payroll role)
"""

import csv
import os
import random

random.seed(42)

# ── Lookup tables ─────────────────────────────────────────────────────────────

DEPT_ROLES = {
    "Research & Development": [
        "Software Engineer", "Senior Developer", "Data Analyst",
        "DevOps Engineer", "Tech Lead", "QA Engineer", "Product Manager",
    ],
    "Technology": [
        "Software Engineer", "Senior Developer", "Data Analyst",
        "DevOps Engineer", "System Architect", "Cloud Engineer",
    ],
    "Sales": [
        "Sales Executive", "Senior Sales Executive", "Sales Manager",
        "Account Manager", "Business Development Manager", "Sales Analyst",
    ],
    "Marketing": [
        "Marketing Analyst", "Digital Marketing Manager", "Content Strategist",
        "Brand Manager", "Marketing Executive", "Growth Manager",
    ],
    "Operations": [
        "Operations Manager", "Operations Analyst", "Process Engineer",
        "Shift Manager", "Ops Coordinator",
    ],
    "Manufacturing": [
        "Production Supervisor", "Quality Analyst", "Manufacturing Engineer",
        "Process Technician", "Production Manager",
    ],
    "Supply Chain": [
        "Logistics Coordinator", "Supply Chain Analyst", "Procurement Manager",
        "Inventory Specialist", "Logistics Manager",
    ],
    "Finance": [
        "Finance Manager", "Financial Analyst", "Senior Financial Analyst",
        "Budget Analyst", "Finance Executive",
    ],
    "Accounting": [
        "Accountant", "Senior Accountant", "Audit Manager",
        "Tax Analyst", "Accounts Executive",
    ],
    "Payroll": [
        "Payroll Manager", "Payroll Executive", "Compensation Analyst",
        "Benefits Administrator", "Payroll Specialist",
    ],
}

SALARY_RANGE = {
    "Research & Development": (4500, 18000),
    "Technology":             (4000, 17000),
    "Sales":                  (2500, 12000),
    "Marketing":              (3000, 11000),
    "Operations":             (2000,  8000),
    "Manufacturing":          (1800,  7000),
    "Supply Chain":           (2500,  8500),
    "Finance":                (4000, 15000),
    "Accounting":             (3500, 12000),
    "Payroll":                (3500, 10000),
}

ATTRITION = {
    "Research & Development": 0.10,
    "Technology":             0.12,
    "Sales":                  0.22,
    "Marketing":              0.18,
    "Operations":             0.16,
    "Manufacturing":          0.14,
    "Supply Chain":           0.15,
    "Finance":                0.08,
    "Accounting":             0.07,
    "Payroll":                0.06,
}

OVERTIME = {
    "Research & Development": 0.25,
    "Technology":             0.28,
    "Sales":                  0.40,
    "Marketing":              0.22,
    "Operations":             0.30,
    "Manufacturing":          0.35,
    "Supply Chain":           0.28,
    "Finance":                0.15,
    "Accounting":             0.12,
    "Payroll":                0.10,
}

EDU_FIELD = {
    "Research & Development": ["Computer Science", "Engineering", "Mathematics", "Physics"],
    "Technology":             ["Computer Science", "Information Technology", "Engineering"],
    "Sales":                  ["Marketing", "Business Administration", "Commerce", "Communication"],
    "Marketing":              ["Marketing", "Communication", "Business Administration"],
    "Operations":             ["Engineering", "Business Administration", "Industrial Management"],
    "Manufacturing":          ["Engineering", "Industrial Management", "Mechanical"],
    "Supply Chain":           ["Supply Chain Management", "Business Administration", "Logistics"],
    "Finance":                ["Finance", "Economics", "Business Administration"],
    "Accounting":             ["Accounting", "Finance", "Commerce"],
    "Payroll":                ["Human Resources", "Business Administration", "Accounting"],
}

GENDERS         = ["Male", "Female"]
MARITAL         = ["Single", "Married", "Divorced"]
TRAVEL          = ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]


# ── Row builder ───────────────────────────────────────────────────────────────

def _employee(depts: list[str], age_range: tuple[int,int] = (22, 58)) -> dict:
    dept    = random.choice(depts)
    age     = random.randint(*age_range)
    role    = random.choice(DEPT_ROLES.get(dept, ["Executive"]))

    lo, hi  = SALARY_RANGE.get(dept, (3000, 10000))
    senior  = any(kw in role.lower() for kw in
                  ["senior", "manager", "lead", "architect", "director", "cfo"])
    salary  = int(random.uniform(lo, hi) * (random.uniform(1.25, 1.45) if senior else 1.0))
    salary  = min(salary, 22000)

    yrs_co  = max(0, min(age - 22, random.randint(0, age - 21)))
    yrs_role = random.randint(0, min(yrs_co, 10))
    yrs_mgr  = random.randint(0, min(yrs_co, 10))
    yrs_promo= random.randint(0, min(yrs_co + 1, 10))

    return {
        "Age":                     age,
        "Attrition":               "Yes" if random.random() < ATTRITION.get(dept, 0.15) else "No",
        "BusinessTravel":          random.choices(TRAVEL,   weights=[40, 45, 15])[0],
        "Department":              dept,
        "DistanceFromHome":        random.randint(1, 40),
        "Education":               random.choices([1,2,3,4,5], weights=[5,15,45,25,10])[0],
        "EducationField":          random.choice(EDU_FIELD.get(dept, ["Business Administration"])),
        "EnvironmentSatisfaction": random.choices([1,2,3,4], weights=[10,20,40,30])[0],
        "Gender":                  random.choices(GENDERS, weights=[60,40])[0],
        "JobInvolvement":          random.choices([1,2,3,4], weights=[5,20,50,25])[0],
        "JobLevel":                random.choices([1,2,3,4,5], weights=[20,30,25,15,10])[0],
        "JobRole":                 role,
        "JobSatisfaction":         random.choices([1,2,3,4], weights=[10,20,40,30])[0],
        "MaritalStatus":           random.choices(MARITAL, weights=[35,50,15])[0],
        "MonthlyIncome":           salary,
        "NumCompaniesWorked":      random.randint(0, 8),
        "OverTime":                "Yes" if random.random() < OVERTIME.get(dept, 0.25) else "No",
        "PercentSalaryHike":       random.randint(11, 25),
        "PerformanceRating":       random.choices([3, 4], weights=[85, 15])[0],
        "TotalWorkingYears":       max(0, age - 22 + random.randint(-2, 3)),
        "WorkLifeBalance":         random.choices([1,2,3,4], weights=[10,20,45,25])[0],
        "YearsAtCompany":          yrs_co,
        "YearsInCurrentRole":      yrs_role,
        "YearsSinceLastPromotion": yrs_promo,
        "YearsWithCurrManager":    yrs_mgr,
    }


def _write(filename: str, depts: list[str], n: int,
           age_range: tuple[int,int] = (22, 58)):
    rows = [_employee(depts, age_range) for _ in range(n)]
    out  = os.path.join(os.path.dirname(__file__), filename)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"  OK  {filename:35s}  {n:>4} employees")


# ── Generate ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nGenerating HR datasets...\n")

    _write(
        "IT_Department_2024.csv",
        ["Research & Development", "Technology"],
        200,
    )
    _write(
        "Sales_Team_2024.csv",
        ["Sales", "Marketing"],
        150,
    )
    _write(
        "Operations_2024.csv",
        ["Operations", "Manufacturing", "Supply Chain"],
        250,
    )
    _write(
        "Finance_Payroll_2024.csv",    # RESTRICTED — payroll role only
        ["Finance", "Accounting", "Payroll"],
        100,
        age_range=(25, 60),
    )

    print("\nDone! Upload each CSV via the Upload Data tab.\n")
