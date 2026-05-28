-- ============================================================
-- Fill NULL values in the 'Employes' dataset
-- Run in Supabase → SQL Editor (uses run_employee_write RPC
-- or paste directly into the SQL editor)
-- ============================================================

-- 1. Gender (deterministic 60/40 based on age+salary)
UPDATE employees
SET gender = CASE
    WHEN (age + COALESCE(monthly_income, 0)) % 5 < 3 THEN 'Male'
    ELSE 'Female'
END
WHERE source_file = 'Employes' AND gender IS NULL;

-- 2. Job Role — based on department
UPDATE employees
SET job_role = CASE department
    WHEN 'HR'         THEN CASE (age % 3)
                           WHEN 0 THEN 'HR Manager'
                           WHEN 1 THEN 'HR Executive'
                           ELSE        'HR Recruiter' END
    WHEN 'Finance'    THEN CASE (age % 3)
                           WHEN 0 THEN 'Finance Manager'
                           WHEN 1 THEN 'Financial Analyst'
                           ELSE        'Accountant' END
    WHEN 'IT'         THEN CASE (age % 4)
                           WHEN 0 THEN 'Software Engineer'
                           WHEN 1 THEN 'Senior Developer'
                           WHEN 2 THEN 'Data Analyst'
                           ELSE        'DevOps Engineer' END
    WHEN 'Sales'      THEN CASE (age % 3)
                           WHEN 0 THEN 'Sales Manager'
                           WHEN 1 THEN 'Sales Executive'
                           ELSE        'Account Manager' END
    WHEN 'Marketing'  THEN CASE (age % 3)
                           WHEN 0 THEN 'Marketing Manager'
                           WHEN 1 THEN 'Marketing Analyst'
                           ELSE        'Brand Manager' END
    WHEN 'Operations' THEN CASE (age % 3)
                           WHEN 0 THEN 'Operations Manager'
                           WHEN 1 THEN 'Operations Analyst'
                           ELSE        'Process Engineer' END
    ELSE CASE (age % 3)
         WHEN 0 THEN 'Senior Executive'
         WHEN 1 THEN 'Executive'
         ELSE        'Associate' END
END
WHERE source_file = 'Employes' AND job_role IS NULL;

-- 3. Attrition (~15% rate, deterministic)
UPDATE employees
SET attrition = CASE
    WHEN (age + COALESCE(monthly_income, 0)) % 7 = 0 THEN 'Yes'
    ELSE 'No'
END
WHERE source_file = 'Employes' AND attrition IS NULL;

-- 4. Overtime (~28% rate, deterministic)
UPDATE employees
SET overtime = CASE
    WHEN age % 4 = 0 THEN 'Yes'
    ELSE 'No'
END
WHERE source_file = 'Employes' AND overtime IS NULL;

-- 5. Education level (1-5)
UPDATE employees
SET education = CASE (COALESCE(monthly_income, 5000) / 2000 % 5)
    WHEN 0 THEN 2
    WHEN 1 THEN 3
    WHEN 2 THEN 3
    WHEN 3 THEN 4
    ELSE 3
END
WHERE source_file = 'Employes' AND education IS NULL;

-- 6. Job Satisfaction (1=Low … 4=Very High)
UPDATE employees
SET job_satisfaction = CASE (age % 4)
    WHEN 0 THEN 2
    WHEN 1 THEN 3
    WHEN 2 THEN 3
    ELSE 4
END
WHERE source_file = 'Employes' AND job_satisfaction IS NULL;

-- 7. Work-Life Balance (1=Bad … 4=Best)
UPDATE employees
SET work_life_balance = CASE (COALESCE(monthly_income, 5000) % 4)
    WHEN 0 THEN 2
    WHEN 1 THEN 3
    WHEN 2 THEN 3
    ELSE 4
END
WHERE source_file = 'Employes' AND work_life_balance IS NULL;

-- 8. Performance Rating (3=Excellent, 4=Outstanding — 85/15 split)
UPDATE employees
SET performance_rating = CASE
    WHEN (age + COALESCE(monthly_income, 0)) % 7 = 0 THEN 4
    ELSE 3
END
WHERE source_file = 'Employes' AND performance_rating IS NULL;

-- 9. Marital Status
UPDATE employees
SET marital_status = CASE (age % 3)
    WHEN 0 THEN 'Single'
    WHEN 1 THEN 'Married'
    ELSE        'Married'
END
WHERE source_file = 'Employes' AND marital_status IS NULL;

-- 10. Business Travel
UPDATE employees
SET business_travel = CASE (COALESCE(monthly_income, 5000) % 3)
    WHEN 0 THEN 'Non-Travel'
    WHEN 1 THEN 'Travel_Rarely'
    ELSE        'Travel_Frequently'
END
WHERE source_file = 'Employes' AND business_travel IS NULL;

-- 11. Environment Satisfaction (1-4)
UPDATE employees
SET environment_satisfaction = CASE (age % 4)
    WHEN 0 THEN 3
    WHEN 1 THEN 3
    WHEN 2 THEN 4
    ELSE 2
END
WHERE source_file = 'Employes' AND environment_satisfaction IS NULL;

-- 12. Distance From Home (1-40 miles)
UPDATE employees
SET distance_from_home = (age % 35) + 1
WHERE source_file = 'Employes' AND distance_from_home IS NULL;

-- 13. Num Companies Worked
UPDATE employees
SET num_companies_worked = CASE
    WHEN age < 28 THEN 0
    WHEN age < 35 THEN 1
    WHEN age < 45 THEN (age % 3)
    ELSE (age % 5)
END
WHERE source_file = 'Employes' AND num_companies_worked IS NULL;

-- 14. Total Working Years
UPDATE employees
SET total_working_years = GREATEST(0, age - 22 + (COALESCE(monthly_income, 5000) % 3) - 1)
WHERE source_file = 'Employes' AND total_working_years IS NULL;

-- 15. Years At Company
UPDATE employees
SET years_at_company = GREATEST(0, LEAST(total_working_years, (age % 15)))
WHERE source_file = 'Employes' AND years_at_company IS NULL;

-- 16. Job Level (1-5)
UPDATE employees
SET job_level = CASE
    WHEN COALESCE(monthly_income, 0) < 3000  THEN 1
    WHEN COALESCE(monthly_income, 0) < 6000  THEN 2
    WHEN COALESCE(monthly_income, 0) < 10000 THEN 3
    WHEN COALESCE(monthly_income, 0) < 15000 THEN 4
    ELSE 5
END
WHERE source_file = 'Employes' AND job_level IS NULL;

-- Confirm
SELECT
    COUNT(*)                                        AS total_rows,
    COUNT(*) FILTER (WHERE gender          IS NULL) AS null_gender,
    COUNT(*) FILTER (WHERE job_role        IS NULL) AS null_job_role,
    COUNT(*) FILTER (WHERE attrition       IS NULL) AS null_attrition,
    COUNT(*) FILTER (WHERE overtime        IS NULL) AS null_overtime,
    COUNT(*) FILTER (WHERE job_satisfaction IS NULL) AS null_satisfaction
FROM employees
WHERE source_file = 'Employes';
