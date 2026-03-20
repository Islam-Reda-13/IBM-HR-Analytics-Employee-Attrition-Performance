-- IBM HR Attrition Dataset — Initial Exploration
-- Table: hr.ibm_attrition

SELECT
    Attrition,
    COUNT(*)                                            AS employee_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct
FROM hr.ibm_attrition
GROUP BY Attrition
ORDER BY employee_count DESC;


SELECT
    COUNT(DISTINCT StandardHours)  AS distinct_standard_hours,
    COUNT(DISTINCT EmployeeCount)  AS distinct_employee_count,
    COUNT(DISTINCT Over18)         AS distinct_over18
FROM hr.ibm_attrition;


SELECT
    Department,
    COUNT(*)                                                   AS total,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)        AS attrition_count,
    ROUND(SUM(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 2) AS attrition_rate_pct
FROM hr.ibm_attrition
GROUP BY Department
ORDER BY attrition_rate_pct DESC;


SELECT
    JobRole,
    COUNT(*)                                                   AS total,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)        AS attrition_count,
    ROUND(SUM(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 2) AS attrition_rate_pct
FROM hr.ibm_attrition
GROUP BY JobRole
ORDER BY attrition_rate_pct DESC;


SELECT
    OverTime,
    COUNT(*)                                                   AS total,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)        AS attrition_count,
    ROUND(SUM(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 2) AS attrition_rate_pct
FROM hr.ibm_attrition
GROUP BY OverTime
ORDER BY attrition_rate_pct DESC;


SELECT
    BusinessTravel,
    COUNT(*)                                                   AS total,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)        AS attrition_count,
    ROUND(SUM(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 2) AS attrition_rate_pct
FROM hr.ibm_attrition
GROUP BY BusinessTravel
ORDER BY attrition_rate_pct DESC;


SELECT
    MaritalStatus,
    COUNT(*)                                                   AS total,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)        AS attrition_count,
    ROUND(SUM(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 2) AS attrition_rate_pct
FROM hr.ibm_attrition
GROUP BY MaritalStatus
ORDER BY attrition_rate_pct DESC;


SELECT
    Attrition,
    ROUND(AVG(MonthlyIncome), 2) AS avg_monthly_income,
    MIN(MonthlyIncome)           AS min_monthly_income,
    MAX(MonthlyIncome)           AS max_monthly_income,
    ROUND(AVG(DailyRate), 2)     AS avg_daily_rate,
    ROUND(AVG(HourlyRate), 2)    AS avg_hourly_rate
FROM hr.ibm_attrition
GROUP BY Attrition;


SELECT
    Attrition,
    ROUND(AVG(JobSatisfaction), 2)          AS avg_job_satisfaction,
    ROUND(AVG(EnvironmentSatisfaction), 2)  AS avg_env_satisfaction,
    ROUND(AVG(RelationshipSatisfaction), 2) AS avg_rel_satisfaction,
    ROUND(AVG(WorkLifeBalance), 2)          AS avg_work_life_balance,
    ROUND(AVG(JobInvolvement), 2)           AS avg_job_involvement
FROM hr.ibm_attrition
GROUP BY Attrition;


SELECT
    Attrition,
    ROUND(AVG(YearsAtCompany), 2)          AS avg_years_at_company,
    ROUND(AVG(YearsInCurrentRole), 2)      AS avg_years_in_role,
    ROUND(AVG(YearsSinceLastPromotion), 2) AS avg_years_since_promo,
    ROUND(AVG(YearsWithCurrManager), 2)    AS avg_years_with_manager,
    ROUND(AVG(TotalWorkingYears), 2)       AS avg_total_exp
FROM hr.ibm_attrition
GROUP BY Attrition;


SELECT
    Attrition,
    ROUND(AVG(Age), 2) AS avg_age,
    MIN(Age)           AS min_age,
    MAX(Age)           AS max_age
FROM hr.ibm_attrition
GROUP BY Attrition;


SELECT
    CASE
        WHEN MonthlyIncome < 5000  THEN '0-5K'
        WHEN MonthlyIncome < 10000 THEN '5K-10K'
        WHEN MonthlyIncome < 15000 THEN '10K-15K'
        ELSE '15K-20K'
    END                                                        AS income_group,
    COUNT(*)                                                   AS total,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)        AS attrition_count,
    ROUND(SUM(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 2) AS attrition_rate_pct
FROM hr.ibm_attrition
GROUP BY income_group
ORDER BY attrition_rate_pct DESC;
