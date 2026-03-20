-- IBM HR Attrition Dataset — KPI Views and Risk Profiling
-- Table: hr.ibm_attrition

CREATE OR REPLACE VIEW hr.v_attrition_kpis AS
SELECT
    COUNT(*)                                                          AS total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)               AS total_attrition,
    ROUND(SUM(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 2) AS overall_attrition_rate_pct,
    ROUND(AVG(MonthlyIncome), 2)                                      AS avg_monthly_income,
    ROUND(AVG(CASE WHEN Attrition = 'Yes' THEN MonthlyIncome END), 2) AS avg_income_attrited,
    ROUND(AVG(CASE WHEN Attrition = 'No'  THEN MonthlyIncome END), 2) AS avg_income_retained
FROM hr.ibm_attrition;


CREATE OR REPLACE VIEW hr.v_role_attrition AS
SELECT
    Department,
    JobRole,
    COUNT(*)                                                          AS headcount,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)               AS attrition_count,
    ROUND(SUM(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 2) AS attrition_rate_pct,
    ROUND(AVG(MonthlyIncome), 2)                                      AS avg_income,
    ROUND(AVG(JobSatisfaction), 2)                                    AS avg_job_satisfaction,
    ROUND(AVG(YearsAtCompany), 2)                                     AS avg_tenure
FROM hr.ibm_attrition
GROUP BY Department, JobRole
ORDER BY attrition_rate_pct DESC;


CREATE OR REPLACE VIEW hr.v_satisfaction_summary AS
SELECT
    Attrition,
    JobRole,
    ROUND(AVG(JobSatisfaction), 2)          AS avg_job_satisfaction,
    ROUND(AVG(EnvironmentSatisfaction), 2)  AS avg_env_satisfaction,
    ROUND(AVG(RelationshipSatisfaction), 2) AS avg_rel_satisfaction,
    ROUND(AVG(WorkLifeBalance), 2)          AS avg_work_life_balance,
    ROUND(AVG(JobInvolvement), 2)           AS avg_job_involvement,
    COUNT(*)                                AS employee_count
FROM hr.ibm_attrition
GROUP BY Attrition, JobRole
ORDER BY JobRole, Attrition;


SELECT
    EmployeeNumber, Age, JobRole, Department, MonthlyIncome,
    OverTime, BusinessTravel, MaritalStatus, YearsAtCompany,
    YearsSinceLastPromotion, JobSatisfaction, EnvironmentSatisfaction,
    WorkLifeBalance, Attrition
FROM hr.ibm_attrition
WHERE
    OverTime          = 'Yes'
    AND MonthlyIncome < 5000
    AND MaritalStatus = 'Single'
    AND BusinessTravel IN ('Travel_Frequently', 'Travel_Rarely')
    AND JobSatisfaction   <= 2
    AND WorkLifeBalance   <= 2
ORDER BY MonthlyIncome ASC;


SELECT
    OverTime,
    CASE
        WHEN MonthlyIncome < 5000  THEN '0-5K'
        WHEN MonthlyIncome < 10000 THEN '5K-10K'
        WHEN MonthlyIncome < 15000 THEN '10K-15K'
        ELSE '15K-20K'
    END                                                               AS income_group,
    COUNT(*)                                                          AS total,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)               AS attrition_count,
    ROUND(SUM(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 2) AS attrition_rate_pct
FROM hr.ibm_attrition
GROUP BY OverTime, income_group
ORDER BY attrition_rate_pct DESC;


SELECT
    StockOptionLevel,
    COUNT(*)                                                          AS total,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)               AS attrition_count,
    ROUND(SUM(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 2) AS attrition_rate_pct
FROM hr.ibm_attrition
GROUP BY StockOptionLevel
ORDER BY StockOptionLevel;


SELECT
    CASE
        WHEN YearsSinceLastPromotion = 0 THEN 'Recently Promoted'
        WHEN YearsSinceLastPromotion <= 2 THEN '1-2 Years'
        WHEN YearsSinceLastPromotion <= 5 THEN '3-5 Years'
        ELSE '5+ Years'
    END                                                               AS promotion_gap,
    COUNT(*)                                                          AS total,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)               AS attrition_count,
    ROUND(SUM(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 2) AS attrition_rate_pct
FROM hr.ibm_attrition
GROUP BY promotion_gap
ORDER BY attrition_rate_pct DESC;


SELECT
    TrainingTimesLastYear,
    COUNT(*)                                                          AS total,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)               AS attrition_count,
    ROUND(SUM(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 2) AS attrition_rate_pct
FROM hr.ibm_attrition
GROUP BY TrainingTimesLastYear
ORDER BY TrainingTimesLastYear;
