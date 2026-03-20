import pandas as pd
import numpy as np

df = pd.read_csv("C:\\Users\\LEGION\\Downloads\\files\\ibm.csv")
print(f"Raw shape: {df.shape}")

constant_cols = ["StandardHours", "EmployeeCount", "Over18", "EmployeeNumber"]
df.drop(columns=constant_cols, inplace=True)
print(f"After dropping constants: {df.shape}")

print("\nMissing values per column:")
print(df.isnull().sum()[df.isnull().sum() > 0])

df["Income_JobSatisfaction"] = df["MonthlyIncome"] * df["JobSatisfaction"]
df["JobRole_Department"] = df["JobRole"] + "_" + df["Department"]

df["AgeGroup"] = pd.cut(df["Age"], bins=[13, 18, 45, 60], labels=["Teens", "Adults", "Mid Adults"])
df["IncomeGroup"] = pd.cut(df["MonthlyIncome"], bins=[0, 5000, 10000, 15000, 20000], labels=["0-5K", "5K-10K", "10K-15K", "15K-20K"])
df["ExperienceGroup"] = pd.cut(df["TotalWorkingYears"], bins=[0, 10, 20, 30, 40], labels=["0-10", "10-20", "20-30", "30-40"])

df.to_csv("ibm_clean.csv", index=False)
print(f"\nClean dataset saved -> ibm_clean.csv  |  shape: {df.shape}")
print("\nColumns:")
print(df.columns.tolist())
