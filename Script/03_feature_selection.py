import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, SelectKBest, chi2
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
os.makedirs("plots", exist_ok=True)

TEXT_COLOR   = "#2E4053"
ACCENT_COLOR = "#1F618D"

df = pd.read_csv("ibm_clean.csv")

df["Attrition_Yes"] = (df["Attrition"] == "Yes").astype(int)
df.drop(columns=["Attrition"], inplace=True)
df.drop(columns=["AgeGroup", "IncomeGroup", "ExperienceGroup"], inplace=True, errors="ignore")
df = pd.get_dummies(df, drop_first=True, dtype=int)

for col in ["Yes", "Attrition_Yes_Yes"]:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

print(f"Encoded shape: {df.shape}")

X = df.drop("Attrition_Yes", axis=1)
y = df["Attrition_Yes"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("\n[1/3] Random Forest feature importance ...")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_sm, y_train_sm)

importance_df = pd.DataFrame({
    "Feature":    X.columns,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False).reset_index(drop=True)

print(importance_df.head(15).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(data=importance_df.head(20), x="Importance", y="Feature", color=ACCENT_COLOR, ax=ax)
ax.set_title("Top 20 Features — RF Importance", fontsize=14, color=TEXT_COLOR, weight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("plots/07_rf_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n[2/3] RFECV ...")

clf_rf = RandomForestClassifier(random_state=1)
rfecv  = RFECV(estimator=clf_rf, step=1, cv=5, scoring="accuracy")
rfecv.fit(X_train, y_train)

optimal_features = list(X_train.columns[rfecv.support_])
print(f"Optimal number of features: {rfecv.n_features_}")
print(f"Selected features: {optimal_features}")

scores = rfecv.cv_results_["mean_test_score"]
x_vals = list(range(1, len(scores) + 1))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x_vals, scores, color=ACCENT_COLOR)
ax.set_xlim(min(x_vals), max(x_vals))
ax.set_ylim(min(scores) - 0.01, max(scores) + 0.01)
ax.set_xlabel("Number of Features Selected")
ax.set_ylabel("CV Accuracy")
ax.set_title("RFECV — Accuracy vs Feature Count", color=TEXT_COLOR, weight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("plots/08_rfecv_curve.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n[3/3] Chi-squared SelectKBest ...")

selector = SelectKBest(chi2, k=10)
selector.fit(X_train, y_train)

feature_scores = pd.DataFrame({
    "Feature": X_train.columns,
    "Score":   selector.scores_
}).sort_values("Score", ascending=False).reset_index(drop=True)

top_10 = feature_scores.head(10)
print(top_10.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=top_10, x="Score", y="Feature", color=ACCENT_COLOR, ax=ax)
ax.set_title("Top 10 Features — Chi-Squared Score", fontsize=14, color=TEXT_COLOR, weight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("plots/09_chi2_feature_scores.png", dpi=150, bbox_inches="tight")
plt.close()

df_rfecv = df[optimal_features + ["Attrition_Yes"]]
df_rfecv.to_csv("features_rfecv.csv", index=False)
print(f"\nSaved RFECV feature set -> features_rfecv.csv  ({df_rfecv.shape[1] - 1} features)")

chi2_features = [f for f in top_10["Feature"].tolist() if f in df.columns]
df_chi2 = df[chi2_features + ["Attrition_Yes"]]
df_chi2.to_csv("features_chi2.csv", index=False)
print(f"Saved Chi2 feature set  -> features_chi2.csv  ({df_chi2.shape[1] - 1} features)")

df.to_csv("features_full.csv", index=False)
print(f"Saved full feature set  -> features_full.csv  ({df.shape[1] - 1} features)")

print("\nFeature selection complete.")