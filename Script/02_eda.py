import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
import squarify
import warnings

warnings.filterwarnings("ignore")
os.makedirs("plots", exist_ok=True)

ATTRITION_COLOR  = "#9a09ae"
RETENTION_COLOR  = "#15ae68"
TEXT_COLOR       = "#2E4053"
ACCENT_COLOR     = "#1F618D"
BACKGROUND_COLOR = "#fcfcfc"

df = pd.read_csv("ibm_clean.csv")
print(f"Dataset shape: {df.shape}")
print(f"\nAttrition distribution:\n{df['Attrition'].value_counts()}")

num_columns = df.select_dtypes(include=["number"]).columns
n_cols = 5
n_rows = (len(num_columns) // n_cols) + (len(num_columns) % n_cols > 0)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(45, n_rows * 5))
axes = axes.flatten()

for i, col in enumerate(num_columns):
    sns.kdeplot(ax=axes[i], x=col, hue="Attrition", data=df, fill=True,
                palette=[ATTRITION_COLOR, RETENTION_COLOR])
    axes[i].set_title(f"{col} vs. Attrition",
                      fontdict={"family": "Serif", "color": TEXT_COLOR, "size": 16, "weight": "bold"})
    axes[i].set_facecolor(BACKGROUND_COLOR)
    axes[i].spines["top"].set_visible(False)
    axes[i].spines["right"].set_visible(False)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig("plots/01_numeric_kde_attrition.png", dpi=150, bbox_inches="tight")
plt.close()

object_columns = df.select_dtypes(include=["object"]).columns.difference(["Attrition"])
n_cols = 3
n_rows = (len(object_columns) // n_cols) + (len(object_columns) % n_cols > 0)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 5))
axes = axes.flatten()

for i, col in enumerate(object_columns):
    sns.countplot(ax=axes[i], x=col, data=df, hue="Attrition",
                  palette=[ATTRITION_COLOR, RETENTION_COLOR])
    axes[i].set_title(col, fontdict={"family": "Serif", "color": TEXT_COLOR, "weight": "bold", "size": 16})
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")
    axes[i].tick_params(axis="x", rotation=45)
    axes[i].spines["top"].set_visible(False)
    axes[i].spines["right"].set_visible(False)
    axes[i].set_facecolor(BACKGROUND_COLOR)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig("plots/02_categorical_countplot_attrition.png", dpi=150, bbox_inches="tight")
plt.close()

satisfaction_features = [
    "EnvironmentSatisfaction", "JobInvolvement", "JobSatisfaction",
    "PerformanceRating", "RelationshipSatisfaction", "WorkLifeBalance"
]

df_encoded = df.copy()
df_encoded["Attrition_Yes"] = (df_encoded["Attrition"] == "Yes").astype(int)

fig = plt.figure(figsize=(18, 12), dpi=100)
gs = fig.add_gridspec(2, 3)
gs.update(wspace=0.3, hspace=0.4)
axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

cmap_attr = matplotlib.colors.LinearSegmentedColormap.from_list("attr", [ATTRITION_COLOR, "#f6f5f5"], N=256)

for ax, feat in zip(axes, satisfaction_features):
    pivot = df_encoded.groupby(feat)["Attrition_Yes"].mean().to_frame().T
    sns.heatmap(pivot, ax=ax, cmap=cmap_attr, annot=True, fmt=".2f", cbar=False)
    ax.set_title(feat, fontsize=12, color=TEXT_COLOR, weight="bold")
    ax.set_ylabel("")

plt.suptitle("Attrition Rate by Satisfaction & Rating Scores", fontsize=16, color=TEXT_COLOR, weight="bold")
plt.savefig("plots/03_satisfaction_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

numeric_cols = df_encoded.select_dtypes(include="number").columns
corr_matrix  = df_encoded[numeric_cols].corr()

plt.figure(figsize=(20, 20))
sns.heatmap(corr_matrix, annot=True, cmap="cividis", fmt=".1f")
plt.title("Full Correlation Matrix", fontsize=16, color=TEXT_COLOR)
plt.savefig("plots/04_correlation_matrix_full.png", dpi=150, bbox_inches="tight")
plt.close()

filtered = corr_matrix[(corr_matrix >= 0.7) | (corr_matrix <= -0.7)]
plt.figure(figsize=(13, 8))
sns.heatmap(filtered, annot=True, cmap="cividis", fmt=".1f")
plt.title("Strong Correlations (|r| ≥ 0.7)", fontsize=14, color=TEXT_COLOR)
plt.savefig("plots/05_correlation_matrix_filtered.png", dpi=150, bbox_inches="tight")
plt.close()

attrition_by_income = (
    df_encoded.groupby("IncomeGroup")["Attrition_Yes"]
    .mean()
    .reset_index()
    .rename(columns={"Attrition_Yes": "AttritionRate"})
)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=attrition_by_income, x="IncomeGroup", y="AttritionRate",
            palette=[ATTRITION_COLOR] * len(attrition_by_income), ax=ax)
ax.set_title("Attrition Rate by Monthly Income Group", fontsize=14, color=TEXT_COLOR, weight="bold")
ax.set_xlabel("Income Group (USD)", color=TEXT_COLOR)
ax.set_ylabel("Attrition Rate", color=TEXT_COLOR)
ax.set_facecolor(BACKGROUND_COLOR)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("plots/06_attrition_by_income_group.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nEDA complete. All plots saved to /plots.")
