import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    ConfusionMatrixDisplay, classification_report
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
os.makedirs("plots", exist_ok=True)

TEXT_COLOR  = "#2E4053"
TRAIN_COLOR = "#003f5c"
TEST_COLOR  = "#7a9ac6"

PARAM_GRIDS = {
    "Decision Tree":       {"criterion": ["gini", "entropy"], "max_depth": [5], "min_samples_split": [20], "min_samples_leaf": [5]},
    "Random Forest":       {"n_estimators": [100, 200], "max_depth": [10, 20, None]},
    "Gradient Boosting":   {"n_estimators": [100], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
    "Logistic Regression": {"C": [0.1, 1, 10], "solver": ["liblinear"], "max_iter": [1000]},

    "KNN":                 {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
}

MODELS = {
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Random Forest":       RandomForestClassifier(random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),

    "KNN":                 KNeighborsClassifier(),
}

SCALERS = {
    "StandardScaler": StandardScaler(),
    "MinMaxScaler":   MinMaxScaler(),
    "RobustScaler":   RobustScaler(),
}

FEATURE_SETS = {
    "Full":  "features_full.csv",
    "RFECV": "features_rfecv.csv",
    "Chi2":  "features_chi2.csv",
}


def evaluate_feature_set(name, path):
    df = pd.read_csv(path)
    X  = df.drop("Attrition_Yes", axis=1)
    y  = df["Attrition_Yes"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    best_params = {}
    for model_name, model in MODELS.items():
        gs = GridSearchCV(estimator=model, param_grid=PARAM_GRIDS[model_name],
                          cv=5, scoring="accuracy", verbose=0, n_jobs=-1)
        gs.fit(X_train_sm, y_train_sm)
        best_params[model_name] = gs.best_params_
        print(f"  [{name}] {model_name}: best CV acc = {gs.best_score_:.4f}  params = {gs.best_params_}")

    results = []
    for model_name, model_cls in [
        ("Decision Tree",       DecisionTreeClassifier),
        ("Random Forest",       RandomForestClassifier),
        ("Gradient Boosting",   GradientBoostingClassifier),
        ("Logistic Regression", LogisticRegression),

        ("KNN",                 KNeighborsClassifier),
    ]:
        for scaler_name, scaler in SCALERS.items():
            Xtr = scaler.fit_transform(X_train_sm)
            Xte = scaler.transform(X_test)

            kwargs = best_params[model_name].copy()
            if model_name == "Logistic Regression":
                kwargs["max_iter"] = 1000
            if model_name in ("Decision Tree", "Random Forest", "Gradient Boosting"):
                kwargs["random_state"] = 42

            mdl = model_cls(**kwargs)
            mdl.fit(Xtr, y_train_sm)

            results.append({
                "FeatureSet": name,
                "Model":      model_name,
                "Scaler":     scaler_name,
                "Train Acc":  round(accuracy_score(y_train_sm, mdl.predict(Xtr)), 4),
                "Test Acc":   round(accuracy_score(y_test, mdl.predict(Xte)), 4),
                "F1":         round(f1_score(y_test, mdl.predict(Xte), zero_division=0), 4),
            })

    return pd.DataFrame(results)


all_results = []
for fs_name, fs_path in FEATURE_SETS.items():
    print(f"\n=== Feature set: {fs_name} ===")
    all_results.append(evaluate_feature_set(fs_name, fs_path))

results_df = pd.concat(all_results, ignore_index=True)
results_df.sort_values("Test Acc", ascending=False, inplace=True)
results_df.to_csv("model_results.csv", index=False)

print("\n── Top 10 configurations by Test Accuracy ──")
print(results_df.head(10).to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for ax, fs_name in zip(axes, FEATURE_SETS.keys()):
    sub   = results_df[(results_df["FeatureSet"] == fs_name) & (results_df["Scaler"] == "RobustScaler")]
    x     = np.arange(len(sub))
    bar_w = 0.35
    ax.bar(x - bar_w / 2, sub["Train Acc"], bar_w, label="Train", color=TRAIN_COLOR)
    ax.bar(x + bar_w / 2, sub["Test Acc"],  bar_w, label="Test",  color=TEST_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["Model"], rotation=30, ha="right", fontsize=9)
    ax.set_title(f"{fs_name} Features — RobustScaler", color=TEXT_COLOR, weight="bold")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.set_facecolor("#fcfcfc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("plots/10_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n── Final model evaluation ──")

df_best = pd.read_csv("features_rfecv.csv")
X = df_best.drop("Attrition_Yes", axis=1)
y = df_best["Attrition_Yes"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

scaler = RobustScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

model = LogisticRegression(C=0.58, penalty="l1", solver="liblinear", max_iter=1000)
model.fit(X_train_sc, y_train)

y_pred = model.predict(X_test_sc)
print(classification_report(y_test, y_pred, target_names=["Retained", "Attrition"]))

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=["Retained", "Attrition"]).plot(ax=ax, colorbar=False)
ax.set_title("Confusion Matrix — Logistic Regression (RFECV features)", color=TEXT_COLOR, weight="bold")
plt.tight_layout()
plt.savefig("plots/11_confusion_matrix_final.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nModelling complete. Results saved -> model_results.csv")