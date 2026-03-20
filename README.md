# Employee Attrition Prediction — IBM HR Dataset

End-to-end attrition analysis and classification project on the IBM HR Analytics dataset (1,470 employee records, 35 attributes). The pipeline covers data cleaning, exploratory analysis, feature selection via three methods, and multi-model classification with hyperparameter tuning.

---

## Repository Structure

```
employee-attrition/
├── data/
│   ├── ibm.csv
│   ├── ibm_clean.csv
│   ├── features_full.csv
│   ├── features_rfecv.csv
│   └── features_chi2.csv
├── scripts/
│   ├── 01_data_cleaning.py
│   ├── 02_eda.py
│   ├── 03_feature_selection.py
│   ├── 04_modelling.py
│   └── 05_dashboard.py
├── sql/
│   ├── 01_exploration.sql
│   └── 02_attrition_analysis.sql
├── plots/
├── results/
│   └── model_results.csv
└── README.md
```

---

## Pipeline

### 1. Data Cleaning

Drops four constant/identifier columns, engineers an interaction feature (monthly income × job satisfaction), creates a compound role identifier, and bins age, income, and experience into labelled groups for EDA.

### 2. Exploratory Data Analysis

Produces 6 saved figures covering KDE distributions, categorical count plots, satisfaction rating heatmaps, full and filtered correlation matrices, and attrition rate by income bracket.

Key finding: 60% of attrition occurred in the lowest income bracket (under $5,000/month).

### 3. Feature Selection

Three methods applied independently:

| Method | Output |
|---|---|
| Random Forest importance (top 20) | plots/07_rf_feature_importance.png |
| RFECV (5-fold CV) | data/features_rfecv.csv |
| Chi-squared SelectKBest (top 10) | data/features_chi2.csv |

### 4. Modelling

- Class imbalance handled with SMOTE on training split only
- GridSearchCV with 5-fold CV across all models
- Models: Logistic Regression, Random Forest, Decision Tree, Gradient Boosting, KNN
- Scalers compared: StandardScaler, MinMaxScaler, RobustScaler
- Feature sets compared: Full, RFECV, Chi2

Best configuration: Logistic Regression (C=0.58, L1 penalty, liblinear) + RobustScaler on RFECV features — 91% accuracy, 0.90 F1-score.

### 5. Dashboard

Generates plots/dashboard.png — a matplotlib dashboard covering KPI cards, attrition by department/role, overtime impact, marital status, and income group breakdowns.

---

## SQL Analysis

Standard SQL (PostgreSQL syntax), adaptable to any relational database.

sql/01_exploration.sql — attrition rate by department, role, overtime, travel frequency, marital status, income bucket, and tenure.

sql/02_attrition_analysis.sql — reusable KPI views, high-risk employee profile query, promotion gap analysis, and training frequency breakdown.

---

## Usage

Run scripts in order from the project root:

```bash
python scripts/01_data_cleaning.py
python scripts/02_eda.py
python scripts/03_feature_selection.py
python scripts/04_modelling.py
python scripts/05_dashboard.py
```

---

## Requirements

```bash
pip install -r requirements.txt
```
