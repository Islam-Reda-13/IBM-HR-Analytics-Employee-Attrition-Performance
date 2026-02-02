# Employee Attrition Prediction System

A machine learning project that predicts employee attrition using IBM HR Analytics data. The system leverages advanced data preprocessing, feature engineering, and multiple classification algorithms to identify employees at risk of leaving the organization.

##  Project Overview

This project analyzes employee data to predict attrition patterns and provides actionable insights for HR decision-making. By identifying key factors contributing to employee turnover, organizations can implement targeted retention strategies.

**Key Results:**
- 91% test accuracy with SVM classifier
- 0.90 F1-score on minority class prediction
- 95% test recall achieved through SMOTE oversampling
- Identified salary and job level as top attrition predictors

##  Dataset

- **Source:** IBM HR Analytics Employee Attrition Dataset
- **Size:** 1,470 employee records
- **Features:** 35 attributes including demographics, job characteristics, compensation, and satisfaction metrics
- **Target Variable:** Attrition (Yes/No) - binary classification problem
- **Class Imbalance:** Handled using SMOTE oversampling techniques

##  Technologies Used

### Core Libraries
- **Python 3.x** - Primary programming language
- **Pandas & NumPy** - Data manipulation and numerical operations
- **scikit-learn** - Machine learning algorithms and preprocessing
- **imbalanced-learn** - SMOTE and oversampling techniques

### Visualization
- **Matplotlib** - Statistical plotting and visualizations
- **Seaborn** - Enhanced statistical graphics
- **Plotly Express** - Interactive visualizations
- **PyWaffle** - Waffle chart visualizations
- **Squarify** - Treemap visualizations

### Machine Learning Models
- Support Vector Machine (SVM)
- Logistic Regression
- Naive Bayes

##  Key Features

### 1. Data Preprocessing
- Missing value detection and handling
- Removal of constant and irrelevant features
- Outlier treatment using IQR method
- Categorical encoding via one-hot encoding
- Feature scaling (StandardScaler, MinMaxScaler, RobustScaler)

### 2. Exploratory Data Analysis (EDA)
- Distribution analysis of numerical and categorical variables
- Attrition patterns across age groups, salary ranges, and job levels
- Correlation analysis to identify multicollinearity
- Interactive visualizations including:
  - KDE plots for continuous variables
  - Waffle charts for satisfaction metrics
  - Treemaps for categorical distributions
  - Dumbbell plots for comparative analysis

### 3. Feature Engineering
- Created interaction features:
  - `Income_JobSatisfaction` - Combined monetary and satisfaction factors
  - `JobRole_Department` - Cross-categorical relationship
- Binned continuous variables:
  - Age categories (Teens, Adults, Mid Adults)
  - Working years ranges (0-10, 10-20, 20-30, 30-40)
  - Monthly income brackets (0-5K, 5K-10K, 10K-15K, 15K-20K)

### 4. Handling Class Imbalance
Evaluated 5 oversampling techniques:
- **SMOTE** (Selected for final model)
- ADASYN
- RandomOverSampler
- BorderlineSMOTE
- SVMSMOTE

### 5. Multicollinearity Management
Removed highly correlated features (correlation > 0.7):
- JobLevel
- YearsAtCompany
- YearsInCurrentRole
- PerformanceRating
- TotalWorkingYears

##  Model Performance

### Final Model: SVM with RobustScaler and SMOTE

```
Classification Report:
                 precision    recall  f1-score   support
Class 0 (No)        0.87      0.96      0.91       250
Class 1 (Yes)       0.95      0.86      0.90       244

Accuracy:           0.91
Macro Avg:          0.91      0.91      0.91
Weighted Avg:       0.91      0.91      0.91
```

**Training Accuracy:** 95.08%  
**Test Accuracy:** 90.89%

### Model Comparison Results

| Model | Scaler | Test Accuracy | Best For |
|-------|--------|---------------|----------|
| SVM | RobustScaler | 90.89% | **Final Selection** |
| SVM | StandardScaler | 89.47% | Balanced performance |
| Logistic Regression | MinMaxScaler | 87.23% | Interpretability |
| Naive Bayes | StandardScaler | 82.14% | Baseline comparison |

##  Key Insights

1. **Salary Impact:** 60% of attrition occurs in the 0-5K monthly income bracket
2. **Job Level Correlation:** Higher job levels show significantly lower attrition rates (26% at Level 1 vs 8% at Level 5)
3. **Age Factor:** Younger employees show higher attrition rates
4. **Working Years:** Employees with fewer years of experience are more likely to leave
5. **Department Distribution:** R&D department represents 60%+ of the dataset


##  Getting Started

### Prerequisites
```bash
Python 3.8+
pip
```

### Installation

1. Clone the repository
```bash
git clone https://github.com/Islam-Reda-13/employee-attrition-prediction.git
cd employee-attrition-prediction
```

2. Install required packages
```bash
pip install -r requirements.txt
```

3. Download the IBM HR Analytics dataset
- Place `ibm.csv` in the project root directory

4. Run the notebook
```bash
jupyter notebook Samsung_ibm_Mickey_Final.pdf
```


##  Methodology

### 1. Data Quality Assessment
- Checked for missing values (none found)
- Identified and removed duplicate records
- Removed constant columns (StandardHours, EmployeeCount)

### 2. Exploratory Analysis
- Univariate analysis of all features
- Bivariate analysis between features and target
- Multivariate correlation analysis

### 3. Feature Selection
- Correlation-based removal (threshold: 0.7)
- Domain knowledge-based feature creation
- PCA visualization for oversampling validation

### 4. Model Training Pipeline
```
Data → Cleaning → Feature Engineering → Train/Test Split → 
SMOTE Oversampling → Scaling → Model Training → Evaluation
```

### 5. Evaluation Metrics
- Accuracy Score
- Precision & Recall
- F1-Score
- Confusion Matrix
- Classification Report

