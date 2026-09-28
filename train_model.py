import pandas as pd
# Load the Palo Alto Networks employee dataset
df = pd.read_csv("data/Palo Alto Networks.csv")

# Display basic information
print("========================================")
print("PALO ALTO NETWORKS ATTRITION PROJECT")
print("========================================")

print("\nDataset loaded successfully!")

print("\nNumber of employees:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nColumn names:")
for column in df.columns:
    print("-", column)

print("\nFirst 5 records:")
print(df.head()) 

# ============================================
# DATASET QUALITY CHECK
# ============================================

print("\n========================================")
print("DATASET QUALITY CHECK")
print("========================================")

# 1. Check for missing values
print("\nMissing values in each column:")

missing_values = df.isnull().sum()

print(missing_values)

print("\nTotal missing values:", missing_values.sum())


# 2. Check for duplicate rows
duplicate_count = df.duplicated().sum()

print("\nDuplicate rows:", duplicate_count)


# 3. Check Attrition distribution
print("\nAttrition distribution:")

attrition_count = df["Attrition"].value_counts()

print(attrition_count)


# 4. Calculate attrition percentage
print("\nAttrition percentage:")

attrition_percentage = (
    df["Attrition"]
    .value_counts(normalize=True)
    * 100
)

print(attrition_percentage.round(2))


# 5. Basic numerical statistics
print("\nBasic numerical statistics:")

print(df.describe())

# ============================================
# BASIC EDA ANALYSIS
# ============================================

print("\n========================================")
print("EXPLORATORY DATA ANALYSIS")
print("========================================")

# Attrition by Department
print("\nAttrition by Department:")
print(
    pd.crosstab(
        df["Department"],
        df["Attrition"]
    )
)

# Attrition by Overtime
print("\nAttrition by Overtime:")
print(
    pd.crosstab(
        df["OverTime"],
        df["Attrition"]
    )
)

# Attrition by Job Role
print("\nAttrition by Job Role:")
print(
    pd.crosstab(
        df["JobRole"],
        df["Attrition"]
    )
)

# Attrition by Job Satisfaction
print("\nAttrition by Job Satisfaction:")
print(
    pd.crosstab(
        df["JobSatisfaction"],
        df["Attrition"]
    )
)

# Attrition by Work-Life Balance
print("\nAttrition by Work-Life Balance:")
print(
    pd.crosstab(
        df["WorkLifeBalance"],
        df["Attrition"]
    )
)

# ============================================
# ATTRITION RATE ANALYSIS
# ============================================

print("\n========================================")
print("ATTRITION RATE ANALYSIS")
print("========================================")


# 1. Attrition rate by Department
print("\nAttrition rate by Department:")

department_rate = (
    df.groupby("Department")["Attrition"]
    .mean()
    .mul(100)
    .round(2)
)

print(department_rate)


# 2. Attrition rate by Overtime
print("\nAttrition rate by Overtime:")

overtime_rate = (
    df.groupby("OverTime")["Attrition"]
    .mean()
    .mul(100)
    .round(2)
)

print(overtime_rate)


# 3. Attrition rate by Job Role
print("\nAttrition rate by Job Role:")

jobrole_rate = (
    df.groupby("JobRole")["Attrition"]
    .mean()
    .mul(100)
    .round(2)
    .sort_values(ascending=False)
)

print(jobrole_rate)


# 4. Attrition rate by Job Satisfaction
print("\nAttrition rate by Job Satisfaction:")

satisfaction_rate = (
    df.groupby("JobSatisfaction")["Attrition"]
    .mean()
    .mul(100)
    .round(2)
)

print(satisfaction_rate)


# 5. Attrition rate by Work-Life Balance
print("\nAttrition rate by WorkLifeBalance:")

worklife_rate = (
    df.groupby("WorkLifeBalance")["Attrition"]
    .mean()
    .mul(100)
    .round(2)
)

print(worklife_rate)

# ============================================
# FEATURE ENGINEERING
# ============================================

print("\n========================================")
print("FEATURE ENGINEERING")
print("========================================")


# 1. Income to Experience Ratio
# Higher income relative to experience may indicate different
# career/retention patterns.

df["IncomeExperienceRatio"] = (
    df["MonthlyIncome"] /
    (df["TotalWorkingYears"] + 1)
)


# 2. Promotion Delay
# Measures how long the employee has been since their last promotion.

df["PromotionDelay"] = (
    df["YearsAtCompany"] -
    df["YearsSinceLastPromotion"]
)


# 3. Engagement Score
# Combines job involvement, job satisfaction,
# environment satisfaction and relationship satisfaction.

df["EngagementScore"] = (
    df["JobInvolvement"] +
    df["JobSatisfaction"] +
    df["EnvironmentSatisfaction"] +
    df["RelationshipSatisfaction"]
)


# 4. Workload Stress Flag
# Employees working overtime are flagged as higher workload.

df["WorkloadStressFlag"] = (
    df["OverTime"] == "Yes"
).astype(int)


# Display engineered features
print("\nNew features created:")

print(
    df[
        [
            "IncomeExperienceRatio",
            "PromotionDelay",
            "EngagementScore",
            "WorkloadStressFlag"
        ]
    ].head()
)


print("\nUpdated dataset shape:")
print(df.shape)

# ============================================
# PREPARE FEATURES AND TARGET
# ============================================

print("\n========================================")
print("PREPARING FEATURES AND TARGET")
print("========================================")


# Target variable
# Attrition is what our model will predict.

y = df["Attrition"]


# Feature variables
# Remove Attrition because it is the target.

X = df.drop("Attrition", axis=1)


# Display the results
print("\nFeature dataset shape:")
print(X.shape)

print("\nTarget dataset shape:")
print(y.shape)


# Display target distribution
print("\nTarget distribution:")
print(y.value_counts())


# Display feature columns
print("\nFeatures used by the model:")

for column in X.columns:
    print("-", column)
    
    # ============================================
# TRAIN / TEST SPLIT
# ============================================

print("\n========================================")
print("TRAIN / TEST SPLIT")
print("========================================")

from sklearn.model_selection import train_test_split


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Display dataset sizes
print("\nTraining features:", X_train.shape)
print("Testing features:", X_test.shape)

print("\nTraining target:", y_train.shape)
print("Testing target:", y_test.shape)


# Check attrition distribution
print("\nTraining attrition distribution:")
print(y_train.value_counts())

print("\nTesting attrition distribution:")
print(y_test.value_counts())

# ============================================
# DATA PREPROCESSING
# ============================================

print("\n========================================")
print("DATA PREPROCESSING")
print("========================================")

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Identify categorical columns
categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


# Identify numerical columns
numerical_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


print("\nCategorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)


# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)


# Fit preprocessing ONLY on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform testing data using the same preprocessing
X_test_processed = preprocessor.transform(X_test)


print("\nProcessed training data shape:")
print(X_train_processed.shape)

print("\nProcessed testing data shape:")
print(X_test_processed.shape)

# ============================================
# HANDLE CLASS IMBALANCE USING SMOTE
# ============================================

print("\n========================================")
print("SMOTE CLASS BALANCING")
print("========================================")

from imblearn.over_sampling import SMOTE


# Create SMOTE object
smote = SMOTE(
    random_state=42
)


# Apply SMOTE ONLY to training data
X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train_processed,
    y_train
)


# Display class distribution before SMOTE
print("\nBefore SMOTE:")
print(y_train.value_counts())


# Display class distribution after SMOTE
print("\nAfter SMOTE:")
print(y_train_balanced.value_counts())


# Display shapes
print("\nBalanced training features:")
print(X_train_balanced.shape)

print("\nBalanced training target:")
print(y_train_balanced.shape)

# ============================================
# LOGISTIC REGRESSION MODEL
# ============================================

print("\n========================================")
print("LOGISTIC REGRESSION")
print("========================================")

from sklearn.linear_model import LogisticRegression


# Create Logistic Regression model
logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# Train the model
logistic_model.fit(
    X_train_balanced,
    y_train_balanced
)


print("\nLogistic Regression training completed successfully!")


# Generate predictions on the test set
logistic_predictions = logistic_model.predict(
    X_test_processed
)


# Generate probability of attrition
logistic_probabilities = logistic_model.predict_proba(
    X_test_processed
)[:, 1]


print("\nFirst 10 predictions:")
print(logistic_predictions[:10])


print("\nFirst 10 attrition probabilities:")
print(logistic_probabilities[:10])

# ============================================
# EVALUATE LOGISTIC REGRESSION
# ============================================

print("\n========================================")
print("LOGISTIC REGRESSION EVALUATION")
print("========================================")

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# Calculate evaluation metrics
logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

logistic_precision = precision_score(
    y_test,
    logistic_predictions
)

logistic_recall = recall_score(
    y_test,
    logistic_predictions
)

logistic_f1 = f1_score(
    y_test,
    logistic_predictions
)

logistic_roc_auc = roc_auc_score(
    y_test,
    logistic_probabilities
)


# Display metrics
print("\nAccuracy:", round(logistic_accuracy, 4))
print("Precision:", round(logistic_precision, 4))
print("Recall:", round(logistic_recall, 4))
print("F1 Score:", round(logistic_f1, 4))
print("ROC-AUC:", round(logistic_roc_auc, 4))


# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    logistic_predictions
))


# Detailed classification report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        logistic_predictions
    )
)


# ============================================
# RANDOM FOREST MODEL
# ============================================

print("\n========================================")
print("RANDOM FOREST")
print("========================================")

from sklearn.ensemble import RandomForestClassifier


# Create Random Forest model
random_forest_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)


# Train the model
random_forest_model.fit(
    X_train_balanced,
    y_train_balanced
)


print("\nRandom Forest training completed successfully!")


# Generate predictions
rf_predictions = random_forest_model.predict(
    X_test_processed
)


# Generate attrition probabilities
rf_probabilities = random_forest_model.predict_proba(
    X_test_processed
)[:, 1]


print("\nFirst 10 predictions:")
print(rf_predictions[:10])


print("\nFirst 10 attrition probabilities:")
print(rf_probabilities[:10])



# ============================================
# EVALUATE RANDOM FOREST
# ============================================

print("\n========================================")
print("RANDOM FOREST EVALUATION")
print("========================================")


# Calculate evaluation metrics
rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

rf_precision = precision_score(
    y_test,
    rf_predictions
)

rf_recall = recall_score(
    y_test,
    rf_predictions
)

rf_f1 = f1_score(
    y_test,
    rf_predictions
)

rf_roc_auc = roc_auc_score(
    y_test,
    rf_probabilities
)


# Display metrics
print("\nAccuracy:", round(rf_accuracy, 4))
print("Precision:", round(rf_precision, 4))
print("Recall:", round(rf_recall, 4))
print("F1 Score:", round(rf_f1, 4))
print("ROC-AUC:", round(rf_roc_auc, 4))


# Confusion Matrix
print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        rf_predictions
    )
)


# Detailed classification report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        rf_predictions
    )
)

# ============================================
# GRADIENT BOOSTING MODEL
# ============================================

print("\n========================================")
print("GRADIENT BOOSTING")
print("========================================")

from sklearn.ensemble import GradientBoostingClassifier


# Create Gradient Boosting model
gradient_boosting_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)


# Train the model
gradient_boosting_model.fit(
    X_train_balanced,
    y_train_balanced
)


print("\nGradient Boosting training completed successfully!")


# Generate predictions
gb_predictions = gradient_boosting_model.predict(
    X_test_processed
)


# Generate attrition probabilities
gb_probabilities = gradient_boosting_model.predict_proba(
    X_test_processed
)[:, 1]


print("\nFirst 10 predictions:")
print(gb_predictions[:10])


print("\nFirst 10 attrition probabilities:")
print(gb_probabilities[:10])


# ============================================
# EVALUATE GRADIENT BOOSTING
# ============================================

print("\n========================================")
print("GRADIENT BOOSTING EVALUATION")
print("========================================")


# Calculate evaluation metrics
gb_accuracy = accuracy_score(
    y_test,
    gb_predictions
)

gb_precision = precision_score(
    y_test,
    gb_predictions
)

gb_recall = recall_score(
    y_test,
    gb_predictions
)

gb_f1 = f1_score(
    y_test,
    gb_predictions
)

gb_roc_auc = roc_auc_score(
    y_test,
    gb_probabilities
)


# Display metrics
print("\nAccuracy:", round(gb_accuracy, 4))
print("Precision:", round(gb_precision, 4))
print("Recall:", round(gb_recall, 4))
print("F1 Score:", round(gb_f1, 4))
print("ROC-AUC:", round(gb_roc_auc, 4))


# Confusion Matrix
print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        gb_predictions
    )
)


# ============================================
# MODEL COMPARISON
# ============================================

print("\n========================================")
print("MODEL COMPARISON")
print("========================================")


comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting"
    ],

    "Accuracy": [
        logistic_accuracy,
        rf_accuracy,
        gb_accuracy
    ],

    "Precision": [
        logistic_precision,
        rf_precision,
        gb_precision
    ],

    "Recall": [
        logistic_recall,
        rf_recall,
        gb_recall
    ],

    "F1 Score": [
        logistic_f1,
        rf_f1,
        gb_f1
    ],

    "ROC-AUC": [
        logistic_roc_auc,
        rf_roc_auc,
        gb_roc_auc
    ]
})


# Round values for easier reading
comparison[
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]
] = comparison[
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]
].round(4)


print("\n")
print(comparison.to_string(index=False))


# ============================================
# EMPLOYEE ATTRITION RISK SCORING
# ============================================

print("\n========================================")
print("EMPLOYEE ATTRITION RISK SCORING")
print("========================================")


# Use Gradient Boosting probabilities as risk scores
risk_scores = gb_probabilities


# Convert probability into percentage
risk_percentages = risk_scores * 100


# Create risk categories
def get_risk_category(score):

    if score < 0.30:
        return "Low"

    elif score <= 0.60:
        return "Medium"

    else:
        return "High"


risk_categories = [
    get_risk_category(score)
    for score in risk_scores
]


# Create employee IDs
employee_ids = [
    f"PAN{number:04d}"
    for number in range(1, len(df) + 1)
]


# Create risk report for test employees
risk_report = pd.DataFrame({
    "EmployeeID": employee_ids[:len(X_test)],
    "ActualAttrition": y_test.values,
    "RiskScore": risk_scores,
    "RiskPercentage": risk_percentages,
    "RiskCategory": risk_categories
})


# Round risk values
risk_report["RiskScore"] = risk_report["RiskScore"].round(4)
risk_report["RiskPercentage"] = risk_report["RiskPercentage"].round(2)


print("\nFirst 10 employee risk scores:")
print(
    risk_report.head(10).to_string(index=False)
)


# Risk category distribution
print("\nRisk category distribution:")
print(
    risk_report["RiskCategory"].value_counts()
)


# ============================================
# GENERATE RISK SCORES FOR ALL EMPLOYEES
# ============================================

print("\n========================================")
print("GENERATING COMPLETE EMPLOYEE RISK REPORT")
print("========================================")


# Process the complete employee dataset
X_all_processed = preprocessor.transform(X)


# Generate attrition probability for every employee
all_risk_scores = gradient_boosting_model.predict_proba(
    X_all_processed
)[:, 1]


# Convert probability to percentage
all_risk_percentages = all_risk_scores * 100


# Create risk categories
all_risk_categories = [
    get_risk_category(score)
    for score in all_risk_scores
]


# Create Employee IDs
all_employee_ids = [
    f"PAN{number:04d}"
    for number in range(1, len(df) + 1)
]


# Create complete risk report
risk_report_all = df.copy()


# Add Employee ID
risk_report_all.insert(
    0,
    "EmployeeID",
    all_employee_ids
)


# Add risk information
risk_report_all["RiskScore"] = all_risk_scores
risk_report_all["RiskPercentage"] = all_risk_percentages
risk_report_all["RiskCategory"] = all_risk_categories


# Round risk values
risk_report_all["RiskScore"] = (
    risk_report_all["RiskScore"].round(4)
)

risk_report_all["RiskPercentage"] = (
    risk_report_all["RiskPercentage"].round(2)
)


# Display first 10 employees
print("\nFirst 10 employees with risk scores:")

print(
    risk_report_all[
        [
            "EmployeeID",
            "Age",
            "JobRole",
            "MonthlyIncome",
            "OverTime",
            "Attrition",
            "RiskScore",
            "RiskPercentage",
            "RiskCategory"
        ]
    ].head(10).to_string(index=False)
)


# Display risk distribution
print("\nRisk category distribution:")

print(
    risk_report_all["RiskCategory"]
    .value_counts()
)


# SAVE MODEL PERFORMANCE METRICS

import json
import os

os.makedirs("models", exist_ok=True)

model_metrics = {
    "Logistic Regression": {
        "Accuracy": logistic_accuracy,
        "Precision": logistic_precision,
        "Recall": logistic_recall,
        "F1 Score": logistic_f1,
        "ROC-AUC": logistic_roc_auc
    },

    "Random Forest": {
        "Accuracy": rf_accuracy,
        "Precision": rf_precision,
        "Recall": rf_recall,
        "F1 Score": rf_f1,
        "ROC-AUC": rf_roc_auc
    },

    "Gradient Boosting": {
        "Accuracy": gb_accuracy,
        "Precision": gb_precision,
        "Recall": gb_recall,
        "F1 Score": gb_f1,
        "ROC-AUC": gb_roc_auc
    }
}

with open("models/model_metrics.json", "w") as file:
    json.dump(model_metrics, file, indent=4)

print("\nModel performance metrics saved to models/model_metrics.json")




# Save complete risk report
risk_report_all.to_csv(
    "employee_risk_report.csv",
    index=False
)


print("\nComplete risk report saved successfully!")

print("\nFile created:")
print("employee_risk_report.csv")


# ============================================
# SAVE TRAINED MODELS
# ============================================

print("\n========================================")
print("SAVING MODELS")
print("========================================")

import joblib
import os


# Make sure models folder exists
os.makedirs("models", exist_ok=True)


# Save Gradient Boosting model
joblib.dump(
    gradient_boosting_model,
    "models/gradient_boosting_model.pkl"
)


# Save Logistic Regression model
joblib.dump(
    logistic_model,
    "models/logistic_regression_model.pkl"
)


# Save Random Forest model
joblib.dump(
    random_forest_model,
    "models/random_forest_model.pkl"
)


# Save preprocessing pipeline
joblib.dump(
    preprocessor,
    "models/preprocessor.pkl"
)


print("\nModels saved successfully!")

print("\nSaved files:")
print("- models/gradient_boosting_model.pkl")
print("- models/logistic_regression_model.pkl")
print("- models/random_forest_model.pkl")
print("- models/preprocessor.pkl")