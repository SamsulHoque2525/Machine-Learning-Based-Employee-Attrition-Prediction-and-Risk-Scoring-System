import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from frontend.components import (
    load_css,
    render_header,
    render_kpi_cards
)

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Employee Attrition Risk System",
    page_icon="📊",
    layout="wide"
)

load_css()


# ============================================
# LOAD RISK REPORT
# ============================================

@st.cache_data
def load_data():

    df = pd.read_csv("employee_risk_report.csv")

    return df


df = load_data()


# LOAD TRAINED MODELS

@st.cache_resource
def load_models():
    logistic_model = joblib.load(
        "models/logistic_regression_model.pkl"
    )

    random_forest_model = joblib.load(
        "models/random_forest_model.pkl"
    )

    gradient_boosting_model = joblib.load(
        "models/gradient_boosting_model.pkl"
    )

    preprocessor = joblib.load(
        "models/preprocessor.pkl"
    )

    return (
        logistic_model,
        random_forest_model,
        gradient_boosting_model,
        preprocessor
    )


(
    logistic_model,
    random_forest_model,
    gradient_boosting_model,
    preprocessor
) = load_models()



# ============================================
# HEADER
# ============================================

render_header()

# ============================================
# KEY METRICS
# ============================================

total_employees = len(df)

high_risk = len(
    df[df["RiskCategory"] == "High"]
)

medium_risk = len(
    df[df["RiskCategory"] == "Medium"]
)

low_risk = len(
    df[df["RiskCategory"] == "Low"]
)


render_kpi_cards(
    total_employees,
    high_risk,
    medium_risk,
    low_risk
)

# ============================================
# SIDEBAR FILTERS
# ============================================

st.sidebar.header("🔎 Filters")


# Department filter
departments = sorted(
    df["Department"].unique()
)

selected_department = st.sidebar.selectbox(
    "Department",
    ["All"] + departments
)


# Job Role filter
job_roles = sorted(
    df["JobRole"].unique()
)

selected_job_role = st.sidebar.selectbox(
    "Job Role",
    ["All"] + job_roles
)


# Risk category filter
selected_risk = st.sidebar.multiselect(
    "Risk Category",
    ["Low", "Medium", "High"],
    default=["Low", "Medium", "High"]
)


# Minimum risk threshold
risk_threshold = st.sidebar.slider(
    "Minimum Risk Score (%)",
    min_value=0,
    max_value=100,
    value=0,
    step=5
)


# ============================================
# APPLY FILTERS
# ============================================

filtered_df = df.copy()


if selected_department != "All":

    filtered_df = filtered_df[
        filtered_df["Department"] == selected_department
    ]


if selected_job_role != "All":

    filtered_df = filtered_df[
        filtered_df["JobRole"] == selected_job_role
    ]


if selected_risk:

    filtered_df = filtered_df[
        filtered_df["RiskCategory"].isin(selected_risk)
    ]


filtered_df = filtered_df[
    filtered_df["RiskPercentage"] >= risk_threshold
]


# ============================================
# FILTERED EMPLOYEE COUNT
# ============================================

st.sidebar.markdown("---")

st.sidebar.metric(
    "Employees Matching Filters",
    len(filtered_df)
)


# ==========================================
# EXECUTIVE SUMMARY / KEY INSIGHTS
# ==========================================

st.markdown("---")
st.header("📌 Executive Summary")

# Calculate summary statistics
total_filtered = len(filtered_df)

if total_filtered > 0:

    high_risk_percentage = (
        len(filtered_df[filtered_df["RiskCategory"] == "High"])
        / total_filtered
        * 100
    )

    medium_risk_percentage = (
        len(filtered_df[filtered_df["RiskCategory"] == "Medium"])
        / total_filtered
        * 100
    )

    low_risk_percentage = (
        len(filtered_df[filtered_df["RiskCategory"] == "Low"])
        / total_filtered
        * 100
    )

    attrition_count = filtered_df["Attrition"].sum()

    attrition_rate = (
        attrition_count
        / total_filtered
        * 100
    )

    overtime_count = len(
        filtered_df[filtered_df["OverTime"] == "Yes"]
    )

    overtime_percentage = (
        overtime_count
        / total_filtered
        * 100
    )

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "High Risk %",
            f"{high_risk_percentage:.2f}%"
        )

    with col2:
        st.metric(
            "Medium Risk %",
            f"{medium_risk_percentage:.2f}%"
        )

    with col3:
        st.metric(
            "Actual Attrition %",
            f"{attrition_rate:.2f}%"
        )

    with col4:
        st.metric(
            "Overtime %",
            f"{overtime_percentage:.2f}%"
        )

    # Written insights
    st.subheader("Key Observations")

    st.write(
        f"• **{high_risk_percentage:.2f}%** of employees "
        f"currently fall into the High Risk category."
    )

    st.write(
        f"• **{medium_risk_percentage:.2f}%** of employees "
        f"fall into the Medium Risk category."
    )

    st.write(
        f"• The observed attrition rate in the filtered "
        f"dataset is **{attrition_rate:.2f}%**."
    )

    st.write(
        f"• **{overtime_percentage:.2f}%** of employees "
        f"have reported working overtime."
    )

else:

    st.info(
        "No employees are available for the selected filters."
    )


# ============================================
# RISK DISTRIBUTION
# ============================================

st.markdown("---")



risk_distribution = (
    filtered_df["RiskCategory"]
    .value_counts()
    .reindex(
        ["Low", "Medium", "High"],
        fill_value=0
    )
)

st.bar_chart(risk_distribution)


# ==========================================
# RISK DISTRIBUTION PERCENTAGE
# ==========================================

st.subheader("Risk Distribution Percentage")

total_filtered = risk_distribution.sum()

if total_filtered > 0:
    risk_percentage = (
        risk_distribution / total_filtered * 100
    ).round(2)

    st.dataframe(
        pd.DataFrame({
            "Risk Category": risk_percentage.index,
            "Employees": risk_distribution.values,
            "Percentage": risk_percentage.values
        }),
        use_container_width=True
    )

    st.bar_chart(risk_percentage)

else:
    st.info("No employees available for the selected filters.")


# ============================================
# RISK TABLE
# ============================================

st.header("Employee Risk Overview")

st.dataframe(
    filtered_df[
        [
            "EmployeeID",
            "Department",
            "JobRole",
            "MonthlyIncome",
            "OverTime",
            "RiskPercentage",
            "RiskCategory"
        ]
    ],
    use_container_width=True
)


# ==========================================
# DOWNLOAD FILTERED RISK REPORT
# ==========================================

st.subheader("📥 Download Risk Report")

download_columns = [
    "EmployeeID",
    "Department",
    "JobRole",
    "Age",
    "MonthlyIncome",
    "OverTime",
    "JobSatisfaction",
    "WorkLifeBalance",
    "YearsAtCompany",
    "RiskScore",
    "RiskPercentage",
    "RiskCategory",
    "Attrition"
]

download_data = filtered_df[download_columns].copy()

csv_data = download_data.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Filtered Employee Risk Report",
    data=csv_data,
    file_name="employee_risk_report_filtered.csv",
    mime="text/csv"
)


# DEPARTMENT-WISE RISK ANALYSIS

st.markdown("---")
st.header("🏢 Department-wise Risk Analysis")

department_risk = pd.crosstab(
    filtered_df["Department"],
    filtered_df["RiskCategory"]
)

department_risk = department_risk.reindex(
    columns=["Low", "Medium", "High"],
    fill_value=0
)

st.dataframe(
    department_risk,
    use_container_width=True
)

st.bar_chart(department_risk)


# ==========================================
# DEPARTMENT ATTRITION ANALYSIS
# ==========================================

st.subheader("📈 Department Attrition Analysis")

department_attrition = (
    filtered_df.groupby("Department")["Attrition"]
    .agg(
        Total_Employees="count",
        Employees_Left="sum"
    )
    .reset_index()
)

department_attrition["Attrition_Rate"] = (
    department_attrition["Employees_Left"]
    / department_attrition["Total_Employees"]
    * 100
).round(2)

st.dataframe(
    department_attrition,
    use_container_width=True
)

st.subheader("Department Attrition Rate (%)")

department_attrition_chart = (
    department_attrition
    .set_index("Department")["Attrition_Rate"]
)

st.bar_chart(department_attrition_chart)



# JOB ROLE-WISE RISK ANALYSIS

st.markdown("---")
st.header("👔 Job Role-wise Risk Analysis")

jobrole_risk = pd.crosstab(
    filtered_df["JobRole"],
    filtered_df["RiskCategory"]
)

jobrole_risk = jobrole_risk.reindex(
    columns=["Low", "Medium", "High"],
    fill_value=0
)

st.dataframe(
    jobrole_risk,
    use_container_width=True
)

st.bar_chart(jobrole_risk)


# ============================================
# INDIVIDUAL EMPLOYEE RISK PROFILE
# ============================================

st.markdown("---")

st.header("👤 Individual Employee Risk Profile")


# Employee selector
employee_ids = filtered_df["EmployeeID"].tolist()

if len(employee_ids) > 0:

    selected_employee = st.selectbox(
        "Select Employee ID",
        employee_ids
    )

    # Get selected employee
    employee = filtered_df[
        filtered_df["EmployeeID"] == selected_employee
    ].iloc[0]


    # Employee risk information
    risk_score = employee["RiskPercentage"]
    risk_category = employee["RiskCategory"]


    # Display risk information
    col1, col2, col3 = st.columns(3)


    with col1:
        st.metric(
            "Employee ID",
            employee["EmployeeID"]
        )


    with col2:
        st.metric(
            "Risk Score",
            f"{risk_score:.2f}%"
        )


    with col3:
        st.metric(
            "Risk Category",
            risk_category
        )


    # Employee details
    st.subheader("Employee Details")


    detail_col1, detail_col2 = st.columns(2)


    with detail_col1:

        st.write(
            "**Department:**",
            employee["Department"]
        )

        st.write(
            "**Job Role:**",
            employee["JobRole"]
        )

        st.write(
            "**Age:**",
            employee["Age"]
        )

        st.write(
            "**Monthly Income:**",
            f"₹{employee['MonthlyIncome']:,.0f}"
        )

        st.write(
            "**Years at Company:**",
            employee["YearsAtCompany"]
        )


    with detail_col2:

        st.write(
            "**Overtime:**",
            employee["OverTime"]
        )

        st.write(
            "**Job Satisfaction:**",
            employee["JobSatisfaction"]
        )

        st.write(
            "**Work-Life Balance:**",
            employee["WorkLifeBalance"]
        )

        st.write(
            "**Actual Attrition:**",
            "Yes" if employee["Attrition"] == 1 else "No"
        )
        
        

else:

    st.warning(
        "No employees match the selected filters."
    )
    
   
   
   
   
   
    # WHAT-IF RISK ANALYSIS

st.markdown("---")
st.header("🔄 What-If Risk Analysis")

st.write(
    "Explore how changes in selected employee factors may affect "
    "their risk profile."
)

if len(employee_ids) > 0:

    whatif_overtime = st.selectbox(
        "What-If Overtime",
        ["Yes", "No"],
        index=0 if employee["OverTime"] == "Yes" else 1
    )

    whatif_satisfaction = st.slider(
        "What-If Job Satisfaction",
        min_value=1,
        max_value=4,
        value=int(employee["JobSatisfaction"])
    )

    whatif_worklife = st.slider(
        "What-If Work-Life Balance",
        min_value=1,
        max_value=4,
        value=int(employee["WorkLifeBalance"])
    )

   

    # ============================================================
    # MODEL-BASED WHAT-IF PREDICTION
    # ============================================================

    whatif_employee = employee.copy()

    # Apply What-If changes
    whatif_employee["OverTime"] = whatif_overtime
    whatif_employee["JobSatisfaction"] = whatif_satisfaction
    whatif_employee["WorkLifeBalance"] = whatif_worklife

    # Recalculate engineered features
    whatif_employee["IncomeExperienceRatio"] = (
        whatif_employee["MonthlyIncome"]
        / (whatif_employee["TotalWorkingYears"] + 1)
    )

    whatif_employee["PromotionDelay"] = (
        whatif_employee["YearsAtCompany"]
        - whatif_employee["YearsSinceLastPromotion"]
    )

    whatif_employee["EngagementScore"] = (
        whatif_employee["JobInvolvement"]
        + whatif_employee["JobSatisfaction"]
        + whatif_employee["EnvironmentSatisfaction"]
        + whatif_employee["RelationshipSatisfaction"]
    )

    whatif_employee["WorkloadStressFlag"] = int(
        whatif_employee["OverTime"] == "Yes"
    )

    # Convert employee data into DataFrame
    whatif_df = pd.DataFrame([whatif_employee])

    # Remove target variable
    if "Attrition" in whatif_df.columns:
        whatif_df = whatif_df.drop("Attrition", axis=1)

    # Apply the same preprocessing used during model training
    whatif_processed = preprocessor.transform(whatif_df)

    # Predict attrition probability
    whatif_probability = (
        gradient_boosting_model
        .predict_proba(whatif_processed)[0][1]
    )

    # Convert probability to percentage
    whatif_risk = whatif_probability * 100

    # Determine risk category
    if whatif_risk < 30:
        whatif_category = "Low"
    elif whatif_risk <= 60:
        whatif_category = "Medium"
    else:
        whatif_category = "High"

    # ============================================================
    # DISPLAY WHAT-IF RESULT
    # ============================================================

    st.markdown("---")

    st.subheader("🎯 What-If Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Original Risk",
            f"{risk_score:.2f}%"
        )

    with col2:
        st.metric(
            "What-If Predicted Risk",
            f"{whatif_risk:.2f}%"
        )

    st.info(
        f"**Predicted Risk Category: {whatif_category}**"
    )

    risk_change = whatif_risk - risk_score

    if risk_change > 0:
        st.warning(
            f"Risk increased by **{risk_change:.2f} percentage points**."
        )

    elif risk_change < 0:
        st.success(
            f"Risk decreased by **{abs(risk_change):.2f} percentage points**."
        )

    else:
        st.info(
            "The predicted risk remained unchanged."
        )

    
    # ==========================================
# MODEL PERFORMANCE
# ==========================================

st.markdown("---")
st.header("📊 Model Performance")

st.write(
    "Performance of the machine learning models "
    "evaluated on the held-out test dataset."
)

import json
import os

metrics_file = "models/model_metrics.json"

if os.path.exists(metrics_file):

    with open(metrics_file, "r") as file:
        model_metrics = json.load(file)

    # Convert saved metrics into DataFrame
    model_results = pd.DataFrame(model_metrics).T

    # Display metrics
    st.subheader("Test-Set Performance")

    st.dataframe(
        model_results.round(4),
        use_container_width=True
    )

    # Individual model metrics
    st.subheader("Model Metrics")

    for model_name, metrics in model_metrics.items():

        st.markdown(f"### {model_name}")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Accuracy",
                f"{metrics['Accuracy']:.4f}"
            )

        with col2:
            st.metric(
                "Precision",
                f"{metrics['Precision']:.4f}"
            )

        with col3:
            st.metric(
                "Recall",
                f"{metrics['Recall']:.4f}"
            )

        with col4:
            st.metric(
                "F1 Score",
                f"{metrics['F1 Score']:.4f}"
            )

        with col5:
            st.metric(
                "ROC-AUC",
                f"{metrics['ROC-AUC']:.4f}"
            )

    # ROC-AUC comparison
    st.subheader("ROC-AUC Comparison")

    roc_auc_chart = model_results["ROC-AUC"]

    st.bar_chart(roc_auc_chart)

else:

    st.error(
        "Model metrics file not found. "
        "Please run train_model.py first."
    )
# ==========================================
# FEATURE IMPORTANCE
# ==========================================

st.markdown("---")
st.header("🔍 Feature Importance")

st.write(
    "The chart below shows the features that contribute most "
    "to the Gradient Boosting model's predictions."
)

# Get feature names after preprocessing
feature_names = preprocessor.get_feature_names_out()

# Get feature importance
importance_values = gradient_boosting_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})

# Sort by importance
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

# Show top 15 features
top_features = feature_importance.head(15)

st.dataframe(
    top_features,
    use_container_width=True
)

# Chart
st.subheader("Top 15 Important Features")

st.bar_chart(
    top_features.set_index("Feature")
)