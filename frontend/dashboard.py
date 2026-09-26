# ============================================================
# EMPLOYEE RISK INTELLIGENCE
# MAIN DASHBOARD FRONTEND
# ============================================================

import json
import os

import pandas as pd
import streamlit as st

from frontend.components import (
    render_kpi_cards,
    render_section_title,
    render_employee_card,
    render_risk_badge,
    render_divider,
    render_footer,
)


# ============================================================
# MAIN DASHBOARD
# ============================================================

def render_dashboard(
    df,
    logistic_model,
    random_forest_model,
    gradient_boosting_model,
    preprocessor
):

    # ========================================================
    # SIDEBAR
    # ========================================================

    st.sidebar.markdown("## 🎛️ Dashboard Filters")

    st.sidebar.markdown(
        "Use the filters below to explore employee attrition risk."
    )

    # --------------------------------------------------------
    # Department Filter
    # --------------------------------------------------------

    departments = sorted(
        df["Department"].dropna().unique().tolist()
    )

    selected_departments = st.sidebar.multiselect(
        "Department",
        departments,
        default=departments
    )

    # --------------------------------------------------------
    # Job Role Filter
    # --------------------------------------------------------

    job_roles = sorted(
        df["JobRole"].dropna().unique().tolist()
    )

    selected_job_roles = st.sidebar.multiselect(
        "Job Role",
        job_roles,
        default=job_roles
    )

    # --------------------------------------------------------
    # Risk Category Filter
    # --------------------------------------------------------

    risk_categories = ["Low", "Medium", "High"]

    selected_risk_categories = st.sidebar.multiselect(
        "Risk Category",
        risk_categories,
        default=risk_categories
    )

    # --------------------------------------------------------
    # Overtime Filter
    # --------------------------------------------------------

    overtime_options = sorted(
        df["OverTime"].dropna().unique().tolist()
    )

    selected_overtime = st.sidebar.multiselect(
        "OverTime",
        overtime_options,
        default=overtime_options
    )

    # --------------------------------------------------------
    # Age Range
    # --------------------------------------------------------

    min_age = int(df["Age"].min())
    max_age = int(df["Age"].max())

    age_range = st.sidebar.slider(
        "Age Range",
        min_value=min_age,
        max_value=max_age,
        value=(min_age, max_age)
    )

    # --------------------------------------------------------
    # Monthly Income Range
    # --------------------------------------------------------

    min_income = int(df["MonthlyIncome"].min())
    max_income = int(df["MonthlyIncome"].max())

    income_range = st.sidebar.slider(
        "Monthly Income Range",
        min_value=min_income,
        max_value=max_income,
        value=(min_income, max_income)
    )

    # ========================================================
    # APPLY FILTERS
    # ========================================================

    filtered_df = df[
        df["Department"].isin(selected_departments)
        & df["JobRole"].isin(selected_job_roles)
        & df["RiskCategory"].isin(selected_risk_categories)
        & df["OverTime"].isin(selected_overtime)
        & df["Age"].between(age_range[0], age_range[1])
        & df["MonthlyIncome"].between(
            income_range[0],
            income_range[1]
        )
    ].copy()

    # ========================================================
    # FILTERED EMPLOYEE COUNT
    # ========================================================

    if len(filtered_df) == 0:

        st.warning(
            "No employees match the selected filters."
        )

        return

    # ========================================================
    # KPI CARDS
    # ========================================================

    total_employees = len(filtered_df)

    high_risk = len(
        filtered_df[
            filtered_df["RiskCategory"] == "High"
        ]
    )

    medium_risk = len(
        filtered_df[
            filtered_df["RiskCategory"] == "Medium"
        ]
    )

    low_risk = len(
        filtered_df[
            filtered_df["RiskCategory"] == "Low"
        ]
    )

    render_kpi_cards(
        total_employees,
        high_risk,
        medium_risk,
        low_risk
    )

    # ========================================================
    # RISK DISTRIBUTION
    # ========================================================

    render_divider()

    render_section_title(
        "📊 Risk Distribution",
        "Distribution of predicted employee attrition risk."
    )

    risk_distribution = (
        filtered_df["RiskCategory"]
        .value_counts()
        .reindex(
            ["Low", "Medium", "High"],
            fill_value=0
        )
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Risk Categories")

        st.bar_chart(
            risk_distribution
        )

    with col2:

        st.subheader("Risk Distribution Percentage")

        total_risk = risk_distribution.sum()

        if total_risk > 0:

            risk_percentage = (
                risk_distribution
                / total_risk
                * 100
            ).round(2)

            risk_percentage_df = pd.DataFrame(
                {
                    "Risk Category":
                        risk_percentage.index,

                    "Employees":
                        risk_distribution.values,

                    "Percentage":
                        risk_percentage.values
                }
            )

            st.dataframe(
                risk_percentage_df,
                use_container_width=True,
                hide_index=True
            )

    # ========================================================
    # EMPLOYEE RISK TABLE
    # ========================================================

    render_divider()

    render_section_title(
        "👥 Employee Risk Report",
        "Employees included in the current filtered view."
    )

    table_columns = [
        "EmployeeID",
        "Department",
        "JobRole",
        "Age",
        "MonthlyIncome",
        "OverTime",
        "RiskPercentage",
        "RiskCategory",
        "Attrition"
    ]

    available_columns = [
        column
        for column in table_columns
        if column in filtered_df.columns
    ]

    display_df = filtered_df[
        available_columns
    ].copy()

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # DEPARTMENT ANALYSIS
    # ========================================================

    render_divider()

    render_section_title(
        "🏢 Department Analysis",
        "Actual attrition and predicted risk by department."
    )

    department_risk = (
        filtered_df
        .groupby("Department")
        .agg(
            Total_Employees=("EmployeeID", "count"),
            Average_Risk=("RiskPercentage", "mean"),
            High_Risk_Employees=(
                "RiskCategory",
                lambda x: (x == "High").sum()
            ),
            Employees_Left=("Attrition", "sum")
        )
        .reset_index()
    )

    department_risk["Average_Risk"] = (
        department_risk["Average_Risk"]
        .round(2)
    )

    department_risk["Attrition_Rate"] = (
        department_risk["Employees_Left"]
        / department_risk["Total_Employees"]
        * 100
    ).round(2)

    st.dataframe(
        department_risk,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Average Predicted Risk by Department")

    department_chart = (
        department_risk
        .set_index("Department")["Average_Risk"]
    )

    st.bar_chart(
        department_chart
    )

    # ========================================================
    # JOB ROLE ANALYSIS
    # ========================================================

    render_divider()

    render_section_title(
        "💼 Job Role Analysis",
        "Attrition and predicted risk across job roles."
    )

    jobrole_risk = (
        filtered_df
        .groupby("JobRole")
        .agg(
            Total_Employees=("EmployeeID", "count"),
            Average_Risk=("RiskPercentage", "mean"),
            High_Risk_Employees=(
                "RiskCategory",
                lambda x: (x == "High").sum()
            ),
            Employees_Left=("Attrition", "sum")
        )
        .reset_index()
    )

    jobrole_risk["Average_Risk"] = (
        jobrole_risk["Average_Risk"]
        .round(2)
    )

    jobrole_risk["Attrition_Rate"] = (
        jobrole_risk["Employees_Left"]
        / jobrole_risk["Total_Employees"]
        * 100
    ).round(2)

    st.dataframe(
        jobrole_risk,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Average Predicted Risk by Job Role")

    jobrole_chart = (
        jobrole_risk
        .set_index("JobRole")["Average_Risk"]
    )

    st.bar_chart(
        jobrole_chart
    )

    # ========================================================
    # INDIVIDUAL EMPLOYEE PROFILE
    # ========================================================

    render_divider()

    render_section_title(
        "👤 Individual Employee Profile",
        "Select an employee to inspect their risk profile."
    )

    employee_ids = (
        filtered_df["EmployeeID"]
        .astype(str)
        .tolist()
    )

    selected_employee_id = st.selectbox(
        "Select Employee ID",
        employee_ids
    )

    employee = filtered_df[
        filtered_df["EmployeeID"].astype(str)
        == selected_employee_id
    ].iloc[0]

    employee_risk = float(
        employee["RiskPercentage"]
    )

    employee_category = employee[
        "RiskCategory"
    ]

    col1, col2 = st.columns([2, 1])

    with col1:

        render_employee_card(
            employee_id=employee["EmployeeID"],
            department=employee["Department"],
            job_role=employee["JobRole"],
            risk_score=employee_risk,
            risk_category=employee_category
        )

    with col2:

        st.subheader("Current Risk")

        st.metric(
            "Predicted Risk",
            f"{employee_risk:.2f}%"
        )

        render_risk_badge(
            employee_category
        )

    # ========================================================
    # EMPLOYEE DETAILS
    # ========================================================

    st.subheader("Employee Details")

    detail_columns = [
        "Age",
        "Gender",
        "MaritalStatus",
        "BusinessTravel",
        "DistanceFromHome",
        "Education",
        "EducationField",
        "JobLevel",
        "MonthlyIncome",
        "NumCompaniesWorked",
        "OverTime",
        "JobSatisfaction",
        "EnvironmentSatisfaction",
        "JobInvolvement",
        "RelationshipSatisfaction",
        "WorkLifeBalance",
        "TotalWorkingYears",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
        "YearsWithCurrManager",
        "TrainingTimesLastYear"
    ]

    available_detail_columns = [
        column
        for column in detail_columns
        if column in employee.index
    ]

    employee_details = pd.DataFrame(
        {
            "Feature": available_detail_columns,
            "Value": [
                employee[column]
                for column in available_detail_columns
            ]
        }
    )

    st.dataframe(
        employee_details,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # WHAT-IF ANALYSIS
    # ========================================================

    render_divider()

    render_section_title(
        "🎯 What-If Risk Analysis",
        "Modify selected employee factors and recalculate predicted risk."
    )

    whatif_col1, whatif_col2, whatif_col3 = st.columns(3)

    with whatif_col1:

        whatif_overtime = st.selectbox(
            "OverTime",
            ["Yes", "No"],
            index=(
                0
                if employee["OverTime"] == "Yes"
                else 1
            )
        )

    with whatif_col2:

        whatif_satisfaction = st.slider(
            "Job Satisfaction",
            min_value=1,
            max_value=4,
            value=int(
                employee["JobSatisfaction"]
            )
        )

    with whatif_col3:

        whatif_worklife = st.slider(
            "Work-Life Balance",
            min_value=1,
            max_value=4,
            value=int(
                employee["WorkLifeBalance"]
            )
        )

    # --------------------------------------------------------
    # CREATE WHAT-IF EMPLOYEE
    # --------------------------------------------------------

    whatif_employee = employee.copy()

    whatif_employee["OverTime"] = (
        whatif_overtime
    )

    whatif_employee["JobSatisfaction"] = (
        whatif_satisfaction
    )

    whatif_employee["WorkLifeBalance"] = (
        whatif_worklife
    )

    # --------------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------------

    whatif_employee[
        "IncomeExperienceRatio"
    ] = (
        whatif_employee["MonthlyIncome"]
        / (
            whatif_employee["TotalWorkingYears"]
            + 1
        )
    )

    whatif_employee[
        "PromotionDelay"
    ] = (
        whatif_employee["YearsAtCompany"]
        - whatif_employee[
            "YearsSinceLastPromotion"
        ]
    )

    whatif_employee[
        "EngagementScore"
    ] = (
        whatif_employee["JobInvolvement"]
        + whatif_employee["JobSatisfaction"]
        + whatif_employee[
            "EnvironmentSatisfaction"
        ]
        + whatif_employee[
            "RelationshipSatisfaction"
        ]
    )

    whatif_employee[
        "WorkloadStressFlag"
    ] = int(
        whatif_employee["OverTime"] == "Yes"
    )

    # --------------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------------

    whatif_df = pd.DataFrame(
        [whatif_employee]
    )

    if "Attrition" in whatif_df.columns:

        whatif_df = whatif_df.drop(
            "Attrition",
            axis=1
        )

    if "RiskScore" in whatif_df.columns:

        whatif_df = whatif_df.drop(
            "RiskScore",
            axis=1
        )

    if "RiskPercentage" in whatif_df.columns:

        whatif_df = whatif_df.drop(
            "RiskPercentage",
            axis=1
        )

    if "RiskCategory" in whatif_df.columns:

        whatif_df = whatif_df.drop(
            "RiskCategory",
            axis=1
        )

    # --------------------------------------------------------
    # WHAT-IF PREDICTION
    # --------------------------------------------------------

    try:

        whatif_processed = (
            preprocessor.transform(
                whatif_df
            )
        )

        whatif_probability = (
            gradient_boosting_model
            .predict_proba(
                whatif_processed
            )[0][1]
        )

        whatif_risk = (
            whatif_probability * 100
        )

        if whatif_risk < 30:

            whatif_category = "Low"

        elif whatif_risk <= 60:

            whatif_category = "Medium"

        else:

            whatif_category = "High"

        st.markdown("---")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                "Original Risk",
                f"{employee_risk:.2f}%"
            )

        with result_col2:

            st.metric(
                "What-If Predicted Risk",
                f"{whatif_risk:.2f}%"
            )

        st.info(
            f"Predicted Risk Category: "
            f"**{whatif_category}**"
        )

        risk_change = (
            whatif_risk - employee_risk
        )

        if risk_change > 0:

            st.warning(
                f"Risk increased by "
                f"**{risk_change:.2f} percentage points**."
            )

        elif risk_change < 0:

            st.success(
                f"Risk decreased by "
                f"**{abs(risk_change):.2f} "
                f"percentage points**."
            )

        else:

            st.info(
                "The predicted risk remained unchanged."
            )

    except Exception as error:

        st.error(
            f"What-If prediction could not be calculated: "
            f"{error}"
        )

    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    render_divider()

    render_section_title(
        "📊 Model Performance",
        "Test-set performance of the trained machine learning models."
    )

    metrics_file = (
        "models/model_metrics.json"
    )

    if os.path.exists(metrics_file):

        with open(
            metrics_file,
            "r",
            encoding="utf-8"
        ) as file:

            model_metrics = json.load(
                file
            )

        metrics_df = (
            pd.DataFrame(
                model_metrics
            ).T
        )

        st.subheader(
            "Test-Set Model Performance"
        )

        st.dataframe(
            metrics_df.round(4),
            use_container_width=True
        )

        for model_name, metrics in model_metrics.items():

            st.markdown(
                f"### {model_name}"
            )

            metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = (
                st.columns(5)
            )

            with metric_col1:

                st.metric(
                    "Accuracy",
                    f"{metrics['Accuracy']:.4f}"
                )

            with metric_col2:

                st.metric(
                    "Precision",
                    f"{metrics['Precision']:.4f}"
                )

            with metric_col3:

                st.metric(
                    "Recall",
                    f"{metrics['Recall']:.4f}"
                )

            with metric_col4:

                st.metric(
                    "F1 Score",
                    f"{metrics['F1 Score']:.4f}"
                )

            with metric_col5:

                st.metric(
                    "ROC-AUC",
                    f"{metrics['ROC-AUC']:.4f}"
                )

        st.subheader(
            "ROC-AUC Comparison"
        )

        st.bar_chart(
            metrics_df["ROC-AUC"]
        )

    else:

        st.error(
            "Model metrics file not found. "
            "Please run train_model.py first."
        )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    render_divider()

    render_section_title(
        "🔎 Feature Importance",
        "Features contributing to the Gradient Boosting model."
    )

    try:

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

        importances = (
            gradient_boosting_model
            .feature_importances_
        )

        feature_importance_df = (
            pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Importance": importances
                }
            )
            .sort_values(
                "Importance",
                ascending=False
            )
            .head(20)
        )

        st.dataframe(
            feature_importance_df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            "Top 20 Feature Importance"
        )

        feature_chart = (
            feature_importance_df
            .set_index("Feature")["Importance"]
            .sort_values()
        )

        st.bar_chart(
            feature_chart
        )

    except Exception as error:

        st.warning(
            f"Feature importance could not be displayed: "
            f"{error}"
        )

    # ========================================================
    # FILTERED REPORT DOWNLOAD
    # ========================================================

    render_divider()

    render_section_title(
        "📥 Download Risk Report",
        "Download the currently filtered employee risk report."
    )

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

    available_download_columns = [
        column
        for column in download_columns
        if column in filtered_df.columns
    ]

    download_data = filtered_df[
        available_download_columns
    ].copy()

    csv_data = (
        download_data
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        label="📥 Download Filtered Employee Risk Report",
        data=csv_data,
        file_name="employee_risk_report_filtered.csv",
        mime="text/csv"
    )

    # ========================================================
    # EXECUTIVE SUMMARY
    # ========================================================

    render_divider()

    render_section_title(
        "📌 Executive Summary",
        "High-level workforce observations based on the current filters."
    )

    total_filtered = len(
        filtered_df
    )

    if total_filtered > 0:

        high_risk_percentage = (
            len(
                filtered_df[
                    filtered_df["RiskCategory"]
                    == "High"
                ]
            )
            / total_filtered
            * 100
        )

        medium_risk_percentage = (
            len(
                filtered_df[
                    filtered_df["RiskCategory"]
                    == "Medium"
                ]
            )
            / total_filtered
            * 100
        )

        low_risk_percentage = (
            len(
                filtered_df[
                    filtered_df["RiskCategory"]
                    == "Low"
                ]
            )
            / total_filtered
            * 100
        )

        attrition_count = (
            filtered_df["Attrition"]
            .sum()
        )

        attrition_rate = (
            attrition_count
            / total_filtered
            * 100
        )

        overtime_count = len(
            filtered_df[
                filtered_df["OverTime"]
                == "Yes"
            ]
        )

        overtime_percentage = (
            overtime_count
            / total_filtered
            * 100
        )

        summary_col1, summary_col2, summary_col3, summary_col4 = (
            st.columns(4)
        )

        with summary_col1:

            st.metric(
                "High Risk %",
                f"{high_risk_percentage:.2f}%"
            )

        with summary_col2:

            st.metric(
                "Medium Risk %",
                f"{medium_risk_percentage:.2f}%"
            )

        with summary_col3:

            st.metric(
                "Actual Attrition %",
                f"{attrition_rate:.2f}%"
            )

        with summary_col4:

            st.metric(
                "Overtime %",
                f"{overtime_percentage:.2f}%"
            )

        st.subheader(
            "Key Observations"
        )

        st.write(
            f"• **{high_risk_percentage:.2f}%** "
            f"of employees currently fall into "
            f"the High Risk category."
        )

        st.write(
            f"• **{medium_risk_percentage:.2f}%** "
            f"of employees fall into "
            f"the Medium Risk category."
        )

        st.write(
            f"• The observed attrition rate "
            f"in the filtered dataset is "
            f"**{attrition_rate:.2f}%**."
        )

        st.write(
            f"• **{overtime_percentage:.2f}%** "
            f"of employees have reported "
            f"working overtime."
        )

    # ========================================================
    # FOOTER
    # ========================================================

    render_footer()