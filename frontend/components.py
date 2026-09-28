# ============================================================
# EMPLOYEE RISK INTELLIGENCE
# FRONTEND COMPONENTS
# ============================================================

import streamlit as st
from pathlib import Path


# ============================================================
# LOAD GLOBAL CSS
# ============================================================

def load_css():
    """
    Load the main dashboard stylesheet from styles.css.
    """

    css_path = Path(__file__).parent / "styles.css"

    if css_path.exists():

        with open(css_path, "r", encoding="utf-8") as file:
            css = file.read()

        st.markdown(
            f"<style>{css}</style>",
            unsafe_allow_html=True
        )

    else:

        st.error(
            "Frontend stylesheet not found: frontend/styles.css"
        )


# ============================================================
# DASHBOARD HEADER
# ============================================================

def render_header():

    header_html = (
        '<div class="dashboard-header">'
        '<div class="header-left">'
        '<div class="header-icon">🛡️</div>'
        '<div>'
        '<div class="dashboard-title">Employee Risk Intelligence</div>'
        '<div class="dashboard-subtitle">'
        'Machine Learning-Based Employee Attrition '
        'Prediction &amp; Risk Scoring'
        '</div>'
        '</div>'
        '</div>'
        '<div class="header-status">'
        '<span class="status-dot"></span>'
        'ML SYSTEM ACTIVE'
        '</div>'
        '</div>'
    )

    st.markdown(
        header_html,
        unsafe_allow_html=True
    )

    description_html = (
        '<div class="dashboard-description">'
        'Monitor employee attrition risk, identify high-risk '
        'profiles, analyze workforce patterns, and explore '
        'potential risk changes using machine learning predictions.'
        '</div>'
    )

    st.markdown(
        description_html,
        unsafe_allow_html=True
    )


# ============================================================
# SECTION HEADER
# ============================================================

def render_section_title(title, subtitle=None):

    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True
    )

    if subtitle:

        st.markdown(
            f'<div class="dashboard-subtitle">{subtitle}</div>',
            unsafe_allow_html=True
        )


# ============================================================
# KPI SECTION
# ============================================================

def render_kpi_cards(
    total_employees,
    high_risk,
    medium_risk,
    low_risk
):

    st.markdown(
        "### Workforce Risk Overview"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Employees",
            f"{total_employees:,}",
            help="Total employees in the current dataset."
        )

    with col2:

        st.metric(
            "High Risk",
            f"{high_risk:,}",
            help="Employees with predicted attrition risk above 60%."
        )

    with col3:

        st.metric(
            "Medium Risk",
            f"{medium_risk:,}",
            help="Employees with predicted attrition risk between 30% and 60%."
        )

    with col4:

        st.metric(
            "Low Risk",
            f"{low_risk:,}",
            help="Employees with predicted attrition risk below 30%."
        )


# ============================================================
# RISK BADGE
# ============================================================

def render_risk_badge(risk_category):

    category = str(risk_category).lower()

    if category == "high":

        css_class = "risk-high"
        icon = "🔴"

    elif category == "medium":

        css_class = "risk-medium"
        icon = "🟡"

    else:

        css_class = "risk-low"
        icon = "🟢"

    badge_html = (
        f'<span class="risk-badge {css_class}">'
        f'{icon} {risk_category}'
        f'</span>'
    )

    st.markdown(
        badge_html,
        unsafe_allow_html=True
    )


# ============================================================
# EMPLOYEE PROFILE CARD
# ============================================================

def render_employee_card(
    employee_id,
    department,
    job_role,
    risk_score,
    risk_category
):

    employee_html = (
        '<div class="employee-card">'
        '<div class="employee-card-title">'
        '👤 Employee Risk Profile'
        '</div>'
        f'<p><strong>Employee ID:</strong> {employee_id}</p>'
        f'<p><strong>Department:</strong> {department}</p>'
        f'<p><strong>Job Role:</strong> {job_role}</p>'
        f'<p><strong>Predicted Risk:</strong> {risk_score:.2f}%</p>'
        f'<p><strong>Risk Category:</strong> {risk_category}</p>'
        '</div>'
    )

    st.markdown(
        employee_html,
        unsafe_allow_html=True
    )


# ============================================================
# INFORMATION CARD
# ============================================================

def render_info_card(
    title,
    value,
    description=""
):

    info_html = (
        '<div class="employee-card">'
        f'<div class="employee-card-title">{title}</div>'
        f'<div style="font-size:1.8rem;'
        f'font-weight:750;'
        f'color:#0f172a;'
        f'margin-bottom:6px;">'
        f'{value}'
        '</div>'
        f'<div style="color:#64748b;'
        f'font-size:0.88rem;">'
        f'{description}'
        '</div>'
        '</div>'
    )

    st.markdown(
        info_html,
        unsafe_allow_html=True
    )


# ============================================================
# DIVIDER
# ============================================================

def render_divider():

    st.markdown(
        "<hr>",
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

def render_footer():

    footer_html = (
        '<hr>'
        '<div style="text-align:center;'
        'color:#94a3b8;'
        'font-size:0.82rem;'
        'padding:15px 0;">'
        'Employee Risk Intelligence System'
        '<br>'
        'Machine Learning-Based Employee Attrition Prediction'
        '</div>'
    )

    st.markdown(
        footer_html,
        unsafe_allow_html=True
    )