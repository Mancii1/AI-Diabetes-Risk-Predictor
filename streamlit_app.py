import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time

# -----------------------------------------------------------------------------
# Page configuration – must be the first Streamlit command
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom CSS – modern design system, Nielsen-friendly
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* ---------- Global ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1e293b;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }
    #MainMenu, header, footer {visibility: hidden;}

    /* ---------- Sidebar dark theme ---------- */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stSelectSlider label {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] .sidebar-section-title {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stCaption {
        color: #cbd5e1 !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        background-color: #334155;
        color: white;
        border: none;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #475569;
    }

    /* ---------- Headers ---------- */
    .hero {
        margin-bottom: 1.5rem;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1d4ed8, #0ea5e9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #64748b;
        margin-top: 0.2rem;
    }

    /* ---------- Cards ---------- */
    .card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }
    .card-title {
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.6rem;
        color: #0f172a;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ---------- Prediction result card ---------- */
    .result-card {
        border-radius: 20px;
        padding: 1.8rem 1.5rem;
        background: #ffffff;
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        border-left: 6px solid #cbd5e1;
        display: flex;
        align-items: center;
        gap: 1.5rem;
        margin-bottom: 1.2rem;
    }
    .result-card.risk-0 {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left-color: #16a34a;
    }
    .result-card.risk-1 {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-left-color: #f59e0b;
    }
    .result-card.risk-2 {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border-left-color: #dc2626;
    }
    .risk-icon {
        font-size: 3.5rem;
        line-height: 1;
    }
    .risk-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
    }
    .risk-value {
        font-size: 2rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0.1rem 0;
    }
    .risk-confidence {
        font-size: 1rem;
        font-weight: 600;
        color: #475569;
    }
    .top-driver {
        font-size: 0.85rem;
        color: #475569;
        margin-top: 0.5rem;
    }

    /* ---------- Metric pills ---------- */
    .metrics-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.2rem;
    }
    .metric-pill {
        background: #ffffff;
        border-radius: 12px;
        padding: 0.7rem 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        flex: 1;
        text-align: center;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* ---------- HTML bar charts ---------- */
    .bar-chart {
        margin-top: 0.5rem;
    }
    .bar-row {
        display: flex;
        align-items: center;
        margin-bottom: 0.45rem;
    }
    .bar-label {
        width: 32%;
        font-size: 0.82rem;
        font-weight: 500;
        color: #334155;
        padding-right: 0.8rem;
        text-align: right;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .bar-track {
        flex: 1;
        height: 22px;
        background: #f1f5f9;
        border-radius: 6px;
        position: relative;
        overflow: hidden;
    }
    .bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.5s ease;
        position: relative;
    }
    .bar-fill.positive {
        background: linear-gradient(90deg, #f87171, #dc2626);
    }
    .bar-fill.negative {
        background: linear-gradient(90deg, #4ade80, #16a34a);
    }
    .bar-fill.neutral {
        background: linear-gradient(90deg, #93c5fd, #1d4ed8);
    }
    .bar-value {
        width: 12%;
        font-size: 0.75rem;
        font-weight: 600;
        text-align: left;
        padding-left: 0.5rem;
    }
    .bar-value.pos { color: #dc2626; }
    .bar-value.neg { color: #16a34a; }

    /* ---------- Confidence stacked bar ---------- */
    .confidence-bar {
        display: flex;
        height: 36px;
        border-radius: 10px;
        overflow: hidden;
        margin-bottom: 0.5rem;
    }
    .confidence-seg {
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 700;
        font-size: 0.8rem;
        transition: width 0.5s ease;
    }
    .seg-0 { background: #16a34a; }
    .seg-1 { background: #f59e0b; }
    .seg-2 { background: #dc2626; }
    .legend {
        display: flex;
        gap: 1rem;
        font-size: 0.75rem;
        color: #475569;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }
    .legend-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
        border: none;
    }
    .stButton > button[kind="primary"] {
        background-color: #2563eb;
        color: white;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #1d4ed8;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37,99,235,0.3);
    }
    .stButton > button[kind="secondary"] {
        background-color: #e2e8f0;
        color: #1e293b;
    }
    .stButton > button[kind="secondary"]:hover {
        background-color: #cbd5e1;
    }

    /* ---------- Sidebar typography ---------- */
    .sidebar-title {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #f8fafc;
    }
    .sidebar-caption {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-bottom: 1rem;
    }
    .sidebar-section-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0.8rem 0 0.3rem 0;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: #f1f5f9;
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: #ffffff;
        box-shadow: 0 -2px 4px rgba(0,0,0,0.05);
    }

    /* ---------- Footer ---------- */
    .footer {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e2e8f0;
        text-align: center;
        color: #94a3b8;
        font-size: 0.8rem;
    }

    /* ---------- Custom suggestion cards ---------- */
    .suggestion-card {
        background: #f8fafc;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        border: 1px solid #e2e8f0;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    .suggestion-icon {
        font-size: 1.5rem;
    }
    .suggestion-content {
        flex: 1;
    }
    .suggestion-title {
        font-weight: 600;
        color: #0f172a;
    }
    .suggestion-detail {
        font-size: 0.8rem;
        color: #475569;
    }
    .suggestion-target {
        background: #e2e8f0;
        border-radius: 999px;
        padding: 0.1rem 0.7rem;
        font-size: 0.7rem;
        font-weight: 600;
        color: #1e293b;
        white-space: nowrap;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Feature metadata
# -----------------------------------------------------------------------------
FEATURE_DISPLAY = {
    'HighBP': 'High Blood Pressure',
    'HighChol': 'High Cholesterol',
    'CholCheck': 'Cholesterol Check (5 yrs)',
    'BMI': 'Body Mass Index (BMI)',
    'Smoker': 'Smoker',
    'Stroke': 'Prior Stroke',
    'HeartDiseaseorAttack': 'Heart Disease or Attack',
    'PhysActivity': 'Physical Activity (30 days)',
    'Fruits': 'Eats Fruits (daily)',
    'Veggies': 'Eats Vegetables (daily)',
    'HvyAlcoholConsump': 'Heavy Alcohol Use',
    'AnyHealthcare': 'Healthcare Coverage',
    'NoDocbcCost': 'Skipped Care Due to Cost',
    'GenHlth': 'General Health',
    'MentHlth': 'Poor Mental Health Days',
    'PhysHlth': 'Poor Physical Health Days',
    'DiffWalk': 'Difficulty Walking',
    'Sex': 'Sex',
    'Age': 'Age Group',
    'Education': 'Education Level',
    'Income': 'Household Income',
}

FEATURE_ORDER = list(FEATURE_DISPLAY.keys())

BINARY_FEATURES = [
    'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk'
]

AGE_LABELS = {
    1: "18–24", 2: "25–29", 3: "30–34", 4: "35–39", 5: "40–44",
    6: "45–49", 7: "50–54", 8: "55–59", 9: "60–64", 10: "65–69",
    11: "70–74", 12: "75–79", 13: "80+"
}

EDUCATION_LABELS = {
    1: "Never attended", 2: "Elementary", 3: "Some HS",
    4: "HS graduate", 5: "Some college", 6: "College graduate+"
}

INCOME_LABELS = {
    1: "<$10k", 2: "$10–15k", 3: "$15–20k", 4: "$20–25k",
    5: "$25–35k", 6: "$35–50k", 7: "$50–75k", 8: ">$75k"
}

GENHLTH_LABELS = {
    1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"
}

# -----------------------------------------------------------------------------
# Dummy prediction logic (replace with real model later)
# -----------------------------------------------------------------------------
def _clamp_score(x):
    return max(0.0, min(0.95, x))

def _risk_score(row):
    """Compute a deterministic risk score from 0 to 0.95."""
    bmi = row['BMI']
    age = row['Age']
    genh = row['GenHlth']
    physact = row['PhysActivity']
    fruits = row['Fruits']
    veggies = row['Veggies']
    highbp = row['HighBP']
    highchol = row['HighChol']
    smoker = row['Smoker']
    stroke = row['Stroke']
    heart = row['HeartDiseaseorAttack']
    diffwalk = row['DiffWalk']
    physh = row['PhysHlth']
    menth = row['MentHlth']
    alcohol = row['HvyAlcoholConsump']
    no_doc = row['NoDocbcCost']
    income = row['Income']
    edu = row['Education']
    any_health = row['AnyHealthcare']

    score = 0.0

    # Body & lifestyle
    if bmi >= 30: score += 0.20
    elif bmi >= 25: score += 0.10
    if highbp == 1: score += 0.16
    if highchol == 1: score += 0.08
    if smoker == 1: score += 0.08
    if physact == 0: score += 0.06
    if fruits == 0 and veggies == 0: score += 0.04

    # General health
    if genh >= 4: score += 0.14
    elif genh == 3: score += 0.06
    if physh >= 15: score += 0.05
    elif physh >= 8: score += 0.03
    if menth >= 15: score += 0.03
    elif menth >= 8: score += 0.02

    # Medical history
    if stroke == 1: score += 0.05
    if heart == 1: score += 0.04
    if diffwalk == 1: score += 0.05

    # Access & demographics
    if no_doc == 1: score += 0.02
    if income <= 3: score += 0.03
    if edu <= 3: score += 0.02
    if alcohol == 1: score += 0.03

    # Protective factors
    if physact == 1: score -= 0.02
    if fruits == 1: score -= 0.02
    if veggies == 1: score -= 0.01

    return _clamp_score(score)

def dummy_predict(features_df):
    """Return (predicted_class_index, probabilities_array)."""
    row = features_df.iloc[0]
    score = _risk_score(row)

    # Soft class boundaries
    p_no = 1 / (1 + np.exp(-(0.42 - score) * 8))
    p_dia = (1 / (1 + np.exp(-(score - 0.60) * 8))) * 0.88
    p_pre = max(0.02, 1 - p_no - p_dia)

    # Normalize
    total = p_no + p_pre + p_dia
    probs = np.array([p_no, p_pre, p_dia]) / total
    return int(np.argmax(probs)), probs

# -----------------------------------------------------------------------------
# Dummy LIME explanation (local feature contributions)
# -----------------------------------------------------------------------------
def dummy_lime_explanation(features_df):
    """Return list of (feature_key, contribution_value) sorted by abs."""
    row = features_df.iloc[0]
    contribs = {}

    # BMI
    bmi = row['BMI']
    if bmi >= 30: contribs['BMI'] = 0.20
    elif bmi >= 25: contribs['BMI'] = 0.10
    else: contribs['BMI'] = -0.05

    # Blood pressure & cholesterol
    contribs['HighBP'] = 0.18 if row['HighBP'] == 1 else -0.06
    contribs['HighChol'] = 0.08 if row['HighChol'] == 1 else -0.05

    # Smoking & alcohol
    contribs['Smoker'] = 0.10 if row['Smoker'] == 1 else -0.04
    contribs['HvyAlcoholConsump'] = 0.05 if row['HvyAlcoholConsump'] == 1 else -0.02

    # Physical activity
    contribs['PhysActivity'] = -0.10 if row['PhysActivity'] == 1 else 0.10

    # Diet
    contribs['Fruits'] = -0.05 if row['Fruits'] == 1 else 0.04
    contribs['Veggies'] = -0.04 if row['Veggies'] == 1 else 0.03

    # General health
    gh = row['GenHlth']
    if gh >= 4: contribs['GenHlth'] = 0.12
    elif gh == 3: contribs['GenHlth'] = 0.05
    else: contribs['GenHlth'] = -0.08

    # Days health
    contribs['PhysHlth'] = 0.06 if row['PhysHlth'] >= 15 else (0.03 if row['PhysHlth'] >= 8 else -0.02)
    contribs['MentHlth'] = 0.04 if row['MentHlth'] >= 15 else (0.02 if row['MentHlth'] >= 8 else -0.01)

    # Medical history
    contribs['Stroke'] = 0.04 if row['Stroke'] == 1 else -0.02
    contribs['HeartDiseaseorAttack'] = 0.04 if row['HeartDiseaseorAttack'] == 1 else -0.03
    contribs['DiffWalk'] = 0.04 if row['DiffWalk'] == 1 else -0.02

    # Access & demographics
    contribs['NoDocbcCost'] = 0.02 if row['NoDocbcCost'] == 1 else -0.01
    contribs['Income'] = 0.03 if row['Income'] <= 3 else (0.0 if row['Income'] <= 5 else -0.02)
    contribs['Education'] = 0.02 if row['Education'] <= 3 else (-0.02 if row['Education'] >= 5 else 0.0)
    contribs['Age'] = 0.05 if row['Age'] >= 9 else (0.02 if row['Age'] >= 6 else -0.03)
    contribs['Sex'] = 0.01 if row['Sex'] == 1 else -0.01

    return sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)

# -----------------------------------------------------------------------------
# Dummy counterfactual generator
# -----------------------------------------------------------------------------
def dummy_counterfactual(features_df, predicted_class):
    """Return dict with suggestions, impact, and modified row."""
    row = features_df.iloc[0].copy()
    modified = row.copy()
    suggestions = []

    # BMI
    bmi = row['BMI']
    if bmi >= 30:
        target_bmi = max(27.0, bmi - 4)
        suggestions.append({
            "icon": "⚖️",
            "title": "Lose weight",
            "detail": f"Reducing BMI from {bmi:.1f} to {target_bmi:.1f} can significantly lower risk.",
            "current": f"BMI {bmi:.1f}",
            "target": f"BMI {target_bmi:.1f}"
        })
        modified['BMI'] = target_bmi
    elif bmi >= 25:
        target_bmi = max(23.0, bmi - 2)
        suggestions.append({
            "icon": "⚖️",
            "title": "Improve BMI",
            "detail": f"Lowering BMI from {bmi:.1f} to {target_bmi:.1f} is beneficial.",
            "current": f"BMI {bmi:.1f}",
            "target": f"BMI {target_bmi:.1f}"
        })
        modified['BMI'] = target_bmi

    # Blood pressure
    if row['HighBP'] == 1:
        suggestions.append({
            "icon": "🫀",
            "title": "Manage blood pressure",
            "detail": "Work with a doctor on medication or lifestyle to control hypertension.",
            "current": "High BP",
            "target": "Normal BP"
        })
        modified['HighBP'] = 0

    # Smoking
    if row['Smoker'] == 1:
        suggestions.append({
            "icon": "🚭",
            "title": "Quit smoking",
            "detail": "Stopping smoking improves cardiovascular health within weeks.",
            "current": "Smoker",
            "target": "Non‑smoker"
        })
        modified['Smoker'] = 0

    # Physical activity
    if row['PhysActivity'] == 0:
        suggestions.append({
            "icon": "🚶",
            "title": "Increase physical activity",
            "detail": "Aim for at least 150 minutes of moderate exercise per week.",
            "current": "Inactive",
            "target": "Active ≥30 min/day"
        })
        modified['PhysActivity'] = 1

    # Diet
    if row['Fruits'] == 0:
        suggestions.append({
            "icon": "🍎",
            "title": "Eat more fruits",
            "detail": "A daily serving of fruit helps reduce risk.",
            "current": "Low fruit intake",
            "target": "Daily fruit"
        })
        modified['Fruits'] = 1
    if row['Veggies'] == 0:
        suggestions.append({
            "icon": "🥦",
            "title": "Eat more vegetables",
            "detail": "Vegetables are protective against diabetes.",
            "current": "Low veggie intake",
            "target": "Daily vegetables"
        })
        modified['Veggies'] = 1

    # General health
    if row['GenHlth'] >= 3:
        suggestions.append({
            "icon": "🌟",
            "title": "Improve overall health",
            "detail": "Better self‑rated health correlates with lower risk. Regular checkups help.",
            "current": f"GenHlth {row['GenHlth']} ({GENHLTH_LABELS[row['GenHlth']]})",
            "target": f"GenHlth {max(1, row['GenHlth']-1)}"
        })
        modified['GenHlth'] = max(1, row['GenHlth'] - 1)

    if not suggestions:
        suggestions.append({
            "icon": "✅",
            "title": "Keep up the good work",
            "detail": "Maintain current healthy lifestyle and continue regular screening.",
            "current": "Healthy profile",
            "target": "Sustained health"
        })

    before_probs = dummy_predict(features_df)[1]
    after_probs = dummy_predict(pd.DataFrame([modified]))[1]
    after_class = int(np.argmax(after_probs))

    return {
        "suggestions": suggestions,
        "impact": {
            "before": before_probs,
            "after": after_probs,
            "after_class": after_class,
        },
        "modified": modified.to_dict()
    }

# -----------------------------------------------------------------------------
# Dummy global importance (SHAP-style)
# -----------------------------------------------------------------------------
@st.cache_data
def dummy_global_importance():
    importance = {
        'BMI': 0.22,
        'HighBP': 0.18,
        'Age': 0.14,
        'GenHlth': 0.11,
        'PhysActivity': 0.09,
        'Smoker': 0.08,
        'HighChol': 0.06,
        'DiffWalk': 0.04,
        'Fruits': 0.04,
        'Veggies': 0.03,
        'Income': 0.02,
        'Other': 0.05
    }
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

# -----------------------------------------------------------------------------
# Helper: format feature values for display
# -----------------------------------------------------------------------------
def format_feature_value(col, val):
    if col in BINARY_FEATURES:
        return "Yes" if val == 1 else "No"
    if col == 'BMI':
        return f"{val:.1f}"
    if col == 'GenHlth':
        return f"{int(val)} – {GENHLTH_LABELS[int(val)]}"
    if col == 'Age':
        return f"{int(val)} ({AGE_LABELS[int(val)]})"
    if col == 'Education':
        return f"{int(val)} – {EDUCATION_LABELS[int(val)]}"
    if col == 'Income':
        return f"{int(val)} – {INCOME_LABELS[int(val)]}"
    if col in ['MentHlth', 'PhysHlth']:
        return f"{int(val)} days"
    if col == 'Sex':
        return "Male" if val == 1 else "Female"
    return str(val)

# -----------------------------------------------------------------------------
# HTML builders for custom charts
# -----------------------------------------------------------------------------
def lime_chart_html(items, row_values):
    max_abs = max(abs(v) for _, v in items) if items else 1
    rows = []
    for feat, val in items:
        pct = min(100, int((abs(val) / max_abs) * 100))
        direction = 'positive' if val > 0 else 'negative'
        fill_style = f"width: {pct}%; background: {'#ef4444' if val > 0 else '#22c55e'};"
        value_display = f"{val:+.2f}"
        rows.append(f"""
        <div class="bar-row">
            <div class="bar-label" title="{FEATURE_DISPLAY.get(feat, feat)}">{FEATURE_DISPLAY.get(feat, feat)} <span style="font-size:0.7rem;color:#64748b;">({format_feature_value(feat, row_values[feat])})</span></div>
            <div class="bar-track">
                <div class="bar-fill {direction}" style="{fill_style}"></div>
            </div>
            <div class="bar-value {'pos' if val > 0 else 'neg'}">{value_display}</div>
        </div>
        """)
    return "\n".join(rows)

def confidence_bar_html(probs, class_names):
    colors = ['#16a34a', '#f59e0b', '#dc2626']
    segs = []
    for i, p in enumerate(probs):
        pct = p * 100
        label = f"{pct:.0f}%" if pct >= 10 else ""
        segs.append(f'<div class="confidence-seg seg-{i}" style="width:{pct:.2f}%;">{label}</div>')
    legend = "".join(
        f'<span class="legend-item"><span class="legend-dot" style="background:{colors[i]}"></span>{class_names[i]} {probs[i]:.0%}</span>'
        for i in range(3)
    )
    return f'<div class="confidence-bar">{"".join(segs)}</div><div class="legend">{legend}</div>'

def global_importance_html(importance_dict):
    max_val = max(importance_dict.values()) if importance_dict else 1
    rows = []
    for feat, val in importance_dict.items():
        pct = int((val / max_val) * 100)
        color = f"linear-gradient(90deg, #93c5fd, #1d4ed8)"
        rows.append(f"""
        <div class="bar-row">
            <div class="bar-label" style="color:#475569;">{feat}</div>
            <div class="bar-track">
                <div class="bar-fill neutral" style="width:{pct}%; background:{color};"></div>
            </div>
            <div class="bar-value" style="color:#1d4ed8;">{val:.2f}</div>
        </div>
        """)
    return "\n".join(rows)

# -----------------------------------------------------------------------------
# Preset profiles & reset
# -----------------------------------------------------------------------------
PROFILES = {
    "🥗 Low‑risk (healthy)": {
        'HighBP': 0, 'HighChol': 0, 'CholCheck': 1, 'BMI': 23.4,
        'Smoker': 0, 'Stroke': 0, 'HeartDiseaseorAttack': 0,
        'PhysActivity': 1, 'Fruits': 1, 'Veggies': 1, 'HvyAlcoholConsump': 0,
        'AnyHealthcare': 1, 'NoDocbcCost': 0, 'GenHlth': 2,
        'MentHlth': 0, 'PhysHlth': 0, 'DiffWalk': 0, 'Sex': 0,
        'Age': 4, 'Education': 5, 'Income': 6
    },
    "👤 Average adult": {
        'HighBP': 0, 'HighChol': 0, 'CholCheck': 1, 'BMI': 27.6,
        'Smoker': 0, 'Stroke': 0, 'HeartDiseaseorAttack': 0,
        'PhysActivity': 1, 'Fruits': 1, 'Veggies': 1, 'HvyAlcoholConsump': 0,
        'AnyHealthcare': 1, 'NoDocbcCost': 0, 'GenHlth': 3,
        'MentHlth': 3, 'PhysHlth': 2, 'DiffWalk': 0, 'Sex': 1,
        'Age': 7, 'Education': 4, 'Income': 4
    },
    "⚠️ High‑risk": {
        'HighBP': 1, 'HighChol': 1, 'CholCheck': 1, 'BMI': 35.2,
        'Smoker': 1, 'Stroke': 0, 'HeartDiseaseorAttack': 0,
        'PhysActivity': 0, 'Fruits': 0, 'Veggies': 0, 'HvyAlcoholConsump': 0,
        'AnyHealthcare': 1, 'NoDocbcCost': 1, 'GenHlth': 4,
        'MentHlth': 15, 'PhysHlth': 18, 'DiffWalk': 1, 'Sex': 1,
        'Age': 11, 'Education': 3, 'Income': 3
    }
}

DEFAULTS = {
    'HighBP': 1, 'HighChol': 0, 'CholCheck': 1, 'BMI': 31.0,
    'Smoker': 0, 'Stroke': 0, 'HeartDiseaseorAttack': 0,
    'PhysActivity': 0, 'Fruits': 1, 'Veggies': 1, 'HvyAlcoholConsump': 0,
    'AnyHealthcare': 1, 'NoDocbcCost': 0, 'GenHlth': 3,
    'MentHlth': 3, 'PhysHlth': 4, 'DiffWalk': 0, 'Sex': 0,
    'Age': 8, 'Education': 4, 'Income': 4
}

def apply_profile(profile_name):
    profile = PROFILES[profile_name]
    for k, v in profile.items():
        st.session_state[k] = v

def reset_to_defaults():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state.cf_visible = False
    st.session_state.cf_result = None

def generate_cf():
    input_df = pd.DataFrame([{k: st.session_state[k] for k in FEATURE_ORDER}])
    pred_class, _ = dummy_predict(input_df)
    cf_result = dummy_counterfactual(input_df, pred_class)
    st.session_state.cf_result = cf_result
    st.session_state.cf_visible = True
    st.toast("Counterfactual suggestions ready ✅")

def apply_cf_changes():
    modified = st.session_state.cf_result["modified"]
    for k, v in modified.items():
        st.session_state[k] = v
    st.session_state.cf_visible = False
    st.toast("Applied suggested changes to inputs ✨")

def hide_cf():
    st.session_state.cf_visible = False

# -----------------------------------------------------------------------------
# Initialize session state (run before widget creation)
# -----------------------------------------------------------------------------
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("cf_visible", False)
st.session_state.setdefault("cf_result", None)
st.session_state.setdefault("feedback_log", [])
st.session_state.setdefault("fb_flash", False)

# -----------------------------------------------------------------------------
# Sidebar – Inputs and presets
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">🧪 Patient Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-caption">Adjust any indicator — results update instantly</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🥗 Low‑risk", key="profile_low", use_container_width=True):
            apply_profile("🥗 Low‑risk (healthy)")
    with col2:
        if st.button("⚠️ High‑risk", key="profile_high", use_container_width=True):
            apply_profile("⚠️ High‑risk")
    if st.button("👤 Average adult", key="profile_avg", use_container_width=True):
        apply_profile("👤 Average adult")
    if st.button("↺ Reset to defaults", key="reset", use_container_width=True):
        reset_to_defaults()
    st.markdown("---")

    def binary_widget(col, label, help=None):
        val = st.session_state[col]
        user_choice = st.selectbox(
            label,
            options=[0, 1],
            index=val,
            format_func=lambda x: "Yes" if x == 1 else "No",
            key=col,
            help=help
        )
        return user_choice

    st.markdown('<div class="sidebar-section-title">🫀 Medical history</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    with cols[0]:
        binary_widget('HighBP', 'High Blood Pressure', help="Diagnosed with high blood pressure")
        binary_widget('HighChol', 'High Cholesterol', help="Diagnosed with high cholesterol")
        binary_widget('CholCheck', 'Cholesterol Check (5 yrs)', help="Had cholesterol check in past 5 years")
    with cols[1]:
        binary_widget('Stroke', 'Prior Stroke', help="Ever diagnosed with a stroke")
        binary_widget('HeartDiseaseorAttack', 'Heart Disease or Attack', help="Ever diagnosed with heart disease or heart attack")
        binary_widget('DiffWalk', 'Difficulty Walking', help="Serious difficulty walking or climbing stairs")

    st.markdown('<div class="sidebar-section-title">🏃 Lifestyle</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    with cols[0]:
        binary_widget('Smoker', 'Smoker', help="Smoked at least 100 cigarettes in lifetime")
        binary_widget('PhysActivity', 'Physical Activity (30 days)', help="Exercised in past 30 days (excluding job)")
        binary_widget('HvyAlcoholConsump', 'Heavy Alcohol Use', help="Men >14 drinks/week, women >7")
    with cols[1]:
        binary_widget('Fruits', 'Eats Fruits (daily)', help="Consumes fruit 1+ times per day")
        binary_widget('Veggies', 'Eats Vegetables (daily)', help="Consumes vegetables 1+ times per day")

    st.markdown('<div class="sidebar-section-title">⚖️ Body & well‑being</div>', unsafe_allow_html=True)
    bmi_val = st.slider(
        'Body Mass Index (BMI)',
        min_value=12.0, max_value=60.0, step=0.1,
        key='BMI',
        format="%.1f",
        help="Weight(kg)/height(m)² — healthy 18.5–24.9"
    )
    genh_val = st.select_slider(
        'General Health',
        options=list(range(1, 6)),
        key='GenHlth',
        format_func=lambda v: f"{v} – {GENHLTH_LABELS[v]}",
        help="Self‑rated health: 1=Excellent, 5=Poor"
    )
    menth_val = st.slider(
        'Poor Mental Health Days',
        min_value=0, max_value=30, step=1,
        key='MentHlth',
        format="%d days",
        help="Days of poor mental health in past 30 days"
    )
    physh_val = st.slider(
        'Poor Physical Health Days',
        min_value=0, max_value=30, step=1,
        key='PhysHlth',
        format="%d days",
        help="Days of poor physical health in past 30 days"
    )

    st.markdown('<div class="sidebar-section-title">🏥 Healthcare access</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    with cols[0]:
        binary_widget('AnyHealthcare', 'Healthcare Coverage', help="Has any kind of health care coverage")
    with cols[1]:
        binary_widget('NoDocbcCost', 'Skipped Care Due to Cost', help="Could not see doctor in past year due to cost")

    st.markdown('<div class="sidebar-section-title">👤 About you</div>', unsafe_allow_html=True)
    sex_val = st.radio(
        'Sex',
        options=[0, 1],
        index=st.session_state['Sex'],
        format_func=lambda v: "Female" if v == 0 else "Male",
        key='Sex',
        horizontal=True,
        help="Biological sex"
    )
    age_val = st.select_slider(
        'Age Group',
        options=list(range(1, 14)),
        key='Age',
        format_func=lambda v: f"{v} – {AGE_LABELS[v]}",
        help="Age category (1=18–24, 13=80+)"
    )
    edu_val = st.select_slider(
        'Education Level',
        options=list(range(1, 7)),
        key='Education',
        format_func=lambda v: f"{v} – {EDUCATION_LABELS[v]}",
        help="1=Never attended, 6=College graduate+"
    )
    income_val = st.select_slider(
        'Household Income',
        options=list(range(1, 9)),
        key='Income',
        format_func=lambda v: f"{v} – {INCOME_LABELS[v]}",
        help="Annual household income category"
    )

    st.markdown("---")
    st.caption("📋 Features: CDC BRFSS 2015 (21 indicators)  \n🧠 Model: rule‑based placeholder (swap with real ML)")
    with st.expander("ℹ️ About this prototype", expanded=False):
        st.markdown("""
        - **Purpose:** Demonstrate Human‑in‑the‑Loop AI & XAI concepts.
        - **Data:** The 21 features are derived from the CDC's Behavioral Risk Factor Surveillance System (BRFSS) 2015.
        - **Model:** Currently a deterministic dummy for demo; replace with a trained ensemble.
        - **Privacy:** No data is stored; feedback is session‑only.
        - **Disclaimer:** This is not a medical device.
        """)

# -----------------------------------------------------------------------------
# Main area
# -----------------------------------------------------------------------------
st.markdown('<div class="hero"><div class="hero-title">Diabetes Risk Predictor</div><div class="hero-subtitle">Interactive risk assessment with explainable AI — adjust inputs to see predictions and counterfactual scenarios</div></div>', unsafe_allow_html=True)

input_df = pd.DataFrame([{k: st.session_state[k] for k in FEATURE_ORDER}])
pred_class, probs = dummy_predict(input_df)
score = _risk_score(input_df.iloc[0])

CLASS_NAMES = ["No Diabetes", "Prediabetes", "Diabetes"]
CLASS_META = [
    {"icon": "🟢", "desc": "Current indicators suggest no diabetes. Keep up healthy habits and routine screening."},
    {"icon": "🟠", "desc": "Indicators are in the prediabetes range. Lifestyle changes now can prevent progression."},
    {"icon": "🔴", "desc": "Indicators are consistent with diabetes. Please consult a healthcare professional."},
]

lime_items = dummy_lime_explanation(input_df)
top_driver_key, top_driver_val = lime_items[0]
top_driver_display = f"{FEATURE_DISPLAY[top_driver_key]} ({format_feature_value(top_driver_key, st.session_state[top_driver_key])})"

active_risk_factors = sum([
    st.session_state['Smoker'] == 1,
    st.session_state['HvyAlcoholConsump'] == 1,
    st.session_state['PhysActivity'] == 0,
    (st.session_state['Fruits'] == 0 and st.session_state['Veggies'] == 0),
    st.session_state['HighBP'] == 1,
    st.session_state['HighChol'] == 1,
    st.session_state['Stroke'] == 1,
    st.session_state['HeartDiseaseorAttack'] == 1,
    st.session_state['DiffWalk'] == 1,
])

# Metrics row
st.markdown('<div class="metrics-row">', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-pill"><div class="metric-value">{st.session_state["BMI"]:.1f}</div><div class="metric-label">BMI</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-pill"><div class="metric-value">{GENHLTH_LABELS[st.session_state["GenHlth"]]}</div><div class="metric-label">General Health</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-pill"><div class="metric-value">{score:.2f}</div><div class="metric-label">Risk Score</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-pill"><div class="metric-value">{active_risk_factors} of 9</div><div class="metric-label">Risk Factors</div></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Prediction card
meta = CLASS_META[pred_class]
st.markdown(f"""
<div class="result-card risk-{pred_class}">
    <div class="risk-icon">{meta['icon']}</div>
    <div style="flex:1;">
        <div class="risk-label">Predicted Class</div>
        <div class="risk-value">{CLASS_NAMES[pred_class]}</div>
        <div class="risk-confidence">Confidence: {probs[pred_class]*100:.0f}%</div>
        <div class="top-driver">Top driver: <strong>{top_driver_display}</strong> ({top_driver_val:+.2f})</div>
        <div style="font-size:0.8rem; color:#64748b; margin-top:0.3rem;">{meta['desc']}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Confidence
st.markdown('<div class="card"><div class="card-title">📊 Prediction Confidence</div>', unsafe_allow_html=True)
st.markdown(confidence_bar_html(probs, CLASS_NAMES), unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Tabs
tab_lime, tab_shap, tab_guide = st.tabs(["🔎 Local Explanation (LIME)", "🌍 Global Importance (SHAP)", "❓ How to Read This"])

with tab_lime:
    top_8 = lime_items[:8]
    top_8_sorted = sorted(top_8, key=lambda x: x[1], reverse=True)
    st.markdown('<div class="card"><div class="card-title">🔎 LIME‑style Feature Contributions (Top 8)</div>', unsafe_allow_html=True)
    st.markdown(lime_chart_html(top_8_sorted, st.session_state), unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.8rem;color:#64748b;">🔴 right = increases risk &nbsp;|&nbsp; 🟢 left = decreases risk &nbsp;|&nbsp; Values are simulated local contributions.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab_shap:
    global_imp = dummy_global_importance()
    top_imp = dict(list(global_imp.items())[:10])
    st.markdown('<div class="card"><div class="card-title">🌍 Global Feature Importance (SHAP‑style, Top 10)</div>', unsafe_allow_html=True)
    st.markdown(global_importance_html(top_imp), unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.8rem;color:#64748b;">Simulated population‑level importance — what drives risk across the whole dataset, not for one individual.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab_guide:
    st.markdown("""
    <div class="card">
    <div class="card-title">❓ How to Read This Dashboard</div>

    **🟢🟠🔴 Risk Level** — The large colored card shows the predicted diabetes risk class based on the current inputs.  
    **📊 Confidence** — The stacked bar shows the model's certainty for each class.  
    **🔎 LIME Chart** — Local explanation for this specific patient. Red bars push toward higher risk, green bars push toward lower risk. The number in parentheses shows the actual feature value.  
    **🌍 SHAP Chart** — Global importance across all patients. It tells you which features matter most for the population, not just one person.  
    **💡 Counterfactuals** — Click "Suggest changes" to see minimal actions that could lower the risk. You can even apply them to the inputs with one click to see the effect.

    **Disclaimer:** This is a prototype demonstration. Outputs are simulated and must not be used for medical decisions.
    </div>
    """, unsafe_allow_html=True)

# Counterfactual section
st.markdown('<div class="card"><div class="card-title">💡 What‑If Exploration (Counterfactuals)</div>', unsafe_allow_html=True)

if not st.session_state.cf_visible:
    if st.button("🪄 Suggest changes to lower risk", type="primary", key="cf_button"):
        generate_cf()
else:
    cf = st.session_state.cf_result
    if cf:
        for s in cf["suggestions"]:
            st.markdown(f"""
            <div class="suggestion-card">
                <div class="suggestion-icon">{s["icon"]}</div>
                <div class="suggestion-content">
                    <div class="suggestion-title">{s["title"]}</div>
                    <div class="suggestion-detail">{s["detail"]}</div>
                </div>
                <div class="suggestion-target">{s["target"]}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### 📈 Simulated impact of applying suggestions")
        col1, col2, col3 = st.columns(3)
        for i, (name, color) in enumerate(zip(CLASS_NAMES, ['#16a34a','#f59e0b','#dc2626'])):
            before = cf["impact"]["before"][i] * 100
            after = cf["impact"]["after"][i] * 100
            delta = after - before
            with col1 if i == 0 else col2 if i == 1 else col3:
                st.metric(
                    label=name,
                    value=f"{after:.0f}%",
                    delta=f"{delta:+.1f} pp",
                    delta_color="off",
                )
        if cf["impact"]["after_class"] != pred_class:
            st.success(f"🎉 Predicted class would change to **{CLASS_NAMES[cf['impact']['after_class']]}**.")
        else:
            st.info("Predicted class stays the same, but probabilities shift toward lower risk.")

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("⚡ Apply changes to my inputs", type="secondary", key="apply_cf"):
                apply_cf_changes()
        with col_btn2:
            if st.button("🫥 Hide suggestions", key="hide_cf"):
                hide_cf()
    st.markdown('</div>', unsafe_allow_html=True)

# Feedback section
st.markdown('<div class="card"><div class="card-title">💬 Feedback</div>', unsafe_allow_html=True)
col1, col2 = st.columns([1, 2])
with col1:
    helpful = st.radio(
        "Was this explanation helpful?",
        ["Yes", "No", "Somewhat"],
        horizontal=True,
        key="fb_helpful",
        help="Your feedback helps improve the system"
    )
with col2:
    comment = st.text_area(
        "Additional comments (optional)",
        key="fb_comment",
        help="Share any suggestions or issues you encountered"
    )

if st.button("Submit feedback", key="fb_submit"):
    entry = {
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Helpful": helpful,
        "Comment": comment if comment else "—",
    }
    st.session_state.feedback_log.append(entry)
    st.session_state.fb_flash = True
    st.toast("Feedback saved 💜")

if st.session_state.fb_flash:
    st.success("✅ Thank you! Your feedback was recorded.")

if st.session_state.feedback_log:
    with st.expander("🗂️ Feedback log (this session)", expanded=False):
        fb_df = pd.DataFrame(st.session_state.feedback_log)
        st.dataframe(fb_df, use_container_width=True, hide_index=True)
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    ⚠️ Prototype demonstration — outputs are simulated for illustrative purposes only and must not be used for medical decisions.<br>
    Features: CDC BRFSS 2015 · Model: rule‑based placeholder · UI: Human‑in‑the‑Loop + XAI demo
</div>
""", unsafe_allow_html=True)