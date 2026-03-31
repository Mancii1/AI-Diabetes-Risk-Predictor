from pyexpat import model

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Diabetes Risk AI", layout="wide")

# -------------------------
# LOAD MODEL BUNDLE (CACHED)
# -------------------------

@st.cache_resource
def load_bundle():

    bundle = joblib.load("diabetes_model_bundle.pkl")

    if isinstance(bundle, dict):
        model = bundle["model"]
        scaler = bundle.get("scaler", None)
        feature_names = bundle.get("features", None)
    else:
        model = bundle
        scaler = None
        feature_names = None

    if feature_names is None:
        feature_names = [
            "HighBP","HighChol","CholCheck","BMI","Smoker","Stroke",
            "HeartDiseaseorAttack","PhysActivity","Fruits","Veggies",
            "HvyAlcoholConsump","AnyHealthcare","NoDocbcCost","GenHlth",
            "MentHlth","PhysHlth","DiffWalk","Sex","Age","Education","Income"
        ]

    xgb_model = model.named_estimators_["xgb"]

    explainer = shap.TreeExplainer(xgb_model)

    return model, scaler, feature_names, explainer


model, scaler, feature_names, explainer = load_bundle()

st.title("AI Diabetes Risk Prediction Dashboard")

st.write(
"Predict diabetes risk using health indicators and explain the decision using SHAP."
)

# -------------------------
# INPUT PANEL
# -------------------------

st.sidebar.header("Patient Health Information")

HighBP = st.sidebar.selectbox("High Blood Pressure", [0,1])
HighChol = st.sidebar.selectbox("High Cholesterol", [0,1])
CholCheck = st.sidebar.selectbox("Cholesterol Check", [0,1])
BMI = st.sidebar.slider("BMI",10.0,60.0,25.0)

Smoker = st.sidebar.selectbox("Smoker", [0,1])
Stroke = st.sidebar.selectbox("Stroke History", [0,1])
HeartDiseaseorAttack = st.sidebar.selectbox("Heart Disease", [0,1])

PhysActivity = st.sidebar.selectbox("Physical Activity", [0,1])
Fruits = st.sidebar.selectbox("Consumes Fruits", [0,1])
Veggies = st.sidebar.selectbox("Consumes Vegetables", [0,1])

HvyAlcoholConsump = st.sidebar.selectbox("Heavy Alcohol Consumption", [0,1])

AnyHealthcare = st.sidebar.selectbox("Access to Healthcare", [0,1])
NoDocbcCost = st.sidebar.selectbox("Doctor visit prevented by cost", [0,1])

GenHlth = st.sidebar.slider("General Health",1,5)
MentHlth = st.sidebar.slider("Mental Health Days (bad)",0,30)
PhysHlth = st.sidebar.slider("Physical Health Days (bad)",0,30)

DiffWalk = st.sidebar.selectbox("Difficulty Walking", [0,1])

Sex = st.sidebar.selectbox("Sex (0=Female,1=Male)", [0,1])

Age = st.sidebar.slider("Age Category",1,13)

Education = st.sidebar.slider("Education Level",1,6)

Income = st.sidebar.slider("Income Level",1,8)

# -------------------------
# PREDICTION
# -------------------------

if st.sidebar.button("Predict Risk"):

    features = pd.DataFrame([[

        HighBP,
        HighChol,
        CholCheck,
        BMI,
        Smoker,
        Stroke,
        HeartDiseaseorAttack,
        PhysActivity,
        Fruits,
        Veggies,
        HvyAlcoholConsump,
        AnyHealthcare,
        NoDocbcCost,
        GenHlth,
        MentHlth,
        PhysHlth,
        DiffWalk,
        Sex,
        Age,
        Education,
        Income

    ]], columns=feature_names)

    if scaler is not None:
        features_scaled = scaler.transform(features)
    else:
        features_scaled = features

    prediction = model.predict(features_scaled)
    probs = model.predict_proba(features_scaled)

    # -------------------------
    # RESULT PANEL
    # -------------------------

    st.header("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:

        if prediction[0] == 0:
            st.success("Low Diabetes Risk")
        elif prediction[0] == 1:
            st.warning("Prediabetes Risk")
        else:
            st.error("High Diabetes Risk")

        st.subheader("Risk Probabilities")

        prob_df = pd.DataFrame({
            "Risk Level": ["Low","Prediabetes","High"],
            "Probability": probs[0]
        })

        st.bar_chart(prob_df.set_index("Risk Level"))

    # -------------------------
    # SHAP EXPLANATION
    # -------------------------

    shap_values = explainer(features)

    with col2:

        st.subheader("Most Influential Factor")

        vals = np.abs(shap_values.values).mean(0)

        importance = pd.DataFrame({
            "Feature":feature_names,
            "Importance":np.ravel(vals)
        }).sort_values(by="Importance",ascending=False)

        st.write(
            f"Top contributing factor: **{importance.iloc[0]['Feature']}**"
        )

        st.dataframe(importance.head(5))

    # -------------------------
    # WATERFALL EXPLANATION
    # -------------------------

    st.header("Patient-Level Explanation")

    fig = plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

    # -------------------------
    # GLOBAL FEATURE IMPORTANCE
    # -------------------------

    st.header("Overall Model Feature Importance")

    fig2 = plt.figure()
    shap.plots.bar(shap_values, show=False)
    st.pyplot(fig2)