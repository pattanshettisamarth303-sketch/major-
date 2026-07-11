from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("Heart Disease Prediction")

if not MODEL_PATH.exists():
    st.error("Model not found. Run `python src/train_models.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

with st.form("prediction_form"):
    age = st.slider("Age", 20, 85, 52)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest pain type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting blood pressure", 80, 220, 130)
    chol = st.number_input("Cholesterol", 100, 600, 245)
    fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    restecg = st.selectbox("Resting ECG result", [0, 1, 2])
    thalach = st.number_input("Maximum heart rate achieved", 60, 230, 150)
    exang = st.selectbox("Exercise-induced angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak = st.number_input("ST depression", 0.0, 7.0, 1.0, step=0.1)
    slope = st.selectbox("ST segment slope", [0, 1, 2])
    ca = st.selectbox("Major vessels colored by fluoroscopy", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3])
    submitted = st.form_submit_button("Predict")

if submitted:
    row = pd.DataFrame(
        [
            {
                "age": age,
                "sex": sex,
                "cp": cp,
                "trestbps": trestbps,
                "chol": chol,
                "fbs": fbs,
                "restecg": restecg,
                "thalach": thalach,
                "exang": exang,
                "oldpeak": oldpeak,
                "slope": slope,
                "ca": ca,
                "thal": thal,
            }
        ]
    )
    probability = float(model.predict_proba(row)[0, 1])
    prediction = int(probability >= 0.5)

    st.metric("Heart disease probability", f"{probability:.1%}")
    if prediction:
        st.warning("Prediction: likely heart disease risk.")
    else:
        st.success("Prediction: lower predicted heart disease risk.")

    st.caption("Educational model output only. It is not a medical diagnosis.")
