import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load the model, scaler, and SHAP background
model = joblib.load('../models/diabetes_model.pkl')
scaler = joblib.load('../models/scaler.pkl')
background = pd.read_csv('../data/processed/shap_background.csv')

# Load feature template (with all one-hot columns used during training)
template = pd.read_csv('../data/processed/template_input.csv')
template.iloc[0] = 0  # Reset all values to 0

# ✅ Robust feature name extraction
try:
    # For XGBoost
    booster = model.get_booster()
    model_features = booster.feature_names if booster.feature_names is not None else template.columns.tolist()
except Exception:
    try:
        # For sklearn models
        model_features = model.feature_names_in_.tolist()
    except Exception:
        # Fallback to template
        model_features = template.columns.tolist()

template_features = template.columns.tolist()

# ✅ Check for feature name mismatch
if set(template_features) != set(model_features):
    st.error("⚠ Template feature mismatch with the trained model!\n\n"
             f"🧩 Missing from template: {set(model_features) - set(template_features)}\n"
             f"🧩 Extra in template: {set(template_features) - set(model_features)}")
    st.stop()

# Initialize SHAP explainer
explainer = shap.Explainer(model, background)

# Streamlit app layout
st.set_page_config(page_title="Healthcare Readmission Risk", layout="centered")
st.title("🏥 Patient Readmission Risk Predictor")
st.markdown("Estimate the risk of readmission using a trained machine learning model and SHAP explainability.")

# Sidebar inputs
st.sidebar.header("📋 Patient Information")
number_inpatient = st.sidebar.slider("Number of Inpatient Visits", 0, 20, 1)
discharge_disposition_id = st.sidebar.selectbox("Discharge Disposition ID", list(range(1, 30)))
number_diagnoses = st.sidebar.slider("Number of Diagnoses", 1, 20, 5)
time_in_hospital = st.sidebar.slider("Time in Hospital (days)", 1, 14, 5)
diabetesMed = st.sidebar.selectbox("Diabetes Medication Prescribed?", ["Yes", "No"])
num_medications = st.sidebar.slider("Number of Medications Prescribed", 1, 50, 10)

# Construct the input row from the template
input_data = template.copy()
input_data.iloc[0] = 0  # reset all values

input_data.at[0, 'number_inpatient'] = number_inpatient
input_data.at[0, 'discharge_disposition_id'] = discharge_disposition_id
input_data.at[0, 'number_diagnoses'] = number_diagnoses
input_data.at[0, 'time_in_hospital'] = time_in_hospital
input_data.at[0, 'diabetesMed'] = 1 if diabetesMed == "Yes" else 0
input_data.at[0, 'num_medications'] = num_medications

# Optional: Display entered input
st.subheader("📄 Patient Input Summary")
st.dataframe(input_data.T[input_data.T[0] != 0])

# Scale input
input_scaled = scaler.transform(input_data)

# Prediction
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]
confidence = "🔴 Low Confidence" if 0.4 <= probability <= 0.6 else "🟢 High Confidence"

# Output
st.subheader("🧪 Prediction Result")
st.write(f"🔹 **Risk of Readmission**: {'✅ Yes' if prediction == 1 else '❌ No'}")
st.write(f"📊 **Probability of Readmission**: {probability:.2%}")
st.write(f"🎯 **Confidence**: {confidence}")

# SHAP Explanation
st.subheader("🔍 SHAP Explanation")
shap_values = explainer(input_data)

# Plot SHAP waterfall
# st.set_option('deprecation.showPyplotGlobalUse', False)
# Fix: Explicitly create figure
fig, ax = plt.subplots(figsize=(10, 6))
shap.plots.waterfall(shap_values[0], show=False)  # show=False prevents immediate display
st.pyplot(fig)
