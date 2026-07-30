import streamlit as st
import pandas as pd
import joblib
import numpy as np
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)
@st.cache_resource
def load_models():
    """Load and cache ML model, scaler, and feature columns."""
    try:
        model   = joblib.load("KNN_heart.pkl")
        scaler  = joblib.load("scaler.pkl")
        columns = joblib.load("columns.pkl")
        return model, scaler, columns
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}")
        st.stop()  # Halt app if models are missing


model, scaler, expected_columns = load_models()
st.sidebar.title("❤️ Heart AI Predictor")
st.sidebar.info(
    """
    **ML-Based Heart Disease Prediction**

    - **Model:** KNN Classifier
    - **Features:** 11 clinical inputs
    - **Developed by:** Mahadi 🚀
    """
)
st.sidebar.warning(
    "⚠️ This tool is for educational purposes only. "
    "Always consult a medical professional."
)
st.title("❤️ Heart Disease Prediction")
st.caption("An ML-powered system to estimate heart disease risk")

st.markdown(
    "Enter patient health information below. "
    "The model will analyze patterns and **predict risk**."
)

st.divider()
with st.form("prediction_form"):

    st.subheader("🩺 Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider(
            "Age",
            min_value=18,
            max_value=100,
            value=40,
            help="Patient's age in years"
        )

        sex = st.selectbox(
            "Gender",
            options=["M", "F"],
            help="Biological sex of the patient"
        )

        chest_pain = st.selectbox(
            "Chest Pain Type",
            options=["ATA", "NAP", "TA", "ASY"],
            help=(
                "ATA: Atypical Angina | "
                "NAP: Non-Anginal Pain | "
                "TA: Typical Angina | "
                "ASY: Asymptomatic"
            )
        )

        resting_bp = st.number_input(
            "Resting Blood Pressure (mmHg)",
            min_value=80,
            max_value=200,
            value=120,
            help="Resting blood pressure in mmHg"
        )

        cholesterol = st.number_input(
            "Cholesterol Level (mg/dL)",
            min_value=100,
            max_value=600,
            value=200,
            help="Serum cholesterol in mg/dL"
        )

    with col2:
        fasting_bs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL",
            options=[0, 1],
            format_func=lambda x: "Yes (>120)" if x == 1 else "No (≤120)",
            help="1 if fasting blood sugar > 120 mg/dL"
        )

        resting_ecg = st.selectbox(
            "Resting ECG Results",
            options=["Normal", "ST", "LVH"],
            help=(
                "Normal | "
                "ST: ST-T wave abnormality | "
                "LVH: Left Ventricular Hypertrophy"
            )
        )

        max_hr = st.slider(
            "Maximum Heart Rate (bpm)",
            min_value=60,
            max_value=220,
            value=150,
            help="Maximum heart rate achieved"
        )

        exercise_angina = st.selectbox(
            "Exercise-Induced Angina",
            options=["Y", "N"],
            format_func=lambda x: "Yes" if x == "Y" else "No",
            help="Angina induced by exercise"
        )

        oldpeak = st.slider(
            "Oldpeak (ST Depression)",
            min_value=0.0,
            max_value=6.0,
            value=1.0,
            step=0.1,
            help="ST depression induced by exercise relative to rest"
        )

    st_slope = st.selectbox(
        "ST Slope",
        options=["Up", "Flat", "Down"],
        help="Slope of the peak exercise ST segment"
    )

    st.divider()
    submitted = st.form_submit_button(
        "🔍 Predict Heart Risk",
        use_container_width=True
    )
def build_input_dataframe(
    age, sex, chest_pain, resting_bp,
    cholesterol, fasting_bs, resting_ecg,
    max_hr, exercise_angina, oldpeak,
    st_slope, expected_columns
):
    """
    Build a properly encoded input DataFrame
    matching the training feature columns.
    """
    # Numeric features
    raw_input = {
        'Age':         age,
        'RestingBP':   resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS':   fasting_bs,
        'MaxHR':       max_hr,
        'Oldpeak':     oldpeak,
    }

    # One-hot encoded features
    one_hot_fields = {
        'Sex':              sex,
        'ChestPainType':    chest_pain,
        'RestingECG':       resting_ecg,
        'ExerciseAngina':   exercise_angina,
        'ST_Slope':         st_slope,
    }

    for prefix, value in one_hot_fields.items():
        raw_input[f"{prefix}_{value}"] = 1
    input_df = pd.DataFrame([raw_input])
    input_df = input_df.reindex(
        columns=expected_columns,
        fill_value=0
    )

    return input_df


if submitted:
    
    input_df = build_input_dataframe(
        age, sex, chest_pain, resting_bp,
        cholesterol, fasting_bs, resting_ecg,
        max_hr, exercise_angina, oldpeak,
        st_slope, expected_columns
    )


    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)[0]


    probability = None
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(scaled_input)[0][1]

    # ── Display Results ───
    st.divider()
    st.subheader("📋 Prediction Results")

    if prediction == 1:
        st.error(
            "### ⚠️ High Risk Detected\n\n"
            "The model predicts a **higher likelihood** of heart disease. "
            "Please consult a qualified healthcare professional immediately."
        )
    else:
        st.success(
            "### ✅ Low Risk Detected\n\n"
            "The model predicts a **lower likelihood** of heart disease. "
            "Maintain a healthy lifestyle and have regular check-ups."
        )

    # Metrics row
    if probability is not None:  
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.metric(
                label="Risk Probability",
                value=f"{probability * 100:.1f}%"
            )
        with col_b:
            st.metric(
                label="Prediction",
                value="High Risk" if prediction == 1 else "Low Risk"
            )
        with col_c:
            st.metric(
                label="Confidence",
                value=f"{max(probability, 1 - probability) * 100:.1f}%"
            )

    # Show input summary
    with st.expander("📊 View Input Summary"):
        summary = {
            "Age": age,
            "Gender": sex,
            "Chest Pain": chest_pain,
            "Resting BP": f"{resting_bp} mmHg",
            "Cholesterol": f"{cholesterol} mg/dL",
            "Fasting BS": "Yes" if fasting_bs == 1 else "No",
            "Resting ECG": resting_ecg,
            "Max Heart Rate": f"{max_hr} bpm",
            "Exercise Angina": exercise_angina,
            "Oldpeak": oldpeak,
            "ST Slope": st_slope
        }
        st.table(pd.DataFrame(
            summary.items(),
            columns=["Feature", "Value"]
        ))

st.divider()
st.caption(
    "Built with Python · Scikit-learn · Streamlit | By Mahadi 🚀  \n"
    "⚠️ For educational use only — not a substitute for medical advice."
)
