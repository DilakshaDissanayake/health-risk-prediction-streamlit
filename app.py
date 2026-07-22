import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

st.set_page_config(
    page_title="Student Health Risk Prediction System",
    page_icon="💚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        :root {
            --bg: #f3faf5;
            --bg-alt: #e8f7ec;
            --surface: rgba(255, 255, 255, 0.82);
            --surface-strong: rgba(255, 255, 255, 0.94);
            --border: rgba(34, 197, 94, 0.14);
            --text: #163220;
            --text-soft: rgba(22, 50, 32, 0.72);
            --accent: #16a34a;
            --accent-2: #22c55e;
            --accent-3: #10b981;
            --shadow: 0 18px 45px rgba(22, 50, 32, 0.12);
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --bg: #08130c;
                --bg-alt: #0d1d12;
                --surface: rgba(10, 22, 14, 0.78);
                --surface-strong: rgba(10, 22, 14, 0.92);
                --border: rgba(134, 239, 172, 0.16);
                --text: #ecfdf3;
                --text-soft: rgba(236, 253, 243, 0.76);
                --accent: #4ade80;
                --accent-2: #22c55e;
                --accent-3: #34d399;
                --shadow: 0 18px 52px rgba(0, 0, 0, 0.34);
            }
        }

        .stApp {
            background:
                radial-gradient(circle at top left, color-mix(in srgb, var(--accent) 18%, transparent), transparent 30%),
                radial-gradient(circle at top right, color-mix(in srgb, var(--accent-3) 16%, transparent), transparent 26%),
                linear-gradient(135deg, var(--bg) 0%, var(--bg-alt) 100%);
            color: var(--text);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .hero-card, .panel-card, .result-card {
            background: var(--surface);
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
            backdrop-filter: blur(14px);
            border-radius: 24px;
        }

        .hero-card {
            padding: 1.5rem 1.75rem;
            margin-bottom: 1rem;
            background:
                linear-gradient(135deg, color-mix(in srgb, var(--surface-strong) 82%, transparent), var(--surface)),
                var(--surface);
        }

        .hero-title {
            font-size: 2.4rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 0.35rem;
            color: var(--text);
        }

        .hero-subtitle {
            font-size: 1rem;
            color: var(--text-soft);
            line-height: 1.6;
        }

        .badge-row {
            display: flex;
            gap: 0.65rem;
            flex-wrap: wrap;
            margin-top: 1rem;
        }

        .badge {
            padding: 0.42rem 0.8rem;
            border-radius: 999px;
            font-size: 0.85rem;
            background: color-mix(in srgb, var(--accent) 12%, transparent);
            border: 1px solid color-mix(in srgb, var(--accent) 20%, transparent);
            color: var(--text);
        }

        .panel-card {
            padding: 1.25rem 1.25rem 0.85rem;
        }

        .section-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--text);
            margin-bottom: 0.25rem;
        }

        .section-copy {
            color: var(--text-soft);
            font-size: 0.92rem;
            margin-bottom: 1rem;
        }

        .result-card {
            padding: 1.25rem 1.4rem;
        }

        .result-label {
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.78rem;
            color: var(--text-soft);
            margin-bottom: 0.45rem;
        }

        .result-value {
            font-size: 2rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }

        .result-note {
            color: var(--text-soft);
            line-height: 1.6;
        }

        .result-card {
            background:
                linear-gradient(135deg, color-mix(in srgb, var(--surface-strong) 82%, transparent), var(--surface)),
                var(--surface);
        }

        .stTextInput > div > div,
        .stNumberInput > div > div,
        .stSelectbox > div > div {
            background: var(--surface-strong) !important;
            border-color: var(--border) !important;
            color: var(--text) !important;
            border-radius: 14px !important;
        }

        label,
        .stMarkdown,
        .stSelectbox label,
        .stNumberInput label {
            color: var(--text) !important;
        }

        .stForm {
            margin-top: 0.25rem;
        }

        div[data-testid="stForm"] {
            background: transparent;
        }

        button[kind="formSubmit"] {
            width: 100%;
            border-radius: 14px !important;
            border: none !important;
            background: linear-gradient(135deg, var(--accent), var(--accent-3)) !important;
            color: white !important;
            font-weight: 700 !important;
            padding: 0.85rem 1rem !important;
            box-shadow: 0 14px 30px color-mix(in srgb, var(--accent) 26%, transparent);
        }

        button[kind="formSubmit"]:hover {
            transform: translateY(-1px);
            box-shadow: 0 18px 34px color-mix(in srgb, var(--accent) 36%, transparent);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# 1. Load trained model and artifacts
@st.cache_resource
def load_resources():
    try:
        model = tf.keras.models.load_model(
            "model.h5",
            compile=False
        )

        scaler = joblib.load("scaler.pkl")
        ohe = joblib.load("onehotencoder.pkl")

        return model, scaler, ohe

    except Exception as e:
        st.error("Failed to load AI model resources.")
        st.exception(e)
        st.stop()
    

model, scaler, ohe = load_resources()

# 2. UI Layout
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Student Health Risk Prediction System</div>
        <div class="hero-subtitle">
            Enter a few lifestyle signals and get an instant health-condition estimate.
            The interface now uses a light green system with dark-mode matching for a cleaner experience.
        </div>
        <div class="badge-row">
            <div class="badge">Fast prediction</div>
            <div class="badge">Lifestyle-aware model</div>
            <div class="badge">Responsive layout</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("prediction_form"):
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Personal and lifestyle details</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">These values should match the way the model was trained.</div>', unsafe_allow_html=True)

    left_col, right_col = st.columns(2, gap="large")

    with left_col:
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
        sleep_duration = st.number_input("Sleep Duration (hours)", min_value=0.0, max_value=15.0, value=7.0)
        diet_type = st.selectbox("Diet Type", ["balanced", "non-veg", "veg"])
        stress_level = st.selectbox("Stress Level", ["low", "medium", "high"])
        sleep_quality = st.selectbox("Sleep Quality", ["average", "good", "poor"])

    with right_col:
        physical_activity_level = st.selectbox("Physical Activity Level", ["active", "moderate", "sedentary"])
        smoking_alcohol = st.selectbox("Smoking/Alcohol", ["no", "occasional", "yes"])
        gender = st.selectbox("Gender", ["female", "male", "other"])
        step_count = st.number_input("Step Count", min_value=0, max_value=20000, value=5000)
        calorie_expenditure = st.number_input("Calorie Expenditure", min_value=0.0, max_value=3000.0, value=500.0)

    bottom_col1, bottom_col2 = st.columns(2, gap="large")
    with bottom_col1:
        exercise_duration = st.number_input("Exercise Duration (mins)", min_value=0.0, max_value=300.0, value=30.0)
        heart_rate = st.number_input("Heart Rate", min_value=40, max_value=200, value=75)
    with bottom_col2:
        water_intake = st.number_input("Water Intake (liters)", min_value=0.0, max_value=5.0, value=2.0)

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    
    submit_button = st.form_submit_button("Predict Health Condition")
    st.markdown("</div>", unsafe_allow_html=True)

# 3. Prediction Logic
if submit_button:
    input_data = pd.DataFrame({
        'bmi': [bmi],
        'sleep_duration': [sleep_duration],
        'diet_type': [diet_type],
        'stress_level': [stress_level],
        'sleep_quality': [sleep_quality],
        'physical_activity_level': [physical_activity_level],
        'smoking_alcohol': [smoking_alcohol],
        'gender': [gender],
        'step_count': [step_count],
        'calorie_expenditure': [calorie_expenditure],
        'exercise_duration': [exercise_duration],
        'heart_rate': [heart_rate],
        'water_intake': [water_intake]
    })
    
    # Use the exact column order the saved preprocessors were fit on.
    num_cols = list(getattr(scaler, 'feature_names_in_', [
        'bmi', 'sleep_duration', 'step_count', 'calorie_expenditure',
        'exercise_duration', 'heart_rate', 'water_intake'
    ]))
    cat_cols = list(getattr(ohe, 'feature_names_in_', ['stress_level']))
    
    num_scaled = scaler.transform(input_data[num_cols])
    cat_encoded = ohe.transform(input_data[cat_cols])
    
    # Combine
    final_input = np.concatenate([num_scaled, cat_encoded], axis=1)
    
    # Predict
    prediction = model.predict(final_input)
    predicted_class = np.argmax(prediction, axis=1)
    
    labels = ["At-Risk", "Fit", "Unhealthy"]
    result = labels[predicted_class[0]]

    result_styles = {
        "At-Risk": ("#eab308", "Your inputs suggest elevated risk. Consider improving rest, activity, and recovery habits."),
        "Fit": ("#16a34a", "Your profile looks balanced. Keep the current routine consistent and monitor long-term trends."),
        "Unhealthy": ("#dc2626", "The model flags a higher-risk pattern. Review sleep, nutrition, and stress-related habits."),
    }
    result_color, result_message = result_styles[result]

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Prediction result</div>
            <div class="result-value" style="color: {result_color};">{result}</div>
            <div class="result-note">{result_message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    