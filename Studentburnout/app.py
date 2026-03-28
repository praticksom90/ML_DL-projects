import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Student Burnout Analytics V3",
    layout="wide",
    page_icon="🛡️"
)

# 2. PROFESSIONAL STYLING (FIXED CSS)
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { 
        width: 100%; 
        border-radius: 8px; 
        height: 3.5em; 
        background-color: #ff4b4b; 
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        border: none;
    }
    div[data-testid="stMetricValue"] { font-size: 24px; }
    </style>
    """, unsafe_allow_html=True)

# 3. ASSET LOADING
@st.cache_resource
def load_v3_assets():
    try:
        model = joblib.load('burnout_model_v3.pkl')
        scaler = joblib.load('scaler_v3.pkl')
        features = joblib.load('features_v3.pkl')
        return model, scaler, features
    except Exception as e:
        return None, None, None

model, scaler, features = load_v3_assets()

# 4. ERROR HANDLING IF FILES ARE MISSING
if model is None:
    st.error("❌ **V3 Files Not Found!**")
    st.info("Pratick, please run your Jupyter Notebook (`model1.ipynb`) first to generate the `_v3.pkl` files in this folder.")
    st.stop()

# 5. HEADER
st.title("🎓 Student Burnout Risk Predictor and Advisor")
st.markdown(" ")
st.write("Analyze academic pressure and lifestyle habits to predict burnout risk and get recovery steps.")
st.markdown("---")

# 6. INPUT SECTION
# Mapping dictionaries
mental_map = {"Yes": 8.5, "No": 2.0, "Don't Know": 5.0}
qual_map = {"Poor": 0, "Average": 1, "Good": 2}

# Using columns for better layout
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📊 Academic & Lifestyle Habits")
    age = st.number_input("Age", 16, 30, 20)
    
    # Nested columns for sliders
    s1, s2 = st.columns(2)
    with s1:
        cgpa = st.slider("Current CGPA", 0.0, 10.0, 7.5, step=0.1)
        study_hrs = st.slider("Daily Study Hours", 0.0, 15.0, 5.0, step=0.1)
    with s2:
        attendance = st.slider("Attendance %", 0.0, 100.0, 80.0, step=0.1)
        sleep_hrs = st.slider("Daily Sleep Hours", 0.0, 12.0, 7.5, step=0.1)
    
    screen_time = st.slider("Daily Screen Time (Hrs)", 0.0, 18.0, 6.0, step=0.1)
    phys_act = st.slider("Physical Activity (Hrs)", 0.0, 6.0, 1.0, step=0.1)

with col_right:
    st.subheader("🧠 Mental Well-being Check")
    anx = st.radio("Do you experience frequent anxiety?", ["No", "Yes", "Don't Know"], horizontal=True)
    dep = st.radio("Do you feel low or depressed?", ["No", "Yes", "Don't Know"], horizontal=True)
    prs = st.radio("Is academic pressure overwhelming?", ["No", "Yes", "Don't Know"], horizontal=True)
    fin = st.radio("Are you stressed about finances?", ["No", "Yes", "Don't Know"], horizontal=True)
    
    st.markdown("---")
    s_qual = st.select_slider("Sleep Quality", options=["Poor", "Average", "Good"], value="Average")
    # Sidebar or Selectbox for text categories
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    course = st.selectbox("Course", ["BTech", "BSc", "BCA", "MTech", "MBA", "MCA"])
    year = st.selectbox("Academic Year", ["1st", "2nd", "3rd", "4th"])
    s_level_ui = st.selectbox("Self-Perceived Stress", ["Low", "Medium", "High"])

# 7. PREDICTION LOGIC
st.markdown("---")
if st.button("🚀 ANALYZE MY BURNOUT RISK"):
    # Encoding inputs
    g_enc = {"Male": 1, "Female": 0, "Other": 2}[gender]
    c_enc = {"BTech": 0, "BSc": 1, "BCA": 2, "MTech": 3, "MBA": 4, "MCA": 5}[course]
    y_enc = {"1st": 0, "2nd": 1, "3rd": 2, "4th": 3}[year]
    sl_enc = {"Low": 0, "Medium": 1, "High": 2}[s_level_ui]
    i_qual_val = 2 # Assuming Good internet as default

    # Prepare Data for Model
    input_data = {
        'age': age, 'gender': g_enc, 'course': c_enc, 'year': y_enc,
        'daily_study_hours': study_hrs, 'daily_sleep_hours': sleep_hrs,
        'screen_time_hours': screen_time, 'stress_level': sl_enc,
        'anxiety_score': mental_map[anx], 'depression_score': mental_map[dep],
        'academic_pressure_score': mental_map[prs], 'financial_stress_score': mental_map[fin],
        'social_support_score': 5.0, # Defaulting for stability
        'physical_activity_hours': phys_act,
        'sleep_quality': qual_map[s_qual], 'attendance_percentage': attendance,
        'cgpa': cgpa, 'internet_quality': i_qual_val
    }

    # DataFrame and Inference
    input_df = pd.DataFrame([input_data])[features]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    
    risk_labels = {0: "Low", 1: "Medium", 2: "High"}
    risk = risk_labels[prediction]

    # Display Result
    if risk == "High":
        st.error(f"## Prediction: {risk} Burnout Risk")
        st.write("⚠️ Your data suggests significant signs of exhaustion. Immediate lifestyle changes are recommended.")
    elif risk == "Medium":
        st.warning(f"## Prediction: {risk} Burnout Risk")
        st.write("⚠️ You are managing, but your stress levels are approaching a tipping point.")
    else:
        st.success(f"## Prediction: {risk} Burnout Risk")
        st.write("✅ You seem to have a healthy balance between academics and rest.")

    # 8. DYNAMIC RECOMMENDATION ENGINE
    st.markdown("---")
    st.subheader("💡 Personalized Recovery & Growth Plan")
    
    # Columnar Advice
    rec_col1, rec_col2 = st.columns(2)
    
    with rec_col1:
        st.markdown("**🛡️ Restoration & Lifestyle**")
        # Optimized Sleep Logic
        if sleep_hrs < 6.5:
            st.error(f"🛌 **Sleep Debt:** {sleep_hrs}h is too low. Lack of deep sleep kills focus. Aim for a 'Recovery Sleep' of 8 hours tonight.")
        elif sleep_hrs < 7.5:
            st.warning("🛌 **Sleep Gap:** You're close to optimal. Add 30-45 mins of sleep to see a 20% jump in daily alertness.")
        elif sleep_hrs >= 7.5 and s_qual == "Poor":
            st.info("🛌 **Sleep Quality:** You have the hours, but not the depth. No caffeine after 2 PM and keep your room cool.")

        # Screen Time
        if screen_time > 7:
            st.error("📱 **Digital Burn:** High screen time is overstimulating your brain. Use a 'Digital Fast' from 9 PM to 7 AM.")
        elif screen_time > 5:
            st.warning("📱 **Screen Balance:** Moderate usage detected. Replace 1 hour of phone use with physical reading or a hobby.")

    with rec_col2:
        st.markdown("**📈 Academic & Stress Strategy**")
        # Study vs CGPA Logic
        if study_hrs > 9:
            st.warning("📚 **Hyper-Focus Risk:** Studying over 9 hours often leads to 'Brain Fog'. Use the 50/10 Pomodoro method.")
        elif study_hrs < 3 and cgpa < 7.5:
            st.info("✍️ **Academic Momentum:** Small, consistent study blocks (2 hours) will reduce exam-time panic later.")

        # Mental Load
        if anx == "Yes" or prs == "Yes":
            st.info("🧘 **Stress Triage:** Since you feel pressured, try 'Time-Boxing'—only plan 3 major tasks per day to avoid overwhelm.")
        
        # Activity
        if phys_act < 0.5:
            st.warning("🏃 **Movement Gap:** Zero activity traps cortisol in your body. A 15-minute walk outside will physically 'flush' out stress.")

    # "Perfect Score" Easter Egg
    if risk == "Low" and sleep_hrs >= 7.5 and screen_time <= 4:
        st.balloons()
        st.success("🌟 **Elite Student Status:** Your habits are perfectly optimized. Share your routine with your peers!")