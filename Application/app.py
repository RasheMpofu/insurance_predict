import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(page_title="Insurance Premium Prediction", page_icon="📊", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
    body { background-color: #f4f9fd; font-family: Arial, sans-serif; }
    .sidebar .sidebar-content { background-color: #003366; color: white; }
    h1, h2, h3, h4 { color: #003366; }
    .stButton button { background-color: #ff3333; color: white; border-radius: 5px; }
    .stButton button:hover { background-color: #cc2900; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load the model and dataset
model_path = 'Application/gradient_boosting_model.pkl'
data_path = 'Application/cleaned_data.csv'

# Load model
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error(f"Model file not found at path: {model_path}.")
    st.stop()

# Prediction Section
st.title("Insurance Premium Prediction")
st.markdown("Enter customer details below to predict the insurance premium.")

# Layout: Two columns for inputs
col1, col2 = st.columns(2)

# Create input fields for the user to enter data
with col1:
    age = st.slider("Age", 18, 100, 30)
    annual_income = st.slider("Annual Income", 10000, 200000, 50000)
    credit_score = st.slider("Credit Score", 300, 850, 700)
    gender = st.selectbox("Gender", options=["Male", "Female", "Other"])

with col2:
    health_score = st.slider("Health Score", 0, 100, 70)
    policy_type = st.selectbox("Policy Type", options=["Type 1", "Type 2", "Type 3"])
    insurance_duration = st.slider("Insurance Duration (in years)", 1, 30, 5)

# Map categorical features to numerical values dynamically
gender_mapping = {"Male": 0, "Female": 1, "Other": 2}
policy_type_mapping = {"Type 1": 0, "Type 2": 1, "Type 3": 2}

gender_encoded = gender_mapping[gender]
policy_type_encoded = policy_type_mapping[policy_type]

# Prepare input data
input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender_encoded],
    'Annual Income': [annual_income],
    'Credit Score': [credit_score],
    'Insurance Duration': [insurance_duration],
    'Health Score': [health_score],
    'Policy Type': [policy_type_encoded],
})

# Predict premium
if st.button("Predict Premium"):
    try:
        prediction = model.predict(input_data)
        st.success(f"Predicted Premium: ${prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")