import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# App title
st.title("🩺 Diabetes Prediction Web App")

st.markdown("Enter the patient details below to predict diabetes status.")

# Input fields
pregnancies = st.number_input("Pregnancies", 0, 20, step=1)
glucose = st.number_input("Glucose", 0, 300)
bp = st.number_input("Blood Pressure", 0, 200)
skin = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 1, 120)

# When user clicks Predict
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    scaled = scaler.transform(input_data)
    prediction = model.predict(scaled)
    
    if prediction[0] == 1:
        st.error("⚠️ Diabetic")
    else:
        st.success("✅ Not Diabetic")
