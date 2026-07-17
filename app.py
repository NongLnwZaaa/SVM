import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺"
)

st.title("🩺 Diabetes Prediction using SVM")

model = joblib.load("svm_diabetes_model.pkl")

Pregnancies = st.number_input("Pregnancies", 0, 20, 1)
Glucose = st.number_input("Glucose", 0, 300, 120)
BloodPressure = st.number_input("Blood Pressure", 0, 200, 70)
SkinThickness = st.number_input("Skin Thickness", 0, 100, 20)
Insulin = st.number_input("Insulin", 0, 900, 80)
BMI = st.number_input("BMI", 0.0, 70.0, 25.0)
DiabetesPedigreeFunction = st.number_input(
    "Diabetes Pedigree Function",
    0.0,
    3.0,
    0.5
)
Age = st.number_input("Age", 1, 120, 30)

if st.button("Predict"):

    data = pd.DataFrame({
        "Pregnancies":[Pregnancies],
        "Glucose":[Glucose],
        "BloodPressure":[BloodPressure],
        "SkinThickness":[SkinThickness],
        "Insulin":[Insulin],
        "BMI":[BMI],
        "DiabetesPedigreeFunction":[DiabetesPedigreeFunction],
        "Age":[Age]
    })

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0]

    if prediction == 1:
        st.error("⚠️ Prediction : Diabetes")
    else:
        st.success("✅ Prediction : No Diabetes")

    st.subheader("Probability")

    st.write(f"No Diabetes : {probability[0]*100:.2f}%")
    st.write(f"Diabetes : {probability[1]*100:.2f}%")