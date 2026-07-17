import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load("svm_loan_model.pkl")
label = joblib.load("label_encoder.pkl")

st.title("Loan Approval Prediction using SVM")

st.write("กรอกข้อมูลเพื่อทำนายผลการอนุมัติสินเชื่อ")

# =========================
# Input
# =========================

Age = st.number_input("Age", 18, 100, 30)

Gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

Marital_Status = st.selectbox(
    "Marital Status",
    ["Single", "Married"]
)

Education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

Employment_Status = st.selectbox(
    "Employment Status",
    [
        "Employed",
        "Self-Employed",
        "Unemployed"
    ]
)

Annual_Income = st.number_input(
    "Annual Income",
    value=50000
)

Loan_Amount = st.number_input(
    "Loan Amount",
    value=20000
)

Loan_Term = st.number_input(
    "Loan Term",
    value=36
)

Credit_Score = st.number_input(
    "Credit Score",
    value=700
)

Existing_Loans = st.number_input(
    "Existing Loans",
    value=0
)

Debt_to_Income_Ratio = st.number_input(
    "Debt to Income Ratio",
    value=30.0
)

Property_Area = st.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)

# =========================
# Predict
# =========================

if st.button("Predict"):

    data = pd.DataFrame({

        "Age":[Age],
        "Gender":[Gender],
        "Marital_Status":[Marital_Status],
        "Education":[Education],
        "Employment_Status":[Employment_Status],
        "Annual_Income":[Annual_Income],
        "Loan_Amount":[Loan_Amount],
        "Loan_Term":[Loan_Term],
        "Credit_Score":[Credit_Score],
        "Existing_Loans":[Existing_Loans],
        "Debt_to_Income_Ratio":[Debt_to_Income_Ratio],
        "Property_Area":[Property_Area]

    })

    prediction = model.predict(data)

    result = label.inverse_transform(prediction)

    probability = model.predict_proba(data)

    st.success("Prediction : {}".format(result[0]))

    st.write("Probability")

    st.write(probability)