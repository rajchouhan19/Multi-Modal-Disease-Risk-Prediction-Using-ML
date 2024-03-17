import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import LabelEncoder

diabetes_dataset = pd.read_csv('diabetes_prediction_dataset.csv')

label_encoding = LabelEncoder()
diabetes_dataset["gender"] = label_encoding.fit_transform(diabetes_dataset["gender"])
diabetes_dataset["smoking_history"] = label_encoding.fit_transform(diabetes_dataset["smoking_history"])

x = diabetes_dataset.drop(columns='diabetes', axis=1)
y = diabetes_dataset['diabetes']

scaler = StandardScaler()
scaler.fit(x)
x_standardized = scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_standardized, y, test_size=0.2, stratify=y, random_state=2)

classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)

st.title("Diabetes Prediction App")

st.header("Input Data")
st.write(" 1-MALE and 0-Female")
gender = st.radio("Gender", ["1", "0"])
age = st.slider("Age", 20, 100, 40)
st.write(" 1-Yes and 0-No")
hypertension = st.selectbox("Hypertension", ["0","1"])
st.write(" 1-Yes and 0-No")
heart_disease = st.selectbox("Heart Disease", ["0","1"])
smoking_history = st.selectbox("Smoking History", ["No info","Never","Ever" ,"Former", "Current"])
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
HbA1c_level = st.slider("HbA1c Level", 4.0, 15.0, 7.0)
blood_glucose_level = st.slider("Blood Glucose Level", 50.0, 300.0, 150.0)

gender_encoded = 1 if gender == "Female" else 0
hypertension_encoded = 1 if hypertension == "Yes" else 0
heart_disease_encoded = 1 if heart_disease == "Yes" else 0
smoking_history_encoded = {"No info": 0, "Never": 1, "Ever":2, "Former":3, "Current":4}[smoking_history]
input_data = np.array([[gender_encoded, age, hypertension_encoded, heart_disease_encoded, 
                        smoking_history_encoded, bmi, HbA1c_level, blood_glucose_level]])

input_data_standardized = scaler.transform(input_data)

if input_data_standardized.shape[1] != x_train.shape[1]:
    st.error("Number of features in input data does not match the training data.")
else:
    st.write("Note: Prediction may take some time due to large dataset.")
    prediction = classifier.predict(input_data_standardized)
    if st.button("Predict"):
        
        st.header("Prediction")
        st.write("Result:", "Diabetic" if prediction[0] == 1 else "Non-Diabetic")
