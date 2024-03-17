import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
h_data = pd.read_csv('heart_disease_data.csv')

X = h_data.drop(columns='target', axis=1)
Y = h_data['target']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Streamlit app
st.title("Heart Disease Prediction App")

# Input form for user to enter parameters
age = st.slider("Age", min_value=29, max_value=77, step=1, value=52)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
cp = st.slider("Chest Pain Type", min_value=0, max_value=3, step=1, value=0)
trestbps = st.slider("Resting Blood Pressure", min_value=94, max_value=200, step=1, value=128)
chol = st.slider("Serum Cholesterol", min_value=126, max_value=564, step=1, value=204)
fbs = st.selectbox("Fasting Blood Sugar", [0, 1], format_func=lambda x: 'True' if x == 1 else 'False')
restecg = st.slider("Resting Electrocardiographic Results", min_value=0, max_value=2, step=1, value=1)
thalach = st.slider("Maximum Heart Rate Achieved", min_value=71, max_value=202, step=1, value=156)
exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
oldpeak = st.slider("Oldpeak", min_value=0.0, max_value=6.2, step=0.1, value=1.0)
slope = st.slider("Slope of the Peak Exercise ST Segment", min_value=0, max_value=2, step=1, value=1)
ca = st.slider("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=4, step=1, value=0)
thal = st.slider("Thalassemia", min_value=0, max_value=3, step=1, value=0)

if st.button('Check Heart Attack Test Result'):
    input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
    prediction = model.predict(input_data)

# Display the prediction result
    if prediction[0] == 0:
        st.write('Person may have no heart disease')
    else:
        st.write('Person may have heart disease')
