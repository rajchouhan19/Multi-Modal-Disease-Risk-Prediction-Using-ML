import numpy as np
import pandas as pd
import streamlit as st
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder


lung_cancer_dataset = pd.read_csv('survey_lung_cancer.csv')

x = lung_cancer_dataset.iloc[:, 0:15]
y = lung_cancer_dataset.iloc[:, -1]

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

le = LabelEncoder()
le.fit(y)
y = le.transform(y)

scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

n_estimators = 100  
random_forest = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
random_forest.fit(x_train, y_train)

st.title("Lung Cancer Risk Prediction")

gender = st.selectbox("Please select your gender", ["M", "F"])
age = st.slider("Select your age", min_value=40, max_value=100)
st.subheader("2-YES and 1-NO")
smoke = st.selectbox("Do you smoke?", ["2", "1"])
yellow_fingers = st.selectbox("Do you have yellow fingers?", ["2", "1"])
anxiety = st.selectbox("Do you have Anxiety?", ["2", "1"])
peer_pressure = st.selectbox("Do you have Peer Pressure to smoke?", ["2", "1"])
chronic_disease = st.selectbox("Do you have a chronic disease?", ["2", "1"])
fatigue = st.selectbox("Do you feel fatigue?", ["2", "1"])
allergy = st.selectbox("Do you have any type of allergy?", ["2", "1"])
wheezing_issue = st.selectbox("Do you have wheezing issue?", ["2", "1"])
alcohol = st.selectbox("Do you consume alcohol?", ["2", "1"])
cough = st.selectbox("Do you cough a lot?", ["2", "1"])
short_of_breath = st.selectbox("Do you get short of breath frequently?", ["2", "1"])
difficulty_swallowing = st.selectbox("Is it difficult to swallow food sometimes for you?", ["2", "1"])
chest_pain = st.selectbox("Do you feel chest pain?", ["2", "1"])

user_inputs = [[gender, age, smoke, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue,
                allergy, wheezing_issue, alcohol, cough, short_of_breath, difficulty_swallowing, chest_pain]]
user_inputs_df = pd.DataFrame(user_inputs, columns=lung_cancer_dataset.columns[:-1])
user_inputs_transformed = np.array(ct.transform(user_inputs_df))

user_inputs_scaled = scaler.transform(user_inputs_transformed)

if st.button("Check Lung Cancer test result"):
    prediction = random_forest.predict(user_inputs_scaled)
    predicted_class = le.inverse_transform(prediction)[0]
    st.write("Lung Cancer Risk:", predicted_class)
