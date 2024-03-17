import streamlit as st
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def main():
    df = pd.read_csv('parkinsons_data')

    X = df.drop(columns=['name', 'status'], axis=1)
    Y = df['status']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = svm.SVC(kernel='linear')
    model.fit(X_train, Y_train)

    st.subheader("Predict Parkinson's Disease:")
    st.write("Enter the values for the following features:")

    mdvp_features = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
        "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
        "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA"
    ]

    other_features = [
        "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
    ]

    mdvp_input_data = {}
    other_input_data = {}

    for mdvp_feature in mdvp_features:
        value = st.number_input(mdvp_feature, step=0.001, format="%.6f")
        mdvp_input_data[mdvp_feature] = value

    for other_feature in other_features:
        value = st.number_input(other_feature, step=0.001, format="%.6f")
        other_input_data[other_feature] = value

    input_data = np.array(list(mdvp_input_data.values()) + list(other_input_data.values())).reshape(1, -1)
    std_data = scaler.transform(input_data)

    prediction = model.predict(std_data)

    st.subheader("Prediction Result:")
    if prediction[0] == 0:
        st.write("This person is not likely to have Parkinson's disease.")
    else:
        st.write("This person is likely to have Parkinson's disease.")

if __name__ == "__main__":
    main()
