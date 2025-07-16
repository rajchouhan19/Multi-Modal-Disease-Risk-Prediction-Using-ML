# Multi-Modal Disease Risk Predictor

## Overview

This project is an intelligent web application designed to predict the risk of various diseases based on user-provided symptoms. It leverages a collection of machine learning models to provide a real-time risk assessment, serving as a powerful preliminary diagnostic aid.

The system is built with a user-friendly interface using Streamlit, allowing users to easily input their symptoms and receive an instant prediction. The backend is powered by robust machine learning models trained on comprehensive medical datasets.

## Key Features

- **Interactive UI:** A clean and simple web interface built with Streamlit for easy user interaction.
- **Multi-Disease Prediction:** Capable of predicting the risk for several diseases, including Diabetes, Heart Disease, and Parkinson's.
- **Machine Learning Core:** Utilizes a suite of well-established machine learning algorithms like Support Vector Machine (SVM), Logistic Regression, and others for accurate predictions.
- **Data-Driven:** Trained and validated on standard medical datasets to ensure the reliability of the predictions.
- **Modular Code:** The project is organized with separate scripts for each disease model, making it easy to extend and maintain.

## How It Works

The application operates on a straightforward principle:

1.  **User Input:** The user selects a disease they want to check for from the sidebar.
2.  **Symptom Data Entry:** A form is dynamically generated for the selected disease, prompting the user to enter relevant medical data (e.g., glucose level, blood pressure, specific symptoms).
3.  **Model Loading:** The application loads the corresponding pre-trained machine learning model (`.sav` file) for the chosen disease.
4.  **Prediction:** The user's input data is fed into the model, which processes it and outputs a prediction.
5.  **Display Result:** The application clearly displays the result to the user, indicating whether they are at risk for the disease or not.

## Technologies & Libraries

- **Backend:** Python
- **Machine Learning:** Scikit-learn, Pandas, NumPy
- **Web Framework:** Streamlit
- **Data Handling:** Pickle (for loading/saving models)

## Setup and Installation

Follow these steps to get the project running on your local machine.

#### 1. Clone the Repository
```bash
git clone [https://github.com/rajchouhan19/Multi-Modal-Disease-Risk-Prediction-Using-ML.git](https://github.com/rajchouhan19/Multi-Modal-Disease-Risk-Prediction-Using-ML.git)
cd Multi-Modal-Disease-Risk-Prediction-Using-ML
```

#### 2. Create and Activate a Virtual Environment
```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

#### 3. Install Dependencies
This project's dependencies can be installed via `pip`.
```bash
pip install streamlit scikit-learn pandas numpy
```

#### 4. Ensure Model Files are Present
The pre-trained model files (e.g., `diabetes_model.sav`, `heart_disease_model.sav`) are required for the application to work. Make sure they are present in the project's root directory.

## Usage

To run the web application, execute the following command in your terminal from the project's root directory:

```bash
streamlit run multiple_disease_prediction.py
```

This will start the Streamlit server and open the application in your default web browser. From there, you can navigate using the sidebar to select a disease and enter symptoms to get a prediction.
