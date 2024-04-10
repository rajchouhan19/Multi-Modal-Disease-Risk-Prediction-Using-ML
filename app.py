import streamlit as st
import webbrowser
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

def main():
   
    st.sidebar.image("images\Doctor.png", use_column_width=True)  
    selected_page = st.sidebar.radio("Navigation Menu", ["Home", "Diabetes", "Lung Cancer", "Heart Attack", "Parkinson's"])

    if selected_page == "Diabetes":
        show_diabetes_page()

    elif selected_page == "Lung Cancer":
       show_lung_cancer_page()
       
    elif selected_page == "Heart Attack":
        st.title("Heart Attack Risk Prediction")
        show_heart_attack_page()

    elif selected_page == "Parkinson's":
        st.title("Parkinson's Disease Risk Prediction")
        show_parkinsons_page()

    elif selected_page == "Home":
        show_home_page()

def show_home_page():
    image_path = 'images\logo.png'
    st.image(image_path, use_column_width=True)
    st.title("Multimodal Disease Risk Predictor")
    st.write("Multimodal Disease Risk Predictor (MMDRP) is a holistic health analysis platform. This app predicts the risk of various diseases based on multimodal data.")
    st.subheader("Diseases and their Symptoms")
    html_content = """
    <div style="font-size: 18px; line-height: 1.5;">
        <h4>Diabetes:</h4>
        Diabetes is a chronic condition that affects how the body processes blood sugar (glucose). 
        There are two main types: <br>Type 1: where the immune system attacks and destroys insulin-producing cells 
        <br> Type 2: where the body doesn't use insulin properly. Both types result in elevated blood sugar levels, 
        leading to various complications if not managed properly.<br><br>
        The symptoms may include:<br>
        <ul>
            <li>Chest discomfort: Pressure, squeezing, fullness, or pain in the chest.</li>
            <li>Upper body pain: Discomfort in the arms, back, neck, jaw, or stomach.</li>
            <li>Shortness of breath: Difficulty breathing, with or without chest discomfort.</li>
            <li>Cold sweats: Profuse sweating, especially when accompanied by other symptoms.</li>
            <li>Nausea or vomiting: Feeling sick to your stomach or vomiting.</li>
            <li>Dizziness: Feeling light-headed or faint.</li>
        </ul>
        <br><br>
        <h4>Lung Cancer:</h4>
        Lung cancer is a type of cancer that begins in the lungs and is characterized by uncontrolled growth of abnormal cells in lung tissue. It is one of the most common cancers worldwide and is a leading cause of cancer-related deaths. Lung cancer can be broadly categorized into two main types: non-small cell lung cancer (NSCLC) and small cell lung cancer (SCLC).
         <br><br>The description of symptoms for lung cancer:
        <ul>
        <li><strong>Persistent Cough:</strong> A long-lasting cough, sometimes with blood.</li>
        <li><strong>Shortness of Breath:</strong> Difficulty breathing, especially with exertion.</li>
        <li><strong>Chest Pain:</strong> Persistent chest discomfort or pain.</li>
        <li><strong>Wheezing:</strong> Noisy breathing or whistling sounds.</li>
        <li><strong>Hoarseness:</strong> Changes in voice, like a raspy or hoarse voice.</li>
        <li><strong>Unexplained Weight Loss:</strong> Losing weight without trying.</li>
        <li><strong>Fatigue:</strong> Feeling tired or weak, even with rest.</li>
        <li><strong>Loss of Appetite:</strong> Not feeling hungry or having no interest in food.</li>
        <li><strong>Coughing up Blood:</strong> Blood in coughed-up mucus.</li>
        <li><strong>Difficulty Swallowing:</strong> Trouble swallowing, especially solids.</li>
        </ul>
        <br><br>
        <h4>Heart Attack (Myocardial Infarction):</h4>
        A heart attack occurs when blood flow to a part of the heart muscle becomes blocked, usually by a blood clot. 
        The lack of blood flow can cause that part of the heart muscle to die. Common symptoms include chest pain or 
        discomfort, shortness of breath, and pain or discomfort in the arms, back, neck, jaw, or stomach. 
        Prompt medical attention is crucial to minimize damage to the heart muscle.
        <br><br>"The symptoms may encompass:"
        <ul>
        <li>Chest Pain or Discomfort: This is the most common symptom. It may feel like pressure, squeezing, fullness, or pain in the center or left side of the chest. The discomfort may last for a few minutes or come and go.</li>
        <li>Pain or Discomfort in Other Areas of the Upper Body: This can include pain or discomfort in one or both arms, the back, neck, jaw, or stomach.</li>
        <li>Shortness of Breath: You may experience difficulty breathing, which can occur with or without chest discomfort.</li>
        <li>Nausea or Vomiting: Some people may feel nauseous or vomit during a heart attack.</li>
        <li>Cold Sweat: You may break out in a cold sweat, often described as feeling clammy or sweaty.</li>
        <li>Lightheadedness or Dizziness: You may feel lightheaded or dizzy, which can sometimes accompany chest pain or discomfort.</li>
        <li>Fatigue: Unexplained fatigue, weakness, or feeling unusually tired, even with rest, can also be a symptom of a heart attack.</li>
        </ul>
        <br><br>
        <h4>Parkinson's Disease:</h4>
        Parkinson's disease is a progressive nervous system disorder that primarily affects movement. 
        It occurs when nerve cells in the brain gradually break down or die, leading to a lack of dopamine—
        a neurotransmitter crucial for smooth and coordinated muscle movements. Symptoms include tremors, 
        stiffness, and difficulties with balance and coordination.
        <br><br>Some common symptoms of Parkinson's disease include:
        <ul>
        <li><strong>Tremors:</strong> Involuntary shaking of a limb, especially at rest.</li>
        <li><strong>Bradykinesia:</strong> Slowness of movement, making simple tasks take longer to complete.</li>
        <li><strong>Rigidity:</strong> Stiffness and inflexibility of the muscles, leading to decreased range of motion.</li>
        <li><strong>Postural instability:</strong> Impaired balance and coordination, often leading to falls.</li>
        <li><strong>Bradyphrenia:</strong> Slowed thinking and difficulty with cognitive tasks such as memory, attention, and problem-solving.</li>
        <li><strong>Micrographia:</strong> Small, cramped handwriting.</li>
        <li><strong>Masked face:</strong> Reduced facial expressions, giving the appearance of a fixed or expressionless face.</li>
        <li><strong>Stooped posture:</strong> Rounded shoulders and forward-leaning posture.</li>
        <li><strong>Freezing episodes:</strong> Brief, involuntary pauses in movement, often when starting to walk or turn.</li>
        <li><strong>Speech changes:</strong> Softening of the voice, monotone speech, or difficulty articulating words.</li>
        <li><strong>Sleep disturbances:</strong> Insomnia, frequent waking during the night, or vivid dreams and nightmares.</li>
        <li><strong>Fatigue:</strong> Persistent feelings of tiredness and lack of energy.</li>
        <li><strong>Depression and anxiety:</strong> Mood changes, including feelings of sadness, apathy, or worry.</li>
        <li><strong>Constipation:</strong> Difficulty passing stools due to slowed digestive function.</li>
        <li><strong>Reduced sense of smell:</strong> Decreased ability to detect and distinguish odors.</li>
        </ul>
        <br><br>        
    </div>
"""

    st.markdown(html_content, unsafe_allow_html=True)
def show_diabetes_page():
    @st.cache_data
    def data_preprocessing():
        diabetes_dataset = pd.read_csv('diabetes_prediction_dataset.csv')

        label_encoding = LabelEncoder()
        diabetes_dataset["gender"] = label_encoding.fit_transform(diabetes_dataset["gender"])
        diabetes_dataset["smoking_history"] = label_encoding.fit_transform(diabetes_dataset["smoking_history"])

        x = diabetes_dataset.drop(columns='diabetes', axis=1)
        y = diabetes_dataset['diabetes']

        scaler = StandardScaler()
        x_standardized = scaler.fit_transform(x)
        pca = PCA(n_components=0.95)  # Keep 95% of variance
        x_pca = pca.fit_transform(x_standardized)
        x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.1, stratify=y, random_state=2)
        return x_train, x_test, y_train, y_test, scaler, pca

    x_train, x_test, y_train, y_test,scaler,pca = data_preprocessing() 
    @st.cache_data
    def train_classifier():
        classifier = svm.SVC(kernel='linear')
        classifier.fit(x_train, y_train)
        return classifier
    classifier= train_classifier()

    st.title("Diabetes Prediction Prediction")

    st.header("Input Data")
    gender = st.radio("Select your gender", ["Male", "Female"])
    age = st.slider("Age", 20, 100, 40)
    hypertension = st.selectbox("Hypertension", ["Yes","No"])
    heart_disease = st.selectbox("Heart Disease", ["Yes","No"])
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
    input_data_pca = pca.transform(input_data_standardized)

    if input_data_standardized.shape[1] != x_train.shape[1]:
        st.error("Number of features in input data does not match the training data.")
    else:
        st.write("Note: Prediction may take some time due to large dataset.")
        prediction = classifier.predict(input_data_pca)
        if st.button("Predict"):
            
            st.header("Prediction")
            st.write("Result:", "You have high risk of being Diabetic" if prediction[0] == 1 else "You have low risk of being Diabetic")

    # Add your Diabetes content here

def show_lung_cancer_page():
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
    age = st.slider("Select your age", min_value=0, max_value=100)
    st.subheader("2-YES and 1-NO")
    smoke = st.selectbox("Do you smoke?", ["2", "1"])
    yellow_fingers = st.selectbox("Do you have yellow fingers?", ["2", "1"])
    anxiety = st.selectbox("Do you have Anxiety?", ["2", "1"])
    peer_pressure = st.selectbox("Do you have Peer Pressure to smoke?", ["2", "1"])
    chronic_disease = st.selectbox("Do you have a chronic disease? (Chronic diseases are conditions that last at least a year and require ongoing medical attention, or limit daily activities.)", ["2", "1"])
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
        # if(predicted_class=='Yes'):
        #     print("You have the high risk of suffering from Lung Cancer")
        # else:
        #     print("You have the low risk of suffering from Lung Cancer")

        st.write("Do you have High Lung Cancer Risk:", predicted_class)

def show_heart_attack_page():
            
            h_data = pd.read_csv('heart_disease_data.csv')

            X = h_data.drop(columns='target', axis=1)
            Y = h_data['target']

            # Split the data into training and testing sets
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

            # Train the logistic regression model
            model = LogisticRegression()
            model.fit(X_train, Y_train)

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
                    st.write('Person may not have heart disease')
                else:
                    st.write('Person may have heart disease')

def show_parkinsons_page():
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
    st.subheader('Feature Names:')
    data = {
    'Feature': ['PPE', 'spread1', 'spread2', 'MDVP:Fo(Hz)', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
                'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'DFA', 'RPDE', 'D2',
                'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'NHR', 'HNR', 'MDVP:Jitter(Abs)', 'MDVP:Jitter(%)',
                'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP'],
    'Description': ['Nonlinear measures of fundamental frequency variation',
                    'Nonlinear measures of fundamental frequency variation',
                    'Nonlinear measures of fundamental frequency variation',
                    'Average vocal fundamental frequency',
                    'Measures of variation in amplitude',
                    'Measures of variation in amplitude',
                    'Measures of variation in amplitude',
                    'Measures of variation in amplitude',
                    'Measures of variation in amplitude',
                    'Measures of variation in amplitude',
                    'Signal fractal scaling exponent',
                    'Nonlinear dynamical complexity measures',
                    'Nonlinear dynamical complexity measures',
                    'Maximum fundamental frequency',
                    'Minimum fundamental frequency',
                    'Measures of the ratio of noise to tonal components in the voice status',
                    'Measures of the ratio of noise to tonal components in the voice status',
                    'Measures of variation in fundamental frequency',
                    'Measures of variation in fundamental frequency',
                    'Measures of variation in fundamental frequency',
                    'Measures of variation in fundamental frequency',
                    'Measures of variation in fundamental frequency']
}

# Create DataFrame
    df = pd.DataFrame(data)

# Display the DataFrame as a table
    st.write(df)

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

    if st.button('Check Parkinsons Test Result'):
        if prediction[0] == 0:
            st.write("This person is not likely to have Parkinson's disease.")
        else:
            st.write("This person is likely to have Parkinson's disease.")

if __name__ == "__main__":
    main()
