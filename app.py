import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(page_title="Disease Prediction App", page_icon="🏥", layout="wide")

# Load the models
try:
    with open("./models/heart_disease_model.pkl", "rb") as file:
        heart_model = pickle.load(file)
    with open("./models/stroke_model.pkl", "rb") as file:
        stroke_model = pickle.load(file)
    with open("./models/thyroid_disease_model.pkl", "rb") as file:
        thyroid_model = pickle.load(file)
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'heart_disease_model.pkl', 'stroke_model.pkl', and 'thyroid_disease_model.pkl' are in the models directory.")
    st.stop()

# Load datasets to get feature information
try:
    heart_df = pd.read_csv("./datasets/heart_disease.csv")
    stroke_df = pd.read_csv("./datasets/stroke.csv")
    thyroid_df = pd.read_csv("./datasets/thyroid_disease.csv")
except FileNotFoundError:
    st.error("Dataset files not found. Please ensure 'heart_disease.csv', 'stroke.csv', and 'thyroid_disease.csv' are in the datasets directory.")
    st.stop()

# Initialize scalers
scaler_heart = StandardScaler()
scaler_stroke = StandardScaler()
scaler_thyroid = StandardScaler()

# Fit scalers on the training data
# Heart Disease: All features are numerical
heart_features = heart_df.drop("target", axis=1)
scaler_heart.fit(heart_features)

# Stroke: Apply one-hot encoding to categorical variables
stroke_features = stroke_df.drop(["stroke", "id"], axis=1)
categorical_cols_stroke = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
stroke_features_encoded = pd.get_dummies(stroke_features, columns=categorical_cols_stroke, drop_first=True)
scaler_stroke.fit(stroke_features_encoded)

# Thyroid: Apply one-hot encoding to categorical variables
thyroid_features = thyroid_df.drop(["Recurred"], axis=1)
categorical_cols_thyroid = ["Gender", "Smoking", "Hx Smoking", "Hx Radiothreapy", "Thyroid Function", 
                           "Physical Examination", "Adenopathy", "Pathology", "Focality", "Risk", 
                           "T", "N", "M", "Stage", "Response"]
thyroid_features_encoded = pd.get_dummies(thyroid_features, columns=categorical_cols_thyroid, drop_first=True)
scaler_thyroid.fit(thyroid_features_encoded)

# Define symptoms, cures, and input ranges
disease_info = {
    "Heart Disease": {
        "symptoms": [
            "Chest pain or discomfort (angina)",
            "Shortness of breath",
            "Fatigue and weakness",
            "Swelling in the legs, ankles, or feet",
            "Pain in the neck, jaw, throat, upper abdomen, or back",
            "Irregular heartbeats (arrhythmias)"
        ],
        "cures": [
            "Lifestyle changes: Healthy diet, regular exercise, quitting smoking",
            "Medications: Statins, beta-blockers, ACE inhibitors",
            "Medical procedures: Angioplasty, bypass surgery",
            "Regular monitoring and check-ups with a cardiologist",
            "Stress management techniques like yoga or meditation"
        ],
        "features": heart_df.drop("target", axis=1).columns.tolist(),
        "input_ranges": {
            "age": (29, 77),
            "sex": (0, 1),
            "cp": (0, 3),
            "trestbps": (94, 200),
            "chol": (126, 564),
            "fbs": (0, 1),
            "restecg": (0, 2),
            "thalach": (71, 202),
            "exang": (0, 1),
            "oldpeak": (0.0, 6.2),
            "slope": (0, 2),
            "ca": (0, 4),
            "thal": (0, 3)
        }
    },
    "Stroke": {
        "symptoms": [
            "Sudden numbness or weakness in the face, arm, or leg",
            "Sudden confusion, trouble speaking, or understanding speech",
            "Sudden trouble seeing in one or both eyes",
            "Sudden trouble walking, dizziness, or loss of balance",
            "Sudden severe headache with no known cause"
        ],
        "cures": [
            "Immediate medical attention (call emergency services)",
            "Medications: Clot-busting drugs, anticoagulants",
            "Rehabilitation: Physical, occupational, and speech therapy",
            "Lifestyle changes: Control blood pressure, reduce cholesterol",
            "Surgical interventions: Carotid endarterectomy, angioplasty"
        ],
        "features": stroke_df.drop(["stroke", "id"], axis=1).columns.tolist(),
        "input_ranges": {
            "gender": list(stroke_df["gender"].unique()),
            "age": (0.08, 82.0),
            "hypertension": (0, 1),
            "heart_disease": (0, 1),
            "ever_married": list(stroke_df["ever_married"].unique()),
            "work_type": list(stroke_df["work_type"].unique()),
            "Residence_type": list(stroke_df["Residence_type"].unique()),
            "avg_glucose_level": (55.12, 271.74),
            "bmi": (10.3, 97.6),
            "smoking_status": list(stroke_df["smoking_status"].unique())
        }
    },
    "Thyroid Disease": {
        "symptoms": [
            "Fatigue and weakness",
            "Weight gain or loss",
            "Dry skin and hair",
            "Sensitivity to cold or heat",
            "Swelling in the neck (goiter)",
            "Irregular heart rate"
        ],
        "cures": [
            "Medications: Levothyroxine for hypothyroidism, anti-thyroid drugs for hyperthyroidism",
            "Radioactive iodine therapy",
            "Surgery: Thyroidectomy in severe cases",
            "Lifestyle changes: Balanced diet, stress management",
            "Regular monitoring of thyroid hormone levels"
        ],
        "features": thyroid_df.drop(["Recurred"], axis=1).columns.tolist(),
        "input_ranges": {
            "Age": (float(thyroid_df["Age"].min()), float(thyroid_df["Age"].max())),
            "Gender": list(thyroid_df["Gender"].unique()),
            "Smoking": list(thyroid_df["Smoking"].unique()),
            "Hx Smoking": list(thyroid_df["Hx Smoking"].unique()),
            "Hx Radiothreapy": list(thyroid_df["Hx Radiothreapy"].unique()),
            "Thyroid Function": list(thyroid_df["Thyroid Function"].unique()),
            "Physical Examination": list(thyroid_df["Physical Examination"].unique()),
            "Adenopathy": list(thyroid_df["Adenopathy"].unique()),
            "Pathology": list(thyroid_df["Pathology"].unique()),
            "Focality": list(thyroid_df["Focality"].unique()),
            "Risk": list(thyroid_df["Risk"].unique()),
            "T": list(thyroid_df["T"].unique()),
            "N": list(thyroid_df["N"].unique()),
            "M": list(thyroid_df["M"].unique()),
            "Stage": list(thyroid_df["Stage"].unique()),
            "Response": list(thyroid_df["Response"].unique())
        }
    }
}

# Sidebar for disease selection
st.sidebar.title("Disease Prediction")
disease = st.sidebar.selectbox("Select Disease", ["Heart Disease", "Stroke", "Thyroid Disease"])

# Main content
st.title(f"{disease} Prediction and Information")
st.markdown("---")

# Tabs for Symptoms, Cures, and Prediction
tab1, tab2, tab3 = st.tabs(["Symptoms", "Cures", "Prediction"])

# Symptoms Tab
with tab1:
    st.header("Symptoms")
    for symptom in disease_info[disease]["symptoms"]:
        st.write(f"- {symptom}")
    st.markdown("**Note**: If you experience these symptoms, consult a healthcare professional immediately.")

# Cures Tab
with tab2:
    st.header("Cures and Treatments")
    for cure in disease_info[disease]["cures"]:
        st.write(f"- {cure}")
    st.markdown("**Note**: Always follow medical advice from professionals for treatment plans.")

# Prediction Tab
with tab3:
    st.header("Predict Disease Risk")
    st.write("Enter the following details to predict the likelihood of the disease. Input sizes are indicated for each feature.")

    # Create input fields for features
    input_data = {}
    cols = st.columns(3)  # Organize inputs in 3 columns for better UI
    for idx, feature in enumerate(disease_info[disease]["features"]):
        with cols[idx % 3]:
            if feature in ["gender", "ever_married", "work_type", "Residence_type", "smoking_status",
                          "Gender", "Smoking", "Hx Smoking", "Hx Radiothreapy", "Thyroid Function",
                          "Physical Examination", "Adenopathy", "Pathology", "Focality", "Risk",
                          "T", "N", "M", "Stage", "Response"]:
                # Categorical features
                options = disease_info[disease]["input_ranges"][feature]
                input_data[feature] = st.selectbox(
                    f"{feature.replace('_', ' ').title()} (Options: {', '.join(options)})",
                    options
                )
            else:
                # Numerical or binary features
                min_val, max_val = disease_info[disease]["input_ranges"][feature]
                step = 0.1 if feature in ["age", "avg_glucose_level", "bmi", "oldpeak", "Age"] else 1.0
                input_data[feature] = st.number_input(
                    f"{feature.replace('_', ' ').title()} (Range: {min_val} to {max_val})",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(min_val),
                    step=step
                )

    # Predict button
    if st.button("Predict"):
        try:
            # Convert input data to DataFrame
            input_df = pd.DataFrame([input_data])

            # Validate input sizes
            for feature in input_data:
                if feature in disease_info[disease]["input_ranges"]:
                    if isinstance(disease_info[disease]["input_ranges"][feature], tuple):
                        min_val, max_val = disease_info[disease]["input_ranges"][feature]
                        if not (min_val <= input_data[feature] <= max_val):
                            st.error(f"{feature.replace('_', ' ').title()} must be between {min_val} and {max_val}.")
                            st.stop()
                    else:
                        if input_data[feature] not in disease_info[disease]["input_ranges"][feature]:
                            st.error(f"{feature.replace('_', ' ').title()} must be one of: {', '.join(disease_info[disease]['input_ranges'][feature])}.")
                            st.stop()

            # Prepare input for prediction
            if disease == "Heart Disease":
                # Ensure correct feature order
                input_array = input_df[disease_info[disease]["features"]].values
                # Scale the input
                input_array_scaled = scaler_heart.transform(input_array)
                model = heart_model
            elif disease == "Stroke":
                # Apply one-hot encoding to match training
                input_df_encoded = pd.get_dummies(input_df, columns=categorical_cols_stroke, drop_first=True)
                # Ensure all expected columns are present (match training data)
                expected_columns = stroke_features_encoded.columns
                for col in expected_columns:
                    if col not in input_df_encoded.columns:
                        input_df_encoded[col] = 0  # Add missing columns with 0
                # Reorder columns to match training data
                input_df_encoded = input_df_encoded[expected_columns]
                input_array = input_df_encoded.values
                # Scale the input
                input_array_scaled = scaler_stroke.transform(input_array)
                model = stroke_model
            else:  # Thyroid Disease
                # Apply one-hot encoding to match training
                input_df_encoded = pd.get_dummies(input_df, columns=categorical_cols_thyroid, drop_first=True)
                # Ensure all expected columns are present (match training data)
                expected_columns = thyroid_features_encoded.columns
                for col in expected_columns:
                    if col not in input_df_encoded.columns:
                        input_df_encoded[col] = 0  # Add missing columns with 0
                # Reorder columns to match training data
                input_df_encoded = input_df_encoded[expected_columns]
                input_array = input_df_encoded.values
                # Scale the input
                input_array_scaled = scaler_thyroid.transform(input_array)
                model = thyroid_model

            # Make prediction
            prediction = model.predict(input_array_scaled)[0]
            probability = model.predict_proba(input_array_scaled)[0][1] * 100  # Probability of positive class

            # Display result
            st.subheader("Prediction Result")
            if prediction == 1:
                st.error(f"High Risk: The model predicts a high likelihood of {disease} recurrence (Probability: {probability:.2f}%).")
                st.write("Please consult a doctor immediately for further evaluation.")
            else:
                st.success(f"Low Risk: The model predicts a low likelihood of {disease} recurrence (Probability: {probability:.2f}%).")
                st.write("Continue monitoring your health and consult a doctor if symptoms appear.")

        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            st.write("Please ensure all inputs are valid and model files are correctly configured.")

# Footer
st.markdown("---")
st.write("Developed with ❤️ using Streamlit")
st.write("**Disclaimer**: This app is for informational purposes only and not a substitute for professional medical advice.")