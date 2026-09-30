# Trigger rebuild
import numpy as np
import streamlit as st
# Title
st.title("Deciduous/Primary Tooth Dimension Based Sex Estimation")

# Summary
st.markdown("""
Sex estimation from dental metrics offers valuable diagnostic evidence in forensic anthropology when skeletal indicators are missing or fragmented. This study evaluated mesiodistal (MD) and buccolingual (BL) dimensions of the complete primary dentition from 500 children aged 3–5 years (250 males, 250 females) from Southwestern India.  

Nine supervised machine learning (ML) classifiers were benchmarked against a classical Logistic Regression Analysis (LRA) baseline on an independent held-out test set (n=100). Gradient Boosting emerged as the top classifier for threshold-dependent performance, achieving 62.0% accuracy and a macro F1-score of 0.6144, outperforming the LRA baseline (54.0% accuracy; F1=0.5370) by 8.0 percentage points. CatBoost offered the most balanced overall profile, combining 60.0% accuracy with high discrimination (ROC-AUC = 0.6632) and optimal probability calibration (Brier score = 0.2309).  
LightGBM maximized class separation (ROC-AUC = 0.6668), whereas Extra Trees yielded the highest precision-recall integration (AUPRC = 0.6785).  

Feature importance consensus highlighted primary tooth dimensions — specifically 81BL, 82MD, 83MD, 71MD, and 54BL — as key predictive drivers. Standalone Decision Trees and Support Vector Machines underperformed relative to the LRA baseline.  

Overall, primary crown sex difference was low (mean percentage dimorphism = 1.4133%). While non-parametric tree ensembles captured complex non-linear feature interactions to systematically surpass traditional linear biostatistical baselines, peak classification accuracy remained modest (62.0%).  

Consequently, calliper-derived primary odontometrics provide useful non-linear predictive signals, but should serve as complementary diagnostic evidence rather than standalone primary indicators in forensic sex estimation.  

This interactive webpage allows users to apply primary tooth dimensions (type in the measurements manually, or upload a .csv file containing the 40 MD + BL dimensions) and estimate sex using the aforementioned best ML models on a similar population.

**NOTE:** This tool is intended for forensic and clinical assistance only. Results should be interpreted by qualified forensic dental professionals and/or forensic anthropologists. This tool does not replace expert forensic or clinical judgment. The models are specific to an Indian dataset, having been trained on a Southwestern Indian population of 3-5-year-olds. It has not been validated in other populations, and results should be interpreted with caution outside this demographic context.
""")

import pandas as pd
import joblib

# --- 1. Load the trained model and feature names ---
# In a real Streamlit app, these files would be in the same directory
# or accessible via a defined path.
# --- Model selection ---
import requests
import io

@st.cache_resource
def load_model_from_github(url):
    response = requests.get(url)
    return joblib.load(io.BytesIO(response.content))

model_choice = st.selectbox(
    "Choose a model for sex estimation:",
    ["Gradient Boosting", "LightGBM", "CatBoost", "Extra Trees"]
)

model_urls = {
    "Gradient Boosting": "https://raw.githubusercontent.com/ashithacharya/DeciduousTeethSexEstimation/main/gradient_boosting_model.pkl",
    "LightGBM": "https://raw.githubusercontent.com/ashithacharya/DeciduousTeethSexEstimation/main/lightgbm_model.pkl",
    "CatBoost": "https://raw.githubusercontent.com/ashithacharya/DeciduousTeethSexEstimation/main/catboost_model.pkl",
    "Extra Trees": "https://raw.githubusercontent.com/ashithacharya/DeciduousTeethSexEstimation/main/extra_trees_model.pkl"
}

model = load_model_from_github(model_urls[model_choice])

# Assuming model is saved in /content/
feature_names = [
    '51MD', '51BL', '52MD', '52BL', '53MD', '53BL', '54MD', '54BL',
    '55MD', '55BL', '61MD', '61BL', '62MD', '62BL', '63MD', '63BL',
    '64MD', '64BL', '65MD', '65BL', '71MD', '71BL', '72MD', '72BL',
    '73MD', '73BL', '74MD', '74BL', '75MD', '75BL', '81MD', '81BL',
    '82MD', '82BL', '83MD', '83BL', '84MD', '84BL', '85MD', '85BL'
]
 # Placeholder: Replace with actual feature names if available or load dynamically

# --- 2. Define the prediction function ---
def predict_individual_sex(model, individual_features_df):
    # Ensure the input DataFrame has the correct feature order
    individual_features_df = individual_features_df[feature_names]
    prediction = model.predict(individual_features_df)[0]
    probability_male = model.predict_proba(individual_features_df)[0][1]  # Probability of class 1 (male)
    return prediction, probability_male

# --- 3. Streamlit UI ---
st.title("Deciduous Tooth Dimension Based Sex Estimation")
st.write('Enter the tooth dimensions below to estimate sex (0=Female, 1=Male).')

# Initialize input dictionary with mean values (or some sensible defaults)
# For a real app, you might save these means and load them too.
# For this example, we'll just use 0.0 as a placeholder or you could embed actual means.
default_values = {feature: 0.0 for feature in feature_names} # Placeholder
# A more robust solution would load mean_features if saved.
# For now, let's just make input fields interactive

input_data = {}
# Create columns for better layout of input fields
cols = st.columns(4) # Adjust number of columns as needed

for i, feature in enumerate(feature_names):
    with cols[i % 4]: # Distribute inputs across columns
        input_data[feature] = st.number_input(
            f'{feature}',
            value=float(default_values[feature]),
            format='%.2f',
            key=f'input_{feature}' # Unique key for each widget
        ) 
uploaded_file = st.file_uploader("Upload a CSV file with tooth dimensions", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    X_input = data[feature_names]
    prediction = model.predict(X_input)
    probability = model.predict_proba(X_input)

    prob_df = pd.DataFrame(probability * 100, columns=["Female (%)", "Male (%)"])
    prob_df.index = [""]  # hides the row index
    st.write("Prediction Probability:")
    st.dataframe(prob_df.round(2))

    # Calculate certainty before using it
    certainty = np.max(probability, axis=1)

    st.write("Certainty:")
    st.markdown(
        """
        <div style='display:flex; justify-content:space-between; font-size:14px;'>
            <span>0%</span><span>100%</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    for c in certainty:
        st.progress(int(c * 100))

# Convert input data to a Pandas DataFrame
individual_df = pd.DataFrame([input_data])

# Prediction button
if st.button('Predict Sex'):
    predicted_sex, prob_male = predict_individual_sex(model, individual_df)
    prob_female = 1 - prob_male

    st.subheader('Prediction Results:')
    st.write(f"**Predicted Sex:** {'Male' if predicted_sex == 1 else 'Female'}")
    st.write(f"**Probability of being Male:** {prob_male:.4f}")
    st.write(f"**Probability of being Female:** {1 - prob_male:.4f}")
