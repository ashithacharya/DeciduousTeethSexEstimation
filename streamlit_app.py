# Trigger rebuild
import numpy as np
import streamlit as st
import pandas as pd
import joblib
import requests
import io

# Title
st.title("Deciduous/Primary Tooth Dimension Based Sex Estimation")

# Summary
st.markdown("""
### 🧭 Study Overview
Sex estimation from dental metrics offers valuable diagnostic evidence in forensic anthropology when skeletal indicators are missing or fragmented.  
This study evaluated mesiodistal (MD) and buccolingual (BL) dimensions of the complete primary dentition from 500 children aged 3–5 years (250 males, 250 females) in Southwestern India.  

---

### ⚙️ Model Performance
Nine supervised machine learning (ML) classifiers were benchmarked against a classical Logistic Regression Analysis (LRA) baseline on an independent held-out test set (n=100).  
**Gradient Boosting** emerged as the top classifier for threshold-dependent performance, achieving 62.0% accuracy and a macro F1-score of 0.6144, outperforming the LRA baseline (54.0% accuracy; F1=0.5370) by 8.0 percentage points.  

**CatBoost** offered the most balanced overall profile, combining 60.0% accuracy with high discrimination (ROC-AUC = 0.6632) and optimal probability calibration (Brier score = 0.2309).  
**LightGBM** maximized class separation (ROC-AUC = 0.6668), whereas **Extra Trees** yielded the highest precision-recall integration (AUPRC = 0.6785).  

Feature importance consensus highlighted primary tooth dimensions—specifically 81BL, 82MD, 83MD, 71MD, and 54BL—as key predictive drivers.  
Standalone Decision Trees and Support Vector Machines underperformed relative to the LRA baseline.  

---

### 🧩 Interpretation
Overall primary crown sex difference was low (mean percentage dimorphism = 1.4133%).  
While non-parametric tree ensembles captured complex non-linear feature interactions to systematically surpass traditional linear biostatistical baselines, peak classification accuracy remained modest (62.0%).  

Consequently, calliper-derived primary odontometrics provide useful non-linear predictive signals, but should serve as complementary diagnostic evidence rather than standalone primary indicators in forensic sex estimation.  

---

### 💻 Interactive Tool
This webpage allows users to input deciduous tooth dimensions (manually or via a `.csv` file containing 40 MD and BL dimensions) and estimate sex using the best-performing ML models.  

---

**NOTE:** This tool is intended for forensic and clinical assistance only.  
Results should be interpreted by qualified forensic dental professionals and/or forensic anthropologists.  
The models are specific to an Indian dataset (Southwestern Indian population of 3–5-year-olds).  
They have not been validated in other populations, and results should be interpreted with caution outside this demographic context.
""")

# --- Model selection ---
@st.cache_resource
def load_model_from_github(url):
    response = requests.get(url)
    return joblib.load(io.BytesIO(response.content))

model_urls = {
    "Gradient Boosting": "https://raw.githubusercontent.com/ashithacharya/DeciduousTeethSexEstimation/main/gradient_boosting_model.pkl",
    "LightGBM": "https://raw.githubusercontent.com/ashithacharya/DeciduousTeethSexEstimation/main/lightgbm_model.pkl",
    "CatBoost": "https://raw.githubusercontent.com/ashithacharya/DeciduousTeethSexEstimation/main/catboost_model.pkl",
    "Extra Trees": "https://raw.githubusercontent.com/ashithacharya/DeciduousTeethSexEstimation/main/extra_trees_model.pkl"
}

# Dropdown for selecting one model
model_choice = st.selectbox(
    "**Choose a model for sex estimation:**",
    list(model_urls.keys())
)
model = load_model_from_github(model_urls[model_choice])

# --- Feature names ---
feature_names = [
    '51MD', '51BL', '52MD', '52BL', '53MD', '53BL', '54MD', '54BL',
    '55MD', '55BL', '61MD', '61BL', '62MD', '62BL', '63MD', '63BL',
    '64MD', '64BL', '65MD', '65BL', '71MD', '71BL', '72MD', '72BL',
    '73MD', '73BL', '74MD', '74BL', '75MD', '75BL', '81MD', '81BL',
    '82MD', '82BL', '83MD', '83BL', '84MD', '84BL', '85MD', '85BL'
]

# --- Prediction function ---
def predict_individual_sex(model, individual_features_df):
    individual_features_df = individual_features_df[feature_names]
    prediction = model.predict(individual_features_df)[0]
    probability_male = model.predict_proba(individual_features_df)[0][1]
    return prediction, probability_male

# --- UI ---
st.markdown("**Enter the tooth dimensions below to estimate sex.**")

default_values = {feature: 0.0 for feature in feature_names}
input_data = {}

cols = st.columns(4)
for i, feature in enumerate(feature_names):
    with cols[i % 4]:
        input_data[feature] = st.number_input(
            f'{feature}',
            value=float(default_values[feature]),
            format='%.2f',
            key=f'input_{feature}'
        )

individual_df = pd.DataFrame([input_data])

# Manual input prediction button
if st.button('Predict Sex (Manual Input)', key='manual_predict'):
    prediction = model.predict(individual_df)
    probability = model.predict_proba(individual_df)

    predicted_sex = prediction[0]
    prob_male = probability[0][1]
    prob_female = probability[0][0]

    st.subheader('Prediction Results (Manual Input):')
    st.write(f"**Predicted Sex:** {'Male' if predicted_sex == 1 else 'Female'}")
    st.write(f"**Probability of being Male:** {prob_male:.2f}")
    st.write(f"**Probability of being Female:** {prob_female:.2f}")

# --- CSV upload prediction ---
st.markdown("**Alternatively, upload a CSV file with tooth dimensions.**")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    X_input = data[feature_names]

    # CSV prediction button
    if st.button('Predict Sex (CSV Upload)', key='csv_predict'):
        prediction = model.predict(X_input)
        probability = model.predict_proba(X_input)

        st.subheader("CSV Prediction Results")

        prob_df = pd.DataFrame(probability * 100, columns=["Female (%)", "Male (%)"])
        prob_df.index = [""]

        styled_prob_df = prob_df.round(1).style.set_properties(**{
            'text-align': 'center'
        }).set_table_styles([{
            'selector': 'th',
            'props': [('text-align', 'center')]
        }])

        st.write("Prediction Probability:")
        st.write(styled_prob_df)

        certainty = np.max(probability, axis=1)
        st.write("Certainty:")
        for c in certainty:
            st.progress(int(c * 100))

# --- Demo dataset prediction ---
@st.cache_data
def load_subset_dataset():
    url = "https://raw.githubusercontent.com/ashithacharya/DeciduousTeethSexEstimation/main/Tooth_Dec_Dimensions_100.csv"
    df = pd.read_csv(url)
    males = df[df["Sex"] == 1].head(10)
    females = df[df["Sex"] == 0].head(10)
    return pd.concat([males, females])

st.markdown("**Or try the demo dataset (20 subjects: 10 males, 10 females).**")
if st.button("Run Predictions on Demo Dataset", key="demo_predict"):
    demo_data = load_subset_dataset()
    X_input = demo_data[feature_names]
    y_true = demo_data["Sex"]

    prediction = model.predict(X_input)
    probability = model.predict_proba(X_input)

    st.subheader("Demo Dataset Prediction Results")
    results_df = pd.DataFrame({
        "True Sex": ["Female" if s == 0 else "Male" for s in y_true],
        "Predicted Sex": ["Female" if p == 0 else "Male" for p in prediction],
        "Prob Male (%)": (probability[:,1] * 100).round(1),
        "Prob Female (%)": (probability[:,0] * 100).round(1)
    })

    results_df.index = np.arange(1, len(results_df) + 1)  # start index at 1
    st.dataframe(results_df)
