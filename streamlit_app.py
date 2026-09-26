import streamlit as st
import pandas as pd
import joblib

# --- 1. Load the trained model and feature names ---
# In a real Streamlit app, these files would be in the same directory
# or accessible via a defined path.
model = joblib.load('gradient_boosting_model.pkl') # Assuming model is saved in /content/
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
st.title('Tooth Dimension Based Sex Estimation')
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

st.markdown("""
**How to run this app:**
1. Save the above code as `streamlit_app.py` in the same directory as `best_sex_estimation_model.joblib` and `feature_names.joblib`.
2. Open your terminal or command prompt.
3. Navigate to that directory.
4. Run the command: `streamlit run streamlit_app.py`
""")
