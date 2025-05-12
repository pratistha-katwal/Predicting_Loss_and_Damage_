import streamlit as st
import joblib
import pandas as pd

# Load the trained models
damage_pipeline = joblib.load('models/damage_pipeline.pkl')
loss_pipeline = joblib.load('models/loss_pipeline.pkl')
metrics = joblib.load('models/model_metrics.pkl')

#Load datatset
df = pd.read_csv('Dataset/dataset.csv')


# Extract unique typologies from your dataset
typologies = df['Building Typology'].unique().tolist()

# Display title and model information
st.title("🏠 Structural Damage & Loss Prediction from Flooding using Machine Learning")
st.write("This app predicts the building damage and loss based on the level and hours of inundation.")

typology = st.selectbox("Select Building Typology", typologies)
inundation = st.number_input("Inundation Level (m)", min_value=0.0, format="%.2f")
hours = st.number_input("Hours of Inundation (hrs)", min_value=0.0, format="%.1f")

if st.button("Predict"):
    hr_lv = inundation * hours
    input_df = pd.DataFrame([{
        'Building Typology': typology,
        'Level of Inundation(m)': inundation,
        'Hours of Inundation(hrs)': hours,
        'hr_lv': hr_lv
    }])
    if inundation == 0 and hours == 0:
        damage_ratio = 0.0
        loss = 0.0
    else:
        damage_ratio = damage_pipeline.predict(input_df)[0]
        input_df['Ratio of Damage(%)'] = damage_ratio
        loss = loss_pipeline.predict(input_df)[0]

    st.success(f"Predicted Ratio of Damage: {damage_ratio:.2f}%")
    st.success(f"Predicted Loss: {loss:.2f} NPR")


st.subheader("Model Information")
st.write("""
The models used for this prediction are **Gradient Boosting Regressor**, which is an ensemble learning algorithm that combines the predictions of multiple weak models to create a more accurate prediction.
""")
st.write(f"R² score for **Damage Prediction Model**: **{metrics['damage_r2']:.4f}**")
st.write(f"R² score for **Loss Prediction Model**: **{metrics['loss_r2']:.4f}**")