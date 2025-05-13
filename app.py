import streamlit as st
import joblib
import pandas as pd

# Loading the trained models
damage_pipeline = joblib.load('models/damage_pipeline_v1.pkl')
loss_pipeline = joblib.load('models/loss_pipeline_v1.pkl')
metrics = joblib.load('models/model_metrics_v1.pkl')

#Loading datatset
df = pd.read_csv('Dataset/dataset.csv')


# Extracting unique typologies from dataset
typologies = df['Building Typology'].unique().tolist()

# Display title and model information
st.title("🏠 Structural Damage & Loss Prediction from Flooding using Machine Learning")

st.write("""
    This Web app predicts the building damage ratio (in %) and monetary loss (in NPR) based on the building typology, level, 
    and duration of inundation.
""")

st.write("""
    Please provide the necessary details below to get the predicted damage ratio (in %) and loss amount (in NPR).
""")

typology = st.selectbox("Select Building Typology", typologies)
inundation = st.number_input("Inundation Level (in meters)", min_value=0.0, format="%.2f")
hours = st.number_input("Hours of Inundation (in hours)", min_value=0.0, format="%.1f")

if st.button("Predict"):
    hr_lv = inundation * hours
    input_df = pd.DataFrame([{
        'Building Typology': typology,
        'Level of Inundation(m)': inundation,
        'Hours of Inundation(hrs)': hours,
        'Combined effect of level and hour of inundation': hr_lv
    }])
    if inundation == 0:
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
    The models used for this prediction are Gradient Boosting Regressors, which are ensemble learning algorithms 
    that combine the predictions of multiple weak learners to produce more accurate predictions.
""")

st.write(f"R² score for **Damage Prediction Model**: **{metrics['damage_r2']:.4f}**")
st.write(f"R² score for **Loss Prediction Model**: **{metrics['loss_r2']:.4f}**")
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-style: italic;'>"
    "Web Developed by Pratistha Katwal<br>"
    "</p>",
    unsafe_allow_html=True
)


