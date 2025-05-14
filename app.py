import streamlit as st
import joblib
import pandas as pd
import plotly.express as px

# Set the page configuration to use the full screen width
st.set_page_config(page_title="Structural Damage & Loss Prediction", layout="wide")

# Loading the trained models
damage_pipeline = joblib.load('models/damage_pipeline_v1.pkl')
loss_pipeline = joblib.load('models/loss_pipeline_v1.pkl')
metrics = joblib.load('models/model_metrics_v1.pkl')
typologies = joblib.load('models/typologies_v1.pkl')



##Display title and model information
st.title("🏠 Structural Damage & Loss Prediction from Flooding using Machine Learning")

st.write("""
    This Web app predicts the building damage ratio (in %) and monetary loss (in NPR) based on the building typology, level, 
    and duration of inundation.
""")
 
st.write("""
    Please provide the necessary details below to get the predicted damage ratio (in %) and loss amount (in NPR).
""")

# Create columns for layout
col1, col2 = st.columns([1.25, 2])  # Define two columns, with the left column smaller

# Place inputs in the left column (col1)
with col1:
    st.header("Input Details")
    typology = st.selectbox("Select Building Typology", typologies)
    inundation = st.number_input("Inundation Level (in meters)", min_value=0.0, value=0.0, format="%.2f", help="Inundation level in meters")
    hours = st.number_input("Hours of Inundation (in hours)", min_value=0.0, value=0.0, format="%.2f", help="Duration of flooding in hours")


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
        if damage_ratio < 1e-3:
            loss = 0.0
        else:
            loss = loss_pipeline.predict(input_df)[0]
        damage_ratio = max(damage_ratio, 0.0)  # Ensure non-negative damage ratio
        loss = max(loss, 0.0)  # Ensure non-negative loss

    st.success(f"Predicted Ratio of Damage: {damage_ratio:.2f}%")
    st.success(f"Predicted Loss: {loss:.2f} NPR")

# Impact Visualization: 3D Scatter Plot
    with col2:
        st.write("")
        if inundation != 0:
            #st.subheader("Impact Visualization")
            fig = px.scatter_3d(input_df, 
                                x='Level of Inundation(m)', 
                                y='Hours of Inundation(hrs)', 
                                z='Ratio of Damage(%)', 
                                color='Building Typology',  # This assigns different colors based on 'Building Typology'
                                title='3D Scatter Plot: Impact of Inundation on Building Damage',
                                labels={'Level of Inundation(m)': 'Level of Inundation (m)', 
                                        'Hours of Inundation(hrs)': 'Hours of Inundation (hrs)', 
                                        'Ratio of Damage(%)': 'Damage Ratio (%)'},
                                color_continuous_scale='Plasma')  # Change the color scale to 'Plasma'

            # Update layout for the plot
            fig.update_layout(
                scene=dict(
                    xaxis=dict(backgroundcolor="white", gridcolor="lightgrey", zerolinecolor="lightgrey"),
                    yaxis=dict(backgroundcolor="white", gridcolor="lightgrey", zerolinecolor="lightgrey"),
                    zaxis=dict(backgroundcolor="white", gridcolor="lightgrey", zerolinecolor="lightgrey"),
                ),
                margin=dict(l=0, r=0, b=0, t=50),  # Set top margin to 50 to accommodate the title
                title=dict(
                    x=0.5,  # Center the title
                    xanchor='center',  # Anchor the title to the center
                    y=0.98,  # Adjust the title's vertical position
                    yanchor='top'  # Anchor the title to the top
                ),
                width=300,  # Set the width of the chart
                height=300  # Set the height of the chart
            )

            # Display the plot
            st.plotly_chart(fig)

# Model Information
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
