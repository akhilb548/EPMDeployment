
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(
    repo_id="indianakhil/engine-predictive-maintenance-model",
    filename="best_model.pkl"
)

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Engine Predictive Maintenance
st.title("Engine Predictive Maintenance App")
st.write(
    "The Engine Predictive Maintenance App predicts whether an industrial engine is operating "
    "**normally** or is **faulty and requires maintenance** based on six real-time sensor readings."
)
st.write("Kindly enter the current sensor readings to check the engine condition.")

# Collect user input
Engine_RPM = st.number_input(
    "Engine RPM (revolutions per minute)", min_value=0.0, max_value=3000.0, value=800.0
)
Lub_Oil_Pressure = st.number_input(
    "Lubricating Oil Pressure (bar)", min_value=0.0, max_value=10.0, value=3.3
)
Fuel_Pressure = st.number_input(
    "Fuel Pressure (bar)", min_value=0.0, max_value=25.0, value=6.5
)
Coolant_Pressure = st.number_input(
    "Coolant Pressure (bar)", min_value=0.0, max_value=10.0, value=2.3
)
Lub_Oil_Temperature = st.number_input(
    "Lubricating Oil Temperature (deg C)", min_value=50.0, max_value=100.0, value=77.6
)
Coolant_Temperature = st.number_input(
    "Coolant Temperature (deg C)", min_value=50.0, max_value=100.0, value=78.1
)

# Package inputs into a DataFrame matching training feature order
input_data = pd.DataFrame([{
    "Engine_RPM": Engine_RPM,
    "Lub_Oil_Pressure": Lub_Oil_Pressure,
    "Fuel_Pressure": Fuel_Pressure,
    "Coolant_Pressure": Coolant_Pressure,
    "Lub_Oil_Temperature": Lub_Oil_Temperature,
    "Coolant_Temperature": Coolant_Temperature
}])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    if prediction == 1:
        st.error(
            f"Warning: Based on the sensor readings provided, the engine is likely FAULTY "
            f"and requires maintenance. (Fault probability: {probability:.1%})"
        )
    else:
        st.success(
            f"Based on the sensor readings provided, the engine is operating NORMALLY. "
            f"(Fault probability: {probability:.1%})"
        )
