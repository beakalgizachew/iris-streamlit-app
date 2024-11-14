# app.py

import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the pre-trained model from .h5 file (ensure you have 'NDVI_Predictor.h5' in the correct location)
model_file = 'NDVI_Predictor.h5'
model = load_model(model_file)

# Streamlit application layout
st.title("NDVI Predictor App (H5 Model)")
st.write("This app uses a pre-trained model to predict NDVI values from input data.")

# Text input box for user to input data
input_text = st.text_area("Enter your data (comma-separated values)")

# Process the input if available
if input_text:
    try:
        # Parse the comma-separated input into a DataFrame
        data = pd.DataFrame([list(map(float, input_text.split(',')))], columns=["feature1", "feature2", "feature3", "feature4"])

        # Display the input data for the user
        st.subheader("Processed Input Data")
        st.write(data)

        # Normalize the input data (using MinMaxScaler as an example)
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)

        # Predict using the model (assuming it is compatible with the processed input)
        prediction = model.predict(data_scaled)

        # Display the prediction result
        st.subheader("Prediction")
        st.write(f"Predicted NDVI Value: {prediction[0]}")

    except Exception as e:
        st.error(f"Error processing the input: {e}")
else:
    st.write("Please enter your data in the text box above.")
