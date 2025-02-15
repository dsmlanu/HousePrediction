import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Load trained model & preprocessor
model = joblib.load("house_price_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Initialize FastAPI app
app = FastAPI()

# Define request body structure
class HouseFeatures(BaseModel):
    feature_values: list  # Example: [3, 2, 1200, 'suburban', 'brick']

@app.post("/predict")
def predict_price(data: HouseFeatures):
    # Convert input to numpy array
    input_data = np.array([data.feature_values])

    # Preprocess input
    input_data_transformed = preprocessor.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_transformed)

    # Return the predicted house price
    return {"predicted_price": prediction[0]}

# Run API server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

import streamlit as st
import requests

# FastAPI URL (Make sure your FastAPI backend is running)
API_URL = "http://0.0.0.0:8000/predict"

# Streamlit UI
st.title("🏡 House Price Prediction App")
st.write("Enter the house details below and get the estimated price!")

# Input fields
num_bedrooms = st.number_input("Number of Bedrooms", min_value=1, step=1)
num_bathrooms = st.number_input("Number of Bathrooms", min_value=1, step=1)
area_sqft = st.number_input("Area (sq ft)", min_value=500, step=10)
location = st.selectbox("Location", ["urban", "suburban", "rural"])
material = st.selectbox("Construction Material", ["brick", "wood", "concrete"])

# Convert categorical inputs
location_map = {"urban": 0, "suburban": 1, "rural": 2}
material_map = {"brick": 0, "wood": 1, "concrete": 2}

# Predict button
if st.button("Predict Price 💰"):
    # Prepare data for API request
    input_data = [num_bedrooms, num_bathrooms, area_sqft, location_map[location], material_map[material]]

    # Send request to FastAPI
    response = requests.post(API_URL, json={"feature_values": input_data})

    if response.status_code == 200:
        price = response.json()["predicted_price"]
        st.success(f"🏡 Estimated House Price: **${price:,.2f}**")
    else:
        st.error("❌ Error: Could not get a prediction. Make sure the FastAPI server is running.")

