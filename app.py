# app.py
import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_iris

# Load model
st.write("Loading model...")
model = joblib.load("iris_model.pkl")
st.write("Model loaded.")

iris = load_iris()

st.title("🌸 Iris Flower Classifier")

# Input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.5)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.0)

# Predict button
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    st.success(f"Predicted species: {iris.target_names[prediction[0]]}")


