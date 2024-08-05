# Run this command in terminal to install all needed resources in requirements.txt
# pip install -r requirements.txt

import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model
model = load_model('pretrained_trash_classification.keras')

# Define the labels (adjust according to your model's classes)
labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def preprocess_image(image):
    """Preprocess the image to the required input shape for the model."""
    image = image.resize((224, 224))  # Adjust size according to model's input size
    image = np.array(image)
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict(image):
    """Predict the class of the image using the loaded model."""
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    return predictions

# Streamlit app
st.title("Trash Classification")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    predictions = predict(image)
    predicted_label = labels[np.argmax(predictions)]
    prediction_prob = np.max(predictions)

    st.write(f"Prediction: {predicted_label} - Confidence score: {prediction_prob*100:.2f}%")
