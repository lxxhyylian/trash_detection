import streamlit as st
import subprocess
import os
from PIL import Image
import numpy as np
import tensorflow as tf

# Install dependencies
subprocess.call(["pip", "install", "-r", "./requirements.txt"])

# Set environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all logs except errors

# Define labels (adjust according to your model's classes)
labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Load model
@st.cache_resource  # Cache the model loading
def load_model():
    model = tf.keras.models.load_model('./pretrained_trash_classification.h5', compile=False)
    return model

with st.spinner('Model is being loaded..'):
    model = load_model()
print("Loading model completed")

# Preprocess the image
@st.cache  # Cache the preprocessing function
def preprocess_image(image):
    """Preprocess the image to the required input shape for the model."""
    image = image.resize((224, 224))  # Adjust size according to model's input size
    image = np.array(image)
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Predict the label
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

# Streamlit app
st.title("Trash Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    predictions = predict(image)
    predicted_label = labels[np.argmax(predictions)]
    prediction_prob = np.max(predictions)

    st.write(f"Prediction: {predicted_label} - Confidence score: {prediction_prob*100:.2f}%")
