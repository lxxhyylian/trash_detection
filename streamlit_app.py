import subprocess
subprocess.call(["pip", "install", "-r", "./requirements.txt"])


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all logs except errors

import streamlit as st
from keras.models import load_model
from keras.optimizers import Adam
from PIL import Image
import numpy as np
import tensorflow as tf

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

# Define the model file path
model_path = './pretrained_trash_classification.h5'

@st.cache_resource
def load_trained_model(model_path):
    model = load_model(model_path)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load the trained model
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
else:
    model = load_trained_model(model_path)
    print("compiling completed")

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

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        predictions = predict(image)
        predicted_label = labels[np.argmax(predictions)]
        prediction_prob = np.max(predictions)

        st.write(f"Prediction: {predicted_label} - Confidence score: {prediction_prob*100:.2f}%")
