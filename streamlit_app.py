import streamlit as st
import subprocess
import os

from PIL import Image
import numpy as np
import pickle 
import tensorflow as tf

subprocess.call(["pip", "install", "-r", "./requirements.txt"])
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all logs except errors
# pickle_in = open('trash_detection/pretrained_model.pkl', 'rb') 
# model = pickle.load(pickle_in)

def load_model():
    model=tf.keras.models.load_model('./pretrained_trash_classification.h5')
    return model
with st.spinner('Model is being loaded..'):
    model=load_model()

def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    print(prediction) 
    return prediction 

# Define the labels (adjust according to your model's classes)
labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def preprocess_image(image):
    """Preprocess the image to the required input shape for the model."""
    image = image.resize((224, 224))  # Adjust size according to model's input size
    image = np.array(image)
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

    

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
