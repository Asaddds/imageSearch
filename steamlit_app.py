import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import streamlit as st
from PIL import Image

# Load the VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Model(inputs=base_model.input, outputs=base_model.output)

# Function to extract features from an image
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

# Function to load database images and extract features
def load_database_images(folder):
    image_paths = [os.path.join(folder, img) for img in os.listdir(folder) if img.endswith(('.jpg', '.png'))]
    features_list = [(img_path, extract_features(img_path)) for img_path in image_paths]
    return features_list

# Function to find the top 5 similar images
def find_top_matches(new_image_path, features_list, top_n=5):
    new_image_features = extract_features(new_image_path)
    similarities = [
        (img_path, cosine_similarity([new_image_features], [features])[0][0])
        for img_path, features in features_list
    ]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# Streamlit App
st.title("Image Similarity Finder")
st.write("Upload an image to find the 5 most similar images from the database.")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png"])
database_folder = "images"  # Folder where database images are stored

if uploaded_image:
    # Display uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Save the uploaded image temporarily
    temp_path = "temp_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_image.getbuffer())

    # Load database images and their features
    st.write("Loading database images...")
    features_list = load_database_images(database_folder)

    # Find the top 5 matches
    st.write("Finding similar images...")
    top_matches = find_top_matches(temp_path, features_list, top_n=5)

    # Display the top 5 matches
    st.write("Top 5 Similar Images:")
    for img_path, similarity in top_matches:
        st.image(img_path, caption=f"Similarity: {similarity:.4f}", use_column_width=True)

    # Remove the temporary file
    os.remove(temp_path)
