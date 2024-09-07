import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# Load the trained model
def load_model():
    model_path = 'car_model.keras'  # Updated the file extension to .keras
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return tf.keras.models.load_model(model_path)

model = load_model()

def load_class_labels():
    # Assuming class labels are saved in a text file
    labels_path = 'class_labels.txt'
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Class labels file not found at {labels_path}")
    with open(labels_path, 'r') as file:
        labels = file.read().splitlines()
    return labels

class_labels = load_class_labels()

def preprocess_image(image):
    image = image.resize((150, 150))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_car_brand_and_model(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    class_index = np.argmax(predictions[0])
    return class_labels[class_index]

def main():
    st.sidebar.title("Car Brand & Model Identifier")
    st.sidebar.image("https://uploads.audi-mediacenter.com/system/production/media/90567/images/72391bd2d21a80a761f0df1bd5bff197d5804daa/A201895_web_1920.jpg?1698421086", use_column_width=True)

    st.sidebar.write("## Navigation")
    page = st.sidebar.radio("Select a page:", ["Introduction", "Upload Image"])

    if page == "Introduction":
        st.title("Welcome to the Car Brand and Model Identifier")
        st.write("""
            This application uses a deep learning model to identify the brand and model of a car based on an image you upload.
            The model was trained on a large dataset of car images to accurately predict various brands and models.
            To use the app, upload an image of a car and click 'Search' to get predictions.
        """)

    elif page == "Upload Image":
        st.title("Upload Car Image")
        st.write("Upload an image of a car to get predictions.")
        
        uploaded_file = st.file_uploader("Choose a car image...", type=["jpg", "jpeg", "png"], key="file_uploader")

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            
            if st.button('Search'):
                st.write("Classifying...")
                with st.spinner('Classifying...'):
                    try:
                        prediction = predict_car_brand_and_model(image)
                        st.write(f"Prediction: {prediction}")
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.write("No image uploaded. Please upload an image to classify.")

if __name__ == "__main__":
    main()
