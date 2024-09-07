# Car Brand and Model Identifier - Image Classification with Streamlit

## Overview
This project uses a pre-trained ResNet50 model to classify car images and identify the car's brand and model. The project is designed to run as a Streamlit web application, where users can upload images for classification.

## Important Note
The trained model, after completion, is approximately 400MB in size. Due to GitHub's file size limitations, the model cannot be uploaded directly to the repository. Therefore, users must clone this repository and train the model locally or use a pre-trained model stored on their machine. This also means that deploying this project directly on Streamlit Cloud is not possible without the model being uploaded separately.

## Instructions for Running the App Locally
Follow these steps to set up and run the project locally on your machine:

### Clone the Repository
To get started, you need to clone this repository to your local machine:

Navigate to the Directory Where You Want to Clone

In the terminal, type the following command:
```bash
git clone https://github.com/Achalavimukthi/car_identifier.git
```

Navigate to the Cloned Repository Folder
Run the following code first
```bash
python train_model.py
```
Wait some time until complete training modle 
then run following code
```bash
streamlit run app.py
```
