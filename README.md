
# Car Brand and Model Identifier - Image Classification with Streamlit

## Overview
This project uses a pre-trained **ResNet50** model to classify car images and identify their brand and model. The application is built with **Streamlit**, allowing users to upload images and get predictions.

## Important Note
The trained model file is approximately **400MB** in size. Due to GitHub's file size limitations, this model cannot be included in the repository. Users must:
1. **Train the model locally** by running the provided script, or
2. **Use a pre-trained model** stored on their own machine.

As a result, deploying this project directly on **Streamlit Cloud** without a separately uploaded model is not feasible.

## Instructions for Running the App Locally

### 1. Clone the Repository
Clone the repository to your local machine using:
```bash
git clone https://github.com/Achalavimukthi/car_identifier.git
```

### 2. Navigate to the Project Directory
Change to the directory where the repository was cloned:
```bash
cd car_identifier
```

### 3. Install Required Dependencies
Install the necessary Python packages. Ensure you have a `requirements.txt` file in the repository. Run:
```bash
pip install -r requirements.txt
```

### 4. Train the Model
Execute the training script to train the model. This step may take some time:
```bash
python train_model.py
```
The model will be saved locally upon completion.

### 5. Run the Streamlit Application
After training, start the Streamlit app with:
```bash
streamlit run app.py
```
Open your browser and navigate to the provided URL to upload car images and receive predictions.

