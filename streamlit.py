import streamlit as st
import os
from PIL import Image
from cnnClassifier.pipeline.prediction import PredictionPipeline

# Set page configuration
st.set_page_config(
    page_title="Kidney Disease Classification",
    page_icon="🔬",
    layout="wide"
)

# Title and description
st.title("Kidney Disease Classification")
st.markdown("Upload a kidney CT scan image to classify if it shows normal kidney or tumor")

# File uploader for images
uploaded_file = st.file_uploader("Choose a kidney CT scan image...", type=["jpg", "jpeg", "png"])

# Create a temp directory for uploaded files if it doesn't exist
temp_dir = os.path.join(os.getcwd(), "temp_uploads")
os.makedirs(temp_dir, exist_ok=True)

# Display pipeline stages in sidebar
with st.sidebar:
    st.header("About this app")
    st.markdown("""
    This application uses a trained CNN model to classify kidney CT scan images as:
    - Normal
    - Tumor
    
    The model was trained with the following pipeline:
    1. Data Ingestion
    2. Base Model Preparation
    3. Model Training
    4. Model Evaluation
    """)

# Function to make prediction
def make_prediction(image_path):
    classifier = PredictionPipeline(image_path)
    result = classifier.predict()
    return result[0]["image"]

# When a file is uploaded
if uploaded_file is not None:
    # Display the image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Save the file to a temporary location
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Make prediction when button is clicked
    if st.button("Classify Image"):
        with st.spinner("Classifying..."):
            # Get prediction
            prediction = make_prediction(temp_file_path)
            
            # Display result
            with col2:
                st.subheader("Classification Result")
                if prediction == "Tumor":
                    st.error(f"Prediction: {prediction}")
                else:
                    st.success(f"Prediction: {prediction}")
                
                # Add additional information based on prediction
                if prediction == "Tumor":
                    st.warning("This image shows signs of a kidney tumor. Please consult with a healthcare professional.")
                else:
                    st.info("This image appears to show a normal kidney.")
            
            # Clean up the temp file
            os.remove(temp_file_path)

# Instructions when no file is uploaded
else:
    st.info("Please upload an image to get started.")
    
    # Example image section
    st.subheader("Example Results")
    st.markdown("""
    The model will classify kidney CT scans into two categories:
    
    - **Normal**: Healthy kidney tissue
    - **Tumor**: Kidney tissue showing signs of tumor
    
    Upload a kidney CT scan image to see the classification results.
    """)

# Footer
st.markdown("---")
st.markdown("Kidney Disease Classification App | Created using Streamlit and CNN model")