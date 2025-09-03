import streamlit as st
from PIL import Image
import io
import tensorflow as tf
import numpy as np
import sys
import os
current_dir = os.path.dirname(__file__)
model_path_eff = os.path.join(current_dir, "models/efficientnet/", "efficient_model.h5")
model_path_meso = os.path.join(current_dir, "models/meso4/", "meso4_full_model.h5")

# process image for mesonet
def preprocess_image(image, target_size=(256, 256)):
    '''
        Process the image, resizing it and normalizing pixel values.
    '''
    image = image.resize(target_size)
    arr = np.asarray(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # shape (1, 256, 256, 3)
    return arr

# process image for efficientnet
def preprocess_image_ef(image, target_size=(224, 224)):
    '''
        Process the image, resizing it and normalizing pixel values.
    '''
    image = image.resize(target_size)
    arr = np.asarray(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # shape (1, 224, 224, 3)
    return arr

# load mesonet model
@st.cache_resource
def load_model_meso():
    model = tf.keras.models.load_model(model_path_meso, compile=False)
    print("[INFO] Model Loaded!")
    return model

# load efficientnet model
@st.cache_resource
def load_model_eff():
    """
    Load the EfficientNet model for deepfake detection
    """
    # model = tf_load_model(model_path)
    model = tf.keras.models.load_model(model_path_eff, compile=False)
    return model

# --- Streamlit App ---
# --- Custom CSS for style ---
st.markdown(
    """
    <style>
    .main-header {
        font-size:2.5rem;
        font-weight:700;
        color:#1a1a2e;
        margin-bottom:0.5em;
        text-align:center;
    }
    .sub-header {
        font-size:1.2rem;
        color:#393e46;
        text-align:center;
        margin-bottom:2em;
    }
    .model-selector {
        margin-bottom:2em;
        padding:1em;
        background-color:#f8f9fa;
        border-radius:5px;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f0f2f6;
        color: #888;
        text-align: center;
        padding: 10px 0;
        font-size: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Logo and Header ---
col1, col2 = st.columns([2,1])
# with col1:
    # st.image("frontend/images/Screenshot_logo.jpg", width=80)
with col1:
    st.markdown('<div class="main-header">🕵️‍♂️ Deepfake Detection App</div>', unsafe_allow_html=True)
with col2:
    st.image("frontend/images/artificial-intelligence.png", width=80) # add image artificial-intelligence.png

st.markdown('<div class="sub-header">Upload an image to check if it is a <b>deepfake</b>.<br>Powered by FastAPI & AI</div>', unsafe_allow_html=True)

# --- Model Selection ---
st.markdown('<div class="model-selector">', unsafe_allow_html=True)
model_type = st.radio(
    "Select Deepfake Detection Model:",
    ["MesoNet-4", "EfficientNet"],
    help="Choose the model you want to use for detection"
)
st.markdown('</div>', unsafe_allow_html=True)

# --- File uploader ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], help="Supported formats: jpg, jpeg, png")

# Load models
# Load both models at startup

if uploaded_file is not None:
    try:
        img_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.info("🔎 Detecting deepfake... Please wait.")

        files = {"file": (uploaded_file.name, img_bytes, uploaded_file.type)}
        
        # Select API endpoint based on model choice
        if model_type == "MesoNet-4":
            preprocessed = preprocess_image(image)
            prediction =  load_model_meso().predict(preprocessed)[0][0]
        else:  # EfficientNet
            preprocessed = preprocess_image_ef(image)
            prediction = load_model_eff().predict(preprocessed)[0][0]

        # Check for prediction
        label = "Fake" if prediction> 0.6 else "Real" 
        confidence = prediction * 100 if label == "Fake" else (1 - prediction) * 100

        # Display result with model type
        st.markdown(f"**Model Used:** {model_type}")
        if label.lower() == "fake":
            st.error(f"🛑 Prediction: {label} (Confidence: {confidence:.2f}%)")
        else:
            st.success(f"✅ Prediction: {label} (Confidence: {confidence:.2f}%)")
    except Exception as e:
        st.error(f"Error: {e}")

# --- Footer ---
st.markdown('<div class="footer">© 2025 Deepfake Detection | Made with <span style="color:#e25555;">♥</span> using Streamlit & FastAPI</div>', unsafe_allow_html=True)
