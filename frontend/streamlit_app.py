import streamlit as st
import requests
from PIL import Image
import io
import os
current_dir = os.path.dirname(__file__)
image_path = os.path.join(current_dir, "images", "artificial-intelligence.png")


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
    st.image(image_path, width=80)

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

if uploaded_file is not None:
    try:
        img_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(img_bytes))
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.info("🔎 Detecting deepfake... Please wait.")

        files = {"file": (uploaded_file.name, img_bytes, uploaded_file.type)}
        
        # Select API endpoint based on model choice
        if model_type == "MesoNet-4":
            api_url = "http://localhost:8000/predict/meso"
        else:  # EfficientNet
            api_url = "http://localhost:8000/predict/efficient"

        # Make prediction request
        response = requests.post(api_url, files=files)
        
        if response.status_code == 200:
            result = response.json()
            label = result.get("label", "Unknown")
            confidence = result.get("confidence", None)
            
            # Display result with model type
            st.markdown(f"**Model Used:** {model_type}")
            
            if confidence is not None:
                confidence_text = f" (Confidence: {confidence:.2f}%)"
                if label.lower() == "fake":
                    st.error(f"🛑 Prediction: {label}")
                elif label.lower() == "real":
                    st.success(f"✅ Prediction: {label}")
                else:
                    st.info(f"Prediction: {label}")
            else:
                st.info(f"Prediction: {label}")
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Error: {e}")

# --- Footer ---
st.markdown('<div class="footer">© 2025 Deepfake Detection | Made with <span style="color:#e25555;">♥</span> using Streamlit & FastAPI</div>', unsafe_allow_html=True)
