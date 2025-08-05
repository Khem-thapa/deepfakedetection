import streamlit as st
import requests
from PIL import Image
import io

st.title("Deepfake Detection App")

st.write("Upload an image to check if it is a deepfake.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Read image bytes for sending to API
        img_bytes = uploaded_file.read()
        # Display the image
        image = Image.open(io.BytesIO(img_bytes))
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")
        st.write("Detecting...")

        files = {"file": (uploaded_file.name, img_bytes, uploaded_file.type)}
        api_url = "http://localhost:8000/predict/"  # Make sure FastAPI is running here
        response = requests.post(api_url, files=files)
        if response.status_code == 200:
            result = response.json()
            label = result.get("label", "Unknown")
            confidence = result.get("confidence", None)
            if confidence is not None:
                st.success(f"Prediction: {label} (Confidence: {confidence:.2f})")
            else:
                st.success(f"Prediction: {label}")
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Error: {e}")
