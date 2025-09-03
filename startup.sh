#!/bin/bash

# Activate the Python environment (Azure Web Apps specific)
source /usr/local/env/bin/activate 2>/dev/null || true

# Set environment variables
export PORT=${PORT:-8000}

# Install any missing dependencies
pip install -r requirements.txt

# Start Streamlit
streamlit run app.py \
    --server.port $PORT \
    --server.address 0.0.0.0 \
    --browser.serverAddress $WEBSITE_HOSTNAME \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --theme.base "light"
