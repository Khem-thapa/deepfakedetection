# Deepfake Detection Project

This project provides an end-to-end solution for detecting deepfake images using a machine learning model, FastAPI backend, and a Streamlit-based interactive UI.

## Project Structure

```
Application/
│
├── api/                # FastAPI backend for deepfake detection
│   ├── apis/           # API route definitions
│   ├── models/         # Model loading and prediction code
│   ├── utils/          # Preprocessing utilities
│   └── main.py         # FastAPI app entry point
│
├── frontend/           # Streamlit UI for user interaction
│   ├── images/         # UI logos and icons
│   └── streamlit_app.py# Streamlit app
│
├── src/                # Core ML code, experiments, and utilities
│   ├── config/         # Configuration files (e.g., config.yaml)
│   ├── data/           # Data loaders for training and evaluation
│   ├── evaluate/       # Model evaluation scripts
│   ├── experiments/    # Experiment runner scripts for different models
│   ├── models/         # Model architectures (e.g., EfficientNet, MesoNet)
│   ├── predict/        # Prediction scripts for different models
│   └── utils/          # Utility modules (config loader, logging, DVC, etc.)
│
├── models/             # Saved and exported model weights/checkpoints
├── results/            # Evaluation results, predictions, and output artifacts
├── start_app.bat       # Batch file to start both API and UI
│
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```
## Models and Results Folders

- `models/`: This folder stores trained model weights, exported checkpoints, or any serialized models ready for inference or deployment.
- `results/`: This folder contains evaluation results, prediction outputs, logs, and other artifacts generated during experiments or inference.

## DVC Files

- `dvc.yaml`: This file defines the DVC (Data Version Control) pipeline stages, including data preparation, training, evaluation, and other reproducible steps. It describes the dependencies, outputs, and commands for each stage, enabling consistent and trackable ML workflows.
- `dvc.lock`: This file records the exact versions (hashes) of data, code, and outputs used or produced in each pipeline stage, ensuring full reproducibility of experiments and results.
## Source Code (`src/`) Overview

The `src` folder contains the core machine learning code and utilities for model training, evaluation, and prediction:

- `config/`: Configuration files for experiments and training (e.g., `config.yaml`).
- `data/`: Data loading scripts for training and evaluation (`data_loader.py`, `eval_data_loader.py`).
- `evaluate/`: Scripts to evaluate trained models (e.g., `evaluate_meso4_model.py`).
- `experiments/`: Scripts to run experiments with different models (e.g., `run_efficientnet.py`, `run_mesonet.py`).
- `models/`: Model architecture definitions and related code (e.g., EfficientNet, MesoNet).
- `predict/`: Scripts for making predictions with trained models (`predict_efficientnet.py`, `predict_mesonet.py`, etc.).
- `utils/`: Utility modules for configuration loading, logging, DVC integration, and MLflow support.

Use these scripts to train, evaluate, and deploy your deepfake detection models as needed.

## Setup Instructions

### 1. Clone the Repository
```
git clone <repo-url>
cd Application
```

### 2. Install Dependencies
It is recommended to use a virtual environment (e.g., conda or venv).

```
pip install -r requirements.txt
cd api
pip install -r requirements.txt
cd ../frontend
pip install -r requirements.txt
```

### 3. Prepare Model and Data
- Place your trained model weights in the appropriate folder (e.g., `api/models/`).
- Organize your datasets as needed in the `dataset/` directory.

### 4. Run the Application

#### Run Both Together
- To start both API and UI in separate terminals:
  ```
  start_app.bat
  ```

### 5. Using the App
- Open your browser and go to the URL shown by Streamlit (usually http://localhost:8501).
- Upload an image to check if it is a deepfake. The result and confidence will be displayed.

## Notes
- Ensure the API server is running before using the Streamlit UI.
- The API endpoint is expected at `http://localhost:8000/predict/`.
- You can customize the UI by editing `frontend/streamlit_app.py` and adding images to `frontend/images/`.

## License
This project is for academic and research purposes.

---
Made with ❤️ using FastAPI & Streamlit.
