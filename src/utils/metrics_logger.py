import os
import json
from datetime import datetime

def log_metrics_to_json(metrics: dict, output_dir: str = "logs", filename: str = "metrics.json"):
    """
    Appends model metrics to a JSON file. Each execution is saved as a separate record with a timestamp.

    Args:
        metrics (dict): Dictionary of model metrics (e.g., {"accuracy": 0.9, "loss": 0.2}).
        output_dir (str): Directory where the JSON file will be stored.
        filename (str): Name of the JSON file (default: "metrics.json").
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics
    }

    # Load existing logs if the file exists
    if os.path.isfile(filepath):
        with open(filepath, "r") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Append new entry
    data.append(log_entry)

    # Write updated log
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
