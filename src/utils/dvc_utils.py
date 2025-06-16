# utils/dvc_utils.py

import yaml
import os

def get_dvc_dataset_version(dvc_file_path="dataset.dvc"):
    """
    Reads the .dvc file and returns the md5 hash of the current dataset version.

    Parameters:
        dvc_file_path (str): Path to the .dvc file (default: "dataset.dvc")

    Returns:
        str: MD5 hash representing the dataset version
    """
    if not os.path.exists(dvc_file_path):
        raise FileNotFoundError(f"{dvc_file_path} not found!")

    with open(dvc_file_path, "r") as f:
        dvc_data = yaml.safe_load(f)

    try:
        return dvc_data["outs"][0]["md5"]
    except (KeyError, IndexError):
        raise ValueError(f"Invalid format in {dvc_file_path}")

