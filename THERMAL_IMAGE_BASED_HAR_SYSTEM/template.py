import os
from pathlib import Path
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

# Project name
project_name = "HAR_heat_image_dataset"  # or use: input("Enter project name: ")

# File structure
list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "research/data_ingestion.ipynb",
    "research/base_model.ipynb",
    "research/hyperparameter_tuning.ipynb",
    "research/model_evaluation.ipynb",
    "research/model_training.ipynb",
    "templates/index.html"
]

# Create folders and files
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir = filepath.parent

    if filedir != Path("."):
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir}")

    if not filepath.exists() or filepath.stat().st_size == 0:
        filepath.touch()
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"File already exists: {filepath}")
