"""
=========================================================
                Utility Functions
=========================================================

Description:
This module provides reusable utility functions for
loading project resources required by the application.

Resources Loaded:
• Cleaned sales dataset
• Trained machine learning model
• Model evaluation metrics
• Feature importance data
=========================================================
"""

from pathlib import Path
import subprocess
import sys

import joblib
import pandas as pd


# ==========================================================
# Project Directory Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "cleaned" / "cleaned_sales.csv"
MODEL_PATH = BASE_DIR / "models" / "sales_model.pkl"
METRICS_PATH = BASE_DIR / "models" / "model_metrics.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_importance.csv"
TRAIN_SCRIPT = BASE_DIR / "models" / "train_model.py"


# ==========================================================
# Load Cleaned Sales Dataset
# ==========================================================

def load_data():
    """
    Load the cleaned sales dataset.
    """
    return pd.read_csv(DATA_PATH)


# ==========================================================
# Generate Model (If Missing)
# ==========================================================

def generate_model():
    """
    Generate the trained model automatically
    if it does not exist.
    """
    if not MODEL_PATH.exists():
        subprocess.run(
            [sys.executable, str(TRAIN_SCRIPT)],
            check=True
        )


# ==========================================================
# Load Trained Machine Learning Model
# ==========================================================

def load_model():
    """
    Load the trained sales prediction model.
    If the model does not exist, generate it first.
    """
    generate_model()
    return joblib.load(MODEL_PATH)


# ==========================================================
# Load Model Performance Metrics
# ==========================================================

def load_metrics():
    """
    Load saved model evaluation metrics.
    Generate them if missing.
    """
    if not METRICS_PATH.exists():
        generate_model()

    return joblib.load(METRICS_PATH)


# ==========================================================
# Load Feature Importance Data
# ==========================================================

def load_feature_importance():
    """
    Load feature importance data.
    Generate it if missing.
    """
    if not FEATURE_PATH.exists():
        generate_model()

    return pd.read_csv(FEATURE_PATH)