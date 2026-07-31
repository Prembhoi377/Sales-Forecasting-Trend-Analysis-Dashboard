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

These helper functions keep the application modular,
organized and easy to maintain.
=========================================================
"""

from pathlib import Path

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


# ==========================================================
# Load Cleaned Sales Dataset
# ==========================================================
def load_data():
    """
    Load the cleaned sales dataset.

    Returns
    -------
    pandas.DataFrame
        Cleaned sales data.
    """
    return pd.read_csv(DATA_PATH)


# ==========================================================
# Load Trained Machine Learning Model
# ==========================================================
def load_model():
    """
    Load the trained sales prediction model.

    Returns
    -------
    object
        Trained machine learning model.
    """
    return joblib.load(MODEL_PATH)


# ==========================================================
# Load Model Performance Metrics
# ==========================================================
def load_metrics():
    """
    Load saved model evaluation metrics.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """
    return joblib.load(METRICS_PATH)


# ==========================================================
# Load Feature Importance Data
# ==========================================================
def load_feature_importance():
    """
    Load feature importance generated during model training.

    Returns
    -------
    pandas.DataFrame
        Feature importance values.
    """
    return pd.read_csv(FEATURE_PATH)