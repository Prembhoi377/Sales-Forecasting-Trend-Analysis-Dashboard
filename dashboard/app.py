import streamlit as st

from dashboard_page import dashboard_page
from performance_page import performance_page
from prediction.prediction import prediction_page
from utils import (
    load_data,
    load_feature_importance,
    load_metrics,
    load_model,
)

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
)

# ==========================================================
# Custom CSS
# ==========================================================

st.markdown(
    """
    <style>

    /* Hide Sidebar */
    section[data-testid="stSidebar"]{
        display:none;
    }

    /* Reduce Top Padding */
    .block-container{
        padding-top:2rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================================
# Load Resources
# ==========================================================

df = load_data()
model = load_model()
metrics = load_metrics()
feature_df = load_feature_importance()

# ==========================================================
# Navigation Tabs
# ==========================================================

dashboard_tab, prediction_tab, performance_tab = st.tabs(
    [
        "📊 Dashboard",
        "🔮 Sales Prediction",
        "📈 Model Performance",
    ]
)

# ==========================================================
# Dashboard
# ==========================================================

with dashboard_tab:
    dashboard_page(df)

# ==========================================================
# Sales Prediction
# ==========================================================

with prediction_tab:
    prediction_page(df, model)

# ==========================================================
# Model Performance
# ==========================================================

with performance_tab:
    performance_page(metrics, feature_df)