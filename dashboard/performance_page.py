"""
=========================================================
              Model Performance Dashboard
=========================================================

Description:
This module presents the performance of the trained
machine learning model through evaluation metrics and
feature importance analysis.

Sections:
• Model Evaluation Metrics
• Feature Importance Analysis
=========================================================
"""

import streamlit as st

from charts import feature_importance_chart


# ==========================================================
# Model Performance Page
# ==========================================================
def performance_page(metrics, feature_df):
    """
    Render the model performance dashboard.

    Parameters
    ----------
    metrics : dict
        Dictionary containing model evaluation metrics.

    feature_df : pandas.DataFrame
        Feature importance values generated during training.
    """

    # ==========================================================
    # Page Header
    # ==========================================================
    st.title("📊 Model Performance")

    st.caption(
        "Evaluate the trained Random Forest Regression model "
        "using standard regression metrics and feature importance."
    )

    st.divider()

    # ==========================================================
    # Performance Metrics
    # ==========================================================
    st.subheader("📈 Model Evaluation Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "R² Score",
            f"{metrics['R2 Score']:.4f}",
        )

        st.metric(
            "Mean Absolute Error (MAE)",
            f"{metrics['MAE']:.2f}",
        )

    with col2:
        st.metric(
            "Root Mean Squared Error (RMSE)",
            f"{metrics['RMSE']:.2f}",
        )

        st.metric(
            "Mean Squared Error (MSE)",
            f"{metrics['MSE']:.2f}",
        )

    st.divider()

    # ==========================================================
    # Model Information
    # ==========================================================
    st.info(
        "Model: Random Forest Regressor\n\n"
        "The model has been trained and evaluated on the cleaned sales dataset."
    )

    st.divider()

    # ==========================================================
    # Feature Importance
    # ==========================================================
    st.subheader("🏆 Top 10 Important Features")

    feature_importance_chart(feature_df)