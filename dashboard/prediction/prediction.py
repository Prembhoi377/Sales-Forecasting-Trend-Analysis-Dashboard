"""
=========================================================
                Sales Prediction Module
=========================================================

Description:
This module enables users to predict future sales
using the trained machine learning model.

Workflow:
• Collect user inputs
• Prepare prediction features
• Generate sales prediction
• Display estimated sales value
=========================================================
"""

import pandas as pd
import streamlit as st


# ==========================================================
# Sales Prediction Page
# ==========================================================
def prediction_page(df, model):
    """
    Render the sales prediction interface.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned sales dataset.

    model : object
        Trained machine learning model.
    """

    # ==========================================================
    # Page Header
    # ==========================================================
    st.title("🤖 Sales Prediction")

    st.caption(
        "Estimate future sales based on business and product "
        "information using the trained machine learning model."
    )

    st.divider()

    # ==========================================================
    # User Input Section
    # ==========================================================

    col1, col2 = st.columns(2)

    with col1:
        year = st.selectbox(
            "Select Year",
            sorted(df["Year"].unique()),
        )

        region = st.selectbox(
            "Select Region",
            sorted(df["Region"].unique()),
        )

        category = st.selectbox(
            "Select Category",
            sorted(df["Category"].unique()),
        )

        ship_mode = st.selectbox(
            "Select Ship Mode",
            sorted(df["Ship Mode"].unique()),
        )

        segment = st.selectbox(
            "Select Segment",
            sorted(df["Segment"].unique()),
        )

        country = st.selectbox(
            "Select Country",
            sorted(df["Country"].unique()),
        )

    with col2:
        state = st.selectbox(
            "Select State",
            sorted(df["State"].unique()),
        )

        city = st.selectbox(
            "Select City",
            sorted(df["City"].unique()),
        )

        sub_category = st.selectbox(
            "Select Sub-Category",
            sorted(df["Sub-Category"].unique()),
        )

        month = st.selectbox(
            "Select Month",
            sorted(df["Month"].unique()),
        )

        month_number = st.selectbox(
            "Select Month Number",
            sorted(df["Month Number"].unique()),
        )

        quarter = st.selectbox(
            "Select Quarter",
            sorted(df["Quarter"].unique()),
        )

    st.divider()

    # ==========================================================
    # Prediction Button
    # ==========================================================

    if st.button("🚀 Predict Sales", use_container_width=True):

        input_data = pd.DataFrame(
            [{
                "Ship Mode": ship_mode,
                "Segment": segment,
                "Country": country,
                "City": city,
                "State": state,
                "Region": region,
                "Category": category,
                "Sub-Category": sub_category,
                "Year": year,
                "Month": month,
                "Month Number": month_number,
                "Quarter": quarter,
            }]
        )

        prediction = model.predict(input_data)[0]

        # ==========================================================
        # Prediction Result
        # ==========================================================

        st.success(
            f"🎯 Estimated Sales: ₹ {prediction:,.2f}"
        )

        with st.expander("View Prediction Details"):
            st.dataframe(
                input_data,
                use_container_width=True,
            )