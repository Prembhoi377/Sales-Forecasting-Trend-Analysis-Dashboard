"""
=========================================================
                Dashboard Visualization Module
=========================================================

Project:
Sales Forecasting & Trend Analysis Dashboard

Description:
This module contains reusable Plotly visualizations
used throughout the dashboard.

=========================================================
"""

import streamlit as st
import plotly.express as px

# ==========================================================
# Global Chart CSS
# ==========================================================

st.markdown("""
<style>

div[data-testid="stVerticalBlockBorderWrapper"]{
    min-height:300px !important;
    padding:8px !important;
    border-radius:12px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# Monthly Sales Trend
# ==========================================================

def monthly_sales_chart(filtered_df):
    """
    Display monthly sales trend.
    """

    sales_trend = (
        filtered_df.groupby(["Month Number", "Month"])["Sales"]
        .sum()
        .reset_index()
        .sort_values("Month Number")
    )

    month_map = {
        "January": "Jan",
        "February": "Feb",
        "March": "Mar",
        "April": "Apr",
        "May": "May",
        "June": "Jun",
        "July": "Jul",
        "August": "Aug",
        "September": "Sep",
        "October": "Oct",
        "November": "Nov",
        "December": "Dec",
    }

    sales_trend["Month"] = sales_trend["Month"].map(month_map)

    fig = px.line(
        sales_trend,
        x="Month",
        y="Sales",
        markers=True,
        title="Monthly Sales Trend",
    )

    fig.update_traces(
        line=dict(width=4),
        marker=dict(size=8),
    )

    fig.update_layout(
        template="plotly_white",
        title=dict(
            text="Monthly Sales Trend",
            x=0.5,
            xanchor="center",
            font=dict(size=18),
        ),
        xaxis_title="Month",
        yaxis_title="Sales",
        hovermode="x unified",
        margin=dict(
            l=15,
            r=15,
            t=45,
            b=15,
        ),
        height=300,
    )

    fig.update_xaxes(
        tickangle=-35,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
        key="monthly_chart",
    )
    # ==========================================================
# Region-wise Sales
# ==========================================================

def region_sales_chart(filtered_df):
    """
    Display sales by region.
    """

    region_sales = (
        filtered_df.groupby("Region")["Sales"]
        .sum()
        .reset_index()
        .sort_values("Sales", ascending=False)
    )

    fig = px.bar(
        region_sales,
        x="Region",
        y="Sales",
        color="Region",
        text_auto=".2s",
        title="Region-wise Sales",
    )

    fig.update_traces(
        textposition="inside",
    )

    fig.update_layout(
        template="plotly_white",
        title=dict(
            text="Region-wise Sales",
            x=0.5,
            xanchor="center",
            font=dict(size=18),
        ),
        showlegend=False,
        xaxis_title="Region",
        yaxis_title="Sales",
        margin=dict(
            l=15,
            r=15,
            t=45,
            b=15,
        ),
        height=300,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
        key="region_chart",
    )


# ==========================================================
# Category-wise Sales
# ==========================================================

def category_sales_chart(filtered_df):
    """
    Display category contribution in total sales.
    """

    category_sales = (
        filtered_df.groupby("Category")["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.pie(
        category_sales,
        names="Category",
        values="Sales",
        hole=0.55,
        title="Category-wise Sales",
    )

    fig.update_traces(
        textinfo="percent+label",
        textfont_size=12,
    )

    fig.update_layout(
        template="plotly_white",
        title=dict(
            text="Category-wise Sales",
            x=0.5,
            xanchor="center",
            font=dict(size=18),
        ),
        height=340,
        margin=dict(
            l=15,
            r=15,
            t=45,
            b=15,
        ),
        legend=dict(
            orientation="h",
            y=-0.15,
            x=0.5,
            xanchor="center",
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
        key="category_chart",
    )
    # ==========================================================
# Year-wise Sales
# ==========================================================

def yearly_sales_chart(filtered_df):
    """
    Display yearly sales comparison.
    """

    year_sales = (
        filtered_df.groupby("Year")["Sales"]
        .sum()
        .reset_index()
        .sort_values("Year")
    )

    fig = px.bar(
        year_sales,
        x="Year",
        y="Sales",
        color="Year",
        text_auto=".2s",
        title="Year-wise Sales",
    )

    fig.update_traces(
        textposition="inside",
    )

    fig.update_layout(
        template="plotly_white",
        title=dict(
            text="Year-wise Sales",
            x=0.5,
            xanchor="center",
            font=dict(size=18),
        ),
        showlegend=False,
        xaxis_title="Year",
        yaxis_title="Sales",
        margin=dict(
            l=15,
            r=15,
            t=45,
            b=15,
        ),
        height=300,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
        key="year_chart",
    )


# ==========================================================
# Top 10 Products
# ==========================================================

def top_products_chart(filtered_df):
    """
    Display top 10 products by sales.
    """

    top_products = (
        filtered_df.groupby("Product Name")["Sales"]
        .sum()
        .reset_index()
        .sort_values("Sales", ascending=False)
        .head(10)
    )

    fig = px.bar(
        top_products,
        x="Sales",
        y="Product Name",
        orientation="h",
        color="Sales",
        text_auto=".2s",
        title="Top 10 Products by Sales",
    )

    fig.update_traces(
        textposition="inside",
    )

    fig.update_layout(
        template="plotly_white",
        title=dict(
            text="Top 10 Products by Sales",
            x=0.5,
            xanchor="center",
            font=dict(size=18),
        ),
        margin=dict(
            l=20,
            r=20,
            t=50,
            b=20,
        ),
        height=340,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
        key="top_products_chart",
    )


# ==========================================================
# Feature Importance
# ==========================================================

def feature_importance_chart(feature_df):
    """
    Display top 10 important features of the trained model.
    """

    top_features = feature_df.head(10).copy()

    top_features["Feature"] = (
        top_features["Feature"]
        .str.replace("categorical__", "", regex=False)
        .str.replace("remainder__", "", regex=False)
        .str.replace("_", " ", regex=False)
    )

    fig = px.bar(
        top_features,
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance",
        title="Top 10 Important Features",
    )

    fig.update_yaxes(
        categoryorder="total ascending",
    )

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
    )

    fig.update_layout(
        template="plotly_white",
        title=dict(
            text="Top 10 Important Features",
            x=0.5,
            xanchor="center",
            font=dict(size=18),
        ),
        xaxis_title="Importance Score",
        yaxis_title="Features",
        margin=dict(
            l=15,
            r=15,
            t=45,
            b=15,
        ),
        height=430,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
        key="feature_importance_chart",
    )