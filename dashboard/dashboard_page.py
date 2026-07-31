"""
=========================================================
                Dashboard Page
=========================================================

Project:
Sales Forecasting & Trend Analysis Dashboard

Description:
Main dashboard with filters, KPI cards and sales analysis.

=========================================================
"""

import streamlit as st

from charts import (
    monthly_sales_chart,
    region_sales_chart,
    category_sales_chart,
    yearly_sales_chart,
    top_products_chart,
)

# ==========================================================
# Global CSS
# ==========================================================

st.markdown("""
<style>

/* ---------------- Page ---------------- */

.block-container{
    padding-top:2rem !important;
    padding-bottom:1rem !important;
    padding-left:2rem !important;
    padding-right:2rem !important;
}

.stTabs{
    margin-top:0.5rem !important;
}

/* ---------------- Filters ---------------- */

div[data-baseweb="select"] > div{
    min-height:42px !important;
    padding-top:2px !important;
    padding-bottom:2px !important;
    border:1px solid #4B5563 !important;
    border-radius:10px !important;
}

div[data-baseweb="select"] > div:hover{
    border-color:#60A5FA !important;
}

div[data-baseweb="select"] > div:focus-within{
    border-color:#3B82F6 !important;
    box-shadow:0 0 0 1px #3B82F6 !important;
}

div[data-baseweb="select"]{
    font-size:13px !important;
}

div[data-baseweb="tag"]{
    font-size:11px !important;
    padding:2px 5px !important;
    margin:1px !important;
}

label[data-testid="stWidgetLabel"] p{
    font-size:14px !important;
}

/* ---------------- KPI Cards ---------------- */

.kpi-card{
    background:#0e1117;
    border:1px solid #30363d;
    border-radius:8px;
    padding:12px 16px;
    height:75px;
    display:flex;
    align-items:center;
    transition:.2s ease;
}

.kpi-card:hover{
    border-color:#4f8bf9;
}

.kpi-title{
    display:flex;
    align-items:center;
    gap:8px;
    color:#c9d1d9;
    font-size:18px;
    font-weight:500;
}

.kpi-value{
    margin-left:18px;
    color:white;
    font-size:24px;
    font-weight:700;
    white-space:nowrap;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# Currency Formatter
# ==========================================================

def format_currency(value):

    if value >= 1_000_000:
        return f"₹ {value/1_000_000:.2f} M"

    elif value >= 1_000:
        return f"₹ {value/1_000:.2f} K"

    return f"₹ {value:,.2f}"


# ==========================================================
# Dashboard
# ==========================================================

def dashboard_page(df):

    # ------------------------------------------------------
    # Header
    # ------------------------------------------------------

    st.title("📊 Sales Forecasting & Trend Analysis Dashboard")

    st.caption(
        "Interactive dashboard powered by Machine Learning, "
        "Streamlit and Plotly for sales analysis, forecasting "
        "and business insights."
    )

    st.write("")

    # ------------------------------------------------------
    # Filters
    # ------------------------------------------------------

    with st.expander("🔍 Filters", expanded=False):

        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(
            [1.2, 1.2, 1.2, 0.6],
            gap="small"
        )

        with filter_col1:

            selected_year = st.multiselect(
                "Year",
                options=sorted(df["Year"].unique()),
                default=sorted(df["Year"].unique()),
            )

        with filter_col2:

            selected_region = st.multiselect(
                "Region",
                options=sorted(df["Region"].unique()),
                default=sorted(df["Region"].unique()),
            )

        with filter_col3:

            selected_category = st.multiselect(
                "Category",
                options=sorted(df["Category"].unique()),
                default=sorted(df["Category"].unique()),
            )

        with filter_col4:

            st.write("")
            st.write("")

            if st.button(
                "🔄 Reset",
                use_container_width=True
            ):
                st.rerun()

    # ------------------------------------------------------
    # Filter Dataset
    # ------------------------------------------------------

    filtered_df = df[
        (df["Year"].isin(selected_year))
        &
        (df["Region"].isin(selected_region))
        &
        (df["Category"].isin(selected_category))
    ]

    if filtered_df.empty:
        st.warning("No records found for the selected filters.")
        return

    # ------------------------------------------------------
    # KPIs
    # ------------------------------------------------------

    total_sales = filtered_df["Sales"].sum()
    average_sales = filtered_df["Sales"].mean()
    total_orders = filtered_df["Order ID"].nunique()
    total_customers = filtered_df["Customer ID"].nunique()

    # ------------------------------------------------------
    # Business Overview
    # ------------------------------------------------------

    st.subheader("📌 Business Overview")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">
                💰 Total Sales
                <span class="kpi-value">
                    {format_currency(total_sales)}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">
                📈 Average Sales
                <span class="kpi-value">
                    {format_currency(average_sales)}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">
                📦 Total Orders
                <span class="kpi-value">
                    {total_orders:,}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">
                👥 Total Customers
                <span class="kpi-value">
                    {total_customers:,}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
            # ==========================================================
    # Sales Analysis
    # ==========================================================

    st.subheader("📈 Sales Analysis")

    monthly_col, spacer1, region_col, spacer2, year_col = st.columns(
        [1, 0.08, 1, 0.08, 1],
        gap="small"
    )

    with monthly_col:
        with st.container(border=True):
            monthly_sales_chart(filtered_df)

    with region_col:
        with st.container(border=True):
            region_sales_chart(filtered_df)

    with year_col:
        with st.container(border=True):
            yearly_sales_chart(filtered_df)

    st.write("")

    # ==========================================================
    # Product Analysis
    # ==========================================================
    
    st.markdown("## 📊 Product Analysis")
    
    category_col, product_col = st.columns(
        [1, 2.4],      # Left smaller, Right much larger
        gap="medium"
    )
    
    with category_col:
        with st.container(border=True):
            category_sales_chart(filtered_df)
    
    with product_col:
        with st.container(border=True):
            top_products_chart(filtered_df)
    
    st.write("")
    # ==========================================================
    # Key Business Insights
    # ==========================================================
    
    st.markdown("## 💡 Key Business Insights")
    
    # Metrics
    total_sales = filtered_df["Sales"].sum()
    
    top_region = filtered_df.groupby("Region")["Sales"].sum().idxmax()
    lowest_region = filtered_df.groupby("Region")["Sales"].sum().idxmin()
    
    top_state = filtered_df.groupby("State")["Sales"].sum().idxmax()
    
    top_category = filtered_df.groupby("Category")["Sales"].sum().idxmax()
    
    top_product = (
        filtered_df.groupby("Product Name")["Sales"]
        .sum()
        .idxmax()
    )
    
    best_month = (
        filtered_df.groupby("Month")["Sales"]
        .sum()
        .idxmax()
    )
    
    top_city = (
        filtered_df.groupby("City")["Sales"]
        .sum()
        .idxmax()
    )
    
    # Layout
    left_col, right_col = st.columns(2)
    
    # ----------------------------------------------------------
    # Performance Highlights
    # ----------------------------------------------------------
    with left_col:
        st.info(f"""
    ### 📊 Performance Highlights
    
    💰 **Total Revenue:** ₹{total_sales:,.2f}
    
    🌍 **Top Performing Region:** {top_region}
    
    📍 **Top Performing State:** {top_state}
    
    📦 **Best Selling Category:** {top_category}
    
    🏆 **Top Selling Product:** {top_product}
    
    📅 **Highest Sales Month:** {best_month}
    """)
    
    # ----------------------------------------------------------
    # Business Recommendations
    # ----------------------------------------------------------
    with right_col:
        st.warning(f"""
    ### 🎯 Business Recommendations
    
    ✅ Focus marketing campaigns in **{top_region}** to maximize revenue.
    
    ✅ Increase inventory for **{top_category}** products.
    
    ✅ Maintain sufficient stock of **{top_product}**.
    
    ⚠️ Improve sales performance in **{lowest_region}** through targeted campaigns.
    
    🚀 Launch promotional offers before **{best_month}** to boost seasonal sales.
    
    📍 Prioritize **{top_city}** for business growth.
    """)
    # ==========================================================
    # Dataset Preview
    # ==========================================================

    with st.expander(
        "📋 View Dataset Preview",
        expanded=False
    ):

        st.caption(
            f"Showing first 10 of {len(filtered_df):,} records "
            "after applying filters."
        )

        st.dataframe(
            filtered_df.head(10),
            use_container_width=True,
            hide_index=True,
        )