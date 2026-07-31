# 📊 Sales Forecasting & Trend Analysis Dashboard

An interactive Machine Learning dashboard built using **Streamlit**, **Scikit-learn**, and **Plotly** to analyze historical sales data, identify business trends, and forecast future sales through interactive visualizations and actionable business insights.

---

## 👥 Project Contributors

- **L. Surya Deepthi Sri**
- **Irene Ramala**
- **Prem Jagannath Bhoi**

---

## 📌 Project Overview

This project analyzes historical sales data to identify sales trends, evaluate business performance, and predict future sales using Machine Learning. The dashboard provides interactive visualizations, forecasting capabilities, and business insights to support data-driven decision-making.

---

## ✨ Features

- 📈 Sales Forecasting using Random Forest Regression
- 📊 Interactive KPI Dashboard
- 🌍 Region-wise Sales Analysis
- 📦 Category-wise Performance Analysis
- 🏆 Top Selling Products
- 📅 Monthly & Yearly Sales Trends
- 💡 Key Business Insights & Recommendations
- 📉 Feature Importance Visualization
- 🎛️ Interactive Filters
- 📋 Dataset Preview
- 📈 Model Performance Metrics

---

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Dashboard:** Streamlit
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Visualization:** Plotly
- **Model Serialization:** Joblib

---

## 📂 Project Structure

```text
Sales-Forecasting-Trend-Analysis-Dashboard/
│
├── dashboard/
│   ├── prediction/
│   ├── app.py
│   ├── charts.py
│   ├── dashboard_page.py
│   ├── performance_page.py
│   └── utils.py
│
├── data/
│   ├── raw/
│   └── cleaned/
│
├── images/
│
├── models/
│   ├── train_model.py
│   ├── feature_importance.csv
│   └── model_metrics.pkl
│
├── notebooks/
├── presentation/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📷 Dashboard Preview

### 🏠 Dashboard

![Dashboard](images/dashboard_home.png)

### 📈 Sales Forecasting

![Forecast](images/forecasting_page.png)

### 💡 Business Insights

![Business Insights](images/business_insights.png)

### 📊 Model Performance

![Model Performance](images/model_performance.png)

---

# 🚀 Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/Prembhoi377/Sales-Forecasting-Trend-Analysis-Dashboard.git
```

### 2. Navigate to the project directory

```bash
cd Sales-Forecasting-Trend-Analysis-Dashboard
```

### 3. Install the required dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate the trained Machine Learning model

```bash
python models/train_model.py
```

This command automatically generates:

- `models/sales_model.pkl`
- `models/model_metrics.pkl`
- `models/feature_importance.csv`

### 5. Run the Streamlit dashboard

```bash
streamlit run dashboard/app.py
```

---

## 🤖 Machine Learning Model

- **Algorithm:** Random Forest Regressor
- **Objective:** Sales Forecasting
- **Input:** Historical Sales Data
- **Output:** Future Sales Prediction

> **Note:** The trained model (`sales_model.pkl`) is not included in this repository because it exceeds GitHub's file size limit. Run `python models/train_model.py` before launching the dashboard to generate the trained model automatically.

---

## 📈 Model Evaluation Metrics

The model is evaluated using the following regression metrics:

- **R² Score**
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**

---

## 💼 Business Value

This dashboard enables businesses to:

- Monitor overall sales performance
- Identify high-performing regions and product categories
- Discover top-selling products
- Analyze monthly and yearly sales trends
- Forecast future sales
- Support data-driven business decisions

---

## 🔮 Future Enhancements

- AI-powered Sales Recommendations
- Real-time Sales Data Integration
- Export Reports (PDF/Excel)
- Cloud Deployment
- User Authentication

---

## 📜 License

This project is developed for educational and academic purposes.
```
