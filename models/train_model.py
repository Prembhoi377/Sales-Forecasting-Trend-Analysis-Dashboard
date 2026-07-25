"""
===========================================================
                SALES FORECASTING MODEL
===========================================================

Project:
Sales Forecasting & Trend Analysis Dashboard

Author:
Deepthi

Description:
This script trains a Random Forest Regression model
to predict Sales using the cleaned Superstore dataset.

Workflow:
1. Import Libraries
2. Load Dataset
3. Feature Selection
4. Data Preprocessing
5. Train-Test Split
6. Train Random Forest Model
7. Evaluate Model
8. Display Feature Importance
9. Save Trained Model
===========================================================
"""

# ==========================================================
# Import Required Libraries
# ==========================================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================================
# Load Cleaned Dataset
# ==========================================================

df = pd.read_csv("../data/cleaned/cleaned_sales.csv")

print("Dataset Loaded Successfully")
print(df.head())

# ==========================================================
# Define Target Variable
# ==========================================================

y = df["Sales"]

# ==========================================================
# Select Features
# ==========================================================

X = df.drop(columns=[
    "Sales",
    "Row ID",
    "Order ID",
    "Customer ID",
    "Customer Name",
    "Product ID",
    "Product Name",
    "Order Date",
    "Ship Date"
])

# ==========================================================
# Identify Categorical Columns
# ==========================================================

categorical_columns = X.select_dtypes(include=["object", "string"]).columns

# ==========================================================
# Create Preprocessor
# ==========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)

# ==========================================================
# Train-Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================================
# Build Machine Learning Pipeline
# ==========================================================

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
])

# ==========================================================
# Train Model
# ==========================================================

print("\nTraining Model...\n")

pipeline.fit(X_train, y_train)

# ==========================================================
# Make Predictions
# ==========================================================

predictions = pipeline.predict(X_test)

# ==========================================================
# Evaluate Model
# ==========================================================

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("----------------------------")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

# ==========================================================
# Feature Importance
# ==========================================================

feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

importances = pipeline.named_steps["model"].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features")
print("----------------------------")
print(importance_df.head(10))

# ==========================================================
# Save Model
# ==========================================================

joblib.dump(pipeline, "sales_model.pkl")

print("\nModel saved successfully as sales_model.pkl")