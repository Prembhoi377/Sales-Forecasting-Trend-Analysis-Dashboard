"""
===========================================================
                SALES FORECASTING MODEL
===========================================================

Project:
Sales Forecasting & Trend Analysis Dashboard

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
9. Save Trained Model & Metrics
===========================================================
"""

# ==========================================================
# Import Required Libraries
# ==========================================================

import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "cleaned" / "cleaned_sales.csv"
MODEL_PATH = BASE_DIR / "models" / "sales_model.pkl"
METRICS_PATH = BASE_DIR / "models" / "model_metrics.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_importance.csv"

# ==========================================================
# Create Models Folder (If Not Exists)
# ==========================================================

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)_PATH = BASE_DIR / "models" / "feature_importance.csv"

# ==========================================================
# Load Cleaned Dataset
# ==========================================================

df = pd.read_csv(DATA_PATH)

print("✅ Dataset Loaded Successfully")
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

categorical_columns = X.select_dtypes(
    include=["object", "string"]
).columns

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
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )
    )
])

# ==========================================================
# Train Model
# ==========================================================

print("\n🚀 Training Model...\n")

pipeline.fit(X_train, y_train)

# ==========================================================
# Make Predictions
# ==========================================================

predictions = pipeline.predict(X_test)

# ==========================================================
# Evaluate Model
# ==========================================================

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

print("\n========== Model Performance ==========")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

# ==========================================================
# Feature Importance
# ==========================================================

feature_names = pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()

importances = pipeline.named_steps[
    "model"
].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== Top 10 Important Features ==========")
print(importance_df.head(10))

# ==========================================================
# Save Feature Importance
# ==========================================================

importance_df.to_csv(FEATURE_PATH, index=False)

# ==========================================================
# Save Performance Metrics
# ==========================================================

metrics = {
    "R2 Score": r2,
    "MAE": mae,
    "MSE": mse,
    "RMSE": rmse
}

joblib.dump(metrics, METRICS_PATH)

# ==========================================================
# Save Trained Model
# ==========================================================

joblib.dump(pipeline, MODEL_PATH)

print("\n✅ Model saved as sales_model.pkl")
print("✅ Metrics saved as model_metrics.pkl")
print("✅ Feature Importance saved as feature_importance.csv")