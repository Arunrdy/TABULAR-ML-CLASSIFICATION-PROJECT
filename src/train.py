import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Paths
DATA_FILE = "data/dataset.csv"
MODEL_PATH = "models/xgb_5_features.pkl"
SCALER_PATH = "models/scaler_5_features.pkl"

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_FILE)

# Features and target
feature_cols = ["age", "bp", "cholesterol", "glucose", "heart_rate"]
X = df[feature_cols].values
y = df["target"].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = XGBClassifier(eval_metric="logloss", random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save scaler
joblib.dump(scaler, SCALER_PATH)
print(f"Scaler saved at: {SCALER_PATH}")

# Save model ✅ directly, not inside a list
joblib.dump(model, MODEL_PATH)
print(f"Model saved at: {MODEL_PATH}")
