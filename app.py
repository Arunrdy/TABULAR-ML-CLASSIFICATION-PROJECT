from flask import Flask, render_template, request
import numpy as np
import joblib
import csv
import os
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load trained 5-feature model
MODEL_PATH = "models/xgb_5_features.pkl"
SCALER_PATH = "models/scaler_5_features.pkl"  # Make sure you save the scaler during training
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Please run train.py first to generate model and scaler.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Path to dataset
DATA_FILE = "data/dataset.csv"

@app.route("/")
def home():
    return render_template("form.html", result=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read form values
        age = float(request.form["age"])
        bp = float(request.form["bp"])
        cholesterol = float(request.form["cholesterol"])
        glucose = float(request.form["glucose"])
        heart_rate = float(request.form["heart_rate"])

        # Convert input to array
        input_data = np.array([[age, bp, cholesterol, glucose, heart_rate]])

        # Scale input using the same scaler as training
        input_scaled = scaler.transform(input_data)

        # Make prediction
        pred = model.predict(input_scaled)[0]
        result = "SICK" if pred == 1 else "HEALTHY"

        # Append new data to dataset.csv
        if not os.path.exists(DATA_FILE):
            # If dataset doesn't exist, create with header
            with open(DATA_FILE, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["age", "bp", "cholesterol", "glucose", "heart_rate", "target"])

        with open(DATA_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([age, bp, cholesterol, glucose, heart_rate, pred])

        return render_template("form.html", result=result)

    except Exception as e:
        return render_template("form.html", result=f"Error: {e}")

if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), debug=False)

