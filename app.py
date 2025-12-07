from flask import Flask, render_template, request
import numpy as np
import joblib
import csv
import os

app = Flask(__name__)

# Paths
MODEL_PATH = "models/xgb_5_features.pkl"
SCALER_PATH = "models/scaler_5_features.pkl"
DATA_FILE = "data/dataset.csv"

# Ensure data dir exists
os.makedirs("data", exist_ok=True)

# Load model and scaler
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Run train.py to generate model.")
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Run train.py to generate scaler.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


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

            # Convert input to numpy array
            input_data = np.array([[age, bp, cholesterol, glucose, heart_rate]])

            # Scale using fitted scaler
            input_scaled = scaler.transform(input_data)

            # Predict
            pred = int(model.predict(input_scaled)[0])
            result = "SICK" if pred == 1 else "HEALTHY"

            # Save to dataset
            new_row = [age, bp, cholesterol, glucose, heart_rate, pred]

            file_exists = os.path.exists(DATA_FILE)

            with open(DATA_FILE, "a", newline="") as f:
                writer = csv.writer(f)

                if not file_exists:
                    writer.writerow(["age", "bp", "cholesterol", "glucose", "heart_rate", "target"])

                writer.writerow(new_row)

            return render_template("form.html", result=result)

        except Exception as e:
            return render_template("form.html", result=f"Error: {str(e)}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
