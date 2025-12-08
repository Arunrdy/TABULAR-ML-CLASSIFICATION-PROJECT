# app.py (replace entire file in GitHub with this)
from flask import Flask, render_template, request
import numpy as np
import joblib
import csv
import os
import sys
import traceback

app = Flask(__name__)

# Paths (adjust if your repo uses different filenames)
MODEL_PATH = "models/xgb_5_features.pkl"
SCALER_PATH = "models/scaler_5_features.pkl"
DATA_FILE = "data/dataset.csv"

# Ensure folders exist
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

def safe_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

# Load model + scaler with defensive checks & debug info
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    safe_print("DEBUG: One or both model files are missing.")
    safe_print("DEBUG: models folder content:", os.listdir("models") if os.path.exists("models") else "models folder not found")
    # We will NOT raise here: keep server alive to show debug on Render
else:
    safe_print("DEBUG: Found model files. Listing models folder:", os.listdir("models"))

# Try to load model and scaler; catch problems and log
model = None
scaler = None
try:
    if os.path.exists(MODEL_PATH):
        raw = joblib.load(MODEL_PATH)
        safe_print("DEBUG: raw model object type:", type(raw))
        # If someone saved [model] or [model, scaler], handle that automatically:
        if isinstance(raw, list):
            # try to find first object that looks like an estimator
            if len(raw) == 0:
                raise ValueError("Loaded list is empty.")
            # naive: assume first item is model
            model = raw[0]
            safe_print("DEBUG: Raw model was a list. Using element [0] as model:", type(model))
        else:
            model = raw
            safe_print("DEBUG: Model loaded as:", type(model))
    else:
        safe_print(f"DEBUG: Model file not found at {MODEL_PATH}")

    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        safe_print("DEBUG: Scaler loaded as:", type(scaler))
    else:
        safe_print(f"DEBUG: Scaler file not found at {SCALER_PATH}")

except Exception as e:
    safe_print("DEBUG: Exception while loading model/scaler:")
    safe_print(traceback.format_exc())

@app.route("/")
def home():
    # If you have a template (form.html or index.html), Flask will render it.
    # If not, this returns a simple HTML form fallback.
    tpl = "form.html" if os.path.exists(os.path.join("templates", "form.html")) else None
    if tpl:
        return render_template(tpl, result=None)
    else:
        return """
        <html><body>
        <h2>Model Server</h2>
        <form action="/predict" method="post">
          Age: <input name="age" value="50"><br>
          BP: <input name="bp" value="120"><br>
          Chol: <input name="cholesterol" value="200"><br>
          Glucose: <input name="glucose" value="100"><br>
          Heart rate: <input name="heart_rate" value="80"><br>
          <button type="submit">Predict</button>
        </form>
        <p>Check logs for debug info.</p>
        </body></html>
        """

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # check model is present
        if model is None:
            return "ERROR: model not loaded. Check logs.", 500
        if scaler is None:
            return "ERROR: scaler not loaded. Check logs.", 500

        age = float(request.form.get("age", 0))
        bp = float(request.form.get("bp", 0))
        cholesterol = float(request.form.get("cholesterol", 0))
        glucose = float(request.form.get("glucose", 0))
        heart_rate = float(request.form.get("heart_rate", 0))

        features = np.array([[age, bp, cholesterol, glucose, heart_rate]])
        features_scaled = scaler.transform(features)
        pred = model.predict(features_scaled)[0]
        result = "SICK" if int(pred) == 1 else "HEALTHY"

        # append to CSV dataset (create header if missing)
        file_exists = os.path.exists(DATA_FILE)
        with open(DATA_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["age", "bp", "cholesterol", "glucose", "heart_rate", "target"])
            writer.writerow([age, bp, cholesterol, glucose, heart_rate, int(pred)])

        return render_template("form.html", result=result) if os.path.exists(os.path.join("templates", "form.html")) else result

    except Exception as e:
        safe_print("DEBUG: Exception in /predict:")
        safe_print(traceback.format_exc())
        return f"Error: {str(e)}", 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    safe_print("Starting app on port", port)
    app.run(host="0.0.0.0", port=port, debug=False)
