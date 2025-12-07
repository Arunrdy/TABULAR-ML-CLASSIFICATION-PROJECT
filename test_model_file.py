import joblib

MODEL_PATH = "models/xgb_5_features.pkl"
SCALER_PATH = "models/scaler_5_features.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("Model loaded:", type(model))
print("Scaler loaded:", type(scaler))
