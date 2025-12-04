# src/evaluate.py
import numpy as np, pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load baseline model
model1 = joblib.load('models/xgb_baseline.pkl')

# Load improved model and columns
m2, cols = joblib.load('models/xgb_improved.pkl')

# Load test arrays
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')

# Baseline predictions (works directly with X_test)
y1 = model1.predict(X_test)
print("Baseline accuracy:", accuracy_score(y_test, y1))
print("Baseline report:\n", classification_report(y_test, y1))

# Recreate DataFrame for improved model (add engineered features)
feat_names = pd.read_csv('data/feature_names.csv', header=None).iloc[:,0].tolist()
Xte = pd.DataFrame(X_test, columns=feat_names)
if len(feat_names) >= 3:
    Xte['sum_3'] = Xte[feat_names[:3]].sum(axis=1)
    Xte['mean_3'] = Xte[feat_names[:3]].mean(axis=1)
if len(feat_names) >= 2:
    Xte['prod_2'] = Xte[feat_names[0]] * Xte[feat_names[1]]

y2 = m2.predict(Xte.values)
print("Improved accuracy:", accuracy_score(y_test, y2))
print("Improved report:\n", classification_report(y_test, y2))
