# src/improve_and_shap.py
import numpy as np, pandas as pd, joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import shap
import matplotlib.pyplot as plt

# Load arrays and feature names
X_train = np.load('data/X_train.npy')
X_test = np.load('data/X_test.npy')
y_train = np.load('data/y_train.npy')
y_test = np.load('data/y_test.npy')
feat_names = pd.read_csv('data/feature_names.csv', header=None).iloc[:,0].tolist()

# Convert to DataFrame to add engineered features
Xtr = pd.DataFrame(X_train, columns=feat_names)
Xte = pd.DataFrame(X_test, columns=feat_names)

# Add three simple engineered features
if len(feat_names) >= 3:
    Xtr['sum_3'] = Xtr[feat_names[:3]].sum(axis=1)
    Xtr['mean_3'] = Xtr[feat_names[:3]].mean(axis=1)
    Xte['sum_3'] = Xte[feat_names[:3]].sum(axis=1)
    Xte['mean_3'] = Xte[feat_names[:3]].mean(axis=1)
if len(feat_names) >= 2:
    Xtr['prod_2'] = Xtr[feat_names[0]] * Xtr[feat_names[1]]
    Xte['prod_2'] = Xte[feat_names[0]] * Xte[feat_names[1]]

# Train improved model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(Xtr.values, y_train)

# Evaluate
y_pred = model.predict(Xte.values)
print("Improved model accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump((model, Xtr.columns.tolist()), 'models/xgb_improved.pkl')

# SHAP explanation
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(Xte)
# Save SHAP summary plot
plt.figure()
shap.summary_plot(shap_values, Xte, show=False)
plt.savefig('figures/shap_summary.png', bbox_inches='tight')
plt.close()
