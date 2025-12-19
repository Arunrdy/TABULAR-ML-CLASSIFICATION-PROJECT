# 🚀 TABULAR-ML-CLASSIFICATION-PROJECT

A simple, clear, and beginner-friendly machine-learning classification system for tabular datasets.  
Upload CSV/Excel or pre-saved NumPy arrays, preprocess, train ML models, evaluate performance, and make predictions through a minimal Flask web UI.

---

## 📌 Quick Highlights
- Minimal Flask web interface (upload → train → predict)  
- Supports CSV/Excel and NumPy `.npy` files  
- Preprocessing + training pipeline (scikit-learn / XGBoost used in this repo)  
- Trained models and scalers saved to `models/`  
- Figures saved to `figures/` for inspection

---

## 🔗 Repository
👉 **GitHub:** https://github.com/Arunrdy/TABULAR-ML-CLASSIFICATION-PROJECT

## ⚙️ Install & Run (short)
```bash
git clone https://github.com/Arunrdy/TABULAR-ML-CLASSIFICATION-PROJECT.git
cd TABULAR-ML-CLASSIFICATION-PROJECT

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
python app.py
```

Open in browser:  
👉 https://tabular-ml-classification-project-ev6n.onrender.com


## 🎯 How to use (summary)
1. Open the web UI.  
2. Upload CSV/Excel **or** use `.npy` files already present.  
3. Select the target column (if CSV) or choose `y.npy`.  
4. Train and view accuracy / evaluation metrics.  
5. Inspect saved figures and models as needed.

---

## 📂 Exact Project Structure — **ANALYZED FROM YOUR LATEST SCREENSHOT**  
I inspected the screenshot you provided and updated the structure to match exactly what’s visible. **No invented filenames** beyond what is shown. If a filename below looks like a typo (see note), you can rename it in your repo.

```
TABULAR-ML-CLASSIFICATION-PROJECT/
│
├── data/
│   ├── dataset.csv          # (visible)
│   ├── X_test.npy           # (visible)
│   ├── X_train.npy          # (visible)
│   ├── X.npy                # (visible)
│   ├── y_test.npy           # (visible)
│   ├── y_train.npy          # (visible)
│   └── y.npy                # (visible)
│
├── figures/                 # (visible)
│   ├── confusion_baseline.png   # (visible)
│   ├── featimp_baseline.png     # (visible) — filename in screenshot uses "featimp"
│   └── shap_summary.png         # (visible)
│
├── models/                  # (visible)
│   ├── scaler_5_features.pkl # (visible)
│   ├── xgb_5_features.pkl    # (visible)
│   ├── xgb_baseline.pkl      # (visible)
│   └── xgb_improved.pkl      # (visible)
│
├── src/                     # (visible)
│   ├── evaluate.py               # (visible)
│   ├── improve_and_shap.py       # (visible)
│   ├── inspect_data.py           # (visible)
│   ├── preprocess.py             # (visible; screenshot shows preprocessing script) 
│   └── train.py                  # (visible)
│
├── static/
│   └── style.css             # (folder visible; file not shown in screenshot but expected)
│
├── templates/
│   └── index.html            # (folder visible; file not shown in screenshot but expected)
│
├── README.md
├── app.py
├── requirements.txt
└── test_model_file.py
```

**Notes & small clarifications from the screenshot analysis**
- The `data/` folder (in the screenshot) contains `dataset.csv` and the `.npy` arrays — I moved those into `data/` in this listing to match the visual layout you showed.
- The figures list exactly matches the three image files visible: `confusion_baseline.png`, `featimp_baseline.png` (screenshot shows `featimp` — I preserved that exact spelling), and `shap_summary.png`.
- The `models/` folder contains four files visible in the screenshot; I listed all four exactly.
- The `src/` folder lists five Python files visible in the screenshot: `evaluate.py`, `improve_and_shap.py`, `inspect_data.py`, `preprocess.py`, and `train.py`.  
  - If your local filename differs (e.g., you suspect `preprocess.py` was spelled with an extra letter in the IDE), I preserved the common/correct spelling `preprocess.py` here and recommend you confirm the actual repo filename and update if needed.

---

## ⚠️ Limitations
- Interface and preprocessing are minimal by design.  
- Intended for small-to-medium datasets and prototyping.

---

## 🔮 Future Enhancements
- Add extra model types (RandomForest, LightGBM) and CV/hyperparameter tuning.  
- Add API endpoints for prediction (e.g., `POST /predict`).  
- Export evaluation reports (CSV) and extended EDA visuals.

---

⭐ If you find this project helpful, consider starring the repository.
