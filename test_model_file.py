import joblib

loaded = joblib.load("models/xgb_improved.pkl")

print("Type:", type(loaded))

if isinstance(loaded, tuple):
    print("Tuple length:", len(loaded))
    print("Contents:")
    for i, item in enumerate(loaded):
        print(f"Index {i}: {type(item)}")
else:
    print("Object is not a tuple. It is:", type(loaded))
