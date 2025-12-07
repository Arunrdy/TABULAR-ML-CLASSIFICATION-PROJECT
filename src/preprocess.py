import pandas as pd
import numpy as np
import os

# Path to dataset
DATA_FILE = "data/dataset.csv"

# Check if dataset exists
if not os.path.exists(DATA_FILE):
    print(f"Error: {DATA_FILE} not found!")
    exit()

# Load dataset
df = pd.read_csv(DATA_FILE)

# Ensure only the 5 features + target exist
expected_cols = ["age", "bp", "cholesterol", "glucose", "heart_rate", "target"]
df = df[expected_cols]

# Fill missing values with median
for col in expected_cols[:-1]:
    df[col] = df[col].fillna(df[col].median())

# Convert all to numeric
df[expected_cols] = df[expected_cols].apply(pd.to_numeric, errors='coerce')

# Drop rows with remaining NaNs
df = df.dropna()

# Save cleaned dataset
df.to_csv(DATA_FILE, index=False)
print(f"Dataset preprocessed and saved to {DATA_FILE}")

# Save feature and label arrays
X = df[expected_cols[:-1]].values
y = df["target"].values
np.save("data/X.npy", X)
np.save("data/y.npy", y)
print("Feature and label arrays saved: X.npy, y.npy")
