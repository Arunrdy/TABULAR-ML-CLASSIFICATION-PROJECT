import pandas as pd

df = pd.read_csv('data/dataset.csv')
print("Rows, Columns:", df.shape)
print("Columns:", df.columns.tolist())
print("First 5 rows:")
print(df.head())

if 'target' in df.columns:
    print("Target distribution:")
    print(df['target'].value_counts())
else:
    print("No column named 'target' - rename your label column to 'target' in the CSV or tell me the name.")
