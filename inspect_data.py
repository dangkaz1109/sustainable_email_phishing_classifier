import pandas as pd
import json

# Load data
df = pd.read_json("data/ephishLLM.json")

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

print("\nValue counts of 'type':")
print(df['type'].value_counts())

if 'Language' in df.columns:
    print("\nValue counts of 'Language':")
    print(df['Language'].value_counts())
else:
    print("\nNo Language column found.")

# Let's see if there are any null values
print("\nNull values count:")
print(df.isnull().sum())
