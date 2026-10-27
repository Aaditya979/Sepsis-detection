import pandas as pd

path = r"D:\STUDY\AI_ML_SOFTWARE project\XG_boost\data\Dataset_cleaned.csv"

df = pd.read_csv(path)

print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

if "SepsisLabel" in df.columns:
    print("\nTarget distribution:")
    print(df["SepsisLabel"].value_counts())
    print("\nTarget proportions:")
    print(df["SepsisLabel"].value_counts(normalize=True))