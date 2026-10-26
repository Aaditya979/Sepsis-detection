
from pathlib import Path
import pandas as pd

# 1. Load the dataset
file_path = Path("data") / "Dataset.csv"
df = pd.read_csv(file_path)

print("Original dataset shape:", df.shape)

# 2. Remove the extra index column
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# 3. Remove rows without a target label or patient ID
df = df.dropna(subset=["SepsisLabel", "Patient_ID"]).copy()

# 4. Convert target and patient ID to integer
df["SepsisLabel"] = df["SepsisLabel"].astype(int)
df["Patient_ID"] = df["Patient_ID"].astype(int)

# 5. Check the cleaned dataset
print("\nShape after basic cleaning:", df.shape)

print("\nSepsis label counts:")
print(df["SepsisLabel"].value_counts())

print("\nSepsis label percentages:")
print((df["SepsisLabel"].value_counts(normalize=True) * 100).round(2))

print("\nNumber of unique patients:", df["Patient_ID"].nunique())

# 6. Check the most-missing columns
missing_percent = (df.isnull().mean() * 100).sort_values(
    ascending=False
)

print("\nTop 15 columns by missing percentage:")
print(missing_percent.head(15).round(2))

# 7. Save the basic cleaned dataset
output_path = Path("data") / "Dataset_cleaned.csv"
df.to_csv(output_path, index=False)

print(f"\nCleaned dataset saved to: {output_path}")
