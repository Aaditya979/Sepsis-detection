
from pathlib import Path
import pandas as pd

# Dataset folder
data_folder = Path("data")

# Find CSV files
csv_files = list(data_folder.glob("*.csv"))

if not csv_files:
    print("No CSV files found in the data folder.")
    print("Please check that you extracted your dataset into data/.")
else:
    print("CSV files found:")
    for file in csv_files:
        print("-", file.name)

    # Load the first CSV file
    df = pd.read_csv(csv_files[0])

    print("\nDataset shape (rows, columns):")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print("\nData types:")
    print(df.dtypes)

    if "SepsisLabel" in df.columns:
        print("\nSepsis label distribution:")
        print(df["SepsisLabel"].value_counts(dropna=False))
