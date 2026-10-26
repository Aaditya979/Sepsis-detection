
import pandas as pd
from pathlib import Path

# Load cleaned dataset
df = pd.read_csv(Path("data") / "Dataset_cleaned.csv")

# Sort patient records chronologically
df = df.sort_values(["Patient_ID", "Hour"]).copy()

# Identify patients with at least one positive sepsis label
positive = df[df["SepsisLabel"] == 1]

first_positive = (
    positive.groupby("Patient_ID")["Hour"]
    .min()
    .rename("FirstSepsisHour")
)

# Attach each patient's first positive hour
df = df.join(first_positive, on="Patient_ID")

print("Total patients:", df["Patient_ID"].nunique())
print("Patients with a positive sepsis label:", len(first_positive))

print("\nFirst positive-label hour statistics:")
print(first_positive.describe())

print("\nExample patient timelines:")
for patient_id in first_positive.index[:5]:
    patient = df[df["Patient_ID"] == patient_id]
    print(f"\nPatient ID: {patient_id}")
    print(patient[[
        "Hour", "HR", "O2Sat", "Temp", "SBP", "MAP",
        "Resp", "SepsisLabel", "FirstSepsisHour"
    ]].head(10).to_string(index=False))

print("\nAnalysis complete.")
