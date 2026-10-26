
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

# 1. Load and prepare the dataset
df = pd.read_csv("data/Dataset_cleaned.csv")
df = df.dropna(subset=["SepsisLabel", "Patient_ID"]).copy()
df["SepsisLabel"] = df["SepsisLabel"].astype(int)
df = df[df["SepsisLabel"].isin([0, 1])].copy()
df = df.dropna(axis=1, how="all")

y = df["SepsisLabel"]
groups = df["Patient_ID"]

X = df.drop(
    columns=["SepsisLabel", "Patient_ID", "Unnamed: 0"],
    errors="ignore"
)
X = X.select_dtypes(include="number")

# 2. Recreate the original 80/20 patient-wise split
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

trainval_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

print("Test patients:", groups.iloc[test_idx].nunique())
print("Test records:", len(y_test))
print("Positive test records:", int(y_test.sum()))

# 3. Load the trained model and selected threshold
model = xgb.XGBClassifier()
model.load_model("models/sepsis_xgb_tuned.json")

with open("models/best_threshold.txt", "r") as f:
    best_threshold = float(f.read().strip())

# 4. Predict probabilities on the untouched test set
y_prob = model.predict_proba(X_test)[:, 1]

print("\nTest ROC-AUC:",
      round(roc_auc_score(y_test, y_prob), 4))

print("Test PR-AUC:",
      round(average_precision_score(y_test, y_prob), 4))

# 5. Compare default and tuned thresholds
for threshold in [0.50, best_threshold]:
    y_pred = (y_prob >= threshold).astype(int)

    print(f"\n{'=' * 15} Threshold: {threshold:.2f} {'=' * 15}")
    print("Accuracy:",
          round(accuracy_score(y_test, y_pred), 4))
    print("Precision:",
          round(precision_score(y_test, y_pred, zero_division=0), 4))
    print("Recall:",
          round(recall_score(y_test, y_pred, zero_division=0), 4))
    print("F1-score:",
          round(f1_score(y_test, y_pred, zero_division=0), 4))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred, zero_division=0
    ))

print("\nEvaluation complete.")
