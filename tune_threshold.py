
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

# 1. Load the dataset
df = pd.read_csv("data/Dataset_cleaned.csv")
df = df.dropna(subset=["SepsisLabel", "Patient_ID"]).copy()
df["SepsisLabel"] = df["SepsisLabel"].astype(int)
df = df[df["SepsisLabel"].isin([0, 1])].copy()
df = df.dropna(axis=1, how="all")

# 2. Separate features, target and patient groups
y = df["SepsisLabel"]
groups = df["Patient_ID"]

X = df.drop(
    columns=["SepsisLabel", "Patient_ID", "Unnamed: 0"],
    errors="ignore"
)
X = X.select_dtypes(include="number")

# 3. Split patients into train+validation and test
split1 = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=42
)
trainval_idx, test_idx = next(
    split1.split(X, y, groups=groups)
)

# Reserve the test set for final evaluation
X_trainval = X.iloc[trainval_idx]
y_trainval = y.iloc[trainval_idx]
groups_trainval = groups.iloc[trainval_idx]

# 4. Split train+validation into training and validation
split2 = GroupShuffleSplit(
    n_splits=1, test_size=0.25, random_state=42
)
train_idx, val_idx = next(
    split2.split(
        X_trainval, y_trainval, groups=groups_trainval
    )
)

X_train = X_trainval.iloc[train_idx]
y_train = y_trainval.iloc[train_idx]

X_val = X_trainval.iloc[val_idx]
y_val = y_trainval.iloc[val_idx]

print("Training patients:", groups.iloc[trainval_idx].iloc[train_idx].nunique())
print("Validation patients:", groups_trainval.iloc[val_idx].nunique())
print("Reserved test patients:", groups.iloc[test_idx].nunique())

# 5. Calculate class imbalance weight using training data only
negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())
scale_pos_weight = negative_count / positive_count

# 6. Train XGBoost
model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.03,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 7. Get validation probabilities
val_prob = model.predict_proba(X_val)[:, 1]

print("\nValidation PR-AUC:", round(
    average_precision_score(y_val, val_prob), 4
))

# 8. Compare classification thresholds
results = []

for threshold in np.arange(0.05, 0.56, 0.05):
    val_pred = (val_prob >= threshold).astype(int)

    precision = precision_score(
        y_val, val_pred, zero_division=0
    )
    recall = recall_score(
        y_val, val_pred, zero_division=0
    )
    f1 = f1_score(
        y_val, val_pred, zero_division=0
    )

    results.append({
        "Threshold": round(float(threshold), 2),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-score": round(f1, 4)
    })

results_df = pd.DataFrame(results)
print("\nThreshold comparison:")
print(results_df.to_string(index=False))

# 9. Select threshold with highest validation F1-score
best_row = results_df.loc[results_df["F1-score"].idxmax()]
best_threshold = float(best_row["Threshold"])

print("\nBest threshold by validation F1-score:")
print(best_row.to_string())

# 10. Save model and selected threshold
Path("models").mkdir(exist_ok=True)

model.save_model("models/sepsis_xgb_tuned.json")

with open("models/best_threshold.txt", "w") as f:
    f.write(str(best_threshold))

print("\nModel saved to models/sepsis_xgb_tuned.json")
print("Threshold saved to models/best_threshold.txt")
print("Test set has NOT been evaluated.")
