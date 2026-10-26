
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. Load and clean the dataset
# --------------------------------------------------

DATA_PATH = Path("data") / "Dataset_cleaned.csv"

df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["SepsisLabel", "Patient_ID"]).copy()

df["SepsisLabel"] = df["SepsisLabel"].astype(int)

# Keep only valid binary target labels
df = df[df["SepsisLabel"].isin([0, 1])].copy()

# Remove columns that contain no usable values
df = df.dropna(axis=1, how="all")

print("Dataset shape:", df.shape)
print("Unique patients:", df["Patient_ID"].nunique())

# --------------------------------------------------
# 2. Separate features, target and patient groups
# --------------------------------------------------

y = df["SepsisLabel"]

# IDs are used for splitting, not as model inputs
groups = df["Patient_ID"]

columns_to_drop = [
    "SepsisLabel",
    "Patient_ID",
    "Unnamed: 0"
]

X = df.drop(
    columns=columns_to_drop,
    errors="ignore"
)

# Remove any non-numeric columns for this baseline
X = X.select_dtypes(include="number")

print("Number of model features:", X.shape[1])
print("Positive label percentage:", round(y.mean() * 100, 2), "%")

# --------------------------------------------------
# 3. Split by patient, not by row
# --------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

train_patients = set(groups.iloc[train_idx])
test_patients = set(groups.iloc[test_idx])

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))
print("Training patients:", len(train_patients))
print("Testing patients:", len(test_patients))
print("Patient overlap:", len(train_patients & test_patients))

if y_train.nunique() < 2 or y_test.nunique() < 2:
    raise ValueError(
        "Both training and testing data must contain labels 0 and 1."
    )

# --------------------------------------------------
# 4. Calculate class imbalance weight
# --------------------------------------------------

negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())

scale_pos_weight = negative_count / positive_count

print("\nTraining negatives:", negative_count)
print("Training positives:", positive_count)
print("Positive class weight:", round(scale_pos_weight, 2))

# --------------------------------------------------
# 5. Train XGBoost
# --------------------------------------------------

model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
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

print("\nTraining XGBoost...")
model.fit(X_train, y_train)

# --------------------------------------------------
# 6. Generate predictions
# --------------------------------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# 7. Evaluate the model
# --------------------------------------------------

print("\n========== MODEL EVALUATION ==========")

print(f"Accuracy          : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision         : {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall            : {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1-score          : {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"ROC-AUC           : {roc_auc_score(y_test, y_prob):.4f}")
print(f"PR-AUC (Average Precision): {average_precision_score(y_test, y_prob):.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# --------------------------------------------------
# 8. Save the model
# --------------------------------------------------

Path("models").mkdir(exist_ok=True)
Path("plots").mkdir(exist_ok=True)

model.save_model("models/sepsis_xgb.json")

# --------------------------------------------------
# 9. Plot confusion matrix
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Predicted 0", "Predicted 1"],
    yticklabels=["Actual 0", "Actual 1"]
)
plt.title("XGBoost Confusion Matrix")
plt.tight_layout()
plt.savefig("plots/confusion_matrix.png", dpi=150)
plt.close()

# --------------------------------------------------
# 10. Plot feature importance
# --------------------------------------------------

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False).head(15)

plt.figure(figsize=(9, 6))
importance.sort_values().plot(kind="barh")
plt.title("Top 15 XGBoost Features")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.savefig("plots/feature_importance.png", dpi=150)
plt.close()

print("\nModel saved: models/sepsis_xgb.json")
print("Confusion matrix saved: plots/confusion_matrix.png")
print("Feature importance saved: plots/feature_importance.png")
print("\nTraining and evaluation completed.")
print("Note: Confirm the dataset's SepsisLabel definition")
print("before interpreting this as early prediction.")
