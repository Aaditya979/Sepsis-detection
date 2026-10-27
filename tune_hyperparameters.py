
from pathlib import Path
from itertools import product

import pandas as pd
import xgboost as xgb

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import average_precision_score

# 1. Load dataset
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
).select_dtypes(include="number")

# 2. Split patients: 80% train/validation, 20% test
split1 = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=42
)
trainval_idx, test_idx = next(
    split1.split(X, y, groups=groups)
)

X_tv = X.iloc[trainval_idx]
y_tv = y.iloc[trainval_idx]
g_tv = groups.iloc[trainval_idx]

# 3. Split train/validation: 75%/25% of train/validation
split2 = GroupShuffleSplit(
    n_splits=1, test_size=0.25, random_state=42
)
train_idx, val_idx = next(
    split2.split(X_tv, y_tv, groups=g_tv)
)

X_train = X_tv.iloc[train_idx]
y_train = y_tv.iloc[train_idx]
X_val = X_tv.iloc[val_idx]
y_val = y_tv.iloc[val_idx]

weight = (y_train == 0).sum() / (y_train == 1).sum()

# 4. Try a small hyperparameter grid
settings = product(
    [3, 5, 7],          # max_depth
    [0.03, 0.05],       # learning_rate
    [200, 300]          # n_estimators
)

results = []

for depth, rate, trees in settings:
    model = xgb.XGBClassifier(
        max_depth=depth,
        learning_rate=rate,
        n_estimators=trees,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_val)[:, 1]
    score = average_precision_score(y_val, probabilities)

    results.append({
        "max_depth": depth,
        "learning_rate": rate,
        "n_estimators": trees,
        "validation_PR_AUC": score
    })

    print(
        f"depth={depth}, rate={rate}, trees={trees}, "
        f"PR-AUC={score:.4f}"
    )

# 5. Rank configurations
results_df = pd.DataFrame(results).sort_values(
    "validation_PR_AUC", ascending=False
)

print("\nBest configurations:")
print(results_df.to_string(index=False))

Path("models").mkdir(exist_ok=True)
results_df.to_csv("models/hyperparameter_results.csv", index=False)

print("\nResults saved to models/hyperparameter_results.csv")
print("Test set was not evaluated.")
