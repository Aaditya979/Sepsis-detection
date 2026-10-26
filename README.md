# AI-Based Early Prediction of Sepsis Using Patient Vital Signs

## Overview

This project aims to develop a machine learning model for predicting sepsis using patient vital signs and clinical measurements. It uses **XGBoost (Extreme Gradient Boosting)** to classify patient records based on the `SepsisLabel` target.

The project focuses on data preprocessing, handling class imbalance, patient-wise model evaluation, classification threshold tuning, and performance analysis.

> **Disclaimer:** This project is an academic machine learning prototype and is not intended for clinical diagnosis or medical decision-making.

## Objectives

- Analyze patient vital signs and clinical data.
- Preprocess the dataset and handle missing values.
- Train an XGBoost binary classification model.
- Address class imbalance using `scale_pos_weight`.
- Evaluate model performance using classification metrics.
- Tune the classification threshold using a validation set.
- Analyze feature importance and model errors.

## Technology Stack

- **Programming Language:** Python
- **Machine Learning:** XGBoost, scikit-learn
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Development Environment:** Visual Studio Code

## Dataset

The dataset contains patient records with clinical measurements and a target column named `SepsisLabel`.

Some of the available features include:

- Heart Rate (`HR`)
- Oxygen Saturation (`O2Sat`)
- Temperature (`Temp`)
- Systolic Blood Pressure (`SBP`)
- Mean Arterial Pressure (`MAP`)
- Respiratory Rate (`Resp`)
- Blood Glucose (`Glucose`)
- Lactate (`Lactate`)
- Age (`Age`)
- ICU Length of Stay (`ICULOS`)

The dataset also includes other laboratory measurements, demographic fields, and patient identifiers.

The target variable is:

- `SepsisLabel = 0`: Negative label
- `SepsisLabel = 1`: Positive label

The precise meaning and timing of the labels depend on the dataset's source and definition. The prediction horizon must be verified before interpreting the model as an early-warning system.

## Project Structure

```text
XG_boost/
│
├── data/
│   ├── Dataset.csv
│   └── Dataset_cleaned.csv
│
├── models/
│   ├── sepsis_xgb.json
│   ├── sepsis_xgb_tuned.json
│   └── best_threshold.txt
│
├── plots/
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── .venv/
├── inspect_data.py
├── prepare_data.py
├── train_model.py
├── tune_threshold.py
├── evaluate_tuned_model.py
├── main.py
└── README.md
```

*Note: This structure describes the intended project files. Some files may not exist until the corresponding steps have been completed.*

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Aaditya979/Sepsis-detection.git
cd Sepsis-detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

## How to Run

Run the scripts in the following order, assuming the corresponding files and input dataset are available:

### 1. Inspect the dataset

```bash
python inspect_data.py
```

### 2. Prepare the data

```bash
python prepare_data.py
```

### 3. Train the baseline XGBoost model

```bash
python train_model.py
```

### 4. Tune the classification threshold

```bash
python tune_threshold.py
```

### 5. Evaluate the tuned model

```bash
python evaluate_tuned_model.py
```

The training scripts save model files and evaluation plots in the `models/` and `plots/` directories.

## Model Evaluation

The baseline patient-wise model produced the following results on its held-out test set:

| Metric | Result |
|---|---:|
| Accuracy | 84.30% |
| Precision | 7.41% |
| Recall | 56.41% |
| F1-score | 13.10% |
| ROC-AUC | 0.7772 |
| PR-AUC (Average Precision) | 0.1069 |

These values are from the baseline model and should not be presented as the results of the tuned model.

The threshold-tuning experiment produced the following validation results at a threshold of `0.55`:

| Metric | Result |
|---|---:|
| Precision | 8.60% |
| Recall | 55.69% |
| F1-score | 14.90% |
| PR-AUC | 0.1010 |

The tuned model's performance on the reserved test set must be evaluated separately before drawing conclusions about generalization.

### Evaluation Metrics

- **Accuracy:** Proportion of all predictions that are correct.
- **Precision:** Proportion of predicted positive cases that are truly positive.
- **Recall:** Proportion of actual positive cases detected by the model.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Measures the model's ability to rank positive cases above negative cases.
- **PR-AUC:** Summarizes the precision-recall trade-off, particularly useful with imbalanced labels.

## Handling Class Imbalance

Positive sepsis labels represent a small proportion of the dataset. The model uses XGBoost's `scale_pos_weight` parameter to give greater weight to positive training examples.

Patient-wise data splitting is used to reduce the risk of the same patient's records appearing in both training and evaluation sets.

## Limitations

- The dataset contains missing clinical measurements.
- The target classes are imbalanced.
- Low precision indicates that many predicted positive records may be false alarms.
- Threshold selection involves a trade-off between precision and recall.
- The target label's exact timing and prediction horizon must be verified.
- Performance on this dataset does not establish clinical effectiveness or safety.

## Future Improvements

- Tune XGBoost hyperparameters using validation data.
- Compare XGBoost with baseline classifiers and other gradient-boosting models.
- Investigate feature importance and prediction errors.
- Evaluate probability calibration and alternative classification thresholds.
- Verify the dataset's label definition and intended prediction horizon.
- Assess performance on independent data before considering real-world applications.

## Author

**Aaditya979**

GitHub Repository: https://github.com/Aaditya979/Sepsis-detection

## License and Dataset Attribution

Add the appropriate project license and cite the original dataset source and its usage terms before redistributing the dataset or derived materials.
