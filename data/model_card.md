---
language: en
license: mit
tags:
  - predictive-maintenance
  - binary-classification
  - engine-health
  - scikit-learn
datasets:
  - indianakhil/engine-predictive-maintenance
metrics:
  - f1
  - accuracy
  - roc_auc
---

# Engine Predictive Maintenance Model

## Model Description
Binary classifier predicting engine health (Normal vs Faulty) from six sensor readings.

- **Model Type**: AdaBoostClassifier
- **Task**: Binary Classification (0=Normal, 1=Faulty)
- **Training Data**: `indianakhil/engine-predictive-maintenance` (19,535 records)
- **Best Hyperparameters**: `{'learning_rate': 0.5, 'n_estimators': 50}`

## Performance (Test Set — 20% holdout)

| Metric | Score |
|---|---|
| Accuracy | 0.6644 |
| Precision | 0.6787 |
| Recall | 0.8883 |
| **F1-Score** | **0.7695** |
| ROC-AUC | 0.6960 |
| CV F1 (5-fold) | 0.7663 |

## Input Features
Engine_RPM, Lub_Oil_Pressure, Fuel_Pressure, Coolant_Pressure,
Lub_Oil_Temperature, Coolant_Temperature

## Usage
```python
from huggingface_hub import hf_hub_download
import joblib, pandas as pd
model = joblib.load(hf_hub_download(
    repo_id='indianakhil/engine-predictive-maintenance-model',
    filename='best_model.pkl'))
pred = model.predict(X)  # 0=Normal, 1=Faulty
prob = model.predict_proba(X)[:, 1]  # Fault probability
```
