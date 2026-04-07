
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier, BaggingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
import xgboost as xgb
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Engine_Predictive_Maintenance")

FEATURES = ['Engine_RPM', 'Lub_Oil_Pressure', 'Fuel_Pressure',
            'Coolant_Pressure', 'Lub_Oil_Temperature', 'Coolant_Temperature']
TARGET = 'Engine_Condition'
DATASET_REPO = "indianakhil/engine-predictive-maintenance"
MODEL_REPO   = "indianakhil/engine-predictive-maintenance-model"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Load train/test from Hugging Face
tr_path = hf_hub_download(repo_id=DATASET_REPO, filename='train.csv',
                           repo_type='dataset', token=os.getenv("HF_TOKEN"))
te_path = hf_hub_download(repo_id=DATASET_REPO, filename='test.csv',
                           repo_type='dataset', token=os.getenv("HF_TOKEN"))
train_data = pd.read_csv(tr_path)
test_data  = pd.read_csv(te_path)

X_train, y_train = train_data[FEATURES], train_data[TARGET]
X_test,  y_test  = test_data[FEATURES],  test_data[TARGET]

# Define models and hyperparameter grids
models_params = {
    'Decision_Tree': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {'max_depth': [5, 10, 15, None], 'criterion': ['gini', 'entropy']}
    },
    'Bagging': {
        'model': BaggingClassifier(random_state=42),
        'params': {'n_estimators': [50, 100], 'max_samples': [0.8, 1.0]}
    },
    'Random_Forest': {
        'model': RandomForestClassifier(random_state=42, n_jobs=-1),
        'params': {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
    },
    'AdaBoost': {
        'model': AdaBoostClassifier(random_state=42),
        'params': {'n_estimators': [50, 100, 200], 'learning_rate': [0.5, 1.0]}
    },
    'Gradient_Boosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}
    },
    'XGBoost': {
        'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False),
        'params': {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1],
                   'max_depth': [3, 5], 'subsample': [0.8, 1.0]}
    }
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_f1 = 0
best_model_name = None
best_model_obj  = None

for model_name, config in models_params.items():
    with mlflow.start_run(run_name=model_name):
        gs = GridSearchCV(config['model'], config['params'],
                          cv=cv, scoring='f1', n_jobs=-1)
        gs.fit(X_train, y_train)
        best_est = gs.best_estimator_
        y_pred   = best_est.predict(X_test)
        y_prob   = best_est.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy':  accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall':    recall_score(y_test, y_pred),
            'f1_score':  f1_score(y_test, y_pred),
            'roc_auc':   roc_auc_score(y_test, y_prob),
            'cv_f1':     gs.best_score_
        }
        mlflow.log_params(gs.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_est, artifact_path='model')

        print(f"{model_name} - F1={metrics['f1_score']:.4f} | Acc={metrics['accuracy']:.4f}")

        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_model_name = model_name
            best_model_obj  = best_est

print(f"\nBest Model: {best_model_name} with F1-Score: {best_f1:.4f}")

# Save best model locally
model_path = "best_model.pkl"
joblib.dump(best_model_obj, model_path)
print(f"Model saved as artifact at: {model_path}")

# Upload to Hugging Face Model Hub
try:
    api.repo_info(repo_id=MODEL_REPO, repo_type='model')
    print(f"Model repository '{MODEL_REPO}' already exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=MODEL_REPO, repo_type='model', private=False)
    print(f"Model repository '{MODEL_REPO}' created.")

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo='best_model.pkl',
    repo_id=MODEL_REPO,
    repo_type='model',
)
print(f"Model registered at: https://huggingface.co/{MODEL_REPO}")
