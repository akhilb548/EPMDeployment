
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, hf_hub_download

FEATURES = ['Engine_RPM', 'Lub_Oil_Pressure', 'Fuel_Pressure',
            'Coolant_Pressure', 'Lub_Oil_Temperature', 'Coolant_Temperature']
TARGET = 'Engine_Condition'

api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_REPO = "indianakhil/engine-predictive-maintenance"

# Load dataset from Hugging Face
file_path = hf_hub_download(
    repo_id=DATASET_REPO, filename='engine_data.csv',
    repo_type='dataset', token=os.getenv("HF_TOKEN")
)
df = pd.read_csv(file_path)
df.columns = FEATURES + [TARGET]
print(f"Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

# Remove duplicates
df = df.drop_duplicates()

# Fill missing values with median
for col in FEATURES:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# IQR-based outlier capping (Winsorization)
for feat in FEATURES:
    Q1, Q3 = df[feat].quantile(0.25), df[feat].quantile(0.75)
    IQR = Q3 - Q1
    df[feat] = df[feat].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

# Stratified 80/20 train-test split
X, y = df[FEATURES], df[TARGET].astype(int)
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)
print(f"Train: {len(Xtrain):,} rows | Test: {len(Xtest):,} rows")

# Upload splits to Hugging Face
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )
    print(f"Uploaded {file_path} to Hugging Face.")
