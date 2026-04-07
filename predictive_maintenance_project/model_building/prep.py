
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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_df = X_train.copy(); train_df[TARGET] = y_train.values
test_df  = X_test.copy();  test_df[TARGET]  = y_test.values

train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)
print(f"Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")

# Upload splits to Hugging Face
for local_file, hf_name in [("train.csv", "train.csv"), ("test.csv", "test.csv")]:
    api.upload_file(
        path_or_fileobj=local_file,
        path_in_repo=hf_name,
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )
    print(f"Uploaded {hf_name} to Hugging Face.")
