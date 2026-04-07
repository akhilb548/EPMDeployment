
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "indianakhil/engine-predictive-maintenance"
repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the dataset repo exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Repository '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Repository '{repo_id}' not found. Creating new repository...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Repository '{repo_id}' created.")

# Step 2: Upload raw dataset
api.upload_folder(
    folder_path="predictive_maintenance_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
print("Raw dataset uploaded to Hugging Face successfully.")
