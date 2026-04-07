
from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="predictive_maintenance_project/deployment",
    repo_id="indianakhil/engine-predictive-maintenance-space",
    repo_type="space",
    path_in_repo="",
)
