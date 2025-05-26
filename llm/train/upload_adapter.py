from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="llm/adapters/Llama-3.1-Minitron_cr_chaptered",
    repo_id="EndlessAI777/Llama-3.1-Minitron_cr_chaptered",
    repo_type="model",
)
