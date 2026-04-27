import os
import sys

# Instructions:
# 1. Install the Hugging Face Hub package: `pip install huggingface_hub`
# 2. Authenticate your machine: `huggingface-cli login`
# 3. Run this script: `python upload_to_hf.py`

try:
    from huggingface_hub import HfApi
except ImportError:
    print("Please install huggingface_hub first: pip install huggingface_hub")
    sys.exit(1)

# --- CONFIGURATION ---
# Replace this with your actual Hugging Face username!
hf_username = "your-username" 
repo_name = f"{hf_username}/florence-2-vqa-lora"
local_dir = "./florence-vqa-lora"
# ---------------------

if not os.path.exists(local_dir):
    print(f"Error: {local_dir} not found. Ensure the model has been downloaded.")
    sys.exit(1)

if hf_username == "your-username":
    print("Please open upload_to_hf.py and replace 'your-username' with your actual Hugging Face username!")
    sys.exit(1)

print(f"Uploading local LoRA weights from {local_dir} to Hugging Face Hub ({repo_name})...")

api = HfApi()

try:
    # Automatically create a public repository if it doesn't exist yet
    api.create_repo(repo_id=repo_name, private=False, exist_ok=True)
    
    # Upload the entire LoRA adapter folder
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_name,
        repo_type="model",
    )
    print(f"\nSuccess! Model uploaded to https://huggingface.co/{repo_name}")
except Exception as e:
    print(f"\nAn error occurred during upload: {e}")
