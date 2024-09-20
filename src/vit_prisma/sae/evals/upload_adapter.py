from huggingface_hub import HfApi, Repository
import os
import shutil
import torch

# Ensure you have the huggingface_hub library installed
# pip install huggingface_hub

# # Set your Hugging Face token
# hf_token = os.environ.get("HF_TOKEN")
# if not hf_token:
#     raise ValueError("Please set the HF_TOKEN environment variable")

# Set the organization and repository name
org_name = "Prisma-Multimodal"
repo_name = "TinyCLIP-Kandinsky-adapter"
full_repo_name = f"{org_name}/{repo_name}"

# Set up the Hugging Face API
api = HfApi()


hf_token = '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'

# Create a new repository in the Prisma-Multimodal organization
try:
    api.create_repo(repo_id=full_repo_name, private=False, token=hf_token)
except Exception as e:
    print(f"Repository already exists or there was an error: {e}")

# Clone the repository
repo_url = f"https://huggingface.co/{full_repo_name}"
local_dir = repo_name
repo = Repository(local_dir=local_dir, clone_from=repo_url, use_auth_token=hf_token)

# Load your adapter
adapter_path = '/network/scratch/s/sonia.joseph/diffusion/tinyclip_adapter/ntilw/adapter_checkpoint_5000.pth'
adapter = torch.load(adapter_path, map_location=torch.device('cpu'))

# Save the adapter in the repository directory
new_adapter_path = os.path.join(local_dir, "adapter_checkpoint_5000.pth")
torch.save(adapter, new_adapter_path)

# Create a README.md file
with open(os.path.join(local_dir, "README.md"), "w") as f:
    f.write(f"# TinyCLIP-Kandinsky Adapter\n\n")
    f.write("This repository contains the TinyCLIP-Kandinsky adapter checkpoint.\n\n")
    f.write("## Usage\n\n")
    f.write("```python\n")
    f.write("from huggingface_hub import hf_hub_download\n")
    f.write("import torch\n\n")
    f.write(f"adapter_path = hf_hub_download(repo_id='{full_repo_name}', filename='adapter_checkpoint_5000.pth')\n")
    f.write("adapter = torch.load(adapter_path, map_location=torch.device('cpu'))\n")
    f.write("```\n")

# Push the changes to the Hugging Face repository
repo.git_add()
repo.git_commit("Add TinyCLIP-Kandinsky adapter")
repo.git_push()

print(f"Adapter uploaded successfully to: {repo_url}")

# Clean up: remove the local repository directory
shutil.rmtree(local_dir)