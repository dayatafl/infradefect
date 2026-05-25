"""
download_model.py — Download trained checkpoint from HuggingFace Hub

Usage:
    python download_model.py

Downloads best_model.pth (~329MB) into checkpoints/ folder.
Required before running the API locally if you don't have the checkpoint.

Requirements:
    pip install huggingface_hub
"""

from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID   = "Ryand12/infradefect-model"
FILENAME  = "best_model.pth"
LOCAL_DIR = Path("checkpoints")

def main():
    LOCAL_DIR.mkdir(exist_ok=True)
    dest = LOCAL_DIR / FILENAME

    if dest.exists():
        print(f"Checkpoint already exists at {dest}")
        return

    print(f"Downloading {FILENAME} from {REPO_ID}...")
    hf_hub_download(
        repo_id   = REPO_ID,
        filename  = FILENAME,
        local_dir = str(LOCAL_DIR),
    )
    print(f"Done. Saved to {dest}")
    print("You can now start the API: uvicorn backend.app:app --host 0.0.0.0 --port 8000")

if __name__ == "__main__":
    main()