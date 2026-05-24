"""
upload_model.py — Upload trained checkpoint to HuggingFace Hub

Run this ONCE after training completes:
    python upload_model.py --checkpoint checkpoints/best_model.pth --repo YOUR_HF_USERNAME/infradefect-model

Requirements:
    pip install huggingface_hub
    huggingface-cli login   (or set HF_TOKEN env var)
"""

import argparse
from pathlib import Path


def upload(checkpoint: str, repo_id: str):
    from huggingface_hub import HfApi, create_repo

    api = HfApi()

    # Create repo if it doesn't exist (private by default)
    print(f"Creating/checking repo: {repo_id}")
    create_repo(repo_id, repo_type="model", exist_ok=True, private=True)

    # Upload checkpoint
    print(f"Uploading {checkpoint} → {repo_id}/best_model.pth ...")
    api.upload_file(
        path_or_fileobj=checkpoint,
        path_in_repo="best_model.pth",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Done. Model available at: https://huggingface.co/{repo_id}")
    print(f"\nSet this in your HF Space secrets:")
    print(f"  HF_MODEL_REPO = {repo_id}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    p.add_argument("--repo", required=True, help="e.g. yourusername/infradefect-model")
    args = p.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        exit(1)

    upload(args.checkpoint, args.repo)
