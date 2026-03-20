"""
Deploy the Capsule Gradio Space to HuggingFace Hub.

Usage:
    HF_TOKEN=hf_xxx python ml-models/upload_space_to_hf.py

Space created:
    moisf56/capsule-medasr-demo
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

HF_USERNAME = "moisf56"
SPACE_ID    = f"{HF_USERNAME}/capsule-medasr-demo"
SPACE_DIR   = Path(__file__).parent / "hf-spaces/medasr-demo"


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    api   = HfApi(token=token)

    try:
        user = api.whoami()
        print(f"Logged in as: {user['name']}")
    except Exception:
        print("Not logged in. Set the HF_TOKEN environment variable.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'─' * 60}")
    print(f"  Space: {SPACE_ID}")
    print(f"{'─' * 60}")

    create_repo(
        repo_id=SPACE_ID,
        repo_type="space",
        space_sdk="gradio",
        private=False,
        exist_ok=True,
    )
    print("  [✓] Space repo ready")

    api.upload_folder(
        folder_path=str(SPACE_DIR),
        repo_id=SPACE_ID,
        repo_type="space",
        commit_message="Deploy Capsule MedASR + MedGemma demo",
    )
    print("  [✓] Space files uploaded")
    print(f"\n  Space: https://huggingface.co/spaces/{SPACE_ID}")


if __name__ == "__main__":
    main()
