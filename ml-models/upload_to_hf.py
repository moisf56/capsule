"""
Upload Capsule quantized models and demo Space to HuggingFace Hub.

Usage:
    pip install huggingface_hub
    HF_TOKEN=hf_xxx python ml-models/upload_to_hf.py

Repos created:
    moisf56/medasr-conformer-ctc-int8-onnx  (model)
    moisf56/medgemma-4b-q3km-gguf           (model)
    moisf56/capsule-medasr-demo             (space)
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

# ── Configuration ────────────────────────────────────────────────────────────

HF_USERNAME = "moisf56"

_HF_MODELS_DIR = Path(__file__).parent / "hf-models"
_ML_MODELS_DIR = Path("/home/mo-saif/Documents/medgemma hackathon/ml-models")
_VOCAB_FILE    = Path(__file__).parent.parent / "mobile/MedGemmaApp/src/medasr_vocab.json"

MODELS = [
    {
        "repo_id": f"{HF_USERNAME}/medasr-conformer-ctc-int8-onnx",
        "card":    _HF_MODELS_DIR / "medasr-int8-onnx/README.md",
        "files": [
            (_ML_MODELS_DIR / "onnx/medasr_int8.onnx", "medasr_int8.onnx"),
            (_VOCAB_FILE,                               "medasr_vocab.json"),
        ],
        "private": False,
    },
    {
        "repo_id": f"{HF_USERNAME}/medgemma-4b-q3km-gguf",
        "card":    _HF_MODELS_DIR / "medgemma-4b-q3km-gguf/README.md",
        "files": [
            (
                _ML_MODELS_DIR / "gguf/medgemma-1.5-4b-it-Q3_K_M.gguf",
                "medgemma-1.5-4b-it-Q3_K_M.gguf",
            ),
        ],
        "private": False,
    },
]

# ── Upload logic ──────────────────────────────────────────────────────────────

def upload_model(api: HfApi, config: dict) -> None:
    repo_id = config["repo_id"]
    print(f"\n{'─' * 60}")
    print(f"  Repo : {repo_id}")
    print(f"{'─' * 60}")

    # 1. Create repo (no-op if it already exists)
    create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=config["private"],
        exist_ok=True,
    )
    print(f"  [✓] Repo ready")

    # 2. Upload model card (README.md)
    card_path = config["card"]
    if not card_path.exists():
        print(f"  [!] Model card not found: {card_path}", file=sys.stderr)
    else:
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add model card",
        )
        print(f"  [✓] Model card uploaded")

    # 3. Upload model files
    for local_path, repo_path in config["files"]:
        if not local_path.exists():
            print(f"  [!] File not found (skipping): {local_path}", file=sys.stderr)
            print(f"      → Place the file there and re-run to upload it.")
            continue

        size_mb = local_path.stat().st_size / 1_048_576
        print(f"  [↑] Uploading {repo_path}  ({size_mb:.0f} MB) ...")

        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add {repo_path}",
        )
        print(f"  [✓] {repo_path} uploaded")

    print(f"\n  HuggingFace page: https://huggingface.co/{repo_id}")


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    # Verify login
    try:
        user = api.whoami()
        print(f"Logged in as: {user['name']}")
    except Exception:
        print(
            "Not logged in. Run:\n"
            "  huggingface-cli login\n"
            "or set the HF_TOKEN environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    for config in MODELS:
        upload_model(api, config)

    print("\nDone.")


if __name__ == "__main__":
    main()
