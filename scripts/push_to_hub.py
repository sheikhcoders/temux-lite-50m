"""Utility to upload the repository contents to the Hugging Face Hub."""

from __future__ import annotations

import os

from huggingface_hub import HfApi, create_repo, upload_folder

REPO_ID = "TheTemuxFamily/Temux-Lite-50M"
LOCAL_DIR = "."


def main() -> None:
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise EnvironmentError("Set the HUGGINGFACE_HUB_TOKEN environment variable before running.")

    api = HfApi(token=token)
    create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True, token=token)

    upload_folder(
        repo_id=REPO_ID,
        folder_path=LOCAL_DIR,
        commit_message="Initial upload of Temux-Lite-50M",
        token=token,
        ignore_patterns=[
            ".git/*",
            "tests/*",
            "*.ipynb_checkpoints/*",
            "weights/*.bin",
            "weights/*.pt",
        ],
    )
    print(f"Pushed to https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
