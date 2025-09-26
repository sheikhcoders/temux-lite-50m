"""Utility to upload repository contents to the Hugging Face Hub safely."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, create_repo, upload_folder

DEFAULT_REPO = "TheTemuxFamily/Temux-Lite-50M"
DEFAULT_IGNORE = [
    ".git/*",
    "tests/*",
    "*.ipynb_checkpoints/*",
    "weights/*.bin",
    "weights/*.pt",
    "weights/*.safetensors",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Target repo id on the Hub.")
    parser.add_argument("--local-dir", default=".", help="Directory to upload (default: current directory).")
    parser.add_argument("--commit-message", default="Sync Temux-Lite-50M template", help="Hub commit message.")
    parser.add_argument(
        "--include-weights",
        action="store_true",
        help="If set, safetensor files in weights/ are uploaded as well.",
    )
    parser.add_argument(
        "--extra-ignore",
        action="append",
        default=[],
        help="Additional glob patterns to exclude (can be used multiple times).",
    )
    return parser.parse_args()


def build_ignore_patterns(include_weights: bool, extra: Iterable[str]) -> list[str]:
    patterns = list(DEFAULT_IGNORE)
    if include_weights:
        patterns = [p for p in patterns if not p.endswith("*.safetensors")]
    patterns.extend(extra)
    return patterns


def main() -> None:
    args = parse_args()
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise EnvironmentError(
            "Set the HUGGINGFACE_HUB_TOKEN environment variable before running this script."
        )

    local_dir = Path(args.local_dir).resolve()
    if not local_dir.exists():
        raise FileNotFoundError(f"{local_dir} does not exist")

    ignore_patterns = build_ignore_patterns(args.include_weights, args.extra_ignore)

    api = HfApi(token=token)
    create_repo(repo_id=args.repo, repo_type="model", exist_ok=True, token=token)
    upload_folder(
        repo_id=args.repo,
        folder_path=str(local_dir),
        commit_message=args.commit_message,
        token=token,
        ignore_patterns=ignore_patterns,
    )
    print(f"Pushed to https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
