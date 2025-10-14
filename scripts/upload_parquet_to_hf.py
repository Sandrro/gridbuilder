"""Upload local Parquet files from the data directory to HuggingFace Hub."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Directory containing Parquet files to upload")
    parser.add_argument("--repo-id", required=True, help="HuggingFace dataset repository id")
    parser.add_argument("--token", help="HuggingFace token; falls back to cached token when omitted")
    parser.add_argument("--branch", default="main", help="Target branch on the HuggingFace Hub")
    parser.add_argument(
        "--path-in-repo",
        default="",
        help="Optional path inside the HuggingFace repository where files will be stored",
    )
    parser.add_argument("--commit-message", default="Upload parquet data", help="Commit message for the upload")
    parser.add_argument("--allow-create", action="store_true", help="Create the repository if it does not exist")
    parser.add_argument("--private", action="store_true", help="Create the repository as private when used with --allow-create")
    parser.add_argument(
        "--patterns",
        nargs="*",
        default=["*.parquet"],
        help="Glob patterns that determine which files inside data-dir will be uploaded",
    )
    return parser.parse_args()


def gather_files(data_dir: Path, patterns: List[str]) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        files.extend(sorted(data_dir.rglob(pattern)))
    unique_files = sorted(set(file for file in files if file.is_file()))
    return unique_files


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory '{data_dir}' does not exist")

    files = gather_files(data_dir, args.patterns)
    if not files:
        raise FileNotFoundError(f"No files matching {args.patterns} were found in {data_dir}")

    try:
        from huggingface_hub import HfApi, HfFolder
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to upload data") from exc

    token = args.token or HfFolder.get_token()
    if not token:
        raise RuntimeError("HuggingFace token not available; provide --token or login via huggingface-cli")

    api = HfApi()
    if args.allow_create:
        logging.info("Ensuring dataset repository %s exists", args.repo_id)
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            token=token,
            private=args.private,
            exist_ok=True,
        )

    logging.info("Uploading %d files from %s to %s", len(files), data_dir, args.repo_id)
    try:
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=str(data_dir),
            path_in_repo=args.path_in_repo or "",
            token=token,
            commit_message=args.commit_message,
            allow_patterns=args.patterns,
            revision=args.branch,
        )
    except Exception as exc:  # pragma: no cover - depends on network state
        raise RuntimeError(f"Failed to upload Parquet files: {exc}") from exc

    logging.info("Upload completed successfully")


if __name__ == "__main__":
    main()
