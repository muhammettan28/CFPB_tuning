#!/usr/bin/env python3
"""Download a Hugging Face training checkpoint to shared TRUBA storage."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("/arf/scratch/muhtan/ccc/hf_models/Qwen3-8B"),
    )
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.model,
        local_dir=args.destination,
        local_dir_use_symlinks=False,
    )
    print(args.destination.resolve())


if __name__ == "__main__":
    main()
