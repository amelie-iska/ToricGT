#!/usr/bin/env python3
"""Publish a ToricGT checkpoint to a Hugging Face model repository."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch
from huggingface_hub import HfApi, create_repo, upload_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--commit-message", default="Upload ToricGT checkpoint")
    return parser.parse_args()


def model_card(checkpoint_path: Path, metadata: dict, private: bool) -> str:
    config = metadata.get("config", {})
    train_config = metadata.get("train_config", {})
    private_line = "private: true\n" if private else ""
    return f"""---
license: other
library_name: pytorch
tags:
  - graph-transformer
  - tokengt
  - soft-moe
  - tropical-attention
  - gflownet
  - toricgt
{private_line.rstrip()}
---

# ToricGT Checkpoint

Author: Amelie Schreiber

This checkpoint is part of the ToricGT research workspace. It contains a
TokenGT-style graph-to-graph model with tropical/ring attention, default
Soft-MoE feed-forward blocks, and an embedding-space GFlowNet policy head.

## Source Checkpoint

- Local path at upload time: `{checkpoint_path}`
- Recorded training step: `{metadata.get("step", "unknown")}`

## Model Config

```json
{json.dumps(config, indent=2, sort_keys=True)}
```

## Train Config

```json
{json.dumps(train_config, indent=2, sort_keys=True)}
```

## Notes

Use the matching ToricGT repository code to load this checkpoint. This upload
does not include raw training data.
"""


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)
    card = model_card(checkpoint, metadata, args.private)
    files = [
        (checkpoint, checkpoint.name),
    ]
    with tempfile.TemporaryDirectory(prefix="toricgt_hf_model_") as tmp:
        readme = Path(tmp) / "README.md"
        readme.write_text(card, encoding="utf-8")
        files.append((readme, "README.md"))
        total_bytes = sum(path.stat().st_size for path, _ in files)
        print({"repo_id": args.repo_id, "files": len(files), "bytes": total_bytes, "dry_run": args.dry_run})
        for path, repo_path in files:
            print(f"{path} -> {repo_path}")
        if args.dry_run:
            return
        api = HfApi()
        api.whoami()
        create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
        if not args.private:
            api.update_repo_settings(args.repo_id, repo_type="model", private=False)
        for path, repo_path in files:
            upload_file(
                path_or_fileobj=str(path),
                path_in_repo=repo_path,
                repo_id=args.repo_id,
                repo_type="model",
                commit_message=args.commit_message,
            )


if __name__ == "__main__":
    main()
