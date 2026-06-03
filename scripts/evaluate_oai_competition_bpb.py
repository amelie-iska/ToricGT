#!/usr/bin/env python3
"""Evaluate a native ToricGT checkpoint on the local OAI competition validation shard.

The cached local validation data is the Parameter-Golf ``fineweb10B_sp1024``
SentencePiece shard.  The native ToricGT random-order LM is byte-level, so this
script decodes the shard back to UTF-8 text and scores the checkpoint on the
resulting bytes.  It logs under ``oai_competition/*`` instead of ``fineweb/*``
because this is native checkpoint evaluation, not the separate FineWeb scaffold.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.train_parameter_golf_random_order import (  # noqa: E402
    RandomOrderLMConfig,
    DenseRandomOrderToricLM,
    build_oai_competition_loader,
    config_get,
    evaluate,
    load_state_dict_with_optional_position_resize,
    read_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="config/train.parameter_golf_all_phases.yaml")
    parser.add_argument("--token-glob", default="")
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--wandb-run-path", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", default="fp32", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=0)
    parser.add_argument("--batches", type=int, default=0)
    parser.add_argument("--sp-tokens-per-decode", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=10017)
    parser.add_argument("--pass-id", type=int, default=0)
    parser.add_argument("--score-first-bias-lr", type=float, default=0.0)
    parser.add_argument("--score-first-bias-decay", type=float, default=0.98)
    parser.add_argument("--score-first-bias-clip", type=float, default=3.0)
    return parser.parse_args()


def config_from_checkpoint(payload: dict[str, Any]) -> RandomOrderLMConfig:
    raw_config = payload.get("config", {})
    if isinstance(raw_config, RandomOrderLMConfig):
        return raw_config
    if not isinstance(raw_config, dict):
        raise ValueError("checkpoint does not contain a RandomOrderLMConfig-compatible config dict")
    allowed = {field.name for field in fields(RandomOrderLMConfig)}
    filtered = {key: value for key, value in raw_config.items() if key in allowed}
    return RandomOrderLMConfig(**filtered)


def log_to_wandb(run_path: str, metrics: dict[str, float], step: int) -> None:
    if not run_path:
        return
    parts = run_path.split("/")
    if len(parts) != 3:
        raise ValueError(f"--wandb-run-path must be entity/project/run_id, got {run_path!r}")
    entity, project, run_id = parts
    import wandb

    run = wandb.init(entity=entity, project=project, id=run_id, resume="allow")
    try:
        run.log(metrics, step=step)
    finally:
        run.finish()


def main() -> None:
    args = parse_args()
    file_config = read_yaml(args.config)
    checkpoint_path = Path(args.checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu")
    model_config = config_from_checkpoint(payload)
    seq_len = int(args.seq_len or model_config.max_seq_len)
    if seq_len > int(model_config.max_seq_len):
        raise ValueError(f"requested seq_len={seq_len} exceeds checkpoint max_seq_len={model_config.max_seq_len}")

    token_glob = args.token_glob or config_get(
        file_config,
        "oai_competition",
        "val_token_glob",
        "amelie-iska/parameter-golf/data/datasets/fineweb10B_sp1024/fineweb_val_*.bin",
    )
    tokenizer_path = args.tokenizer_path or config_get(
        file_config,
        "oai_competition",
        "tokenizer_path",
        "amelie-iska/parameter-golf/data/tokenizers/fineweb_1024_bpe.model",
    )
    batches = int(args.batches or config_get(file_config, "oai_competition", "eval_batches", 8))
    batch_size = int(args.batch_size or config_get(file_config, "training", "batch_size", 2))
    sp_tokens_per_decode = int(
        args.sp_tokens_per_decode or config_get(file_config, "oai_competition", "sp_tokens_per_decode", 4096)
    )
    workers = int(args.workers if args.workers is not None else config_get(file_config, "oai_competition", "workers", 0))

    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    model = DenseRandomOrderToricLM(model_config).to(device)
    load_state_dict_with_optional_position_resize(model, payload["model"], allow_position_resize=False)
    loader = build_oai_competition_loader(
        token_glob=token_glob,
        tokenizer_path=tokenizer_path,
        batch_size=batch_size,
        seq_len=seq_len,
        byte_offset=model_config.byte_offset,
        sp_tokens_per_decode=sp_tokens_per_decode,
        seed=int(args.seed),
        workers=workers,
        repeat=False,
    )
    step = int(payload.get("step", 0))
    result = evaluate(
        model,
        loader,
        device=device,
        batches=batches,
        precision=args.precision,
        pass_id=int(args.pass_id or (int(args.seed) + 40_000 + step)),
        order_samples=1,
        gflownet_samples=1,
        score_first_bias_lr=float(args.score_first_bias_lr),
        score_first_bias_decay=float(args.score_first_bias_decay),
        score_first_bias_clip=float(args.score_first_bias_clip),
    )
    metrics = {
        "oai_competition/loss": float(result["loss"]),
        "oai_competition/bpb": float(result["bpb"]),
        "oai_competition/deterministic_loss": float(result["loss"]),
        "oai_competition/deterministic_bpb": float(result["bpb"]),
        "oai_competition/eval_batches": float(batches),
        "oai_competition/seq_len": float(seq_len),
        "oai_competition/available": 1.0,
        "oai_competition/source_sp1024_decoded_bytes": 1.0,
        "competition/oai_bpb": float(result["bpb"]),
        "bpb/oai_competition": float(result["bpb"]),
    }
    if "bias_norm" in result:
        metrics["oai_competition/score_first_bias_norm"] = float(result["bias_norm"])
    if not math.isfinite(float(result["bpb"])):
        raise RuntimeError(f"non-finite OAI competition BPB: {result['bpb']}")

    summary = {
        "checkpoint": str(checkpoint_path),
        "step": step,
        "token_glob": token_glob,
        "tokenizer_path": tokenizer_path,
        "metric_namespace": "oai_competition",
        "source": "local_fineweb10B_sp1024_validation_decoded_to_utf8_bytes",
        **metrics,
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    log_to_wandb(args.wandb_run_path, metrics, step=step)


if __name__ == "__main__":
    main()
