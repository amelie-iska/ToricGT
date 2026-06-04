#!/usr/bin/env python3
"""Evaluate compact Seq4096 checkpoints on the FineWeb Parameter-Golf shard.

The historical ``evaluate_oai_competition_bpb.py`` script is for native
RandomOrderLM checkpoints.  The OpenAI Parameter-Golf Seq4096 trainer writes a
compact GPT-style checkpoint instead, so this script loads the compact
``train_gpt.py`` implementation, reconstructs the model shape from tensor keys,
and evaluates the checkpoint with the same BPB byte accounting used by training.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import sentencepiece as spm
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPACT_SCRIPT = (
    ROOT
    / "amelie-iska"
    / "parameter-golf"
    / "records"
    / "track_10min_16mb"
    / "2026-03-19_TrainingOptSeq4096"
    / "train_gpt.py"
)
DEFAULT_TOKEN_GLOB = str(ROOT / "amelie-iska" / "parameter-golf" / "data" / "datasets" / "fineweb10B_sp1024" / "fineweb_val_*.bin")
DEFAULT_TOKENIZER = str(ROOT / "amelie-iska" / "parameter-golf" / "data" / "tokenizers" / "fineweb_1024_bpe.model")


@dataclass(frozen=True)
class CompactSeq4096Config:
    vocab_size: int
    num_layers: int
    model_dim: int
    num_heads: int
    num_kv_heads: int
    mlp_mult: int
    tie_embeddings: bool
    bigram_bias: bool
    hash_ngram_bias: bool
    hash_ngram_bias_buckets: int
    hash_ngram_bias_order: int = 3
    tied_embed_init_std: float = 0.005
    logit_softcap: float = 30.0
    rope_base: float = 10000.0
    qk_gain_init: float = 1.5
    bigram_bias_init_std: float = 0.0
    bigram_bias_scale: float = 1.0
    hash_ngram_bias_init_std: float = 0.0
    hash_ngram_bias_scale: float = 1.0
    polarquant_kv_bits: int = 0
    polarquant_train: bool = False
    polarquant_train_sample_tokens: int = 0
    polarquant_eval_sample_tokens: int = 0
    polarquant_seed: int = 271828


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--compact-script", default=str(DEFAULT_COMPACT_SCRIPT))
    parser.add_argument("--token-glob", default=DEFAULT_TOKEN_GLOB)
    parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--wandb-run-path", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--val-batch-size", type=int, default=524_288)
    parser.add_argument("--val-max-sequences", type=int, default=0)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--logit-softcap", type=float, default=30.0)
    parser.add_argument("--rope-base", type=float, default=10000.0)
    parser.add_argument("--qk-gain-init", type=float, default=1.5)
    parser.add_argument("--bigram-bias-scale", type=float, default=1.0)
    parser.add_argument("--hash-ngram-bias-order", type=int, default=3)
    parser.add_argument("--hash-ngram-bias-scale", type=float, default=1.0)
    parser.add_argument("--polarquant-kv-bits", type=int, default=0)
    parser.add_argument("--polarquant-eval-sample-tokens", type=int, default=0)
    parser.add_argument("--polarquant-seed", type=int, default=271828)
    return parser.parse_args()


def load_compact_module(script_path: str | Path):
    path = Path(script_path)
    if not path.exists():
        raise FileNotFoundError(f"compact Seq4096 train script not found: {path}")
    spec = importlib.util.spec_from_file_location("seq4096_train_gpt_for_eval", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import compact Seq4096 train script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def infer_compact_config(
    state_dict: dict[str, torch.Tensor],
    *,
    logit_softcap: float = 30.0,
    rope_base: float = 10000.0,
    qk_gain_init: float = 1.5,
    bigram_bias_scale: float = 1.0,
    hash_ngram_bias_order: int = 3,
    hash_ngram_bias_scale: float = 1.0,
    polarquant_kv_bits: int = 0,
    polarquant_eval_sample_tokens: int = 0,
    polarquant_seed: int = 271828,
) -> CompactSeq4096Config:
    if "tok_emb.weight" not in state_dict:
        raise ValueError("compact Seq4096 checkpoint is missing tok_emb.weight")
    tok_emb = state_dict["tok_emb.weight"]
    if tok_emb.ndim != 2:
        raise ValueError(f"tok_emb.weight must be 2D, got shape={tuple(tok_emb.shape)}")
    vocab_size, model_dim = map(int, tok_emb.shape)
    layer_ids = sorted(
        {
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith("blocks.") and len(key.split(".")) > 2 and key.split(".")[1].isdigit()
        }
    )
    if not layer_ids:
        raise ValueError("compact Seq4096 checkpoint has no blocks.* tensors")
    num_layers = max(layer_ids) + 1
    q_gain = state_dict.get("blocks.0.attn.q_gain")
    c_k = state_dict.get("blocks.0.attn.c_k.weight")
    fc = state_dict.get("blocks.0.mlp.fc.weight")
    if q_gain is None or c_k is None or fc is None:
        raise ValueError("compact Seq4096 checkpoint is missing block-0 attention/MLP tensors")
    num_heads = int(q_gain.numel())
    if num_heads <= 0 or model_dim % num_heads != 0:
        raise ValueError(f"cannot infer valid num_heads from q_gain={tuple(q_gain.shape)} model_dim={model_dim}")
    head_dim = model_dim // num_heads
    num_kv_heads = int(c_k.shape[0]) // head_dim
    if num_kv_heads <= 0 or int(c_k.shape[0]) != num_kv_heads * head_dim:
        raise ValueError(f"cannot infer valid num_kv_heads from c_k.weight={tuple(c_k.shape)}")
    mlp_mult = int(fc.shape[0]) // model_dim
    if mlp_mult <= 0 or int(fc.shape[0]) != mlp_mult * model_dim:
        raise ValueError(f"cannot infer valid mlp_mult from fc.weight={tuple(fc.shape)}")
    tie_embeddings = "lm_head.weight" not in state_dict
    bigram_bias = "prev_token_bias.weight" in state_dict
    hash_ngram_bias = "hash_ngram_bias.weight" in state_dict
    hash_ngram_bias_buckets = int(state_dict["hash_ngram_bias.weight"].shape[0]) if hash_ngram_bias else 2048
    return CompactSeq4096Config(
        vocab_size=vocab_size,
        num_layers=num_layers,
        model_dim=model_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        mlp_mult=mlp_mult,
        tie_embeddings=tie_embeddings,
        bigram_bias=bigram_bias,
        hash_ngram_bias=hash_ngram_bias,
        hash_ngram_bias_buckets=hash_ngram_bias_buckets,
        hash_ngram_bias_order=int(hash_ngram_bias_order),
        logit_softcap=float(logit_softcap),
        rope_base=float(rope_base),
        qk_gain_init=float(qk_gain_init),
        bigram_bias_scale=float(bigram_bias_scale),
        hash_ngram_bias_scale=float(hash_ngram_bias_scale),
        polarquant_kv_bits=int(polarquant_kv_bits),
        polarquant_eval_sample_tokens=int(polarquant_eval_sample_tokens),
        polarquant_seed=int(polarquant_seed),
    )


def checkpoint_kind(payload: dict[str, Any]) -> str:
    model = payload.get("model")
    if isinstance(model, dict) and "tok_emb.weight" in model:
        return "seq4096_compact_gpt"
    if isinstance(payload.get("config"), dict):
        return "random_order_toricgt"
    return "unknown"


def checkpoint_step(payload: dict[str, Any]) -> int:
    try:
        return int(payload.get("step", 0))
    except (TypeError, ValueError):
        return 0


def build_model(module: Any, config: CompactSeq4096Config, state_dict: dict[str, torch.Tensor], device: torch.device, precision: str):
    model = module.GPT(**asdict(config))
    if precision == "bf16" and device.type == "cuda":
        model = model.to(device).bfloat16()
        for submodule in model.modules():
            if isinstance(submodule, module.CastedLinear):
                submodule.float()
        module.restore_low_dim_params_to_fp32(model)
    else:
        model = model.to(device)
    result = model.load_state_dict(state_dict, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"state_dict mismatch missing={result.missing_keys} unexpected={result.unexpected_keys}")
    return model


def evaluate_checkpoint(
    *,
    module: Any,
    checkpoint_path: Path,
    token_glob: str,
    tokenizer_path: str,
    device: torch.device,
    seq_len: int,
    val_batch_size: int,
    val_max_sequences: int,
    precision: str,
    config_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], float, float, CompactSeq4096Config]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint_kind(payload) != "seq4096_compact_gpt":
        raise ValueError(f"checkpoint is not a compact Seq4096 GPT payload: kind={checkpoint_kind(payload)}")
    state_dict = payload["model"]
    config = infer_compact_config(state_dict, **config_kwargs)
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    if int(sp.vocab_size()) != int(config.vocab_size):
        raise ValueError(f"tokenizer vocab_size={int(sp.vocab_size())} does not match checkpoint vocab_size={config.vocab_size}")
    val_tokens = module.load_validation_tokens(token_glob, int(seq_len))
    byte_luts = module.build_sentencepiece_luts(sp, int(config.vocab_size), device)
    model = build_model(module, config, state_dict, device, precision)
    args = SimpleNamespace(
        val_batch_size=int(val_batch_size),
        train_seq_len=int(seq_len),
        val_max_sequences=int(val_max_sequences),
    )
    val_loss, val_bpb = module.eval_val(
        args,
        model,
        rank=0,
        world_size=1,
        device=device,
        grad_accum_steps=1,
        val_tokens=val_tokens,
        base_bytes_lut=byte_luts[0],
        has_leading_space_lut=byte_luts[1],
        is_boundary_token_lut=byte_luts[2],
    )
    return payload, float(val_loss), float(val_bpb), config


def oai_metric_summary(
    *,
    checkpoint: Path,
    step: int,
    val_loss: float,
    val_bpb: float,
    seq_len: int,
    val_max_sequences: int,
    token_glob: str,
    tokenizer_path: str,
    config: CompactSeq4096Config | None = None,
) -> dict[str, Any]:
    eval_scope = "full" if int(val_max_sequences) <= 0 else "sampled"
    metrics: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "step": int(step),
        "token_glob": str(token_glob),
        "tokenizer_path": str(tokenizer_path),
        "source": "local_fineweb10B_sp1024_validation_tokens",
        "metric_namespace": "oai_competition",
        "checkpoint_kind": "seq4096_compact_gpt",
        "oai_competition/loss": float(val_loss),
        "oai_competition/bpb": float(val_bpb),
        "oai_competition/deterministic_loss": float(val_loss),
        "oai_competition/deterministic_bpb": float(val_bpb),
        "oai_competition/seq_len": float(seq_len),
        "oai_competition/eval_scope": eval_scope,
        "oai_competition/val_max_sequences": float(val_max_sequences),
        "oai_competition/available": 1.0,
        "competition/oai_bpb": float(val_bpb),
        "bpb/oai_competition": float(val_bpb),
        "seq4096/oai_competition_bpb": float(val_bpb),
        "seq4096/oai_competition_loss": float(val_loss),
    }
    if config is not None:
        metrics["compact_config"] = asdict(config)
        metrics["seq4096/model_dim"] = float(config.model_dim)
        metrics["seq4096/num_layers"] = float(config.num_layers)
        metrics["seq4096/num_heads"] = float(config.num_heads)
        metrics["seq4096/num_kv_heads"] = float(config.num_kv_heads)
        metrics["seq4096/bigram_bias"] = 1.0 if config.bigram_bias else 0.0
    return metrics


def log_to_wandb(run_path: str, metrics: dict[str, Any], step: int) -> None:
    if not run_path:
        return
    parts = run_path.split("/")
    if len(parts) != 3:
        raise ValueError(f"--wandb-run-path must be entity/project/run_id, got {run_path!r}")
    entity, project, run_id = parts
    import wandb

    numeric = {key: value for key, value in metrics.items() if isinstance(value, (int, float)) and math.isfinite(float(value))}
    run = wandb.init(entity=entity, project=project, id=run_id, resume="allow")
    try:
        run.log(numeric, step=step)
    finally:
        run.finish()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    module = load_compact_module(args.compact_script)
    payload, val_loss, val_bpb, config = evaluate_checkpoint(
        module=module,
        checkpoint_path=checkpoint,
        token_glob=args.token_glob,
        tokenizer_path=args.tokenizer_path,
        device=device,
        seq_len=args.seq_len,
        val_batch_size=args.val_batch_size,
        val_max_sequences=args.val_max_sequences,
        precision=args.precision,
        config_kwargs={
            "logit_softcap": args.logit_softcap,
            "rope_base": args.rope_base,
            "qk_gain_init": args.qk_gain_init,
            "bigram_bias_scale": args.bigram_bias_scale,
            "hash_ngram_bias_order": args.hash_ngram_bias_order,
            "hash_ngram_bias_scale": args.hash_ngram_bias_scale,
            "polarquant_kv_bits": args.polarquant_kv_bits,
            "polarquant_eval_sample_tokens": args.polarquant_eval_sample_tokens,
            "polarquant_seed": args.polarquant_seed,
        },
    )
    summary = oai_metric_summary(
        checkpoint=checkpoint,
        step=checkpoint_step(payload),
        val_loss=val_loss,
        val_bpb=val_bpb,
        seq_len=args.seq_len,
        val_max_sequences=args.val_max_sequences,
        token_glob=args.token_glob,
        tokenizer_path=args.tokenizer_path,
        config=config,
    )
    if not math.isfinite(float(summary["oai_competition/bpb"])):
        raise RuntimeError(f"non-finite Seq4096 competition BPB: {summary['oai_competition/bpb']}")
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    log_to_wandb(args.wandb_run_path, summary, step=int(summary["step"]))


if __name__ == "__main__":
    main()
