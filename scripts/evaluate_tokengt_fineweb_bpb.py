#!/usr/bin/env python3
"""Evaluate TokenGT LM-head checkpoints on FineWeb SP1024 graph-node loss.

This is the TokenGT checkpoint-family analogue of the OAI Parameter-Golf
FineWeb check.  It scores next-token SP1024 targets that are attached to
FineWeb graph vertices by ``CuratedGraphIterableDataset``.  With causal graph
attention and learned LM token embeddings enabled, the byte-normalized metric
uses the same SentencePiece byte accounting as the OAI compact evaluator.
Without those flags it remains a graph-node proxy diagnostic.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from toricgt.cli_config import flatten_cli_config, load_yaml_config  # noqa: E402
from toricgt.config import ModelConfig  # noqa: E402
from toricgt.graph_dataset import CuratedGraphIterableDataset, collate_graph_items  # noqa: E402
from toricgt.graph_tokenizer import GraphBatch  # noqa: E402
from toricgt.model import ToricTokenGT  # noqa: E402
from toricgt.wandb_organization import configure_wandb_metrics, organize_wandb_payload  # noqa: E402


DEFAULT_TOKEN_GLOB = "amelie-iska/parameter-golf/data/datasets/fineweb10B_sp1024/fineweb_val_*.bin"
DEFAULT_TOKENIZER = "amelie-iska/parameter-golf/data/tokenizers/fineweb_1024_bpe.model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="config/train.full_tokengt_got_fineweb_derived.yaml")
    parser.add_argument("--token-glob", default="")
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--wandb-run-path", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="fp32")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--fineweb-tokens-per-graph", type=int, default=0)
    parser.add_argument("--fineweb-stride-tokens", type=int, default=0)
    parser.add_argument("--parquet-batch-size", type=int, default=512)
    return parser.parse_args()


def config_get(flat: dict[str, Any], key: str, default: Any) -> Any:
    value = flat.get(key, default)
    return default if value is None else value


def load_flat_config(path: str | Path) -> dict[str, Any]:
    try:
        return flatten_cli_config(load_yaml_config(path))
    except Exception:
        return {}


def model_config_from_checkpoint(payload: dict[str, Any], flat_config: dict[str, Any]) -> ModelConfig:
    raw = payload.get("config") if isinstance(payload, dict) else None
    if not isinstance(raw, dict):
        raw = flat_config
    allowed = {field.name for field in fields(ModelConfig)}
    filtered = {key: value for key, value in dict(raw).items() if key in allowed}
    return ModelConfig(**filtered)


def autocast_context(device: str, precision: str):
    if not str(device).startswith("cuda") or precision == "fp32":
        return torch.no_grad()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def move_batch(batch: GraphBatch, device: str) -> GraphBatch:
    return GraphBatch(
        node_features=batch.node_features.to(device),
        edge_features=batch.edge_features.to(device),
        edge_index=batch.edge_index.to(device),
        node_mask=batch.node_mask.to(device),
        edge_mask=batch.edge_mask.to(device),
        lm_input_ids=batch.lm_input_ids.to(device) if batch.lm_input_ids is not None else None,
        lm_target_ids=batch.lm_target_ids.to(device) if batch.lm_target_ids is not None else None,
        lm_mask=batch.lm_mask.to(device) if batch.lm_mask is not None else None,
        lm_target_byte_lengths=batch.lm_target_byte_lengths.to(device)
        if batch.lm_target_byte_lengths is not None
        else None,
        lm_target_positions=batch.lm_target_positions.to(device) if batch.lm_target_positions is not None else None,
        lm_input_features=batch.lm_input_features.to(device) if batch.lm_input_features is not None else None,
        node_causal_rank=batch.node_causal_rank.to(device) if batch.node_causal_rank is not None else None,
    )


def token_paths(args: argparse.Namespace, flat: dict[str, Any]) -> list[str]:
    token_glob = args.token_glob or str(config_get(flat, "val_token_glob", DEFAULT_TOKEN_GLOB))
    matches = sorted(glob.glob(token_glob))
    return matches or [token_glob]


def unavailable_summary(checkpoint: Path, reason: str) -> dict[str, Any]:
    return {
        "checkpoint": str(checkpoint),
        "source": "tokengt_graph_node_sp1024_lm",
        "metric_namespace": "fineweb",
        "oai_competition/available": 0.0,
        "oai_competition/bpb_available": 0.0,
        "oai_competition/bpb": None,
        "oai_competition/loss": None,
        "fineweb/val_bpb": None,
        "fineweb/val_loss": None,
        "reason": reason,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    flat = load_flat_config(args.config)
    checkpoint_path = Path(args.checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = model_config_from_checkpoint(payload, flat)
    if not bool(cfg.use_lm_head):
        return unavailable_summary(checkpoint_path, "checkpoint config does not enable TokenGT LM head")
    model_state = payload.get("model", {})
    has_lm_projection = isinstance(model_state, dict) and (
        "lm_head.weight" in model_state
        or (cfg.tie_lm_head_to_token_embeddings and "lm_token_emb.weight" in model_state)
    )
    if not has_lm_projection:
        return unavailable_summary(checkpoint_path, "checkpoint state dict does not contain LM projection weights")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = ToricTokenGT(cfg)
    model.load_state_dict(model_state, strict=True)
    model.to(device)
    model.eval()

    dataset = CuratedGraphIterableDataset(
        token_paths(args, flat),
        cfg,
        parquet_batch_size=int(args.parquet_batch_size),
        interleave_paths=False,
        fineweb_tokenizer_path=args.tokenizer_path or str(config_get(flat, "fineweb_tokenizer_path", DEFAULT_TOKENIZER)),
        fineweb_tokens_per_graph=int(args.fineweb_tokens_per_graph or config_get(flat, "fineweb_tokens_per_graph", 1024)),
        fineweb_stride_tokens=int(args.fineweb_stride_tokens or config_get(flat, "fineweb_stride_tokens", 1024)),
    )
    batch_size = int(args.batch_size or config_get(flat, "batch_size", 2))
    loader = DataLoader(dataset, batch_size=max(1, batch_size), collate_fn=collate_graph_items, num_workers=0)

    total_nll = 0.0
    total_tokens = 0.0
    total_estimated_bytes = 0.0
    batches = 0
    max_batches = max(1, int(args.batches))
    with torch.no_grad():
        for batch, _target in loader:
            batch = move_batch(batch, device)
            if batch.lm_target_ids is None or batch.lm_mask is None:
                continue
            with autocast_context(device, args.precision):
                out = model(batch)
                logits = out.get("lm_logits")
                if logits is None:
                    continue
                mask = batch.lm_mask.to(device=logits.device, dtype=torch.bool) & batch.node_mask.to(
                    device=logits.device,
                    dtype=torch.bool,
                )
                if not bool(mask.any().detach().cpu()):
                    continue
                targets = batch.lm_target_ids.to(device=logits.device, dtype=torch.long).clamp(0, logits.shape[-1] - 1)
                loss = F.cross_entropy(logits.float()[mask], targets[mask])
                tokens = float(mask.sum().detach().cpu())
                if batch.lm_target_byte_lengths is not None:
                    byte_lengths = batch.lm_target_byte_lengths.to(device=logits.device, dtype=logits.float().dtype)[mask]
                    total_estimated_bytes += float(byte_lengths.clamp_min(0.0).sum().detach().cpu())
            total_nll += float(loss.detach().cpu()) * tokens
            total_tokens += tokens
            batches += 1
            if batches >= max_batches:
                break

    if total_tokens <= 0:
        return unavailable_summary(checkpoint_path, "no FineWeb LM target tokens were found during evaluation")
    loss_value = total_nll / total_tokens
    bits_per_token = loss_value / math.log(2.0)
    estimated_bpb = (total_nll / math.log(2.0)) / total_estimated_bytes if total_estimated_bytes > 0 else None
    mean_bytes_per_token = total_estimated_bytes / total_tokens if total_estimated_bytes > 0 else None
    step = int(payload.get("step", 0) or 0)
    causal_tokengt_bpb_available = bool(cfg.use_causal_graph_attention and cfg.use_lm_token_embeddings)
    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "step": step,
        "source": "tokengt_graph_node_sp1024_lm",
        "metric_namespace": "fineweb",
        "note": (
            "Causal TokenGT SP1024 next-token BPB with OAI SentencePiece byte accounting."
            if causal_tokengt_bpb_available
            else "TokenGT graph-node SP1024 next-token bits/token; byte-normalized BPB is a proxy unless causal graph attention and learned LM token embeddings are enabled."
        ),
        "oai_competition/available": 1.0 if causal_tokengt_bpb_available else 0.0,
        "oai_competition/bpb_available": 1.0 if causal_tokengt_bpb_available else 0.0,
        "oai_competition/bpb": float(estimated_bpb) if estimated_bpb is not None and causal_tokengt_bpb_available else None,
        "oai_competition/loss": float(loss_value) if causal_tokengt_bpb_available else None,
        "oai_competition/source_sp1024_graph_nodes": 1.0,
        "oai_competition/official_byte_level_bpb_available": 1.0 if causal_tokengt_bpb_available else 0.0,
        "oai_competition/eval_scope": (
            "causal_tokengt_graph_node_sp1024"
            if causal_tokengt_bpb_available
            else "tokengt_graph_node_sp1024_proxy"
        ),
        "oai_competition/eval_batches": float(batches),
        "oai_competition/eval_tokens": float(total_tokens),
        "tokengt/causal_graph_attention": 1.0 if cfg.use_causal_graph_attention else 0.0,
        "tokengt/learned_lm_token_embeddings": 1.0 if cfg.use_lm_token_embeddings else 0.0,
        "tokengt/tied_lm_head": 1.0 if cfg.tie_lm_head_to_token_embeddings else 0.0,
        "tokengt/recurrent_passes": float(cfg.recurrent_passes),
        "tokengt/lm_position_embeddings": 1.0 if cfg.use_lm_position_embeddings else 0.0,
        "tokengt/lm_toric_position_features": 1.0 if cfg.use_lm_toric_position_features else 0.0,
        "tokengt/lm_context_hash_embeddings": 1.0 if cfg.use_lm_context_hash_embeddings else 0.0,
        "tokengt/lm_revealed_neighbor_hash": 1.0 if cfg.use_lm_revealed_neighbor_hash else 0.0,
        "tokengt/lm_revealed_neighbor_hash_buckets": float(cfg.lm_revealed_neighbor_hash_buckets),
        "tokengt/lm_caseops_features": 1.0 if cfg.use_lm_caseops_features else 0.0,
        "tokengt/lm_smear_gate": 1.0 if cfg.use_lm_smear_gate else 0.0,
        "tokengt/fineweb_graph_node_sp1024_loss": float(loss_value),
        "tokengt/fineweb_graph_node_sp1024_bpt": float(bits_per_token),
        "tokengt/fineweb_graph_node_sp1024_tokens": float(total_tokens),
        "tokengt/restricted_fineweb_graph_node_sp1024_loss": float(loss_value),
        "tokengt/restricted_fineweb_graph_node_sp1024_bpt": float(bits_per_token),
        "tokengt/restricted_fineweb_graph_node_sp1024_tokens": float(total_tokens),
        "03_validation/tokengt_fineweb_graph_node_sp1024_loss": float(loss_value),
        "03_validation/tokengt_fineweb_graph_node_sp1024_bpt": float(bits_per_token),
        "03_validation/tokengt_fineweb_graph_node_sp1024_tokens": float(total_tokens),
        "03_validation/oai_parameter_golf_restricted_fineweb_official_byte_bpb_available": 0.0,
    }
    if estimated_bpb is not None and mean_bytes_per_token is not None:
        summary.update(
            {
                "tokengt/fineweb_graph_node_sp1024_estimated_bpb": float(estimated_bpb),
                "tokengt/fineweb_graph_node_sp1024_estimated_bytes": float(total_estimated_bytes),
                "tokengt/fineweb_graph_node_sp1024_mean_target_bytes_per_token": float(mean_bytes_per_token),
                "tokengt/restricted_fineweb_graph_node_sp1024_estimated_bpb": float(estimated_bpb),
                "tokengt/restricted_fineweb_graph_node_sp1024_estimated_bytes": float(total_estimated_bytes),
                "tokengt/restricted_fineweb_graph_node_sp1024_mean_target_bytes_per_token": float(
                    mean_bytes_per_token
                ),
                "03_validation/tokengt_fineweb_graph_node_sp1024_estimated_bpb": float(estimated_bpb),
                "03_validation/tokengt_fineweb_graph_node_sp1024_estimated_bytes": float(total_estimated_bytes),
                "03_validation/tokengt_fineweb_graph_node_sp1024_mean_target_bytes_per_token": float(
                    mean_bytes_per_token
                ),
                "03_validation/oai_parameter_golf_restricted_fineweb_causal_tokengt_bpb": (
                    float(estimated_bpb) if causal_tokengt_bpb_available else None
                ),
                "competition/oai_bpb": float(estimated_bpb) if causal_tokengt_bpb_available else None,
                "bpb/oai_competition": float(estimated_bpb) if causal_tokengt_bpb_available else None,
            }
        )
    return summary


def write_report(summary: dict[str, Any], output_json: Path) -> None:
    report = output_json.with_name("REPORT.md")
    report.write_text(
        "\n".join(
            [
                "# TokenGT FineWeb Graph-Node SP1024 Metrics",
                "",
                f"- Checkpoint: `{summary.get('checkpoint', 'n/a')}`",
                f"- Step: `{summary.get('step', 'n/a')}`",
                f"- Source: `{summary.get('source', 'n/a')}`",
                f"- Bits/token: `{summary.get('tokengt/fineweb_graph_node_sp1024_bpt', 'n/a')}`",
                f"- Estimated BPB: `{summary.get('tokengt/fineweb_graph_node_sp1024_estimated_bpb', 'n/a')}`",
                f"- Causal TokenGT OAI BPB: `{summary.get('03_validation/oai_parameter_golf_restricted_fineweb_causal_tokengt_bpb', 'n/a')}`",
                f"- Loss: `{summary.get('tokengt/fineweb_graph_node_sp1024_loss', 'n/a')}`",
                f"- Eval tokens: `{summary.get('tokengt/fineweb_graph_node_sp1024_tokens', 'n/a')}`",
                f"- Estimated bytes: `{summary.get('tokengt/fineweb_graph_node_sp1024_estimated_bytes', 'n/a')}`",
                f"- Official OAI byte BPB available: `{summary.get('oai_competition/official_byte_level_bpb_available', 0.0)}`",
                "",
                str(summary.get("note", summary.get("reason", ""))),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def log_to_wandb(run_path: str, summary: dict[str, Any]) -> None:
    if not run_path:
        return
    parts = [part for part in run_path.split("/") if part]
    if len(parts) != 3:
        raise ValueError(f"--wandb-run-path must be entity/project/run_id, got {run_path!r}")
    entity, project, run_id = parts
    import wandb

    run = wandb.init(entity=entity, project=project, id=run_id, resume="allow")
    try:
        configure_wandb_metrics(wandb)
        summary.setdefault("analysis_control/checkpoint_step", int(summary.get("step", 0) or 0))
        run.log(organize_wandb_payload(summary))
    finally:
        run.finish()


def main() -> None:
    args = parse_args()
    summary = evaluate(args)
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_report(summary, output_json)
    log_to_wandb(args.wandb_run_path, summary)
    if "tokengt/fineweb_graph_node_sp1024_bpt" in summary and not math.isfinite(
        float(summary["tokengt/fineweb_graph_node_sp1024_bpt"])
    ):
        raise RuntimeError(
            f"non-finite TokenGT FineWeb graph-node bits/token: {summary['tokengt/fineweb_graph_node_sp1024_bpt']}"
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
