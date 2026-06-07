#!/usr/bin/env python3
"""Evaluate ToricGT with simple inference-time scaling budgets."""

from __future__ import annotations

import argparse
import glob
import os
import json
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from toricgt.cli_config import flatten_cli_config, load_yaml_config, parse_config_path
from toricgt.config import ModelConfig
from toricgt.datasets import TEST_TIME_SCALING_DATASETS
from toricgt.graph_dataset import CuratedGraphIterableDataset, collate_graph_items
from toricgt.graph_tokenizer import GraphBatch
from toricgt.metrics import masked_mse
from toricgt.model import ToricTokenGT


def apply_known_yaml_defaults(parser: argparse.ArgumentParser, path: str | Path | None) -> None:
    """Apply only this script's keys from a broader training YAML."""

    payload = flatten_cli_config(load_yaml_config(path)) if path is not None else {}
    if not payload:
        return
    # The trainer's full YAML contains train/validation paths. For this script,
    # held-out test data should be selected explicitly or by the test-time
    # scaling defaults, never inherited from the training split.
    payload.pop("data_path", None)
    payload.pop("val_data_path", None)
    actions = {action.dest for action in parser._actions if action.dest != "help"}
    parser.set_defaults(**{key: value for key, value in payload.items() if key in actions})


def synthetic_batch(cfg: ModelConfig, batch_size: int, device: str) -> tuple[GraphBatch, torch.Tensor]:
    n = min(16, cfg.max_nodes)
    e = min(48, cfg.max_edges)
    node = torch.randn(batch_size, n, cfg.node_feature_dim, device=device)
    edge = torch.randn(batch_size, e, cfg.edge_feature_dim, device=device)
    edge_index = torch.randint(0, n, (batch_size, e, 2), device=device)
    node_mask = torch.ones(batch_size, n, dtype=torch.bool, device=device)
    edge_mask = torch.ones(batch_size, e, dtype=torch.bool, device=device)
    target = torch.zeros(batch_size, n, cfg.output_dim, device=device)
    copy_dim = min(cfg.output_dim, cfg.node_feature_dim)
    target[..., :copy_dim] = node[..., :copy_dim]
    return GraphBatch(node, edge, edge_index, node_mask, edge_mask), target


def move_batch(batch: GraphBatch, target: torch.Tensor, device: str) -> tuple[GraphBatch, torch.Tensor]:
    return (
        GraphBatch(
            node_features=batch.node_features.to(device),
            edge_features=batch.edge_features.to(device),
            edge_index=batch.edge_index.to(device),
            node_mask=batch.node_mask.to(device),
            edge_mask=batch.edge_mask.to(device),
            lm_input_ids=batch.lm_input_ids.to(device) if batch.lm_input_ids is not None else None,
            lm_target_ids=batch.lm_target_ids.to(device) if batch.lm_target_ids is not None else None,
            lm_mask=batch.lm_mask.to(device) if batch.lm_mask is not None else None,
        ),
        target.to(device),
    )


def pooled_embeddings(out: dict[str, torch.Tensor]) -> torch.Tensor:
    token_mask = out["token_mask"]
    token_embeddings = out["token_embeddings"]
    denom = token_mask.sum(dim=1, keepdim=True).clamp_min(1).to(token_embeddings.dtype)
    return (token_embeddings * token_mask.unsqueeze(-1).to(token_embeddings.dtype)).sum(dim=1) / denom


def sample_gflownet_stats(
    model: ToricTokenGT,
    states: torch.Tensor,
    rollouts: int,
    horizon: int,
    temperature: float,
) -> dict[str, float]:
    if model.gflownet_policy is None or rollouts <= 0 or horizon <= 0:
        return {
            "policy_entropy": 0.0,
            "action_diversity": 0.0,
            "mean_trajectory_logprob": 0.0,
        }
    entropies: list[torch.Tensor] = []
    logprobs: list[torch.Tensor] = []
    sampled_actions: list[torch.Tensor] = []
    for _ in range(rollouts):
        current = states
        rollout_logprob = states.new_zeros(states.shape[0])
        actions_for_rollout: list[torch.Tensor] = []
        for step_idx in range(horizon):
            logits, _ = model.gflownet_policy(current)
            scaled = logits / max(temperature, 1e-6)
            dist = torch.distributions.Categorical(logits=scaled)
            action = dist.sample()
            rollout_logprob = rollout_logprob + dist.log_prob(action)
            entropies.append(dist.entropy())
            actions_for_rollout.append(action)
            # Lightweight deterministic state perturbation from the sampled action.
            # This is an inference-time search diagnostic, not a learned transition model.
            current = current + 0.01 * torch.sin((action.float().unsqueeze(-1) + 1.0) * (step_idx + 1.0))
        logprobs.append(rollout_logprob)
        sampled_actions.append(torch.stack(actions_for_rollout, dim=1))
    action_tensor = torch.stack(sampled_actions, dim=1)
    diversity = []
    for batch_idx in range(action_tensor.shape[0]):
        unique = torch.unique(action_tensor[batch_idx].reshape(-1)).numel()
        diversity.append(unique / max(model.config.gflownet_num_actions, 1))
    return {
        "policy_entropy": float(torch.cat(entropies).mean().detach().cpu()),
        "action_diversity": float(torch.tensor(diversity).mean().detach().cpu()),
        "mean_trajectory_logprob": float(torch.stack(logprobs).mean().detach().cpu()),
    }


def expanded_data_paths(raw_paths: list[str] | str | None) -> list[str] | None:
    if raw_paths is None:
        return None
    values = [raw_paths] if isinstance(raw_paths, str) else list(raw_paths)
    paths: list[str] = []
    for value in values:
        if any(token in value for token in ("*", "?", "[")):
            matches = sorted(glob.glob(value))
            paths.extend(matches or [value])
        else:
            paths.append(value)
    return paths


def make_loader(args: argparse.Namespace, cfg: ModelConfig):
    if args.data_path is None:
        return None
    dataset = CuratedGraphIterableDataset(
        args.data_path,
        cfg,
        parquet_batch_size=args.parquet_batch_size,
        interleave_paths=True,
    )
    return DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_graph_items, num_workers=0)


def default_new_eval_path(args: argparse.Namespace) -> str | None:
    if args.synthetic:
        return None
    if args.data_path is not None:
        return expanded_data_paths(args.data_path)
    eval_dir = Path(args.new_eval_dir)
    test_path = eval_dir / "test.parquet"
    if test_path.exists():
        return [str(test_path)]
    if args.no_auto_curate_new_data:
        raise FileNotFoundError(
            f"{test_path} does not exist. Remove --no-auto-curate-new-data or pass --data-path explicitly."
        )
    eval_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/curate_datasets.py",
        "--output-dir",
        str(eval_dir),
        "--raw-dir",
        args.raw_dir,
        "--chunk-size",
        str(args.curation_chunk_size),
        "--num-workers",
        str(args.curation_workers),
        "--normalize-batch-size",
        str(args.curation_normalize_batch_size),
    ]
    if args.new_data_max_records_per_source > 0:
        cmd.extend(["--max-records-per-source", str(args.new_data_max_records_per_source)])
    for dataset_name in TEST_TIME_SCALING_DATASETS:
        cmd.extend(["--dataset", dataset_name])
    env = os.environ.copy()
    env["PYTHONPATH"] = "src" if not env.get("PYTHONPATH") else f"src:{env['PYTHONPATH']}"
    print(
        json.dumps(
            {
                "event": "curating_default_test_time_scaling_data",
                "output_dir": str(eval_dir),
                "datasets": list(TEST_TIME_SCALING_DATASETS),
            },
            indent=2,
        )
    )
    subprocess.run(cmd, check=True, env=env)
    return [str(test_path)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="YAML file whose keys become CLI defaults.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--data-path",
        action="append",
        default=None,
        help="Curated test Parquet/JSONL path. Defaults to auto-curated OpenAI/NVIDIA held-out data.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--gflownet-horizon", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--mc-dropout", action=argparse.BooleanOptionalAction, default=True, help="Enable dropout during candidate generation.")
    parser.add_argument("--parquet-batch-size", type=int, default=512)
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic batches instead of the default new held-out datasets.")
    parser.add_argument("--new-eval-dir", default="data/test_time_scaling/new_external")
    parser.add_argument("--raw-dir", default="data/raw/hf")
    parser.add_argument("--no-auto-curate-new-data", action="store_true")
    parser.add_argument("--new-data-max-records-per-source", type=int, default=0, help="0 means uncapped.")
    parser.add_argument("--curation-workers", type=int, default=4)
    parser.add_argument("--curation-normalize-batch-size", type=int, default=256)
    parser.add_argument("--curation-chunk-size", type=int, default=5000)
    parser.add_argument("--output-json", default=None)
    apply_known_yaml_defaults(parser, parse_config_path())
    args = parser.parse_args()
    if args.checkpoint is None:
        parser.error("--checkpoint is required unless supplied by --config")
    used_default_new_eval = args.data_path is None and not args.synthetic
    args.data_path = default_new_eval_path(args)

    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    cfg = ModelConfig(**payload["config"])
    model = ToricTokenGT(cfg).to(args.device)
    model.load_state_dict(payload["model"])
    loader = make_loader(args, cfg)
    data_iter = iter(loader) if loader is not None else None

    results: list[dict[str, float | int]] = []
    max_budget = max(args.budgets)
    with torch.no_grad():
        for batch_idx in tqdm(range(args.batches), desc="test-time-scaling"):
            if loader is None:
                batch, target = synthetic_batch(cfg, args.batch_size, args.device)
            else:
                try:
                    batch, target = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    batch, target = next(data_iter)
                batch, target = move_batch(batch, target, args.device)

            candidate_losses: list[float] = []
            candidate_rewards: list[float] = []
            candidate_policy_stats: list[dict[str, float]] = []
            for _ in range(max_budget):
                model.train(args.mc_dropout)
                out = model(batch)
                loss = masked_mse(out["node"], target, batch.node_mask)
                states = pooled_embeddings(out)
                policy_stats = sample_gflownet_stats(
                    model,
                    states,
                    rollouts=1,
                    horizon=args.gflownet_horizon,
                    temperature=args.temperature,
                )
                reward = torch.exp(-loss.detach()).item()
                candidate_losses.append(float(loss.detach().cpu()))
                candidate_rewards.append(float(reward))
                candidate_policy_stats.append(policy_stats)

            for budget in args.budgets:
                losses = candidate_losses[:budget]
                rewards = candidate_rewards[:budget]
                stats = candidate_policy_stats[:budget]
                results.append(
                    {
                        "batch": batch_idx,
                        "budget": budget,
                        "mean_mse": float(sum(losses) / len(losses)),
                        "best_oracle_mse": float(min(losses)),
                        "mean_reward_proxy": float(sum(rewards) / len(rewards)),
                        "best_reward_proxy": float(max(rewards)),
                        "policy_entropy": float(sum(item["policy_entropy"] for item in stats) / len(stats)),
                        "action_diversity": float(sum(item["action_diversity"] for item in stats) / len(stats)),
                        "mean_trajectory_logprob": float(
                            sum(item["mean_trajectory_logprob"] for item in stats) / len(stats)
                        ),
                    }
                )

    summary: dict[str, dict[str, float]] = {}
    for budget in args.budgets:
        rows = [row for row in results if row["budget"] == budget]
        summary[str(budget)] = {
            key: float(sum(float(row[key]) for row in rows) / len(rows))
            for key in (
                "mean_mse",
                "best_oracle_mse",
                "mean_reward_proxy",
                "best_reward_proxy",
                "policy_entropy",
                "action_diversity",
                "mean_trajectory_logprob",
            )
        }
    payload_out = {
        "checkpoint": args.checkpoint,
        "data_path": args.data_path,
        "default_test_time_scaling_datasets": list(TEST_TIME_SCALING_DATASETS) if used_default_new_eval else [],
        "budgets": args.budgets,
        "summary": summary,
    }
    print(json.dumps(payload_out, indent=2))
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"summary": payload_out, "rows": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
