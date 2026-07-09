#!/usr/bin/env python3
"""Export an embedding-space Forest-of-Thought trace.

This utility is the inference/reporting companion to the OAI FoT training head.
It can either:

* load a ToricBLM/OAI checkpoint and run a prompt through the adapted baseline,
  then emit the learned FoT trace; or
* run a deterministic smoke trace over synthetic hidden states to validate the
  forest topology without needing a large checkpoint.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
PARAMETER_GOLF = ROOT / "amelie-iska" / "parameter-golf"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(PARAMETER_GOLF) not in sys.path:
    sys.path.insert(0, str(PARAMETER_GOLF))

from toricgt.embedding_forest_of_thought import EmbeddingFoTConfig, EmbeddingForestOfThoughtHead


def load_train_gpt_module() -> Any:
    module_path = PARAMETER_GOLF / "train_gpt.py"
    spec = importlib.util.spec_from_file_location("toricblm_parameter_golf_train_gpt", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hp(hparams: dict[str, Any], key: str, default: Any) -> Any:
    return hparams.get(key, default)


def build_model(train_gpt: Any, hparams: dict[str, Any]) -> torch.nn.Module:
    return train_gpt.GPT(
        vocab_size=int(hp(hparams, "vocab_size", 8192)),
        num_layers=int(hp(hparams, "num_layers", 12)),
        model_dim=int(hp(hparams, "model_dim", 896)),
        num_heads=int(hp(hparams, "num_heads", 14)),
        num_kv_heads=int(hp(hparams, "num_kv_heads", 7)),
        mlp_mult=int(hp(hparams, "mlp_mult", 2)),
        tie_embeddings=bool(hp(hparams, "tie_embeddings", True)),
        tied_embed_init_std=float(hp(hparams, "tied_embed_init_std", 0.04)),
        logit_softcap=float(hp(hparams, "logit_softcap", 30.0)),
        rope_base=float(hp(hparams, "rope_base", 10000.0)),
        qk_gain_init=float(hp(hparams, "qk_gain_init", 1.5)),
        fineweb_graphify=bool(hp(hparams, "fineweb_graphify", True)),
        tokengt_first_class=bool(hp(hparams, "tokengt_first_class", True)),
        tokengt_graph_radius=int(hp(hparams, "tokengt_graph_radius", 4)),
        tokengt_distance_features=str(hp(hparams, "tokengt_distance_features", "functional")),
        tokengt_token_class_buckets=int(hp(hparams, "tokengt_token_class_buckets", 128)),
        tokengt_position_buckets=int(hp(hparams, "tokengt_position_buckets", 256)),
        tokengt_structural_weight=float(hp(hparams, "tokengt_structural_weight", 0.052)),
        tokengt_edge_weight=float(hp(hparams, "tokengt_edge_weight", 0.038)),
        tokengt_torus_weight=float(hp(hparams, "tokengt_torus_weight", 0.014)),
        tokengt_identifier_dim=int(hp(hparams, "tokengt_identifier_dim", 24)),
        tokengt_identifier_weight=float(hp(hparams, "tokengt_identifier_weight", 0.014)),
        tokengt_endpoint_weight=float(hp(hparams, "tokengt_endpoint_weight", 0.018)),
        tokengt_edge_token_weight=float(hp(hparams, "tokengt_edge_token_weight", 0.016)),
        convextok_dag_features=bool(hp(hparams, "convextok_dag_features", True)),
        convextok_dag_feature_weight=float(hp(hparams, "convextok_dag_feature_weight", 0.024)),
        graph_output_flattening=bool(hp(hparams, "graph_output_flattening", False)),
        graph_output_edge_radius=int(hp(hparams, "graph_output_edge_radius", 4)),
        graph_output_distance_features=str(hp(hparams, "graph_output_distance_features", "functional")),
        graph_output_node_weight=float(hp(hparams, "graph_output_node_weight", 0.0)),
        graph_output_edge_weight=float(hp(hparams, "graph_output_edge_weight", 0.0)),
        graph_output_virtual_edge_tokens=bool(hp(hparams, "graph_output_virtual_edge_tokens", False)),
        graph_output_edge_token_weight=float(hp(hparams, "graph_output_edge_token_weight", 0.0)),
        graph_output_score_correction=bool(hp(hparams, "graph_output_score_correction", False)),
        graph_output_score_correction_weight=float(hp(hparams, "graph_output_score_correction_weight", 0.0)),
        toricblm_mup=bool(hp(hparams, "toricblm_mup", False)),
        mup_width_mult=float(hp(hparams, "mup_width_mult", 1.0)),
        mup_output_mult=float(hp(hparams, "mup_output_mult", 1.0)),
        mup_attention_scale=bool(hp(hparams, "mup_attention_scale", False)),
        mup_attention_base_head_dim=float(hp(hparams, "mup_attention_base_head_dim", 64.0)),
    )


def build_fot_head(hparams: dict[str, Any]) -> EmbeddingForestOfThoughtHead:
    return EmbeddingForestOfThoughtHead(
        EmbeddingFoTConfig(
            dim=int(hp(hparams, "model_dim", 896)),
            num_trees=int(hp(hparams, "oai_fot_num_trees", 5)),
            max_depth=int(hp(hparams, "oai_fot_max_depth", 6)),
            branching=int(hp(hparams, "oai_fot_branching", 4)),
            topk_trees=int(hp(hparams, "oai_fot_topk_trees", 2)),
            hidden_dim=int(hp(hparams, "oai_fot_hidden_dim", 192)),
            max_positions=int(hp(hparams, "oai_fot_max_positions", 192)),
            consensus_buckets=int(hp(hparams, "oai_fot_consensus_buckets", 64)),
            correction_scale=float(hp(hparams, "oai_fot_correction_scale", 0.08)),
            ucb_exploration=float(hp(hparams, "oai_fot_ucb_exploration", 1.25)),
            temperature=float(hp(hparams, "oai_fot_temperature", 0.70)),
            sparse_weight=float(hp(hparams, "oai_fot_sparse_weight", 1.0)),
            ucb_weight=float(hp(hparams, "oai_fot_ucb_weight", 0.5)),
            correction_weight=float(hp(hparams, "oai_fot_correction_weight", 0.5)),
            consensus_weight=float(hp(hparams, "oai_fot_consensus_weight", 0.75)),
            tb_weight=float(hp(hparams, "oai_fot_tb_weight", 1.0)),
            subtb_weight=float(hp(hparams, "oai_fot_subtb_weight", 0.0)),
            complexity_weight=float(hp(hparams, "oai_fot_complexity_weight", 0.05)),
            reward_mode=str(hp(hparams, "oai_fot_reward_mode", "bpb_delta")),
            bpb_delta_weight=float(hp(hparams, "oai_fot_bpb_delta_weight", 1.0)),
            reward_graph_weight=float(hp(hparams, "oai_fot_reward_graph_weight", 0.14)),
            reward_consensus_weight=float(hp(hparams, "oai_fot_reward_consensus_weight", 0.24)),
            reward_complexity_weight=float(hp(hparams, "oai_fot_reward_complexity_weight", 0.02)),
            reward_floor=float(hp(hparams, "oai_fot_reward_floor", 1.0e-4)),
        )
    )


def encode_prompt(tokenizer_path: str, prompt: str) -> list[int]:
    if tokenizer_path.endswith(".json"):
        from convextok import ConvexTokTokenizer

        tokenizer = ConvexTokTokenizer.load_json(tokenizer_path)
        return [int(x) for x in tokenizer.encode(prompt)]
    import sentencepiece as spm

    tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
    return [int(x) for x in tokenizer.encode(prompt, out_type=int)]


def smoke_trace(output: Path) -> None:
    torch.manual_seed(7)
    hparams = {"model_dim": 48, "oai_fot_num_trees": 4, "oai_fot_max_depth": 5, "oai_fot_branching": 3, "oai_fot_max_positions": 60}
    head = build_fot_head(hparams)
    hidden = torch.randn(1, 96, 48)
    target = torch.arange(96).view(1, -1) % 8192
    nll = torch.linspace(0.5, 2.5, 96).view(1, -1)
    trace = head.trace_payload(hidden, target, nll)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(trace, indent=2, sort_keys=True), encoding="utf-8")
    print(f"fot_trace_smoke_written:{output} nodes:{trace['node_count']} edges:{trace['edge_count']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--tokenizer")
    parser.add_argument("--prompt", default="Design a thermostable oxidoreductase and explain the catalytic tradeoffs.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_json)
    if args.smoke:
        smoke_trace(output)
        return
    if not args.checkpoint or not args.tokenizer:
        raise SystemExit("--checkpoint and --tokenizer are required unless --smoke is used")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    hparams = dict(ckpt.get("hyperparameters", {})) if isinstance(ckpt, dict) else {}
    train_gpt = load_train_gpt_module()
    device = torch.device(args.device)
    model = build_model(train_gpt, hparams).to(device)
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt, strict=True)
    model.eval()
    fot = build_fot_head(hparams).to(device)
    if isinstance(ckpt, dict) and ckpt.get("oai_fot") is not None:
        fot.load_state_dict(ckpt["oai_fot"], strict=True)
    fot.eval()

    token_ids = encode_prompt(args.tokenizer, args.prompt)
    if len(token_ids) < 3:
        raise SystemExit("Prompt encoded to fewer than three tokens")
    x = torch.tensor(token_ids[:-1], dtype=torch.long, device=device).view(1, -1)
    y = torch.tensor(token_ids[1:], dtype=torch.long, device=device).view(1, -1)
    with torch.no_grad():
        _, hidden, nll = model.forward_aux(x, y, flatten_graph_output=True)
        trace = fot.trace_payload(hidden, y, nll)
    trace["prompt"] = args.prompt
    trace["checkpoint"] = str(args.checkpoint)
    trace["tokenizer"] = str(args.tokenizer)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(trace, indent=2, sort_keys=True), encoding="utf-8")
    print(f"fot_trace_written:{output} nodes:{trace['node_count']} edges:{trace['edge_count']}")


if __name__ == "__main__":
    main()
