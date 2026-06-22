#!/usr/bin/env python3
"""Generate muP base shapes and a ToricBLM scale-up env config."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TRAIN_GPT = ROOT / "amelie-iska" / "parameter-golf" / "train_gpt.py"
MUP_ROOT = ROOT / "external" / "mup"

if MUP_ROOT.exists():
    sys.path.insert(0, str(MUP_ROOT))


def load_train_module():
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "amelie-iska" / "parameter-golf"))
    if MUP_ROOT.exists():
        sys.path.insert(0, str(MUP_ROOT))
    spec = importlib.util.spec_from_file_location("toricblm_train_gpt_for_mup", TRAIN_GPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {TRAIN_GPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model(
    module,
    *,
    vocab_size: int,
    dim: int,
    layers: int,
    heads: int,
    kv_heads: int,
    mup: bool,
    width_mult: float,
):
    return module.GPT(
        vocab_size=vocab_size,
        num_layers=layers,
        model_dim=dim,
        num_heads=heads,
        num_kv_heads=kv_heads,
        mlp_mult=2,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
        fineweb_graphify=True,
        tokengt_first_class=True,
        tokengt_graph_radius=4,
        tokengt_distance_features="functional",
        tokengt_token_class_buckets=128,
        tokengt_position_buckets=512,
        tokengt_structural_weight=0.040,
        tokengt_edge_weight=0.030,
        tokengt_torus_weight=0.010,
        tokengt_identifier_dim=32,
        tokengt_identifier_weight=0.012,
        tokengt_endpoint_weight=0.016,
        tokengt_edge_token_weight=0.014,
        convextok_dag_features=True,
        convextok_dag_feature_weight=0.020,
        graph_output_flattening=True,
        graph_output_edge_radius=4,
        graph_output_distance_features="functional",
        graph_output_node_weight=0.045,
        graph_output_edge_weight=0.045,
        graph_output_virtual_edge_tokens=True,
        graph_output_edge_token_weight=0.028,
        graph_output_score_correction=True,
        graph_output_score_correction_weight=0.020,
        toricblm_mup=mup,
        mup_width_mult=width_mult,
        mup_output_mult=1.0,
        mup_attention_scale=mup,
        mup_attention_base_head_dim=64.0,
    )


def count_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dim", type=int, default=512)
    parser.add_argument("--delta-dim", type=int, default=768)
    parser.add_argument("--target-dim", type=int, default=1536)
    parser.add_argument("--layers", type=int, default=9)
    parser.add_argument("--base-heads", type=int, default=8)
    parser.add_argument("--delta-heads", type=int, default=12)
    parser.add_argument("--target-heads", type=int, default=12)
    parser.add_argument("--base-kv-heads", type=int, default=4)
    parser.add_argument("--delta-kv-heads", type=int, default=6)
    parser.add_argument("--target-kv-heads", type=int, default=6)
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument(
        "--fineweb-root",
        default=(
            "/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/data/toricgt/"
            "parameter_golf_convextok2048_det_full"
        ),
    )
    parser.add_argument("--tokenizer-filename", default="fineweb_convextok_2048_det.convextok.json")
    parser.add_argument("--dataset-name", default="fineweb10B_convextok2048_det")
    parser.add_argument("--train-batch-tokens", type=int, default=262144)
    parser.add_argument("--tokenizer-label", default="convextok2048_det")
    parser.add_argument("--output-dir", default=str(ROOT / "configs" / "mup"))
    parser.add_argument("--env-path", default=str(ROOT / "configs" / "toricblm_mup_170m_codex55.env"))
    parser.add_argument("--late-start-step", type=int, default=17500)
    args = parser.parse_args()

    from mup import set_base_shapes

    module = load_train_module()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bsh_path = output_dir / (
        f"toricblm_{args.tokenizer_label}_base{args.base_dim}_delta{args.delta_dim}_layers{args.layers}.bsh"
    )
    base = build_model(
        module,
        vocab_size=args.vocab_size,
        dim=args.base_dim,
        layers=args.layers,
        heads=args.base_heads,
        kv_heads=args.base_kv_heads,
        mup=False,
        width_mult=1.0,
    )
    delta = build_model(
        module,
        vocab_size=args.vocab_size,
        dim=args.delta_dim,
        layers=args.layers,
        heads=args.delta_heads,
        kv_heads=args.delta_kv_heads,
        mup=False,
        width_mult=args.delta_dim / args.base_dim,
    )
    target_width_mult = args.target_dim / args.base_dim
    target = build_model(
        module,
        vocab_size=args.vocab_size,
        dim=args.target_dim,
        layers=args.layers,
        heads=args.target_heads,
        kv_heads=args.target_kv_heads,
        mup=True,
        width_mult=target_width_mult,
    )
    set_base_shapes(target, base, delta=delta, savefile=str(bsh_path), rescale_params=False)
    params = {
        "base_params": count_params(base),
        "delta_params": count_params(delta),
        "target_params": count_params(target),
        "base_dim": args.base_dim,
        "delta_dim": args.delta_dim,
        "target_dim": args.target_dim,
        "layers": args.layers,
        "target_width_mult": target_width_mult,
        "base_shapes": str(bsh_path),
        "vocab_size": int(args.vocab_size),
        "tokenizer_label": str(args.tokenizer_label),
    }
    env_path = Path(args.env_path)
    env_path.parent.mkdir(parents=True, exist_ok=True)
    fineweb_root = str(Path(args.fineweb_root).expanduser().resolve())
    tokenizer_path = str(Path(fineweb_root) / "tokenizers" / str(args.tokenizer_filename))
    dataset_path = str(Path(fineweb_root) / "datasets" / str(args.dataset_name))
    late_root = (
        "/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/data/toricgt/"
        "codex55_tot_late_stage"
    )
    env_lines = [
        "# ToricBLM mu-transfer scale-up config generated by scripts/generate_toricblm_mup_config.py",
        f"# target_params={params['target_params']}",
        f"export TORICBLM_MUP=1",
        f"export MUP_BASE_SHAPES={bsh_path}",
        f"export MUP_WIDTH_MULT={target_width_mult:.8f}",
        "export MUP_OUTPUT_MULT=1.0",
        "export MUP_ATTENTION_SCALE=1",
        "export MUP_ATTENTION_BASE_HEAD_DIM=64",
        "export MUP_RESCALE_PARAMS=1",
        "export MUP_MATRIX_LR_SCALE_POWER=1.0",
        "export MUP_SCALAR_LR_SCALE_POWER=0.0",
        "export MUP_TOKEN_LR_SCALE_POWER=0.0",
        "export MUP_AUX_LR_SCALE_POWER=0.5",
        f"export MODEL_DIM={args.target_dim}",
        f"export NUM_LAYERS={args.layers}",
        f"export NUM_HEADS={args.target_heads}",
        f"export NUM_KV_HEADS={args.target_kv_heads}",
        "export MLP_MULT=2",
        f"export VOCAB_SIZE={int(args.vocab_size)}",
        "export TIE_EMBEDDINGS=1",
        f"export TOKENIZER_PATH={tokenizer_path}",
        f"export DATA_PATH={dataset_path}",
        "export TRAIN_SEQ_LEN=1024",
        f"export TRAIN_BATCH_TOKENS={int(args.train_batch_tokens)}",
        "export ITERATIONS=25000",
        "export MAX_WALLCLOCK_SECONDS=0",
        "export WARMUP_STEPS=20",
        "export WARMDOWN_ITERS=9000",
        "export VAL_LOSS_EVERY=1000",
        "export CHECKPOINT_EVERY=1000",
        "export VAL_MAX_TOKENS=8388608",
        "export MATRIX_LR=0.028",
        "export SCALAR_LR=0.028",
        "export TIED_EMBED_LR=0.030",
        "export TOKENGT_FIRST_CLASS_LR=0.00018",
        "export GRAPH_OUTPUT_FLATTENING_LR=0.00018",
        "export FINEWEB_GRAPHIFY=1",
        "export TOKENGT_FIRST_CLASS=1",
        "export TOKENGT_GRAPH_RADIUS=4",
        "export TOKENGT_TOKEN_CLASS_BUCKETS=128",
        "export TOKENGT_POSITION_BUCKETS=512",
        "export CONVEXTOK_DAG_FEATURES=1",
        "export CONVEXTOK_DAG_FEATURE_WEIGHT=0.024",
        "export GRAPH_OUTPUT_FLATTENING=1",
        "export GRAPH_OUTPUT_EDGE_RADIUS=4",
        "export GRAPH_OUTPUT_CALIBRATION_LOSS_WEIGHT=0.010",
        "export GRAPH_OUTPUT_CALIBRATION_EVERY=25",
        "export TORICGT_SIDECAR=1",
        "export REQUIRE_TORICGT_SIDECAR=1",
        "export GRAPH_LM_PRIMARY=1",
        "export REQUIRE_GRAPH_LM_PRIMARY=1",
        "export GRAPH_LM_EVERY=4",
        "export GRAPH_LM_BATCH_SIZE=2",
        "export GRAPH_LM_SEQ_LEN=1024",
        "export GRAPH_LM_LOSS_WEIGHT_START=0.020",
        "export GRAPH_LM_LOSS_WEIGHT=0.075",
        "export TORICGT_SIDECAR_EVERY=4",
        "export TORICGT_SIDECAR_BATCH_SIZE=4",
        "export TORICGT_SIDECAR_SEQ_LEN=256",
        "export GRAPHCG_LOSS_WEIGHT=0.0000025",
        "export ANALOGY_LOSS_WEIGHT=0.000025",
        "export TOKENGT_GRAPH_LOSS_WEIGHT=0.000035",
        "export TRAJECTORY_MEMORY_LOSS_WEIGHT=0.00003",
        "export TORIC_GEOMETRY_LOSS_WEIGHT=0.0000025",
        "export TORIC_VECTOR_BUNDLE_LOSS_WEIGHT=0.0000012",
        "export TORIC_BGG_LOSS_WEIGHT=0.0000008",
        "export KOSZUL_PERSISTENCE_LOSS_WEIGHT=0.0000008",
        "export COMBINATORIAL_TORIC_LOSS_WEIGHT=0.0000005",
        "export DERIVED_SIGNATURE_LOSS_WEIGHT=0.0000003",
        "export OAI_GFLOWNET=1",
        "export OAI_GFLOWNET_EVERY=4",
        "export OAI_GFLOWNET_LOSS_WEIGHT=0.000025",
        "export OAI_GFLOWNET_ENTROPY_WEIGHT=0.000002",
        "export OAI_EMBEDDING_FOT=1",
        "export OAI_FOT_EVERY=4",
        "export OAI_FOT_LOSS_WEIGHT=0.000012",
        "export OAI_FOT_NUM_TREES=4",
        "export OAI_FOT_MAX_DEPTH=5",
        "export OAI_FOT_BRANCHING=4",
        "export OAI_FOT_REWARD_TARGET=0.040",
        "export OAI_FOT_DIVERSITY_TARGET=0.060",
        "export OAI_MTP=1",
        "export OAI_MTP_EVERY=4",
        "export OAI_MTP_LOSS_WEIGHT=0.0015",
        "export OAI_MTP_OFFSETS=2",
        "export AUX_GRAD_ROUTING=1",
        "export BPB_FIRST_AUX_STAGING=1",
        "export BPB_FIRST_CORE_STEPS=500",
        "export BPB_FIRST_RAMP_STEPS=1500",
        "export ADVANCED_LAGRANGIAN_CONTROLLER=1",
        "export GRAPH_DATA_PATH=/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/data/toricgt/curated_hf_shards",
        f"export LATE_GRAPH_TRAIN_GLOB={late_root}/train_3pass/*.parquet",
        f"export LATE_GRAPH_START_STEP={args.late_start_step}",
        "export LATE_GRAPH_MIX_RATIO=0.85",
        "export LATE_GRAPH_UPSAMPLE_PASSES=3",
        "export TORICBLM_TOKENIZER_FAMILY=convextok",
        f"export TORICBLM_TOKENIZER_LABEL={args.tokenizer_label}",
    ]
    env_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
    print(json.dumps({"base_shapes": str(bsh_path), "env_path": str(env_path), **params}, indent=2))


if __name__ == "__main__":
    main()
