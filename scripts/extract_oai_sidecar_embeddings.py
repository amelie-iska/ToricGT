#!/usr/bin/env python3
"""Extract real OAI-baseline hidden-state payloads for ToricGT analyses.

The OAI Parameter-Golf baseline is a byte/token language model rather than the
native TokenGT graph encoder, so the periodic geometric/CAS stack needs a
bridge from an OAI checkpoint to the existing ``toricgt.embedding_payload.v1``
format.  This script loads the exact checkpoint, streams graph-structured
auxiliary Parquet rows through the same tokenizer family used by training
(SentencePiece or ConvexTok), runs ``forward_aux``, and writes hidden states,
per-token NLL, token text, and decoding-order edges.  No synthetic embeddings
are introduced.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import sentencepiece as spm
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from toricgt.oai_sidecar import GraphParquetTokenStream  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-gpt-path", default="amelie-iska/parameter-golf/train_gpt.py")
    parser.add_argument("--tokenizer-path", default="amelie-iska/parameter-golf/data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument(
        "--graph-train-glob",
        default="/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/data/toricgt/curated_hf_shards/train/*.parquet",
    )
    parser.add_argument("--records", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=9)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--mlp-mult", type=int, default=2)
    return parser.parse_args()


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def import_train_gpt(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("toricgt_oai_train_gpt_for_embedding_extract", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import train_gpt module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def pca3(points: np.ndarray) -> np.ndarray:
    x = np.asarray(points, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"PCA input must be rank-2, got {x.shape}")
    x = x - x.mean(axis=0, keepdims=True)
    if x.shape[0] < 2:
        return np.zeros((x.shape[0], 3), dtype=np.float32)
    _u, _s, vt = np.linalg.svd(x, full_matrices=False)
    k = min(3, vt.shape[0])
    projected = x @ vt[:k].T
    if k < 3:
        projected = np.pad(projected, ((0, 0), (0, 3 - k)))
    return projected.astype(np.float32)


def import_convextok() -> Any:
    path = ROOT / "amelie-iska" / "parameter-golf" / "convextok.py"
    spec = importlib.util.spec_from_file_location("toricgt_parameter_golf_convextok", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import ConvexTok from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_tokenizer(path: Path) -> tuple[Any, str, int]:
    if path.suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        if payload.get("tokenizer_type") == "convextok":
            convextok = import_convextok()
            tokenizer = convextok.ConvexTokTokenizer.load_json(path)
            return tokenizer, "convextok", int(tokenizer.vocab_size)
    tokenizer = spm.SentencePieceProcessor(model_file=str(path))
    return tokenizer, "sentencepiece", int(tokenizer.get_piece_size())


def token_piece(tokenizer: Any, tokenizer_kind: str, token_id: int) -> str:
    token_id = int(token_id)
    if tokenizer_kind == "convextok":
        try:
            data = tokenizer.token_bytes(token_id)
        except Exception:
            return f"<tok:{token_id}>"
        if not data:
            return f"<special:{token_id}>"
        text = data.decode("utf-8", errors="replace")
        if text.strip() and "\ufffd" not in text and all(ch.isprintable() or ch.isspace() for ch in text):
            return text.replace("\n", "\\n").replace("\t", "\\t")
        return "0x" + data.hex()
    try:
        return str(tokenizer.id_to_piece(token_id))
    except Exception:
        return f"<tok:{token_id}>"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def payload_edges(length: int) -> np.ndarray:
    if length <= 1:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray([[idx, idx + 1] for idx in range(length - 1)], dtype=np.int64)


def load_state(checkpoint_path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"], checkpoint
    if isinstance(checkpoint, dict):
        return checkpoint, {"schema": "raw_state_dict"}
    raise TypeError(f"checkpoint {checkpoint_path} did not contain a state dict")


def rows_minus_one(
    state: dict[str, torch.Tensor],
    key: str,
    default: int,
) -> int:
    tensor = state.get(key)
    if isinstance(tensor, torch.Tensor) and tensor.ndim >= 1:
        return max(0, int(tensor.shape[0]) - 1)
    return int(default)


def rows_of(
    state: dict[str, torch.Tensor],
    key: str,
    default: int,
) -> int:
    tensor = state.get(key)
    if isinstance(tensor, torch.Tensor) and tensor.ndim >= 1:
        return max(1, int(tensor.shape[0]))
    return int(default)


def infer_checkpoint_vocab_size(state: dict[str, torch.Tensor], default: int) -> int:
    preferred = (
        "embed.weight",
        "emb.weight",
        "token_embedding.weight",
        "tok_embeddings.weight",
        "wte.weight",
    )
    for key in preferred:
        tensor = state.get(key)
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
            return int(tensor.shape[0])
    candidates: list[int] = []
    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
            continue
        if "embed" in key.lower() and int(tensor.shape[0]) >= 260:
            candidates.append(int(tensor.shape[0]))
    if candidates:
        return max(candidates)
    return int(default)


def input_dim_of(
    state: dict[str, torch.Tensor],
    key: str,
    default: int,
) -> int:
    tensor = state.get(key)
    if isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
        return max(1, int(tensor.shape[1]))
    return int(default)


def checkpoint_meta_values(checkpoint_meta: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if isinstance(checkpoint_meta.get("config"), dict):
        values.update(checkpoint_meta["config"])
    if isinstance(checkpoint_meta.get("hyperparameters"), dict):
        values.update(checkpoint_meta["hyperparameters"])
    return values


def meta_int(meta_values: dict[str, Any], key: str, default: int) -> int:
    value = meta_values.get(key)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def meta_float(meta_values: dict[str, Any], key: str, default: float) -> float:
    value = meta_values.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def meta_mode(meta_values: dict[str, Any], key: str, default: str) -> str:
    value = str(meta_values.get(key, default)).strip().lower()
    return value if value in {"embedding", "functional"} else str(default)


def checkpoint_graph_config(
    state: dict[str, torch.Tensor],
    checkpoint_meta: dict[str, Any] | None = None,
) -> dict[str, int | str]:
    """Infer graphification module constructor dimensions from checkpoint tensors.

    ``load_state_dict(strict=False)`` still rejects shape mismatches, so the
    extractor must instantiate the OAI baseline with the same radius and table
    dimensions used by the checkpoint.  These values are structural: they are
    encoded directly in embedding-table row counts and projection input widths
    for older checkpoints, and in checkpoint hyperparameters for functional
    distance-feature checkpoints where no learned distance table is serialized.
    """

    meta_values = checkpoint_meta_values(checkpoint_meta or {})
    tokengt_distance_features = (
        "embedding"
        if "fineweb_tokengt.indegree_emb.weight" in state or "fineweb_tokengt.edge_distance_emb.weight" in state
        else meta_mode(meta_values, "tokengt_distance_features", "functional")
    )
    graph_output_distance_features = (
        "embedding"
        if "graph_output_flattening.edge_distance_emb.weight" in state
        else meta_mode(meta_values, "graph_output_distance_features", tokengt_distance_features)
    )
    tokengt_radius_default = meta_int(meta_values, "tokengt_graph_radius", 4)
    graph_output_radius_default = meta_int(
        meta_values,
        "graph_output_edge_radius",
        tokengt_radius_default,
    )
    tokengt_radius = rows_minus_one(state, "fineweb_tokengt.indegree_emb.weight", tokengt_radius_default)
    tokengt_edge_radius = rows_minus_one(state, "fineweb_tokengt.edge_distance_emb.weight", tokengt_radius)
    if tokengt_edge_radius != tokengt_radius:
        tokengt_radius = min(tokengt_radius, tokengt_edge_radius)
    graph_output_radius = rows_minus_one(
        state,
        "graph_output_flattening.edge_distance_emb.weight",
        graph_output_radius_default,
    )
    identifier_feature_dim = input_dim_of(state, "fineweb_tokengt.identifier_proj.weight", 24)
    endpoint_feature_dim = input_dim_of(state, "fineweb_tokengt.endpoint_proj.weight", 2 * identifier_feature_dim + 1)
    if endpoint_feature_dim > 1:
        inferred_identifier = max(1, (endpoint_feature_dim - 1) // 2)
        if inferred_identifier == identifier_feature_dim:
            identifier_feature_dim = inferred_identifier
    return {
        "tokengt_graph_radius": int(tokengt_radius),
        "graph_output_edge_radius": int(graph_output_radius),
        "tokengt_distance_features": tokengt_distance_features,
        "graph_output_distance_features": graph_output_distance_features,
        "tokengt_token_class_buckets": rows_of(state, "fineweb_tokengt.token_class_emb.weight", 64),
        "tokengt_position_buckets": rows_of(state, "fineweb_tokengt.position_bucket_emb.weight", 256),
        "tokengt_identifier_dim": int(identifier_feature_dim),
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve(args.checkpoint)
    output_dir = resolve(args.output_dir)
    embeddings_dir = output_dir / "embeddings"
    state, checkpoint_meta = load_state(checkpoint_path)
    train_gpt = import_train_gpt(resolve(args.train_gpt_path))
    tokenizer, tokenizer_kind, tokenizer_vocab_size = load_tokenizer(resolve(args.tokenizer_path))
    meta_values = checkpoint_meta_values(checkpoint_meta if isinstance(checkpoint_meta, dict) else {})
    vocab_size = max(
        int(args.vocab_size),
        int(tokenizer_vocab_size),
        infer_checkpoint_vocab_size(state, int(tokenizer_vocab_size)),
    )
    if vocab_size != int(args.vocab_size):
        print(f"embedding_extract:vocab_size_override requested={args.vocab_size} inferred={vocab_size} tokenizer_kind={tokenizer_kind}")
    stream = GraphParquetTokenStream(args.graph_train_glob, tokenizer, int(args.seq_len), int(args.batch_size))
    has_first_class_tokengt = any(key.startswith("fineweb_tokengt.") for key in state)
    has_graph_output_flattening = any(key.startswith("graph_output_flattening.") for key in state)
    graph_config = checkpoint_graph_config(state, checkpoint_meta)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = train_gpt.GPT(
        vocab_size=int(vocab_size),
        num_layers=int(args.num_layers),
        model_dim=int(args.model_dim),
        num_heads=int(args.num_heads),
        num_kv_heads=int(args.num_kv_heads),
        mlp_mult=int(args.mlp_mult),
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
        fineweb_graphify=has_first_class_tokengt or has_graph_output_flattening,
        tokengt_first_class=has_first_class_tokengt,
        tokengt_graph_radius=graph_config["tokengt_graph_radius"],
        tokengt_distance_features=str(graph_config["tokengt_distance_features"]),
        tokengt_token_class_buckets=graph_config["tokengt_token_class_buckets"],
        tokengt_position_buckets=graph_config["tokengt_position_buckets"],
        tokengt_structural_weight=0.050,
        tokengt_edge_weight=0.035,
        tokengt_torus_weight=0.015,
        tokengt_identifier_dim=graph_config["tokengt_identifier_dim"],
        tokengt_identifier_weight=0.014,
        tokengt_endpoint_weight=0.018,
        tokengt_edge_token_weight=0.016,
        convextok_dag_features=bool(has_first_class_tokengt and tokenizer_kind == "convextok"),
        convextok_dag_feature_weight=meta_float(meta_values, "convextok_dag_feature_weight", 0.026),
        graph_output_flattening=has_graph_output_flattening,
        graph_output_edge_radius=graph_config["graph_output_edge_radius"],
        graph_output_distance_features=str(graph_config["graph_output_distance_features"]),
        graph_output_node_weight=0.05,
        graph_output_edge_weight=0.05,
        graph_output_virtual_edge_tokens=True,
        graph_output_edge_token_weight=0.035,
        graph_output_score_correction=True,
        graph_output_score_correction_weight=0.024,
    ).to(device)
    if device.type == "cuda":
        model = model.bfloat16()
        for module in model.modules():
            if isinstance(module, train_gpt.CastedLinear):
                module.float()
    train_gpt.restore_low_dim_params_to_fp32(model)
    load_result = model.load_state_dict(state, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        print(
            "checkpoint_load_non_strict "
            f"missing={len(load_result.missing_keys)} unexpected={len(load_result.unexpected_keys)}"
        )
    if tokenizer_kind == "convextok" and hasattr(model, "set_convextok_features"):
        try:
            base_bytes, _space, _boundary, lp_scores, rank_scores, priced_flags = train_gpt.build_convextok_luts(
                tokenizer,
                int(vocab_size),
                device,
            )
            model.set_convextok_features(
                lp_scores=lp_scores,
                rank_scores=rank_scores,
                priced_flags=priced_flags,
                byte_lengths=base_bytes.to(dtype=torch.float32),
            )
        except Exception as exc:
            print(f"embedding_extract:convextok_lut_warning:{exc}")
    model.eval()

    rows: list[dict[str, Any]] = []
    written = 0
    batch_index = 0
    while written < int(args.records):
        x, y = stream.next_batch(device)
        with torch.no_grad():
            autocast_enabled = device.type == "cuda"
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                _loss, hidden, per_token_nll = model.forward_aux(x, y)
        hidden_np = hidden.detach().float().cpu().numpy()
        nll_np = per_token_nll.detach().float().cpu().numpy()
        tokens_np = y.detach().cpu().numpy()
        for row_index in range(hidden_np.shape[0]):
            if written >= int(args.records):
                break
            h = np.asarray(hidden_np[row_index], dtype=np.float32)
            nll = np.nan_to_num(np.asarray(nll_np[row_index], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            token_ids = np.asarray(tokens_np[row_index], dtype=np.int64)
            projected = pca3(h)
            edges = payload_edges(h.shape[0])
            energy = np.linalg.norm(h, axis=1).astype(np.float32)
            record_id = f"oai_ckpt_{checkpoint_path.stem}_batch{batch_index:03d}_row{row_index:02d}"
            npz_name = f"{record_id}.npz"
            json_name = f"{record_id}.json"
            npz_path = embeddings_dir / npz_name
            json_path = embeddings_dir / json_name
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                npz_path,
                hidden=h,
                projected=projected,
                energy=energy,
                nll=nll,
                edges=edges,
                complex_projected=projected,
                complex_nll=nll,
                complex_mass=(1.0 / np.maximum(1.0, nll)).astype(np.float32),
                complex_edges=edges,
                token_ids=token_ids,
            )
            token_rows = [
                {
                    "token_index": int(idx),
                    "token_id": int(token_ids[idx]),
                    "token_piece": token_piece(tokenizer, tokenizer_kind, int(token_ids[idx])),
                    "nll": float(nll[idx]) if math.isfinite(float(nll[idx])) else 0.0,
                    "decoding_parent": int(idx - 1) if idx > 0 else None,
                }
                for idx in range(int(token_ids.shape[0]))
            ]
            metadata = {
                "schema": "toricgt.embedding_payload.v1",
                "source": "oai_parameter_golf_forward_aux",
                "record_id": record_id,
                "checkpoint": str(checkpoint_path),
                "checkpoint_step": int(checkpoint_meta.get("step", -1)) if isinstance(checkpoint_meta, dict) else -1,
                "checkpoint_run_id": str(checkpoint_meta.get("run_id", "")) if isinstance(checkpoint_meta, dict) else "",
                "tokenizer_kind": tokenizer_kind,
                "tokenizer_path": str(resolve(args.tokenizer_path)),
                "vocab_size": int(vocab_size),
                "arrays": {
                    "hidden": list(h.shape),
                    "projected": list(projected.shape),
                    "nll": list(nll.shape),
                    "edges": list(edges.shape),
                },
                "decoding_order_edges_are_actual": True,
                "embedding_vectors_are_actual_checkpoint_hidden_states": True,
                "comparison_space": "original_hidden_vectors",
                "visualization_space": "PCA3_of_hidden_vectors",
                "checkpoint_graph_config": graph_config,
                "complex_rows": token_rows,
            }
            write_json(json_path, metadata)
            rows.append(
                {
                    "record_index": int(written),
                    "record_id": record_id,
                    "relative_npz": f"embeddings/{npz_name}",
                    "relative_json": f"embeddings/{json_name}",
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_step": metadata["checkpoint_step"],
                    "num_tokens": int(h.shape[0]),
                    "hidden_dim": int(h.shape[1]),
                    "mean_nll": float(np.mean(nll)),
                    "max_nll": float(np.max(nll)),
                }
            )
            written += 1
        batch_index += 1

    manifest = {
        "schema": "toricgt.embedding_payload_manifest.v1",
        "source": "oai_parameter_golf_forward_aux",
        "checkpoint": str(checkpoint_path),
        "checkpoint_graph_config": graph_config,
        "records": rows,
        "payloads": rows,
    }
    write_json(embeddings_dir / "manifest.json", manifest)
    print(json.dumps({"manifest": str(embeddings_dir / "manifest.json"), "records": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
