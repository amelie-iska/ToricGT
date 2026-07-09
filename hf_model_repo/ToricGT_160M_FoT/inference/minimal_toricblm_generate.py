#!/usr/bin/env python3
"""Minimal ToricBLM/ToricGT ConvexTok checkpoint inference.

Run this from a local ToricGT checkout, or pass --toricgt-root.  The script
downloads a checkpoint and tokenizer from the Hugging Face model repository
unless local paths are supplied.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch


DEFAULT_REPO_ID = "AmelieSchreiber/ToricGT_160M_FoT"
DEFAULT_TOKENIZER_FILE = "tokenizers/fineweb_convextok_8192_biomed_det.convextok.json"


def _download_or_local(repo_id: str, value: str) -> str:
    path = Path(value).expanduser()
    if path.exists():
        return str(path)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "Install huggingface_hub or pass a local file path: "
            "python -m pip install huggingface_hub"
        ) from exc
    return hf_hub_download(repo_id=repo_id, filename=value)


def _load_train_module(toricgt_root: Path) -> Any:
    parameter_golf = toricgt_root / "amelie-iska" / "parameter-golf"
    src = toricgt_root / "src"
    train_gpt = parameter_golf / "train_gpt.py"
    if not train_gpt.exists():
        raise SystemExit(f"Could not find {train_gpt}. Pass --toricgt-root /path/to/ToricGT.")
    sys.path.insert(0, str(parameter_golf))
    sys.path.insert(0, str(src))
    spec = importlib.util.spec_from_file_location("toricgt_train_gpt", train_gpt)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not import {train_gpt}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hp(hparams: dict[str, Any], key: str, default: Any) -> Any:
    value = hparams.get(key, default)
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


def _build_model(train_mod: Any, hparams: dict[str, Any]) -> torch.nn.Module:
    return train_mod.GPT(
        vocab_size=_hp(hparams, "vocab_size", 8192),
        num_layers=_hp(hparams, "num_layers", 9),
        model_dim=_hp(hparams, "model_dim", 1536),
        num_heads=_hp(hparams, "num_heads", 12),
        num_kv_heads=_hp(hparams, "num_kv_heads", 6),
        mlp_mult=_hp(hparams, "mlp_mult", 2),
        tie_embeddings=_hp(hparams, "tie_embeddings", True),
        tied_embed_init_std=_hp(hparams, "tied_embed_init_std", 0.02),
        logit_softcap=_hp(hparams, "logit_softcap", 30.0),
        rope_base=_hp(hparams, "rope_base", 10000.0),
        qk_gain_init=_hp(hparams, "qk_gain_init", 0.25),
        fineweb_graphify=_hp(hparams, "fineweb_graphify", True),
        tokengt_first_class=_hp(hparams, "tokengt_first_class", True),
        tokengt_graph_radius=_hp(hparams, "tokengt_graph_radius", 4),
        tokengt_distance_features=_hp(hparams, "tokengt_distance_features", "functional"),
        tokengt_token_class_buckets=_hp(hparams, "tokengt_token_class_buckets", 128),
        tokengt_position_buckets=_hp(hparams, "tokengt_position_buckets", 512),
        tokengt_structural_weight=_hp(hparams, "tokengt_structural_weight", 0.05),
        tokengt_edge_weight=_hp(hparams, "tokengt_edge_weight", 0.035),
        tokengt_torus_weight=_hp(hparams, "tokengt_torus_weight", 0.015),
        tokengt_identifier_dim=_hp(hparams, "tokengt_identifier_dim", 24),
        tokengt_identifier_weight=_hp(hparams, "tokengt_identifier_weight", 0.014),
        tokengt_endpoint_weight=_hp(hparams, "tokengt_endpoint_weight", 0.018),
        tokengt_edge_token_weight=_hp(hparams, "tokengt_edge_token_weight", 0.016),
        convextok_dag_features=_hp(hparams, "convextok_dag_features", True),
        convextok_dag_feature_weight=_hp(hparams, "convextok_dag_feature_weight", 0.024),
        graph_output_flattening=_hp(hparams, "graph_output_flattening", False),
        graph_output_edge_radius=_hp(hparams, "graph_output_edge_radius", 4),
        graph_output_distance_features=_hp(hparams, "graph_output_distance_features", "functional"),
        graph_output_node_weight=_hp(hparams, "graph_output_node_weight", 0.05),
        graph_output_edge_weight=_hp(hparams, "graph_output_edge_weight", 0.05),
        graph_output_virtual_edge_tokens=_hp(hparams, "graph_output_virtual_edge_tokens", True),
        graph_output_edge_token_weight=_hp(hparams, "graph_output_edge_token_weight", 0.035),
        graph_output_score_correction=_hp(hparams, "graph_output_score_correction", True),
        graph_output_score_correction_weight=_hp(hparams, "graph_output_score_correction_weight", 0.024),
        toricblm_mup=_hp(hparams, "toricblm_mup", True),
        mup_width_mult=_hp(hparams, "mup_width_mult", 3.0),
        mup_output_mult=_hp(hparams, "mup_output_mult", 1.0),
        mup_attention_scale=_hp(hparams, "mup_attention_scale", True),
        mup_attention_base_head_dim=_hp(hparams, "mup_attention_base_head_dim", 64.0),
    )


def _load_prompt(path: str | None, prompt: str | None) -> str:
    if prompt:
        return prompt
    if not path:
        raise SystemExit("Pass --prompt or --prompt-file.")
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith(".json"):
        obj = json.loads(text)
        if isinstance(obj, dict) and "prompt" in obj:
            return str(obj["prompt"])
    return text


def _sample_next(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    logits = logits.float()
    if top_k > 0 and top_k < logits.numel():
        values, indices = torch.topk(logits, k=top_k)
        probs = torch.softmax(values / max(float(temperature), 1e-6), dim=-1)
        return int(indices[torch.multinomial(probs, num_samples=1)].item())
    probs = torch.softmax(logits / max(float(temperature), 1e-6), dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--checkpoint", required=True, help="Hub filename or local .pt checkpoint path.")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_FILE, help="Hub filename or local ConvexTok JSON.")
    parser.add_argument("--toricgt-root", default=".", help="Local ToricGT checkout.")
    parser.add_argument("--prompt-file")
    parser.add_argument("--prompt")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--flatten-graph-output", action="store_true", help="Use only for BPB sequence-flattening checkpoints.")
    args = parser.parse_args()

    root = Path(args.toricgt_root).resolve()
    train_mod = _load_train_module(root)
    from convextok import ConvexTokTokenizer

    checkpoint_path = _download_or_local(args.repo_id, args.checkpoint)
    tokenizer_path = _download_or_local(args.repo_id, args.tokenizer)
    tokenizer = ConvexTokTokenizer.load_json(tokenizer_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = payload.get("hyperparameters", {}) if isinstance(payload, dict) else {}
    model = _build_model(train_mod, hparams)
    if hasattr(model, "set_convextok_features"):
        model.set_convextok_features(
            lp_scores=torch.tensor(tokenizer.token_lp_scores(), dtype=torch.float32),
            rank_scores=torch.tensor(tokenizer.token_rank_scores(), dtype=torch.float32),
            priced_flags=torch.tensor(tokenizer.token_is_priced(), dtype=torch.float32),
            byte_lengths=torch.tensor(tokenizer.token_byte_lengths(), dtype=torch.float32),
        )
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=True)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model.to(args.device, dtype=dtype).eval()

    prompt_text = _load_prompt(args.prompt_file, args.prompt)
    ids = [tokenizer.bos_id] + tokenizer.encode(prompt_text)
    generated = list(ids)
    with torch.inference_mode():
        for _ in range(args.max_new_tokens):
            x = torch.tensor([generated[-args.seq_len :]], dtype=torch.long, device=args.device)
            logits = model.logits_for_distill(x, flatten_graph_output=args.flatten_graph_output)[0, -1]
            next_id = _sample_next(logits, args.temperature, args.top_k)
            generated.append(next_id)
            if next_id == tokenizer.eos_id:
                break

    print(tokenizer.decode(generated))


if __name__ == "__main__":
    main()
