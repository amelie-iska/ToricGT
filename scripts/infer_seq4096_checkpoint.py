#!/usr/bin/env python3
"""Run text generation from a compact Seq4096 Parameter-Golf .pt checkpoint."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_seq4096_competition_bpb import (  # noqa: E402
    DEFAULT_COMPACT_SCRIPT,
    DEFAULT_TOKENIZER,
    build_model,
    checkpoint_kind,
    infer_compact_config,
    load_compact_module,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="The next idea is")
    parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER)
    parser.add_argument("--compact-script", default=str(DEFAULT_COMPACT_SCRIPT))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=17)
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


def model_logits(module: Any, model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    x = model.tok_emb(input_ids)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    skips: list[torch.Tensor] = []
    for i in range(model.num_encoder_layers):
        x = model.blocks[i](x, x0)
        skips.append(x)
    for i in range(model.num_decoder_layers):
        if skips:
            x = x + model.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
        x = model.blocks[model.num_encoder_layers + i](x, x0)
    x = model.final_norm(x)
    if model.tie_embeddings:
        logits_proj = F.linear(x, model.tok_emb.weight)
    else:
        logits_proj = model.lm_head(x)
    if getattr(model, "prev_token_bias", None) is not None:
        transition_bias = model.prev_token_bias(input_ids.reshape(-1)).reshape(
            input_ids.shape[0], input_ids.shape[1], -1
        )
        logits_proj = logits_proj + transition_bias.to(dtype=logits_proj.dtype) * model.bigram_bias_scale
    if getattr(model, "hash_ngram_bias", None) is not None:
        bucket_ids = module.hash_ngram_context_ids(
            input_ids,
            model.hash_ngram_bias_buckets,
            model.hash_ngram_bias_order,
        )
        memory_bias = model.hash_ngram_bias(bucket_ids)
        logits_proj = logits_proj + memory_bias.to(dtype=logits_proj.dtype) * model.hash_ngram_bias_scale
    return model.logit_softcap * torch.tanh(logits_proj / model.logit_softcap)


def sample_next(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    next_logits = logits[:, -1, :].float()
    temperature = max(float(temperature), 1e-6)
    next_logits = next_logits / temperature
    if top_k > 0 and top_k < next_logits.size(-1):
        values, indices = torch.topk(next_logits, k=int(top_k), dim=-1)
        probs = torch.softmax(values, dim=-1)
        choice = torch.multinomial(probs, num_samples=1)
        return indices.gather(-1, choice)
    probs = torch.softmax(next_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint_kind(payload) != "seq4096_compact_gpt":
        raise ValueError(f"expected compact Seq4096 checkpoint, got {checkpoint_kind(payload)}")
    module = load_compact_module(args.compact_script)
    config = infer_compact_config(
        payload["model"],
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        bigram_bias_scale=args.bigram_bias_scale,
        hash_ngram_bias_order=args.hash_ngram_bias_order,
        hash_ngram_bias_scale=args.hash_ngram_bias_scale,
        polarquant_kv_bits=args.polarquant_kv_bits,
        polarquant_eval_sample_tokens=args.polarquant_eval_sample_tokens,
        polarquant_seed=args.polarquant_seed,
    )
    model = build_model(module, config, payload["model"], device, args.precision)
    model.eval()
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    ids = sp.encode(args.prompt, out_type=int)
    if not ids:
        bos = int(sp.bos_id())
        ids = [bos if bos >= 0 else 0]
    ids = [int(token) for token in ids if 0 <= int(token) < int(config.vocab_size)]
    input_ids = torch.tensor([ids[-int(args.seq_len) :]], device=device, dtype=torch.long)
    with torch.inference_mode():
        for _ in range(int(args.max_new_tokens)):
            context = input_ids[:, -int(args.seq_len) :]
            autocast_enabled = device.type == "cuda" and args.precision == "bf16"
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                logits = model_logits(module, model, context)
            next_id = sample_next(logits, args.temperature, args.top_k)
            if not torch.isfinite(next_id.float()).all():
                raise RuntimeError("non-finite sampled token")
            input_ids = torch.cat([input_ids, next_id], dim=1)
    generated = input_ids[0].detach().cpu().tolist()
    text = sp.decode(generated)
    print(text)


if __name__ == "__main__":
    main()
