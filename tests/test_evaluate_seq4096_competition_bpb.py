from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_seq4096_competition_bpb.py"


def load_module():
    spec = importlib.util.spec_from_file_location("evaluate_seq4096_competition_bpb", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fake_state_dict(
    *,
    vocab_size: int = 32,
    model_dim: int = 16,
    num_layers: int = 3,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    mlp_mult: int = 3,
    bigram_bias: bool = True,
    hash_ngram_bias: bool = True,
) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {
        "tok_emb.weight": torch.zeros(vocab_size, model_dim),
        "skip_weights": torch.zeros(1, model_dim),
    }
    if bigram_bias:
        state["prev_token_bias.weight"] = torch.zeros(vocab_size, vocab_size)
    if hash_ngram_bias:
        state["hash_ngram_bias.weight"] = torch.zeros(128, vocab_size)
    head_dim = model_dim // num_heads
    for idx in range(num_layers):
        state[f"blocks.{idx}.attn.q_gain"] = torch.zeros(num_heads)
        state[f"blocks.{idx}.attn.c_q.weight"] = torch.zeros(model_dim, model_dim)
        state[f"blocks.{idx}.attn.c_k.weight"] = torch.zeros(num_kv_heads * head_dim, model_dim)
        state[f"blocks.{idx}.attn.c_v.weight"] = torch.zeros(num_kv_heads * head_dim, model_dim)
        state[f"blocks.{idx}.attn.proj.weight"] = torch.zeros(model_dim, model_dim)
        state[f"blocks.{idx}.mlp.fc.weight"] = torch.zeros(mlp_mult * model_dim, model_dim)
        state[f"blocks.{idx}.mlp.proj.weight"] = torch.zeros(model_dim, mlp_mult * model_dim)
        state[f"blocks.{idx}.attn_scale"] = torch.zeros(model_dim)
        state[f"blocks.{idx}.mlp_scale"] = torch.zeros(model_dim)
        state[f"blocks.{idx}.resid_mix"] = torch.zeros(2, model_dim)
    return state


def test_infer_compact_config_from_seq4096_state_dict() -> None:
    module = load_module()

    config = module.infer_compact_config(fake_state_dict())

    assert config.vocab_size == 32
    assert config.model_dim == 16
    assert config.num_layers == 3
    assert config.num_heads == 4
    assert config.num_kv_heads == 2
    assert config.mlp_mult == 3
    assert config.tie_embeddings is True
    assert config.bigram_bias is True
    assert config.hash_ngram_bias is True
    assert config.hash_ngram_bias_buckets == 128


def test_checkpoint_config_sidecar_overrides_polarquant_defaults(tmp_path: Path) -> None:
    module = load_module()
    checkpoint = tmp_path / "run_step_000250.pt"
    payload = {"step": 250, "model": fake_state_dict()}
    (tmp_path / "run_config.json").write_text(
        '{"polarquant_kv_bits": 8, "polarquant_eval_sample_tokens": 256, "polarquant_seed": 123}\n',
        encoding="utf-8",
    )

    overrides = module.checkpoint_config_overrides(checkpoint, payload)
    config = module.infer_compact_config(fake_state_dict(), **overrides)

    assert config.polarquant_kv_bits == 8
    assert config.polarquant_eval_sample_tokens == 256
    assert config.polarquant_seed == 123


def test_infer_compact_config_rejects_non_seq4096_payload() -> None:
    module = load_module()

    with pytest.raises(ValueError, match="tok_emb.weight"):
        module.infer_compact_config({"not_a_seq4096_tensor": torch.zeros(2)})


def test_oai_metric_summary_uses_expected_aliases() -> None:
    module = load_module()

    summary = module.oai_metric_summary(
        checkpoint=Path("/tmp/run_step_000250.pt"),
        step=250,
        val_loss=2.0,
        val_bpb=1.1875,
        seq_len=4096,
        val_max_sequences=16,
        token_glob="data/fineweb_val_*.bin",
        tokenizer_path="data/tokenizers/fineweb_1024_bpe.model",
    )

    assert summary["oai_competition/bpb"] == pytest.approx(1.1875)
    assert summary["competition/oai_bpb"] == pytest.approx(1.1875)
    assert summary["bpb/oai_competition"] == pytest.approx(1.1875)
    assert summary["seq4096/oai_competition_bpb"] == pytest.approx(1.1875)
    assert summary["oai_competition/eval_scope"] == "sampled"
    assert summary["source"] == "local_fineweb10B_sp1024_validation_tokens"


def test_checkpoint_kind_detects_compact_payload() -> None:
    module = load_module()
    payload = {"step": 12, "model": fake_state_dict(bigram_bias=False, hash_ngram_bias=False)}

    assert module.checkpoint_kind(payload) == "seq4096_compact_gpt"
    assert module.checkpoint_step(payload) == 12
