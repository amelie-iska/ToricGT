import importlib.util
from pathlib import Path

import torch


def load_seq4096_module():
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root
        / "amelie-iska"
        / "parameter-golf"
        / "records"
        / "track_10min_16mb"
        / "2026-03-19_TrainingOptSeq4096"
        / "train_gpt.py"
    )
    spec = importlib.util.spec_from_file_location("seq4096_train_gpt", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_bigram_bias_lowers_loss_for_matching_previous_token():
    module = load_seq4096_module()
    model = module.GPT(
        vocab_size=8,
        num_layers=2,
        model_dim=16,
        num_heads=2,
        num_kv_heads=1,
        mlp_mult=2,
        tie_embeddings=True,
        tied_embed_init_std=0.01,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.0,
        bigram_bias=True,
        bigram_bias_init_std=0.0,
        bigram_bias_scale=1.0,
    )
    input_ids = torch.full((2, 4), 1, dtype=torch.long)
    target_ids = torch.full((2, 4), 2, dtype=torch.long)

    base_loss = model(input_ids, target_ids)
    with torch.no_grad():
        model.prev_token_bias.weight[1, 2] = 8.0
    biased_loss = model(input_ids, target_ids)

    assert biased_loss < base_loss
