import torch

from toricgt.random_order_lm import (
    DenseRandomOrderToricLM,
    RandomOrderLMConfig,
    byte_decode,
    byte_encode,
    random_order_batch,
    random_order_permutation,
)


def tiny_config(**overrides):
    values = {
        "vocab_size": 32,
        "max_seq_len": 16,
        "d_model": 32,
        "num_heads": 4,
        "num_layers": 2,
        "recurrent_passes": 1,
        "ffn_multiplier": 2,
        "dropout": 0.0,
        "attention": "hybrid",
        "ring_block_size": 8,
        "seed": 11,
        "byte_offset": 4,
        "bigram_hash_buckets": 64,
        "aux_mtp_offsets": 1,
    }
    values.update(overrides)
    return RandomOrderLMConfig(**values)


def test_random_order_is_content_independent_and_seeded():
    tokens_a = torch.arange(8, dtype=torch.long).view(1, 8) + 4
    tokens_b = torch.flip(tokens_a, dims=[1])
    sample_ids = torch.tensor([123])
    batch_a = random_order_batch(tokens_a, seed=17, sample_ids=sample_ids, bos_token_id=40)
    batch_b = random_order_batch(tokens_b, seed=17, sample_ids=sample_ids, bos_token_id=40)
    assert torch.equal(batch_a.permutation, batch_b.permutation)
    assert not torch.equal(
        random_order_permutation(8, seed=17, sample_id=123),
        random_order_permutation(8, seed=17, sample_id=124),
    )


def test_score_before_update_future_tokens_do_not_affect_current_logit():
    cfg = tiny_config()
    model = DenseRandomOrderToricLM(cfg).eval()
    permutation = torch.arange(8, dtype=torch.long).view(1, 8)
    tokens_a = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]])
    tokens_b = torch.tensor([[4, 5, 6, 7, 20, 21, 22, 23]])
    out_a = model(tokens_a, permutation=permutation, return_order=True)
    out_b = model(tokens_b, permutation=permutation, return_order=True)
    assert torch.allclose(out_a["logits"][:, 4], out_b["logits"][:, 4], atol=1e-6)


def test_dense_random_order_lm_loss_and_generation_shapes():
    cfg = tiny_config(vocab_size=48)
    model = DenseRandomOrderToricLM(cfg)
    tokens = torch.randint(4, cfg.vocab_size, (3, 12))
    out = model(tokens, sample_ids=torch.arange(3), return_order=True)
    assert out["logits"].shape == (3, 12, cfg.vocab_size)
    assert out["loss"].isfinite()
    assert out["mtp_loss"].isfinite()
    assert out["smear_temperature"].isfinite()
    assert out["toric_memory_entropy"].isfinite()
    assert out["contrastive_loss"].isfinite()
    assert out["trajectory_flow_loss"].isfinite()
    assert out["trajectory_kinetic_energy"].isfinite()
    assert out["permutation"].shape == tokens.shape
    generated = model.eval().sample_random_order(length=10, seed=99, sample_id=0, top_k=8)
    assert generated.shape == (10,)


def test_gflownet_adapter_losses_and_multi_sample_scaling():
    cfg = tiny_config(vocab_size=48, use_gflownet_policy=True, gflownet_num_actions=4, gflownet_hidden_dim=16)
    model = DenseRandomOrderToricLM(cfg)
    tokens = torch.randint(4, cfg.vocab_size, (2, 10))
    out = model(tokens, sample_ids=torch.arange(2), sample_gflownet=True)
    assert out["loss"].isfinite()
    assert out["gflownet_loss"].isfinite()
    assert out["gflownet_entropy"].isfinite()
    assert 0.0 <= float(out["gflownet_action_diversity"]) <= 1.0

    scaled = model(tokens, sample_ids=torch.arange(2), gflownet_samples=3)
    assert scaled["logits"].shape == (2, 10, cfg.vocab_size)
    assert scaled["loss"].isfinite()
    assert scaled["single_sample_loss"].isfinite()


def test_gflownet_adapter_preserves_score_before_update():
    cfg = tiny_config(use_gflownet_policy=True, gflownet_num_actions=4, gflownet_hidden_dim=16)
    model = DenseRandomOrderToricLM(cfg).eval()
    permutation = torch.arange(8, dtype=torch.long).view(1, 8)
    tokens_a = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]])
    tokens_b = torch.tensor([[4, 5, 6, 7, 20, 21, 22, 23]])
    out_a = model(tokens_a, permutation=permutation, return_order=True)
    out_b = model(tokens_b, permutation=permutation, return_order=True)
    assert torch.allclose(out_a["logits"][:, 4], out_b["logits"][:, 4], atol=1e-6)


def test_score_first_bias_adaptation_is_finite_and_prefix_safe():
    cfg = tiny_config(vocab_size=48, use_gflownet_policy=True, gflownet_num_actions=4, gflownet_hidden_dim=16)
    model = DenseRandomOrderToricLM(cfg).eval()
    permutation = torch.arange(8, dtype=torch.long).view(1, 8)
    tokens = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]])
    out = model.score_with_bias_adaptation(tokens, permutation=permutation, lr=0.05, gflownet_samples=2)
    assert out["loss"].isfinite()
    assert out["bias_norm"].isfinite()
    assert model.causal_future_permutation_error(tokens, permutation=permutation) < 1e-5


def test_byte_roundtrip():
    text = "ToricGT \u05db\u05bc\u05b8\u05ea\u05b7\u05d1"
    tokens = byte_encode(text, byte_offset=4)
    assert byte_decode(tokens, byte_offset=4) == text
