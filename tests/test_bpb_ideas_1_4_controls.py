import torch

from scripts.adaptive_bpb_annealing import build_adaptive_decision
from toricgt.embedding_forest_of_thought import EmbeddingFoTConfig, EmbeddingForestOfThoughtHead


def test_embedding_fot_bpb_delta_reward_path_backpropagates():
    torch.manual_seed(19)
    cfg = EmbeddingFoTConfig(
        dim=8,
        num_trees=2,
        max_depth=3,
        branching=3,
        topk_trees=1,
        hidden_dim=16,
        max_positions=6,
        consensus_buckets=8,
        reward_mode="bpb_delta",
        bpb_delta_weight=0.75,
    )
    head = EmbeddingForestOfThoughtHead(cfg)
    hidden = torch.randn(2, 7, 8, requires_grad=True)
    target_ids = torch.randint(0, 11, (2, 7))
    per_token_nll = torch.rand(2, 7) + 0.5
    byte_lengths = torch.arange(1, 8).view(1, 7).repeat(2, 1).float()
    lm_head_weight = torch.randn(11, 8)

    out = head(
        hidden,
        target_ids,
        per_token_nll,
        target_byte_lengths=byte_lengths,
        lm_head_weight=lm_head_weight,
        logit_softcap=8.0,
        temperature_multiplier=0.9,
        ucb_multiplier=1.1,
        sparse_multiplier=0.8,
    )

    assert out["oai_fot_enabled"].item() == 1.0
    assert torch.isfinite(out["oai_fot_loss"])
    assert torch.isfinite(out["oai_fot_reward_bpb_delta"])
    assert torch.isfinite(out["oai_fot_reward_raw_byte_nll"])
    assert torch.isfinite(out["oai_fot_reward_corrected_byte_nll"])
    assert torch.isfinite(out["oai_fot_corrected_ce"])

    out["oai_fot_loss"].backward()
    grad_norm = sum(float(p.grad.detach().abs().sum()) for p in head.parameters() if p.grad is not None)
    assert grad_norm > 0.0


def test_adaptive_decision_enables_bpb_first_conflict_and_fot_controls():
    decision = build_adaptive_decision(
        {
            "selected_bpb": 1.31,
            "latest_train": {
                "step": 1000,
                "train_bpb": 1.36,
                "graph_lm_bpb": 1.9,
                "graph_lm_weight": 0.04,
            },
            "train_rows": [
                {"step": 200, "train_bpb": 1.42},
                {"step": 600, "train_bpb": 1.38},
                {"step": 1000, "train_bpb": 1.36},
            ],
        },
        validation={"strict_validation_passed": True},
        sidecar_review={"observed_metric_count": 145},
        run_id="test-run",
        target_bpb=1.19,
    )
    env = decision["env_overrides"]

    assert env["BPB_FIRST_AUX_STAGING"] == "1"
    assert env["AUX_CONFLICT_CONTROLLER"] == "1"
    assert env["OAI_FOT_REWARD_MODE"] == "bpb_delta"
    assert env["OAI_FOT_ADAPTIVE_CONTROL"] == "1"
    assert env["OAI_EMBEDDING_FOT"] == "1"
    assert env["OAI_GFLOWNET"] == "1"
    assert decision["sweep_strategy"] == "family_specific_evidence_weighted_exploration"
