# W&B Sweep

Date: 2026-06-03 UTC

## Active Run

```text
run id: toricgt_seq4096_wandb_ckpt_seed1337_20260603T1743Z
url: https://wandb.ai/amelie-iska-math/toricgt-parameter-golf/runs/toricgt_seq4096_wandb_ckpt_seed1337_20260603T1743Z
launched from parameter-golf commit: 00ebe09
current parameter-golf head: 7dc7d59
```

The active process is a sequential Parameter Golf `TrainingOptSeq4096` run.  It is not the ToricGT random-order all-phases trainer, so it does not directly instantiate GraphCG, GFlowNet graph-of-thought actions, toric geometry losses, Toric BGG, Koszul persistence, topology losses, trajectory memory, QAT, or contrastive/analogical auxiliary losses.

W&B API confirmed these active-run history keys:

```text
_step
optim/lr_scale
optim/muon_momentum
time/step_avg_ms
time/train_ms
train/loss
val/bpb
val/loss
```

At `_step=0`:

```text
val/bpb = 4.107707800251096
val/loss = 6.935692555837313
```

BPB is a full validation metric in this trainer and refreshes every `VAL_LOSS_EVERY=1000` steps.  Between validation events, W&B receives train-loss and optimizer/timing points only.

## Direct Seq4096 Metrics

Commit `7dc7d59` on `amelie-iska/parameter-golf` expands direct reporting for future resumes or launches.  These metrics are available without changing the model objective:

```text
trainer/step
trainer/total_steps
trainer/progress
progress/step
progress/total_steps
progress/fraction
progress/remaining_steps
train/loss
train/perplexity
fineweb/train_loss
fineweb/train_time_ms
val/loss
val/perplexity
val/bpb
val_bpb
bpb
openai_parameter_golf/bpb
fineweb/val_loss
fineweb/val_bpb
fineweb/best_val_bpb
fineweb/target_bpb
fineweb/target_gap_bpb
fineweb/target_reached
bpb/val
bpb/best
bpb/target
bpb/gap_to_target
bpb/improvement_from_initial
bpb/target_reached
time/train_ms
time/train_seconds
time/step_avg_ms
time/steps_per_second
perf/train_tokens_per_second
perf/train_tokens_per_step
optim/lr_scale
optim/muon_momentum
optim/token_lr
optim/matrix_lr
optim/scalar_lr
system/vram_allocated_gb
system/vram_reserved_gb
checkpoint/step
checkpoint/bytes
checkpoint/training_time_ms
artifact/raw_model_bytes
artifact/code_bytes
artifact/raw_total_bytes
artifact/int8_zlib_model_bytes
artifact/int8_zlib_total_bytes
artifact/int8_payload_bytes
artifact/int8_raw_torch_bytes
artifact/int8_payload_ratio
final/int8_zlib_roundtrip_loss
final/int8_zlib_roundtrip_bpb
final/int8_zlib_roundtrip_eval_ms
```

The expanded checkpoint payload also stores:

```text
best_val_bpb
initial_val_bpb
```

This lets a resumed process keep BPB target-gap and improvement reporting consistent.

## Historical ToricGT W&B Sources

The sweep found these first-party W&B paths:

| Source | Role |
|---|---|
| `scripts/train_parameter_golf_random_order.py` | all-phases random-order ToricGT trainer with the largest metric surface |
| `scripts/mirror_fineweb_log_to_wandb.py` | FineWeb text-log mirror for Seq/FineWeb train and BPB metrics |
| `scripts/mirror_fineweb_full_diagnostics_to_wandb.py` | FineWeb curve proxy diagnostics for toric/topology/BGG/tropical dashboards |
| `scripts/evaluate_oai_competition_bpb.py` | OAI BPB evaluator that logs to an existing run |
| `scripts/evaluate_complexity.py` | compressor complexity summaries |
| `scripts/evaluate_reasoning_geometry_suite.py` | reasoning/geometry summaries and images |
| `scripts/evaluate_reasoning_simplex.py` | reasoning-simplex scalar summaries |
| `scripts/train.py` | base ToricGT trainer W&B logging |
| `scripts/supervise_parameter_golf_training.py` and launch scripts | W&B env/run orchestration |

Generated outputs also contain many historical W&B references, but they were excluded from the source sweep except as evidence of previous run conventions.

## Random-Order All-Phases Metric Surface

`scripts/train_parameter_golf_random_order.py` reports the broad ToricGT metric/loss surface.  Important namespaces:

```text
train/*
val/*
oai_competition/*
competition/*
bpb/*
fineweb_calibration/*
metrics_status/*
controller/*
data/*
phase/*
system/*
artifact/*
hf_publish/*
graphcg/*
topology/*
toric/*
tropical/*
bgg_category_o/*
category_o/*
koszul_persistence/*
hessian/*
complexity/*
audit/*
```

Notable training losses and probes in that trainer:

```text
train/gflownet_loss
train/gflownet_entropy
train/gflownet_action_diversity
train/graphcg_loss
train/graphcg_code_loss
train/graphcg_orthogonal_loss
train/graphcg_covariance_loss
train/graphcg_sparsity_loss
train/analogy_lattice_loss
train/analogy_topology_loss
train/analogy_step_topology_loss
train/toric_geometry_loss
train/toric_fan_loss
train/toric_active_face_ce
train/toric_bend_loss
train/toric_binomial_loss
train/toric_moment_loss
train/toric_coxeter_loss
train/toric_braid_loss
train/toric_bgg_loss
train/toric_bgg_d2_residual
train/toric_bgg_resolution_consistency
train/koszul_persistence_loss
train/koszul_exactness_residual
train/koszul_syzygy_residual
train/trajectory_flow_loss
train/trajectory_memory_loss
train/qat_loss
train/contrastive_loss
train/mtp_loss
```

Those are not automatically compatible with the Seq4096 record script.  They are model/objective machinery from the ToricGT random-order branch.  They should be used in one of three ways:

1. Diagnostic sidecars for the sequential branch when they only need logs/curves/checkpoints.
2. Gated auxiliary losses in a separate sequential ToricGT branch after proving no BPB regression.
3. Continued ToricGT reasoning/graph training after a contest-credible BPB model exists.

## FineWeb Mirror Metrics

`scripts/mirror_fineweb_log_to_wandb.py` already defines the most useful Seq/FineWeb monitor set:

```text
trainer/step
trainer/total_steps
trainer/progress
progress/step
progress/total_steps
progress/fraction
progress/remaining_steps
time/train_ms
time/train_seconds
time/step_avg_ms
time/steps_per_second
fineweb/train_loss
fineweb/train_time_ms
fineweb/val_loss
fineweb/val_bpb
fineweb/best_val_bpb
fineweb/target_bpb
fineweb/target_gap_bpb
fineweb/target_reached
train/loss
train/perplexity
val/loss
val/perplexity
val/bpb
bpb/val
bpb/best
bpb/target
bpb/gap_to_target
bpb/improvement_from_initial
bpb/target_reached
```

Commit `7dc7d59` ports these directly into the Seq4096 trainer so the run does not need a separate mirror for basic FineWeb/BPB visibility.

## Full-Diagnostics Mirror Metrics

`scripts/mirror_fineweb_full_diagnostics_to_wandb.py` computes curve/proxy diagnostics from a FineWeb log.  It can be used as a sidecar because it does not require hidden model activations:

```text
metrics_status/fineweb_curve_diagnostics_available
metrics_status/full_toricgt_training_metrics_available
metrics_status/metric_scope_fineweb_curve_proxy
metrics_status/model_hidden_state_available
diagnostics/latest_full_metrics_step
diagnostics/source_code
tropical/bpb_best_so_far
tropical/bpb_target_gap
tropical/bpb_recent_slope
tropical/bpb_minplus_improvement
tropical/bpb_margin_to_second_best
tropical/bpb_plateau_pressure
tropical/bpb_active_face_entropy
toric/shadow_fan_cell_entropy
topology/topology_loss
bgg_category_o/toy_koszul_d2_residual
bgg_category_o/toy_standard_poset_size
bgg_category_o/late_phase_toggle_enabled
category_o/late_phase_toggle_enabled
complexity/run_log_available
complexity/run_log_bytes
complexity/recent_full_log_ncd_lzma
progress/step
progress/total_steps
progress/fraction
```

These are useful now as dashboards and decision aids.  They are not equivalent to active ToricGT training losses in the Seq4096 model.

## OAI BPB Evaluator Metrics

`scripts/evaluate_oai_competition_bpb.py` logs:

```text
oai_competition/loss
oai_competition/bpb
oai_competition/deterministic_loss
oai_competition/deterministic_bpb
oai_competition/eval_batches
oai_competition/seq_len
oai_competition/available
oai_competition/source_sp1024_decoded_bytes
oai_competition/score_first_bias_norm
competition/oai_bpb
bpb/oai_competition
```

For Seq4096, `val/bpb` is already computed over the FineWeb validation shards using the same SentencePiece byte-accounting path in the record script.  A separate OAI evaluator remains useful for cross-checking checkpoints and maintaining naming continuity with ToricGT random-order runs.

## Complexity And Reasoning Evaluators

Additional W&B emitters:

```text
evaluate_complexity.py:
  complexity/samples
  compressor-derived *_mean, *_min, *_max summaries

evaluate_reasoning_geometry_suite.py:
  score/reasoning
  score/bpb_quality
  score/loss_quality
  score/order_robustness
  score/gflownet_diversity
  score/toric_entropy
  score/trajectory_depth
  analysis/images/*

evaluate_reasoning_simplex.py:
  reasoning_simplex/<record-key>
```

These should be run on checkpoints as diagnostics.  They should not steer Seq4096 BPB training until a checkpoint comparison shows no BPB regression.

## Utilization Policy

BPB remains the primary optimization gate:

```text
target: <= 1.2 BPB
primary metric: val/bpb and fineweb/val_bpb
record aliases: bpb, val_bpb, bpb/val, openai_parameter_golf/bpb
```

Use immediately:

```text
train/loss
train/perplexity
val/bpb
fineweb/best_val_bpb
bpb/gap_to_target
time/steps_per_second
perf/train_tokens_per_second
optim/token_lr
optim/matrix_lr
optim/scalar_lr
optim/muon_momentum
system/vram_allocated_gb
artifact/int8_zlib_total_bytes
checkpoint/bytes
```

Use as sidecar diagnostics at checkpoint gates:

```text
tropical/bpb_recent_slope
tropical/bpb_plateau_pressure
tropical/bpb_minplus_improvement
toric/shadow_fan_cell_entropy
topology/topology_loss
bgg_category_o/toy_koszul_d2_residual
complexity/recent_full_log_ncd_lzma
oai_competition/bpb
```

Promote as training auxiliaries only after checkpoint A/B tests show flat or improved BPB:

```text
GraphCG losses
GFlowNet losses
topology losses
toric geometry losses
Toric BGG losses
Koszul persistence losses
trajectory memory losses
QAT losses
contrastive and MTP losses
```

This policy uses the metrics and losses where helpful without sacrificing the contest objective to diagnostics that are not yet proven useful for sequential FineWeb compression.

## Next Operational Step

Let the active run reach step 1000 and write its first checkpoint.  Then resume from that checkpoint with Parameter Golf commit `7dc7d59` so the expanded direct W&B metric set is active without discarding trained state.

After step 1000:

```text
1. Verify val/bpb and checkpoint bytes.
2. Resume from checkpoint with 7dc7d59.
3. Start or run once the full-diagnostics mirror for proxy ToricGT/tropical/topology/BGG metrics.
4. Use BPB trend, throughput, and artifact-size metrics for hyperparameter decisions.
5. Do not enable auxiliary ToricGT losses in the Seq4096 path unless a controlled checkpoint comparison shows no BPB regression.
```
