# ToricGT Status Report

Date: 2026-06-03 UTC

This report records the current state of the ToricGT Parameter-Golf/FineWeb training effort, the evidence returned by the latest automated analyses, and the next engineering decisions.  It is intended to be a working status document rather than a polished paper section.

## Executive Summary

The current supervised recovery run is alive, logging to W&B, producing checkpoints, and generating periodic analysis artifacts.  The run is:

```text
toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z
```

W&B:

```text
https://wandb.ai/amelie-iska-math/toricgt-parameter-golf/runs/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z
```

The latest durable checkpoint with completed analysis is:

```text
checkpoints/parameter_golf_all_phases_fineweb_from0/random_order_step_00005250.pt
```

The latest completed analysis directory is:

```text
outputs/post_resume_analysis/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/step-00005250
```

Training is continuing beyond step 5250.  At the time this report was written, the trainer was around step 5400-5450, the step-5500 watcher was active and waiting for the next checkpoint, and the latest durable checkpoint remained step 5250.

The main conclusion is clear: the recovery intervention improved OAI/FineWeb BPB modestly, but the improvement is too shallow to plausibly reach the target by waiting.  The random-order byte objective is likely the primary bottleneck for FineWeb BPB.  We should keep the current run alive through the next short gate, but the next major action should be a strict sequential FineWeb-first baseline/objective under the same artifact discipline.

## Current Live System

Active tmux sessions:

```text
toricgt_fineweb_bpb_recovery_live
toricgt_fineweb_bpb_recovery_watcher
toricgt_supervisor_fineweb_bpb_recovery
```

Supervisor:

```text
logs/parameter_golf_all_phases/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/supervisor/supervisor_state.json
```

Trainer log:

```text
logs/parameter_golf_all_phases/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/supervisor/restart_001_20260603T124401Z/train.log
```

The supervisor has started periodic analysis/checkin cycles at:

```text
4000, 4250, 4500, 4750, 5000, 5250, 5500 target pending
```

The latest supervisor state reported:

```text
latest_checkpoint_step: 5250
latest_checkpoint: checkpoints/parameter_golf_all_phases_fineweb_from0/random_order_step_00005250.pt
best_oai_competition_bpb: 4.523199440070747
best_val_bpb: 5.349750123909216
train_bpb: 4.594683061499462
highest_analysis_step: 5250
watcher_target_step: 5500
target_bpb: 1.2
```

The target BPB of 1.2 remains aspirational under this run.  It should not be treated as likely under the present random-order architecture.  The supervisor should continue to enforce liveness and periodic review, but the optimization path needs to change if the next checkpoints stay flat.

## Recent Implemented Changes

The key BPB recovery change was committed as:

```text
ce904f3 Tune all-phases run for FineWeb BPB recovery
```

The all-phases config now enables score-before-update legal revealed context:

```yaml
use_revealed_context_prior: true
revealed_context_prior_alpha: 0.25
revealed_context_prior_mode: mixture
revealed_context_prior_weight: 0.10
use_revealed_neighbor_context: true
revealed_neighbor_radius: 2
revealed_neighbor_context_weight: 0.45
```

The checkpoint resume confirmed this path is actually active:

```text
new_parameters_initialized:
  revealed_left_logits.0.weight
  revealed_left_logits.1.weight
  revealed_right_logits.0.weight
  revealed_right_logits.1.weight

optimizer_state_padded_for_new_parameters: new_parameter_states=4
```

The early phase was also made more BPB-focused:

```text
byte_warmup end_step: 12000
fineweb_mix_ratio: 1.0
gflownet_loss_weight: 0.00010
gflownet_entropy_weight: 0.0
graphcg_loss_weight: 0.00005
analogy_lattice_loss_weight: 0.0
contrastive_loss_weight: 0.00015
toric_geometry_loss_weight: 0.0
toric_bgg_loss_weight: 0.0
koszul_persistence_loss_weight: 0.0
```

Full-rank GraphCG remains active from step 0:

```text
graphcg_num_directions: 384
graphcg_min_directions: 384
graphcg_max_directions: 384
```

The Toric BGG, Koszul persistence, topology, memory, and QAT machinery remains instantiated or logged where implemented, but heavy losses are deliberately delayed until late phases.  This is intentional: the current bottleneck is BPB, not lack of algebraic auxiliary structure.

## Checkpoint BPB Trend

The relevant checkpoint metrics are:

| step | train BPB | best validation BPB | best OAI competition BPB |
|---:|---:|---:|---:|
| 3500 | 4.657651 | 5.852095 | 4.539991 |
| 3750 | 4.644115 | 5.852095 | 4.533575 |
| 4000 | 4.605330 | 5.382641 | 4.533575 |
| 4250 | 4.631999 | 5.373152 | 4.533575 |
| 4500 | 4.640970 | 5.359213 | 4.532134 |
| 4750 | 4.564344 | 5.349750 | 4.527964 |
| 5000 | 4.491884 | 5.349750 | 4.523199 |
| 5250 | 4.594683 | 5.349750 | 4.523199 |

The recovery path did help:

```text
best OAI BPB at step 3750: 4.533575
best OAI BPB by step 5000: 4.523199
absolute improvement: about 0.0104 BPB
```

But the improvement is shallow:

```text
step 5000: oai_competition/bpb 4.523199
step 5250: oai_competition/bpb 4.523213, best unchanged
```

This is not the trajectory required for a serious Parameter-Golf score.  The target `1.2 BPB` is more than 3 BPB away.  At this slope, waiting is not a credible strategy.

## Latest Completed Analysis: Step 5250

Latest synopsis:

```text
outputs/post_resume_analysis/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/step-00005250/SYNOPSIS.md
```

Latest metrics report:

```text
outputs/post_resume_analysis/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/step-00005250/metrics/automatic_metrics_report.md
```

Latest adjustment proposal:

```text
outputs/post_resume_analysis/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/step-00005250/metrics/training_adjustment_proposal.md
```

Metric category counts:

```text
as_desired: 427
as_desired_but_not_strong_or_fast_enough: 188
not_as_desired: 160
```

Checkpoint metadata in the synopsis:

```text
train_bpb: 4.594683061499462
best_val_bpb: 5.349750123909216
val_bpb: n/a in checkpoint synopsis
```

Post-resume OAI competition evaluation:

```text
source: local_fineweb10B_sp1024_validation_decoded_to_utf8_bytes
oai_competition/bpb: 4.5242843066323895
oai_competition/loss: 3.1359949111938477
eval_batches: 8
```

The direct training log at checkpoint intervals also reported:

```text
step 4000: oai_competition/bpb 4.54670499689477
step 4250: oai_competition/bpb 4.540763684238064
step 4500: oai_competition/bpb 4.5321342827483715
step 4750: oai_competition/bpb 4.52796404743025
step 5000: oai_competition/bpb 4.523199440070747
step 5250: oai_competition/bpb 4.523212510752212
```

This confirms that the step-5250 checkpoint did not improve the OAI best score.

## Core Metrics From Automatic Analysis

The step-5250 automatic report summarized:

| metric | category | first median | last median | relative change | recent slope / 1k |
|---|---|---:|---:|---:|---:|
| `train/bpb` | as_desired_but_not_strong_or_fast_enough | 4.61178 | 4.55047 | -1.33% | -0.163796 |
| `train/loss` | as_desired_but_not_strong_or_fast_enough | 3.19664 | 3.15415 | -1.33% | -0.113534 |
| `val/bpb` | as_desired_but_not_strong_or_fast_enough | 5.37315 | 5.35421 | -0.35% | +0.035769 |
| `train/gflownet_loss` | as_desired | 1.57784 | 1.50578 | -4.57% | -0.273310 |
| `train/gflownet_entropy` | as_desired_but_not_strong_or_fast_enough | 2.77072 | 2.77154 | +0.03% | +0.000608 |
| `train/gflownet_action_diversity` | as_desired_but_not_strong_or_fast_enough | 0.998467 | 0.998561 | +0.01% | +0.000073 |
| `complexity/val/argmax_byte_accuracy` | as_desired_but_not_strong_or_fast_enough | 0.173828 | 0.173828 | 0.00% | 0 |
| `complexity/val/prediction_target_ncd_lzma_mean` | as_desired_but_not_strong_or_fast_enough | 0.926937 | 0.927142 | +0.02% | -0.019756 |
| `audit/future_permutation_logit_error` | as_desired | 0 | 0 | 0.00% | 0 |

The important interpretation:

1. Training BPB is still moving down locally.
2. Deterministic validation/OAI BPB is barely moving.
3. Score validity is intact: future permutation logit error is zero.
4. Complexity and argmax-byte metrics are not strong enough to indicate a compression breakthrough.
5. The gap between training movement and native OAI BPB movement is the key concern.

## Geometry, Topology, Toric, and Complexity Findings

The step-5250 geometry diagnostics used 4 records and 24 branches:

```text
mean BPB: 7.542314887046814
best BPB: 5.062435150146484
mean answer BPB: 7.694171314142278
best answer BPB: 4.934705768443206
mean MST efficiency: 0.47992876045229343
mean path smoothness: 0.1354596856298935
mean directed topology asymmetry: 0.4589943156735468
mean directed cycle flux: 5.537630829179353e-18
mean radius-HDBSCAN clusters: 0.921875
mean radius-HDBSCAN noise: 0.22715928819444442
mean radius-HDBSCAN stability: 0.7728407118055557
```

The analysis provides useful diagnostic plots, but these are not sufficient promotion signals.  The branch-level geometry BPB is much worse than the native OAI checkpoint BPB.  The geometry branch search does find better branches than its own mean, but the entire branch distribution is still far from a useful byte-compression regime.

The training adjustment proposal returned the following signals:

```text
train_bpb_recent_slope_per_1k: -0.16379559245311212
train_loss_recent_slope_per_1k: -0.11353445309702054
branch_gap_mean_minus_best_bpb: 2.4798797369003296
toric_shadow_mean_margin: 0.02305212647964557
toric_active_face_margin: -2.8859834571679435
graphcg_loss_category: as_desired_but_not_strong_or_fast_enough
graphcg_covariance_category: as_desired
analogy_map_category: not_as_desired
toric_entropy_category: not_as_desired
```

Recommended phase delta from the automated analysis:

```text
lr_multiplier: decrease_or_rollback
graphcg_loss_weight: enable_5e-5_to_1e-4
analogy_lattice_loss_weight: tiny_1e-5_to_2e-5
gflownet_loss_weight: enable_tiny_2.5e-4_to_5e-4
gflownet_entropy_weight: tiny_5e-5_to_1e-4_at_target_2.0
toric_geometry_loss_weight: keep_zero_until_margins_improve
koszul_persistence_loss_weight: keep_zero_until_margins_improve
contrastive_loss_weight: increase_small_while_graphcg_on
```

We should not blindly apply this delta.  It is useful as a diagnostic, but BPB remains the primary gate.  In particular:

1. A lower LR or rollback may be appropriate if OAI BPB worsens after step 5500.
2. GraphCG is already active at full rank with a small weight; increasing it should be tested carefully.
3. GFlowNet branch gap is large, but GFlowNet loss can easily optimize branch-search diagnostics without improving next-byte BPB.
4. Heavy toric, BGG, and Koszul losses should remain off until toric margins and BPB are healthier.

## Latest Plot and Report Locations

Step-5250 summary:

```text
outputs/post_resume_analysis/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/step-00005250/SYNOPSIS.md
```

Metrics:

```text
outputs/post_resume_analysis/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/step-00005250/metrics
```

Key metrics plots:

```text
outputs/post_resume_analysis/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/step-00005250/metrics/core_metric_timeseries.png
outputs/post_resume_analysis/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/step-00005250/metrics/recent_metric_slopes.png
outputs/post_resume_analysis/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/step-00005250/metrics/selected_metric_correlations.png
```

Topology plots:

```text
outputs/post_resume_analysis/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/step-00005250/geometry/topology
```

Trajectory plots:

```text
outputs/post_resume_analysis/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/step-00005250/geometry/trajectories
```

Tetrahedra:

```text
outputs/post_resume_analysis/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/step-00005250/geometry/tetrahedra
```

Simplex:

```text
outputs/post_resume_analysis/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/step-00005250/simplex
```

Useful specific diagrams:

```text
outputs/post_resume_analysis/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/step-00005250/geometry/tetrahedra/reasoning_k_bpb_mst.html
outputs/post_resume_analysis/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/step-00005250/geometry/tetrahedra/reasoning_k_bpb_mst.png
outputs/post_resume_analysis/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/step-00005250/simplex/reasoning_k_bpb_mst_tetrahedron.html
outputs/post_resume_analysis/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/step-00005250/simplex/reasoning_k_bpb_mst_tetrahedron.png
```

## Interpretation: Why FineWeb BPB Is Poor

The poor FineWeb BPB is most likely structural rather than a missing logging issue or a temporary optimizer stall.

The current model is a random-order byte model.  That factorization is score-valid and useful for graph-style reasoning experiments, but it is not naturally aligned with FineWeb byte compression.  FineWeb rewards strict local sequential context: previous bytes, recent subword fragments, whitespace, punctuation, capitalization, common phrase prefixes, and document-level continuation patterns.  Random-order prediction hides much of that local left-to-right context.  Revealed-neighbor features restore some legal context, which is why BPB improved slightly, but not enough to remove the mismatch.

The model is also constrained by the 16 MB artifact budget.  That means every auxiliary structure competes directly with next-byte likelihood capacity.  GraphCG, toric memory, topology, GFlowNet, BGG probes, and Koszul diagnostics are valuable if they improve the predictive distribution.  At present they are mostly useful for audit and interpretability, not for FineWeb BPB.  They should stay logged and staged, but the early optimizer should remain dominated by native OAI/FineWeb likelihood.

The native OAI metric is the real gate.  The current run has:

```text
best_oai_competition_bpb: 4.523199440070747
```

This is very far from 1.2.  The gap is too large to attribute to a small LR setting or a single auxiliary loss coefficient.

## Decisions

### Decision 1: Keep the recovery run alive through the next short gate

Continue the current recovery run through at least step 5500 and preferably step 6000, unless it dies or the next OAI evaluations regress badly.  The run is supervised and useful for measuring whether the revealed-context recovery continues to help.

Gate:

```text
If OAI best BPB improves by at least 0.005-0.010 between steps 5250 and 6000, continue to 6500.
If it remains flat near 4.523 or worsens, stop treating this run as the primary BPB path.
```

### Decision 2: Do not turn on heavy toric/BGG/Koszul losses yet

Toric active-face margins are weak and toric entropy is categorized as not desired.  Heavy toric/Koszul/BGG losses under these conditions would likely fight BPB and make the representation more geometric without improving compression.

Keep:

```text
toric_geometry_loss_weight: 0.0
toric_bgg_loss_weight: 0.0
koszul_persistence_loss_weight: 0.0
```

until native OAI BPB is moving at a credible rate and toric margins are no longer pathological.

### Decision 3: Preserve full-rank GraphCG, but keep it low weight

The user requirement is full-rank GraphCG from step 0.  That is satisfied.  The analysis says GraphCG covariance is as desired, while GraphCG loss is only "not strong or fast enough."  A small increase may be defensible, but not before the next BPB gate.

For now, keep:

```text
graphcg_loss_weight: 0.00005
```

If BPB remains stable and GraphCG diagnostics are still useful, test:

```text
graphcg_loss_weight: 0.00008
```

not a large increase.

### Decision 4: Prepare a sequential FineWeb-first branch

The next serious BPB-improvement action should be a sequential causal FineWeb model/objective under the same artifact discipline.  Random-order should become auxiliary, not primary, for the BPB contest path.

Required properties:

1. Native FineWeb/OAI stream from step 0.
2. Strict left-to-right score-before-update objective.
3. Same artifact size gate.
4. Full-rank GraphCG instantiated and logged from step 0 with small loss.
5. Toric/BGG/topology/Koszul diagnostics on, but losses off until late phases.
6. OAI competition BPB every 250 steps.
7. Supervisor and watcher behavior identical to the current recovery run.

### Decision 5: Add a legal sequential/cache prior

A byte-level compressor should exploit legal local recurrence.  The next branch should test a small score-before-update mixture with:

1. byte unigram prior;
2. byte bigram or short n-gram cache;
3. recent-window adaptive byte cache;
4. learned model logits;
5. interpolation weights logged and ablated.

This must be evaluated in score-before-update form only.  No future validation bytes may influence a scored probability.

## Next Course of Action

### Immediate, while current run continues

1. Let the current run reach checkpoint 5500.
2. Read:

```text
outputs/post_resume_analysis/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/step-00005500
```

once created.

3. Compare OAI BPB at 5500 and 6000 against:

```text
current best: 4.523199440070747
```

4. If no meaningful OAI improvement appears, do not keep tuning random-order hyperparameters as the primary path.

### Engineering work to start next

Create a sequential FineWeb-first training config and launcher:

```text
config/train.parameter_golf_sequential_fineweb.yaml
```

or an equivalent sequential mode in the existing trainer, depending on the least invasive code path.

The model should remain byte-level and artifact-accounted.  If the existing random-order trainer can be adapted cleanly, use it.  If not, introduce a separate sequential trainer to avoid mixing factorization assumptions.

Minimum sequential baseline:

```text
vocab_size: 260
d_model: 384 or adjusted to fit artifact
num_layers: 7
recurrent_passes: 2
attention: causal/hybrid
fineweb_mix_ratio: 1.0
full_rank_graphcg: true
graphcg_loss_weight: 0.00005
gflownet_loss_weight: 0.0 to 0.00010
toric/bgg/koszul losses: 0.0
oai_eval_interval: 250
checkpoint_interval: 250
```

Run gate:

```text
If sequential OAI BPB is not clearly below random-order recovery by step 2000-3000, inspect model implementation and OAI evaluation path.
If sequential OAI BPB is materially better, promote it as the primary BPB track and keep random-order for graph/reasoning auxiliary work.
```

### Hyperparameter policy

Do not chase all metrics simultaneously.  The early phase should optimize this priority order:

1. native OAI competition BPB;
2. ordinary validation BPB;
3. train BPB stability;
4. score validity audits;
5. GraphCG covariance and low-rank/full-rank chart health;
6. GFlowNet branch-gap diagnostics;
7. toric/topology/BGG/Koszul diagnostics.

Loss promotion rule:

```text
No auxiliary loss should be promoted unless native OAI BPB improves or at least remains statistically flat while a predeclared reasoning metric improves.
```

This implies:

```text
Keep toric/BGG/Koszul losses off for now.
Keep GraphCG small.
Keep GFlowNet tiny.
Keep topology as analysis, not optimization.
```

## Risks

### Risk 1: Random-order objective is structurally capped

The evidence points here.  Revealed context helped but did not change the BPB regime.  A sequential baseline is necessary.

### Risk 2: Auxiliary metrics can look good while BPB stays bad

The reports contain many geometry, topology, toric, complexity, and GraphCG metrics.  They are useful, but BPB is the contest gate.  A branch with attractive topology but poor native FineWeb BPB is not a better Parameter-Golf submission.

### Risk 3: OAI eval is small and noisy

The current OAI eval uses 8 batches.  This is enough for quick gating but not enough for final claims.  For serious comparisons, increase OAI eval batches or run repeated evals on fixed checkpoint pairs.

### Risk 4: Hessian diagnostics are currently unreliable

The automatic report showed NaN values in Hessian traces and related quantities.  These should not be used as checkpoint gates until the probe is fixed or disabled.

### Risk 5: Supervisor cannot invent better objectives

The supervisor is doing its job: it keeps training alive and launches analysis.  It can restart a dead or stale run.  It cannot overcome an objective mismatch.  The next improvement requires a deliberate sequential BPB objective.

## Bottom Line

We are in a supervised random-order FineWeb recovery run.  It is alive and producing rigorous periodic reports.  The recovery changes improved best native OAI BPB from about 4.5336 to 4.5232, but the curve flattened again by step 5250.

The correct next decision is not to keep stacking toric, topology, or BGG losses onto this run.  The correct next decision is to keep this run alive through the short gate while preparing a sequential FineWeb-first BPB branch.  Random-order, GraphCG, toric geometry, topology, and Toric BGG should remain valuable research and diagnostic machinery, but the contest BPB path needs a left-to-right compression objective that matches FineWeb.

## 2026-06-03 UTC Follow-up: Sequential branch promoted

The public Parameter-Golf records and local `amelie-iska/parameter-golf` clone show that the relevant contest baseline family is sequential SentencePiece FineWeb training with legal score-first evaluation, not random-order byte completion.  The current official baseline is around the low `1.2` BPB range, while leaderboard records are near `1.06`; this is the performance regime required for the `<= 1.2` target.

The live ToricGT random-order recovery should finish its already-scheduled step-5500 analysis so the record is complete, but it should no longer receive the only active GPU budget.  If the step-5500 OAI score remains near `4.52`, the run should be stopped or demoted to a background research track.  The next primary run should use a sequential FineWeb-first launcher, preferably starting from the standalone `records/track_10min_16mb/2026-03-19_TrainingOptSeq4096/train_gpt.py` or the main `amelie-iska/parameter-golf/train_gpt.py` with equivalent long-context settings.

Operational policy for the pivot:

1. Keep BPB and serialized artifact bytes as the first two gates.
2. Use legal left-to-right scoring and legal score-first test-time adaptation only.
3. Keep GraphCG, toric geometry, Toric BGG, Koszul, Slepian, topology, memory, and GFlowNet as diagnostics or tiny late auxiliaries until the sequential model is below `1.2` BPB.
4. Record every launch command, checkpoint, validation BPB, quantized round-trip BPB, and artifact byte count in `planning/SEQUENTIAL-FINEWEB-PIVOT.md`.
5. Continue the graph/reasoning curriculum only after a contest-credible sequential model exists.

### Launch update

The random-order recovery sessions were stopped before step 5500 because the resumed trainer was still around step 5361, W&B was rejecting some resumed metrics as out-of-order, and the last durable checkpoint remained step 5250.  The available RTX 4090 was reassigned to the sequential FineWeb-first branch.

Active sequential run:

```text
tmux session: toricgt_seq4096_pivot
run id: toricgt_seq4096_pivot_seed1337_20260603T1718Z
log: amelie-iska/parameter-golf/logs/toricgt_seq4096_pivot_seed1337_20260603T1718Z.txt
```

Smoke verification for the patched `TrainingOptSeq4096` record script passed initial validation, one backward step, final validation, int8+zlib export, and quantized round-trip evaluation.  The one-step smoke round-trip BPB was `4.10907054`, as expected for an essentially untrained model, and the int8+zlib total size was `4,972,577` bytes.

### Checkpointed relaunch update

The first sequential launch was stopped at step 10 to add opt-in checkpoint/resume support before committing several hours of GPU time.  The durability patch writes model, optimizer, loader, RNG, step, and accumulated training-time state at validation boundaries when `CHECKPOINT_EVERY` is set.

Checkpoint/resume smoke verification passed:

```text
save smoke: toricgt_seq4096_resume_smoke_a
saved checkpoint: checkpoints/toricgt_seq4096_resume_smoke/toricgt_seq4096_resume_smoke_a_step_000001.pt
checkpoint bytes: 135,612,859
resume smoke: toricgt_seq4096_resume_smoke_b
resume result: loaded step 1, trained through step 2, saved a new checkpoint
post-resume val_bpb: 4.1067
post-resume int8+zlib round-trip val_bpb: 4.10907022
post-resume int8+zlib total size: 4,983,519 bytes
```

Active checkpointed sequential run:

```text
tmux session: toricgt_seq4096_pivot
run id: toricgt_seq4096_wandb_ckpt_seed1337_20260603T1743Z
parameter-golf commit: 00ebe09
wandb: https://wandb.ai/amelie-iska-math/toricgt-parameter-golf/runs/toricgt_seq4096_wandb_ckpt_seed1337_20260603T1743Z
log: amelie-iska/parameter-golf/logs/toricgt_seq4096_wandb_ckpt_seed1337_20260603T1743Z.txt
console log: amelie-iska/parameter-golf/logs/toricgt_seq4096_wandb_ckpt_seed1337_20260603T1743Z.console.txt
checkpoint dir: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_wandb_ckpt_seed1337_20260603T1743Z
checkpoint cadence: every 1000 validation steps
```

W&B is now default-on in the local Parameter Golf record script, using `WANDB_ENTITY=amelie-iska-math` and `WANDB_PROJECT=toricgt-parameter-golf` unless overridden.  The script reads the API key only from `WANDB_API_KEY` or local `keys.txt`; key files are ignored and must not be committed.

The W&B metric sweep is documented in `planning/WANDB-SWEEP.md`.  It records that the active run logs canonical BPB as `val/bpb`, while Parameter Golf commit `7dc7d59` adds expanded direct reporting for progress, target-gap, throughput, artifact, checkpoint, VRAM, BPB aliases, and best-BPB metrics for the next resume or launch.
