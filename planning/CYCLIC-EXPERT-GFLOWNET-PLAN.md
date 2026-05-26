# Cyclic Expert GFlowNet Training Plan

Author: Amelie Schreiber

## Purpose

This plan adds a second training paradigm for ToricGT: cyclic expert specialization
followed by inter-expert GFlowNet distillation. The goal is to make the default
4-expert Soft-MoE less homogeneous. Each expert is first trained on a different
hash-disjoint subset of the training corpus. The expert-to-subset assignment is
then rotated in a cyclic or braided order until every expert has trained on every
subset. After full coverage, experts distill into one another through
embedding-space GFlowNet rewards and graph-of-thought rollouts, so each expert
learns both its own recency-biased skill and the reasoning traces produced by
the others.

## NVIDIA Reasoning Data To Add

The NVIDIA/Nemotron and PhysicalAI sources are useful because they add
procedural reasoning, temporal/spatial reasoning, and physical-scene reasoning
families that are underrepresented in pure math CoT corpora.

| Dataset | License | Role |
|---|---|---|
| `nvidia/Nemotron-RL-ReasoningGym-v1` | CC-BY-4.0 | 15k procedurally generated, algorithmically verifiable RL reasoning tasks across algebra, arithmetic, computation, cognition, geometry, graph theory, logic, and games. |
| `nvidia/Nemotron-Content-Safety-Reasoning-Dataset` | CC-BY-4.0 | Label-justification reasoning traces generated for content-safety/topic-control decisions; useful for verifier and explanation-graph training. |
| `nvidia/PhysicalAI-Traffic-Anomaly-Reasoning` | CC-BY-4.0 | Video QA, scene understanding, temporal reasoning, anomaly detection, and chain-of-thought annotations over transportation videos. |
| `nvidia/PhysicalAI-SpatialIntelligence-Lyra-SDG` | CC-BY-4.0 | Optional 3D/4D spatial reconstruction data; do not include in default text-only curation without a visual/3D graph adapter. |
| `nvidia/PhysicalAI-Spatial-Intelligence-Warehouse` | gated CC-BY-4.0 | Optional synthetic 3D scene-understanding corpus; requires access approval and a separate storage plan. |
| `nvidia/PhysicsNeMo-CFD-Ahmed-Body` | Apache-2.0 | Optional small 3D CFD/geometry field data for graph-physics adapters. |
| `nvidia/PhysicsNeMo-Datacenter-CFD` | Apache-2.0 | Optional OpenFOAM datacenter CFD simulations for physics reasoning over typed geometry/field graphs. |

The implemented default curation manifest adds the three text/reasoning-facing
datasets. The spatial and CFD datasets remain opt-in because they need a
separate converter from 3D geometry, video, or simulation fields into graph
records.

## Dataset Partitioning

Let the training split be partitioned into `K` disjoint subsets by stable row
identity:

```text
subset(row) = blake2b(salt || group_hash || content_hash || record_id) mod K
```

This is implemented at stream time, so the 53GB Parquet splits do not need to be
rewritten. The partition fields preserve leakage-control semantics because
`group_hash` already encodes task-family keys, content hashes, and SimHash
prefixes. For `E=4` experts, the default is `K=4` subsets.

## Braided Expert Schedule

Let `phase_steps` be the number of optimizer steps assigned to one
expert/subset pair. At phase

```text
phase = floor((global_step - start_step) / phase_steps)
active_expert = phase mod E
round = floor(phase / E)
```

The cyclic schedule assigns:

```text
subset = (active_expert + round) mod K
```

The braided schedule assigns:

```text
subset = (active_expert + (K - 1) * round) mod K
```

For `E=K=4`, the braided orders are:

```text
expert 0: 0 -> 3 -> 2 -> 1
expert 1: 1 -> 0 -> 3 -> 2
expert 2: 2 -> 1 -> 0 -> 3
expert 3: 3 -> 2 -> 1 -> 0
```

Thus every expert sees every subset exactly once after four rounds, but the
recency order differs across experts. This is the intended specialization: after
full coverage, every expert has broad exposure, but each remains biased toward
the subset it saw most recently.

## Training Semantics

During an expert-curriculum phase, all Soft-MoE blocks are restricted to the
active expert by masking inactive experts in dispatch and combine weights.
Inactive experts receive no expert-MLP gradients for that step. Shared
attention, tokenization, graph heads, and the GFlowNet policy continue to train
normally. This is a pragmatic compromise between training four separate models
and training one fully mixed Soft-MoE from the beginning.

The active expert restriction is reset during validation so validation measures
the full Soft-MoE model.

## Inter-Expert GFlowNet Distillation

After every expert has seen every subset, the curriculum enters full-coverage
mode. For a step assigned to expert `e`, a different expert `t` is selected as a
teacher. The current batch is evaluated twice:

1. with teacher expert `t` active, under `torch.no_grad`;
2. with student expert `e` active, with gradients.

The student receives an optional expert-distillation loss:

```text
L_distill = masked_mse(student_node_output, teacher_node_output)
```

The embedding-space GFlowNet reward is also shaped by an out-reasoning
advantage:

```text
R = exp(-L_student) * exp(clamp(L_teacher - L_student, -5, 5))
```

When the student beats the teacher on the supervised graph target, the terminal
reward increases. When it underperforms, the reward decreases. This implements
the requested "learn to out-reason the expert" behavior without leaving graph
embedding space.

## Resume Command

Resume from the latest checkpoint with braided expert curriculum:

```bash
tmux new-session -d -s toricgt_train_cyclic_experts '
cd /home/iska/Documents/amelie/bio/ToricGT &&
conda run --no-capture-output -n tokengt env PYTHONPATH=src python scripts/train.py \
  --data-path data/curated/train.parquet \
  --val-data-path data/curated/validation.parquet \
  --resume checkpoints/toricgt_full_30m/toricgt_step_00002000.pt \
  --checkpoint-dir checkpoints/toricgt_full_30m \
  --steps 98000 \
  --attention hybrid \
  --device cuda \
  --d-model 384 \
  --num-heads 8 \
  --num-layers 8 \
  --batch-size 2 \
  --grad-accum-steps 16 \
  --precision bf16 \
  --wandb \
  --gflownet-loss-weight 0.05 \
  --gflownet-space embedding \
  --expert-cyclic-curriculum \
  --expert-curriculum-subsets 4 \
  --expert-curriculum-phase-steps 500 \
  --expert-curriculum-start-step 2000 \
  --expert-curriculum-order braid \
  --expert-curriculum-distill-weight 0.05 \
  --checkpoint-every 2000 \
  --eval-every 1000
'
```

If only `toricgt_step_00001000.pt` is available, use that checkpoint. The run
will still be valid; it will simply discard any unsaved progress from the older
training process.

## W&B Metrics

The new mode reports:

```text
expert_curriculum/phase
expert_curriculum/round
expert_curriculum/active_expert
expert_curriculum/subset_id
expert_curriculum/full_coverage_cycles
expert_curriculum/full_coverage_complete
expert_curriculum/teacher_expert
train/expert_distill_loss
train/expert_teacher_supervised_loss
```

Existing Soft-MoE expert mass and entropy metrics remain active. The expected
early signal is sharper expert mass during active-expert phases and nonzero
distillation metrics only after full coverage.

## Validation And Ablations

Run at least these comparisons:

1. default mixed Soft-MoE baseline;
2. cyclic order without distillation;
3. braided order without distillation;
4. braided order with inter-expert distillation;
5. braided order with GFlowNet reward shaping but no direct MSE distillation;
6. full Soft-MoE continuation after at least one full coverage cycle.

The key diagnostic is whether full-coverage braided training improves held-out
graph-to-graph validation, GFlowNet terminal diversity, and graph-of-thought
rollout quality compared with the default mixed Soft-MoE model.
