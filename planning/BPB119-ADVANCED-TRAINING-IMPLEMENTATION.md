# BPB<1.19 Advanced Training Implementation Plan

Generated: 2026-06-17

This plan implements the high-impact BPB changes discussed in chat, excluding item 3
(`Adaptive BPB-Slope Controller`).  The current training process must not be
interrupted until this checklist is implemented and tested.  After implementation,
run the full analysis suite on the current run, write a report and hyperparameter
update plan, then restart training from step 0 with the new configuration.

Primary objective: OAI FineWeb BPB remains the controlling score.  Graph structure
is the modeling paradigm, but OAI FineWeb output flattening is optional and scoped
only to OAI FineWeb BPB scoring.  General graph data must remain graph structured.

## Implementation Checklist

- [x] Conflict-aware auxiliary gradient routing.
  - Add optional PCGrad-style routing for auxiliary gradients against the primary
    FineWeb BPB gradient.
  - Route `graph_lm_primary` and `toricgt_sidecar` separately.
  - Log gradient cosine, projection scale, conflict flag, and routed gradient norm.
  - Preserve the primary FineWeb gradient exactly.
  - Keep default on for campaign profiles, with a config to disable.

- [x] Frozen baseline logit distillation.
  - Add optional teacher checkpoint loading for the OAI baseline model only.
  - Compute KL from teacher logits to student logits on FineWeb batches during
    early training.
  - Keep teacher frozen and excluded from export.
  - Support a fixed early schedule and a soft linear decay.
  - Log teacher KL, teacher weight, and checkpoint path.

- [x] Graph-LM curriculum with low early weight.
  - Replace fixed `GRAPH_LM_LOSS_WEIGHT` with a schedule:
    `start -> peak`, warmup start/end, optional post-warmup hold.
  - Keep graph data unflattened by default.
  - Use graph LM as primary training data, but keep FineWeb BPB dominant early.
  - Log scheduled graph weight separately from configured peak.

- [x] Reversible CaseOps-style transform for OAI FineWeb.
  - Implement a reversible, local, byte-safe case transform for FineWeb training
    and validation token streams.
  - Keep it optional and default-off unless enabled by campaign profile.
  - Ensure BPB byte accounting remains on original SentencePiece target bytes.
  - Apply the same transform to FineWeb train and validation when enabled.
  - Do not apply CaseOps to general graph data unless explicitly configured later.

- [x] Advanced loss family bandit profiles.
  - Add profile metadata describing active families and theoretical role.
  - Add `next_profile_decision` support for bandit-style exploration:
    exploit best BPB families, explore under-tested family subsets, and preserve
    all metric observability.
  - Do not implement item 3 adaptive intra-run BPB-slope control.
  - The bandit operates only between fresh 1.5K-step runs.

- [x] Retrieval-conditioned auxiliary training.
  - Gate analogy and trajectory-memory losses by retrieval confidence/features.
  - Use lenient thresholds initially.
  - If retrieval evidence is weak, keep diagnostics logged but reduce or zero
    retrieval/analogy gradient pressure for that batch.
  - Log gate score, active fraction, and gated loss weights.

- [x] Uncertainty-weighted toric/tropical/BGG/topological sidecar pressure.
  - Apply high-NLL emphasis to advanced sidecar losses.
  - The mathematical intent is local: shape uncertain/high-entropy decisions,
    not impose global geometry everywhere.
  - Add a normalized uncertainty scalar from per-token NLL.
  - Log uncertainty weight and affected family losses.

- [x] GraphCG full-rank BPB orthogonalization.
  - Keep full-rank GraphCG active from step 0.
  - Add a BPB-alignment/orthogonalization term that compares GraphCG axes with
    the local NLL gradient proxy or hidden-state likelihood direction.
  - Dampen covariance pressure when it conflicts with BPB-critical directions.
  - Log alignment, conflict, and effective GraphCG covariance weight.

- [x] Score-first TTA / warm adapter evaluation hooks.
  - Implement optional score-first adaptation for eval-only experiments:
    score a validation chunk first, then update tiny adapters only for later chunks.
  - Default off for campaign training unless explicitly enabled.
  - Ensure score-before-update semantics are visible in logs.
  - Restrict updates to graph-output flattening and first-class graph adapters.

- [x] BPB-safe TokenGT identifier and graph-output flattening refinement.
  - Add deterministic low-rank node identifiers to the first-class FineWeb
    TokenGT path without adding absolute-position tables large enough to
    threaten the artifact budget.
  - Add endpoint-pair features and virtual local edge-token states for causal
    FineWeb token graphs.
  - Fold graph-valued output states back to the original SentencePiece sequence
    with a gated OAI-FineWeb-only score-correction path.
  - Keep general graph data graph structured; only OAI FineWeb uses sequential
    flattening for BPB scoring.
  - Log identifier, endpoint, edge-token, virtual-edge, and score-correction
    weights independently so the 1.5K-step analysis can adjust them separately.

## Test Plan

- [x] `py_compile` all touched Python files.
- [x] Smoke-test graph stream serialization on a real graph shard.
- [x] Smoke-test OAI model forward paths:
  - FineWeb `forward()` uses optional flattening.
  - graph `forward_aux()` does not flatten by default.
  - teacher distillation can compute KL and backprop.
  - gradient routing can combine primary/aux gradients without NaN.
- [x] Dry-run campaign command and verify environment includes:
  - 1500-step cadence.
  - `GRAPH_LM_PRIMARY=1`.
  - optional OAI-FineWeb-only flattening.
  - all sidecar metric families.
  - conflict-aware routing knobs.
  - teacher distillation knobs.
  - graph-LM schedule knobs.
- [x] Do not interrupt the active training run until tests pass.

## Post-Implementation Protocol

1. Run full current-run analysis without assuming the current run used the new code.
2. Write a timestamped report under `training_notes/`.
3. Include:
   - FineWeb BPB and train BPB.
   - graph-LM BPB.
   - all W&B metrics, including every `toricgt_sidecar/*` metric.
   - graphification/flattening state.
   - gradient-routing metrics once available.
   - sidecar family correlations and mathematical interpretation.
4. Restart training from step 0 with the implemented configuration.
5. Preserve the 1.5K-step loop:
   - target `BPB < 1.19` by step 1500;
   - full checkpoint-backed analysis;
   - Codex review;
   - evidence-based fresh restart if missed.
