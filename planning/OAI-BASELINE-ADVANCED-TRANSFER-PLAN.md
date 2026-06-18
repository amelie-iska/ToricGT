# OAI Baseline Advanced Transfer Plan

This plan transfers the high-value training techniques implemented in the native/random-order ToricGT path into the OAI Parameter-Golf baseline adaptation without breaking the BPB-first contract.  The OAI baseline remains the score-defining model: SentencePiece FineWeb BPB is primary, graphified FineWeb and graph-output flattening are active, graph LM/sidecar streams are auxiliary, and all new advanced methods must be measurable, byte-accounted, and easy to disable.

## Operating Rules

- Preserve BPB as the primary objective and primary restart gate.
- Keep graph-in/graph-out structure active, while preserving OAI FineWeb sequence flattening only for BPB scoring.
- Prefer training-only probes/heads unless a technique directly improves exported BPB after int8+zlib roundtrip.
- Route auxiliary gradients against the primary FineWeb gradient whenever the objective touches the backbone.
- Log every transferred technique to W&B with activation flags, weights, and outcome metrics.
- Treat each technique independently in the adaptive review; do not collapse all advanced losses into one bucket.

## Transfer Items

### 1. Embedding-Space GFlowNet Graph-of-Thought Head

Transfer target: native `RandomOrderLM` GFlowNet policy, action embedding, flow/log-Z objective, entropy metrics, action-diversity metrics, and validation GFlowNet sampling.

OAI design:

- Add a lightweight training-only `OAIEmbeddingGFlowNetHead`.
- Treat each hidden FineWeb/graphified sequence as a graph-of-thought trajectory in embedding space.
- Use token/action buckets as discrete refinement actions and per-token NLL as a terminal reward signal.
- Train with a normalized trajectory-balance residual:
  `logZ + mean(log p_F - log p_B) - log reward`.
- Add entropy-target pressure and action-diversity metrics.
- Route gradients through the same auxiliary-gradient conflict projection used by graph LM and sidecar losses.

Activation now:

- Enable by default in the OAI campaign with a small weight.
- Use only a small number of sequences per step to avoid disrupting throughput.
- Report `oai_gflownet/*` metrics every train log interval.

Later:

- Feed sampled action embeddings back into graph-output flattening or a low-rank logit adapter only after the pure training signal shows positive BPB correlation.

### 2. Score-First Test-Time Adaptation

Transfer target: native score-first output-bias adaptation and OAI test-time-scaled BPB reporting.

OAI design:

- Keep score-before-update semantics: score each validation batch first, then adapt on that batch.
- Make validation adaptation non-destructive by default: restore adapted parameters after evaluation.
- Restrict adaptation to graphification/flattening modules unless explicitly overridden.
- Log `score_first_tta/*` configuration and whether the TTA state is committed or restored.

Activation now:

- Enable non-destructive TTA in the campaign with conservative steps and LR.
- Do not commit TTA updates into training unless evidence shows stable BPB improvement without validation leakage risk.

Later:

- Add paired deterministic-vs-score-first validation in a single evaluation call so W&B can report both `val/deterministic_bpb` and `val/score_first_bpb`.

### 3. Multi-Token Prediction Auxiliary

Transfer target: native MTP offsets.

OAI design:

- Add a small FineWeb-only MTP loss using existing hidden states and tied LM head.
- Predict tokens at offsets 2..K from the current hidden state.
- Gate with low weight, small max sequence count, and aux-gradient routing.

Activation now:

- Enable offset-2 MTP with a small weight.
- Use as early BPB acceleration pressure, then let adaptive restarts adjust weight.

### 4. Order-Sampled / Random-Order Decoding

Transfer target: native random-order autoregressive graph completion and order-sampled validation.

OAI design:

- Preserve OAI FineWeb sequential BPB.
- Add optional graph-stream random reveal only for graph LM records, not for the BPB FineWeb scorer.
- Add order-sampled eval as a separate diagnostic, not as primary BPB.

Activation now:

- Not enabled. It is more invasive than GFlowNet/MTP and can change the score contract.

### 5. Inference-Time GFlowNet Sampling

Transfer target: native GFlowNet-sampled validation and `oai_competition/test_time_scaled_bpb`.

OAI design:

- Stage 1: train/log the embedding GFlowNet head.
- Stage 2: sample low-rank action contexts at validation time and evaluate whether mixture/logit adapters improve BPB.
- Stage 3: only export a GFlowNet/logit adapter if int8+zlib roundtrip improves.

Activation now:

- Train the head and log proxy metrics. Do not alter logits at inference yet.

### 6. Trajectory Flow Regularization

Transfer target: native trajectory-flow loss, kinetic energy, and viscous dissipation metrics.

OAI design:

- Reuse hidden trajectories from FineWeb/graphified graph streams.
- Penalize excessive hidden-state acceleration only when BPB is unstable.

Activation now:

- Not enabled; useful after GFlowNet/MTP metrics are stable.

### 7. Contrastive Retrieval Alignment

Transfer target: native contrastive hidden-state loss.

OAI design:

- Align graphified FineWeb spans and graph LM spans when their graph metadata hashes or local TokenGT signatures match.

Activation now:

- Not enabled; needs paired examples to avoid noisy contrastive pressure.

### 8. Slepian-Pollak Phase Concentration

Transfer target: native `slepian_pollak/*` training loss and toric phase coherence metrics.

OAI design:

- Use as a training-only regularizer on toric phase channels and hidden trajectory summaries.

Activation now:

- Not enabled; leave as sidecar/analysis until BPB-positive correlation is established.

### 9. QAT / PolarQuant Training

Transfer target: native QAT and PolarQuant probes.

OAI design:

- Keep current post-training int8+zlib roundtrip as the hard gate.
- Add QAT only after the current architecture reaches target BPB before export.

Activation now:

- Not enabled. QAT can slow early BPB descent.

### 10. Complexity, Hessian, and Causal-Audit Metrics

Transfer target: native complexity metrics, Hessian probes, future-permutation audit.

OAI design:

- Keep these in periodic analysis for now.
- Promote to in-training W&B only if overhead is low.

Activation now:

- Not enabled in the trainer; already available through full-iteration analysis.

## Initial Activation Set

The first activation should be conservative:

- `OAI_GFLOWNET=1`
- `OAI_GFLOWNET_LOSS_WEIGHT=2e-5`
- `OAI_GFLOWNET_ENTROPY_WEIGHT=2e-6`
- `OAI_GFLOWNET_ENTROPY_TARGET=1.8`
- `OAI_MTP=1`
- `OAI_MTP_LOSS_WEIGHT=0.003`
- `OAI_MTP_OFFSETS=2`
- `SCORE_FIRST_TTA=1`
- `SCORE_FIRST_TTA_STEPS=64`
- `SCORE_FIRST_TTA_LR=2e-5`
- `SCORE_FIRST_TTA_COMMIT=0`

These are expected to help early BPB without changing exported inference behavior.  Adaptive restarts should modulate each weight separately based on train BPB slope, validation BPB, graph LM BPB, sidecar family correlations, GFlowNet entropy/diversity, and MTP regression.

## Adaptive Review Requirements

At each gate:

- Compare train BPB slope against `oai_gflownet/loss`, reward, entropy, action diversity, and gradient conflict.
- Compare MTP loss and MTP weighted loss against train BPB slope.
- Check score-first TTA delta once paired deterministic-vs-score-first evaluation is added.
- Keep graph LM and sidecar family analysis separate: GraphCG, analogy, trajectory memory, toric geometry, vector bundle/1D-cone, BGG, Koszul, and CCA each receive separate interpretation.
- Increase only the families whose metrics improve BPB or improve graph/reasoning diagnostics without BPB regression.
- If GFlowNet entropy collapses, raise entropy weight or lower GFlowNet loss weight.
- If GFlowNet reward improves but BPB worsens, keep GFlowNet as a diagnostic and reduce backbone gradient routing.
- If MTP reduces train BPB but hurts validation BPB, shorten offsets or reduce weight.

## Implementation Status

- [x] Added `OAIEmbeddingGFlowNetHead` to the OAI baseline adaptation as a training-only embedding-space graph-of-thought head.
- [x] Added trajectory-balance, reward, entropy, diversity, score-gap, and log-Z metrics under `oai_gflownet/*`.
- [x] Added FineWeb-only multi-token prediction under `oai_mtp/*`, using existing hidden states and the tied LM head.
- [x] Added non-destructive score-first TTA defaults; validation adaptation restores adapted parameters unless `SCORE_FIRST_TTA_COMMIT=1`.
- [x] Activated conservative defaults in `scripts/run_oai_sidecar_bpb_campaign.py` for future campaign runs.
- [x] Extended periodic W&B metric review to classify and describe `oai_gflownet/*`, `oai_mtp/*`, `score_first_tta/*`, `graph_lm_primary/*`, `flattening_calibration/*`, `teacher_distill/*`, and `aux_grad/*` as separate families.
- [x] Verified Python compilation for the modified trainer and campaign/review scripts.
- [x] Ran a CPU smoke test covering OAI GFlowNet gradients, TokenGT graphification, graph-output flattening, and MTP loss.

The remaining items are intentionally staged.  Inference-time GFlowNet logit/action mixing, graph-stream random reveal, trajectory-flow regularization, contrastive retrieval alignment, Slepian-Pollak phase regularization, QAT, and PolarQuant should be enabled only after the first activation set produces BPB-positive evidence or clear diagnostic value without destabilizing primary FineWeb BPB.
