# OpenAI Parameter Golf Competition Modification Plan

Date: 2026-05-26

Status: planning only. Do not implement until approved.

Implementation note for branch `oai`: the plan is being implemented with JEPA
explicitly excluded. The representation-shaping slot is filled by contrastive
hidden-state regularization, stripped multi-token heads, toric memory entropy,
and score-first GFlowNet scaling instead.

## Scope

This plan translates the strongest public lessons from OpenAI's Parameter
Golf retrospective and highlighted submissions into a staged update path for
ToricGT. The goal is to improve the Parameter-Golf candidate while preserving
the core ToricGT commitments:

- random-order autoregressive graph-to-sequence decoding as the default
  competition-facing projection;
- dense compact deployment model for the artifact, not full Soft-MoE in the
  submitted path;
- tropical/ring/toric/graph structure used where it gives byte-efficient
  inductive bias;
- embedding-space graph-of-thought GFlowNet scaling as our legal
  score-first test-time strategy;
- all evaluation and adaptation remain prequential and validation-safe.

The plan intentionally separates changes that preserve checkpoint shape from
changes that require a new training run.

## Sources Reviewed

Primary OpenAI retrospective:

- [What Parameter Golf taught us](https://openai.com/index/what-parameter-golf-taught-us/)

OpenAI-highlighted top three nonrecord submissions:

- [LeWM: JEPA + Mamba2 + U-Net + INT4/FP8 QAT + Brotli](https://github.com/openai/parameter-golf/tree/main/records/track_non_record_16mb/2026-03-26_37M_LeWM_Jepa_Mamba2_10L_UNet_INT4FP8QAT_Brotli)
- [DG Attention: Designator/Guided Attention](https://github.com/openai/parameter-golf/tree/main/records/track_non_record_16mb/2026-03-23_DGAttention_DavidGao)
- [Byte-Level H-Net: dynamic chunking study](https://github.com/openai/parameter-golf/tree/main/records/track_non_record_16mb/2026-03-29_HNet_ByteVsSubword_Study)

Record-track ideas emphasized by the retrospective:

- score-first test-time training, especially per-document LoRA style
  adaptation;
- GPTQ-lite and full-Hessian GPTQ compression paths;
- CaseOps tokenizer sidecar accounting;
- XSA / partial exclusive self-attention;
- SmearGate and BigramHash feature mechanisms;
- mini depth recurrence;
- Muon optimizer, spectral embedding initialization, residual-mix schedules,
  and compiled evaluation;
- auxiliary training-only outputs that are stripped from the final artifact.

## Competition Lessons To Preserve

1. Artifact bytes matter more than nominal parameter count.

   The submitted artifact includes code, model state, tokenizer or byte
   machinery, metadata, and packing overhead. Any new module must be assessed
   by artifact bytes after compression, not by scalar parameter count alone.

2. Evaluation-time compute is valuable only when it is score-first.

   Test-time training and test-time scaling are useful if every scored
   probability depends only on already scored bytes, fixed parameters, and
   legal online state. Future validation bytes must never influence current
   probabilities.

3. Training-only structure is high leverage.

   JEPA heads, auxiliary latent predictors, multi-token prediction heads,
   graph consistency heads, and verifier heads can improve representation
   quality while contributing zero bytes if they are stripped during export.

4. Quantization must be trained and calibrated, not bolted on.

   Strong submissions used QAT or GPTQ-style calibration. For ToricGT, this
   means BPB should be measured after round-trip packed loading, and training
   should periodically expose the model to the quantized grid.

5. Data order and packing are first-class optimization variables.

   The OpenAI article singled out loader and stride choices. Our random-order
   graph projection makes this especially important: order seeds, document
   boundaries, graph trace packing, and held-out graph rows need explicit
   schedules.

## Analysis Of The Three Highlighted Nonrecord Methods

### 1. LeWM / JEPA + SSM + U-Net + QAT

Useful observations:

- JEPA is most valuable as a training-only latent prediction signal.
- SIGReg-style anti-collapse penalties can shape hidden-state geometry.
- U-Net/residual skips help narrow or recurrent sequence models maintain
  gradient flow.
- INT4/FP8 QAT plus Brotli-style packing is a credible path to fitting a much
  larger effective model under the 16 MB cap.
- Sliding-window evaluation reduces cold-start penalty for recurrent or
  stateful sequence mixers.

ToricGT translation:

- Add a training-only graph-latent JEPA head that predicts future random-order
  hidden states and graph-projection hidden summaries.
- Add a SIGReg-like isotropic latent regularizer to GFlowNet state embeddings,
  not to the deployment logits directly.
- Reuse the current dense recurrent blocks and residual mixing instead of
  adopting Mamba2 immediately.
- Add QAT/GPTQ stages before artifact export.

Decision:

- Adopt JEPA/SIGReg as training-only auxiliary losses.
- Do not replace the current dense random-order ToricGT LM with a Mamba2
  backbone in the first implementation pass. That would be a separate radical
  ablation.

### 2. DG Attention / Designator-Guided Payloads

Useful observations:

- DG attention attempted to transmit changes between tokens rather than raw
  absolute payloads.
- The reported DG variant reduced VRAM and artifact size, but did not beat a
  matched standard attention baseline in BPB.
- The useful lesson is not "copy DG attention"; it is that differential
  payloads and learned merge gates can reduce redundant sequence state.

ToricGT translation:

- Add optional differential residual streams:

  ```text
  delta_t = h_t - stopgrad_or_shift(h_{t-1})
  h'_t = h_t + gate_t * W_delta(delta_t)
  ```

- Apply this only inside training or as a tiny branch in the dense model if
  artifact audit permits.
- Use graph projection tags and tropical active-face tags as payload-change
  features.

Decision:

- Do not replace attention with DG attention.
- Add a small differential payload auxiliary or residual gate only after
  no-shape-change loader and quantization updates are complete.

### 3. Byte-Level H-Net / Dynamic Chunking

Useful observations:

- Byte-level dynamic chunking learned whitespace-aligned word-like boundaries
  from raw bytes.
- Compression before the main stage can reduce sequence length and compute.
- Chunking/dechunking interface depth mattered more than simply adding more
  depth to the compressed stage.
- The method was not competitive with top record-track BPB but is highly
  relevant to graph projection and inference scaling.

ToricGT translation:

- Treat H-Net boundaries as a model of learned graph trace segmentation:

  ```text
  bytes -> chunks -> graph/proof segments -> random-order decode
  ```

- Implement a training-only or lightweight runtime boundary head that predicts
  stable byte/span boundaries for graph-projected records.
- Let GFlowNet actions operate over chunk summaries as well as byte positions.
- Use chunk regularity and graph consistency as auxiliary rewards.

Decision:

- Add H-Net style learned chunk summaries as a training/inference scaling
  optional stage only after the core BPB path is stable.
- Avoid making dynamic chunking mandatory in the first competition artifact.

## Record-Track Techniques To Integrate

### A. Score-First Test-Time Adaptation

OpenAI highlighted score-first LoRA test-time training as a valid and important
edge of the rules. ToricGT should implement the same principle through a
competition-safe adaptation loop:

1. score the current chunk or token group;
2. record the log loss;
3. update only allowed online state from already scored bytes;
4. reset at document boundaries or explicit reset markers.

Candidate adapted parameters:

- GFlowNet action bias and temperature;
- low-rank adapters on the final projection;
- LayerNorm gain/bias;
- tiny input embedding offset table;
- graph projection chunk memory.

Do not adapt:

- full dense backbone weights by default;
- tokenizer or byte accounting;
- any state using future validation bytes.

Acceptance gate:

- A causal audit must prove that the probability for byte/token `t` is
  unchanged when bytes after `t` are permuted.

### B. GPTQ / Hessian-Aware Export

Current ToricGT export has simple symmetric quantization. Competition results
suggest we need a stronger path:

- collect calibration activations only from training data or self-generated
  model text, never validation;
- implement GPTQ-lite first, then optional full-Hessian GPTQ for the largest
  matrices;
- compare int8, int6, int4, and mixed int4/FP8 storage;
- always evaluate the loaded packed artifact, not the float checkpoint.

Acceptance gate:

- packed BPB delta from float must be `<= 0.01` for int8/int6 and justified by
  BPB improvement or byte savings for int4.

### C. CaseOps, BigramHash, And SmearGate

These are attractive because they are cheap.

CaseOps:

- represent capitalization/case transforms as side-channel operator tokens or
  byte features;
- preserve exact original byte reconstruction for BPB accounting;
- use only if implementation does not endanger tokenizer-agnostic scoring.

BigramHash:

- add a small hash embedding for adjacent byte or token pairs;
- combine with graph-projection type tags and Hebrew/toric symbolic tags;
- parameter budget can be tiny: e.g. 1024 to 4096 buckets with low dimension.

SmearGate:

- add an input-dependent softmax/logit temperature or previous-token blending
  gate;
- use for random-order decoding to condition confidence on revealed context
  density and graph evidence density.

Acceptance gate:

- each feature must improve validation BPB by at least `0.003` in two local
  seeds or be removed.

### D. Mini Depth Recurrence

The current Parameter-Golf path already uses dense recurrence ideas. The next
step is to make recurrence scheduled:

- start with shallow effective depth for early optimization;
- enable repeated middle blocks after a warmup fraction;
- partially untie only the smallest MLP or gate parameters if artifact audit
  permits;
- log recurrence depth and activation cost to W&B.

Acceptance gate:

- recurrence schedule must not reduce tokens/sec enough to erase BPB gains.

### E. Auxiliary Heads Stripped At Export

Add training-only heads:

- random-order multi-token prediction offsets;
- graph latent JEPA;
- graph consistency verifier;
- tropical active-face prediction;
- GFlowNet reward/flow calibration;
- chunk boundary prediction.

All must be excluded from `parameter_golf_export.py` unless explicitly marked
as deployment parameters.

Acceptance gate:

- artifact byte audit reports stripped parameter count and confirms no
  training-only tensors are serialized.

## GFlowNet Test-Time Scaling Plan

ToricGT's version of competition test-time scaling should be an embedding-space
graph-of-thought sampler, not generic best-of-N text sampling.

At each prefix state:

```text
s_t = (prefix bytes, visible graph projection, random order, chunk summaries,
       hidden state, GFlowNet action trace, legal online memory)
```

The GFlowNet samples actions such as:

- choose a random-order decoding seed/order;
- select a graph projection refinement;
- select a tropical active-face hypothesis;
- select a toric phase or symbolic consistency feature;
- choose a chunk boundary summary;
- adjust legal temperature/confidence;
- read or write a score-first online memory slot;
- terminate the latent deliberation and emit logits.

For scoring, use a mixture distribution:

```text
q(y_t | prefix) =
  average_{order seeds, GFlowNet actions, legal adapted states}
    q_theta(y_t | prefix, order, action_trace, state)
```

Implementation constraints:

- Every sampled branch must be prefix-visible only.
- If adaptation is enabled, score before update.
- Branch weights may come from the GFlowNet policy, but the final distribution
  must remain normalized.
- The number of branches must be configurable for local eval and disabled or
  reduced for time-limited artifact scoring if too slow.

Training objective:

```text
L = L_CE
  + lambda_gfn L_TB_or_SubTB
  + lambda_jepa L_latent_prediction
  + lambda_graph L_graph_consistency
  + lambda_entropy H(action_policy)
```

Reward proxy after score:

```text
R_t = exp(
  - NLL_t
  + alpha * graph_consistency
  + beta * tropical_margin
  - gamma * branch_cost
  - eta * equivariance_error
)
```

This keeps the reinforcement signal aligned with BPB while retaining ToricGT's
graph reasoning diagnostics.

## Staged Implementation Plan

### Stage 0: No Checkpoint-Shape Changes

Safe to apply to the current training run or the next resumed run.

1. Add coprime-stride and document-boundary-aware dataloader schedules.
2. Add graph-record packing modes:
   - random order per sample;
   - random order per epoch;
   - fixed seed for reproducibility only;
   - dynamic seed for default training.
3. Add sliding-window or warm-prefix evaluation for random-order state, while
   preserving score-first accounting.
4. Add W&B metrics:
   - order seed entropy;
   - graph projection density;
   - GFlowNet branch count;
   - branch-normalized BPB;
   - packed artifact BPB delta.
5. Add a validation leakage audit:
   - future-byte permutation test;
   - document-boundary reset test;
   - score-before-update test.

Expected impact:

- modest BPB improvement;
- better reproducibility;
- minimal risk.

### Stage 1: Small Shape Changes With High Leverage

Requires a fresh or intentionally resumed compatible run.

1. Add BigramHash embeddings.
2. Add SmearGate confidence/temperature control.
3. Add CaseOps-like side-channel features only if exact byte accounting is
   straightforward.
4. Add training-only multi-token prediction offsets.
5. Add training-only graph-latent JEPA and SIGReg heads.
6. Add stripped graph consistency and active-face auxiliary heads.

Expected impact:

- BPB improvement from cheap local features;
- better hidden-state geometry for GFlowNet scaling;
- no artifact size increase from stripped heads.

### Stage 2: Quantization And Artifact Export Upgrade

Can proceed in parallel with Stage 1 experiments.

1. Add QAT toggles for int8/int6/int4 grids.
2. Add GPTQ-lite calibration for large matrices.
3. Add optional full-Hessian GPTQ for final candidates.
4. Add mixed storage:
   - int4 for large linear matrices;
   - int6/int8 for sensitive matrices;
   - FP8 or BF16 for embeddings/scales if needed.
5. Add compressor selection across zstd, lzma, and Brotli.
6. Add round-trip artifact evaluation as the only official local BPB.

Expected impact:

- larger effective model under 16 MB;
- fewer surprises at submission time.

### Stage 3: Score-First GFlowNet Test-Time Training

Requires strict validation-safety tests before use.

1. Implement legal online state object.
2. Let the GFlowNet update only:
   - action priors;
   - small low-rank final adapters;
   - LayerNorm scalars;
   - online graph memory.
3. Reset at document boundaries.
4. Add SWA/EMA over adapted parameters within a document if it is strictly
   prefix-derived.
5. Expose config:
   - `tts.gflownet_branches`;
   - `tts.score_first_adaptation`;
   - `tts.adapter_rank`;
   - `tts.reset_on_document`;
   - `tts.max_online_steps`.

Expected impact:

- potentially large evaluation BPB improvement;
- highest leakage risk, so auditing must be strict.

### Stage 4: Optional Hierarchical Chunking

This is useful but should wait until Stages 0 to 3 are measured.

1. Add H-Net style boundary head over bytes or graph-projected tokens.
2. Convert chunks to graph-of-thought states.
3. Let GFlowNet sample chunk refinements and branch over chunk summaries.
4. Keep the chunk head training-only at first.
5. Promote a tiny runtime chunker only if it improves BPB per byte.

Expected impact:

- better long-context use;
- more natural graph projection;
- moderate implementation risk.

### Stage 5: Radical Ablations Only If Needed

These should not block the main competition path.

1. Mamba2 or RWKV-style linear mixer.
2. DG-style differential attention replacement.
3. n-gram trie/hash sidecar model.
4. Parallel residual branches.

Each is compatible with ToricGT theory, but each is large enough to deserve a
separate branch and ablation.

## Priority Order

Implement in this order after approval:

1. Stage 0 dataloader/eval/audit changes.
2. Stage 2 round-trip export and GPTQ-lite scaffolding.
3. Stage 1 BigramHash, SmearGate, and stripped auxiliary heads.
4. Stage 3 score-first GFlowNet test-time adaptation.
5. Stage 4 chunk summaries.
6. Stage 5 radical ablations only after BPB plateaus.

This order maximizes expected BPB improvement per disruption.

## Acceptance Gates

Every accepted modification must satisfy:

- `pytest -q tests` passes.
- Artifact remains below `15,600,000` bytes target, leaving code/metadata
  safety margin below the official `16,000,000` byte cap.
- Round-trip artifact evaluation is reported.
- W&B reports:
  - train and validation BPB;
  - artifact bytes;
  - quantization mode;
  - random-order seed entropy;
  - GFlowNet loss, entropy, branch count, and action diversity;
  - test-time adaptation steps and adapted parameter count when enabled.
- Causal audit passes for:
  - random-order decoding;
  - GFlowNet branch sampling;
  - online adaptation;
  - graph projection;
  - chunk summaries if enabled.

Recommended BPB decision thresholds:

- no-shape-change improvement: keep if repeated delta is `<= -0.003 BPB`;
- checkpoint-shape change: keep if repeated delta is `<= -0.010 BPB`;
- quantization change: keep if artifact BPB is no worse than float by `0.01`
  unless it unlocks a larger model that wins overall;
- test-time scaling: keep if wallclock-normalized BPB improves and causal
  audit passes.

## Non-Goals

- Do not reintroduce full Soft-MoE into the Parameter-Golf artifact unless a
  dense baseline is beaten under the byte cap.
- Do not let GFlowNet or TTT see future validation bytes.
- Do not train on held-out test-time scaling datasets.
- Do not change the core ToricGT research model to chase a single competition
  trick unless ablations justify it.
- Do not submit a nonrecord-style architecture unless it also satisfies the
  official record constraints.

## Expected Final Competition Path

The strongest ToricGT competition candidate should be:

```text
Dense random-order ToricGT LM
  + graph projection into byte/token traces
  + dynamic random order seeds
  + compact GFlowNet action routing
  + BigramHash / CaseOps-style cheap input features
  + SmearGate confidence control
  + stripped JEPA / graph consistency / multi-token auxiliary heads
  + QAT + GPTQ-lite/full-Hessian export
  + score-first GFlowNet test-time scaling
  + optional chunk summaries if BPB-positive
```

This keeps the model faithful to ToricGT while adopting the strongest
competition-proven ideas: legal test-time adaptation, better packing, cheap
feature engineering, training-only representation shaping, and rigorous
artifact-first evaluation.
