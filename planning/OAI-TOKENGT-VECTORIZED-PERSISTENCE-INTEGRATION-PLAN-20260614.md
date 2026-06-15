# OAI TokenGT Fusion And Vectorized Persistence Integration Plan

Date: 2026-06-14

## Purpose

The OAI Parameter-Golf adapter must stop treating graph structure as only a
side loss.  Every random-order byte stream already defines a legal directed
graph: reveal steps are vertices, original byte positions define local
neighborhood edges, and the reveal order supplies the causal rank.  The model
should tokenize that graph in TokenGT style, process node and edge tokens under
a rank-causal graph attention mask, and fuse the resulting node contexts back
into the byte logits.

Analogical memory retrieval must also use vectorized persistence, not only a
small scalar topology summary.  The existing GUDHI audit path computes exact
finite persistence for offline reports, and the training path already has
differentiable persistence landscape/image functions.  The missing piece is a
compact persistence signature inside `TrajectoryRetrievalHead`, so retrieval
teachers and traces explicitly compare trajectories by persistence landscapes,
persistence images, and persistence entropy.

## Constraints

- Score-before-update validity is non-negotiable.  The graph-token fusion path
  may use `previous_tokens`, target positions, reveal ranks, and derived graph
  incidence.  It may not use the current target token or any future unrevealed
  target token.
- The graph-token path must be real model computation: node and edge tokens,
  endpoint features, causal graph attention, and residual fusion into logits.
  It is not acceptable to only log an auxiliary TokenGT-style loss.
- Vectorized persistence must be a first-class retrieval feature with its own
  similarity term, W&B metrics, and trace components.
- Exact GUDHI/Macaulay2/Sage sidecars remain the audit/report path.  The
  in-training persistence signature must be differentiable and cheap enough for
  recurrent byte-model training.
- All new behavior must be configurable and default-on in the full all-phases
  training config, with finite tests proving the enabled paths run.

## Implementation Plan

1. **Native graph-token fusion for the byte model**

   Add `use_tokengt_graph_fusion`, `tokengt_graph_fusion_weight`,
   `tokengt_graph_fusion_layers`, and small fixed feature dimensions to
   `RandomOrderLMConfig`.  Instantiate a `GraphTokenizer`, one or more
   `TransformerBlock`s, and a projection/gate inside `DenseRandomOrderToricLM`.

   Build a prefix-safe `GraphBatch` from `(previous_tokens, target_positions)`:

   - node features: normalized original position, normalized reveal rank,
     strict-prefix previous-token byte-class features, and toric phase features;
   - edge features: source/target position deltas, reveal deltas, local radius
     flags, and phase-difference features;
   - edge index: local-neighborhood directed edges whose source reveal rank is
     no greater than destination reveal rank;
   - node causal rank: reveal rank.

   Tokenize with `GraphTokenizer`, construct a graph attention mask with
   `attention_mask_from_token_mask`, run graph-token transformer layers, and add
   the node-token output to the byte hidden state before logits.

2. **Vectorized persistence signatures for retrieval**

   Extend `TrajectoryMemoryConfig` with persistence weight and vectorization
   dimensions.  Add a differentiable signature builder to
   `TrajectoryRetrievalHead`:

   - normalize and sample each hidden trajectory;
   - compute pairwise distances and finite birth/death diagrams for the local
     0-dimensional Rips filtration using nearest-neighbor deaths;
   - vectorize with `torch_persistence_landscape` and
     `torch_persistence_image`;
   - include persistence entropy, norm, and total persistence.

   Add persistence similarity to the in-batch teacher:

   \[
   T_{ij} =
     w_{\mathrm{GCG}} C_{ij} +
     w_{\Theta} P_{ij} +
     w_{\mathrm{top}} S_{ij} +
     w_{\mathrm{PH}} L_{ij} +
     w_{\mathrm{DAG}} D_{ij} +
     w_{\mathrm{der}} R_{ij} +
     q_j.
   \]

   The trace output must include persistence similarity, candidate persistence
   norms, and the global weight table.

3. **Training script and config wiring**

   Map new config fields through `scripts/train_parameter_golf_random_order.py`.
   Log graph-fusion metrics and trajectory-memory persistence metrics.  Enable
   native TokenGT graph fusion and causal graph loss from step 0 in
   `config/train.parameter_golf_all_phases.yaml`.  Keep Toric BGG loss weight
   late-phase only, but instantiate the probe from step 0 so metrics stay on.

4. **Tests**

   Add tests for:

   - graph fusion produces finite logits, graph-token metrics, and gradients;
   - graph fusion is prefix-safe because it accepts no `target_tokens` and is
     invariant when only target labels change;
   - vectorized persistence signatures are finite, nonzero on nontrivial
     trajectories, and influence retrieval teacher metrics;
   - all-phases config enables TokenGT graph fusion and persistence-weighted
     retrieval.

5. **Documentation**

   Update `README.md` to state that the OAI byte model now has a native
   TokenGT graph-token fusion path and that analogical memory retrieval uses
   vectorized persistence landscapes/images in addition to exact GUDHI/CAS
   sidecar audits.

## Acceptance Criteria

- `pytest tests -q` passes.
- `tests/test_random_order_lm.py` covers graph-token fusion.
- `tests/test_trajectory_memory.py` covers persistence-weighted retrieval.
- `config/train.parameter_golf_all_phases.yaml` enables graph-token fusion and
  causal graph metrics from step 0.
- Documentation and planning notes are committed and pushed.
