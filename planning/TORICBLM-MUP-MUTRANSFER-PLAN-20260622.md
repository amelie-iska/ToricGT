# ToricBLM muP / mu-Transfer Scale-Up Plan

Date: 2026-06-22

This document records the muP review and implementation plan for scaling the ToricGT/OAI-baseline adaptation toward a ToricBLM run at roughly 10x the current parameter count.  The goal is not a Parameter Golf submission artifact; it is a larger biomedical-preparation reasoning model that preserves the best small-model training dynamics while adding capacity.

## Sources Reviewed

- Local repository: `external/mup`
- Local README: `external/mup/README.md`
- Paper: Greg Yang et al., *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer*, arXiv:2203.03466

The README and paper agree on the operational rule: use maximal update parametrization, tune hyperparameters on a small proxy model, then transfer those hyperparameters to a wider model after verifying coordinate checks.  Standard parametrization does not preserve optimal learning rates under width scaling.

## Theory Summary

muP separates coordinates into finite-width and infinite-width axes.  Hidden matrices, embeddings, and readouts are scaled so that feature coordinates and update magnitudes remain comparable when width changes.  Under this parametrization, the best learning rate and multiplier settings for the base model often transfer to a much wider model without re-sweeping.

Important consequences for ToricBLM:

1. **Width transfer is the main target.**  Depth transfer is less reliable.  The first scaled model should primarily increase width and avoid an aggressive depth change.
2. **Attention must use muP-compatible scaling.**  Transformer attention should use a `1/d`-style scale rather than the usual `1/sqrt(d)` behavior.  In PyTorch SDPA terms, this means passing an explicit scale proportional to `sqrt(base_head_dim) / head_dim`.
3. **Readouts need special handling.**  The current OAI baseline uses tied embeddings.  We therefore emulate the `MuSharedReadout` effect by scaling the hidden state before the tied embedding projection by `output_mult / width_mult`.
4. **Optimizer learning rates must be width-aware.**  The local baseline uses Muon for matrix parameters and Adam for token/scalar/auxiliary groups, not plain MuAdam.  The implementation therefore applies explicit LR scaling by parameter family.
5. **Coordinate checks matter.**  Before committing a long run, small forward/backward smoke checks should verify that activations, logits, losses, and updates remain finite and in a comparable range.

## Current Baseline and Scale Target

The most recent long run used about 18.7M parameters.  The target scale is approximately 10x larger, around 170M parameters.  The first practical target keeps the vocabulary and broad architecture family fixed, increases model width, and keeps the layer count moderate.

Planned generated config:

- Vocabulary: ConvexTok deterministic 2048
- Sequence length: 1024
- Base width for muP shapes: 512
- Delta width for muP shapes: 768
- Target width: 1536
- Layers: 9
- Heads: 12
- KV heads: 6
- Width multiplier: 3.0 relative to the base-width shape file
- Attention: GQA with muP attention scale enabled
- Embeddings: tied
- Graphification: enabled
- TokenGT first-class structure: enabled
- Graph-output flattening for OAI FineWeb BPB: enabled
- GFlowNet: enabled
- Embedding-space FoT: enabled
- Late-stage codex5.5_ToT graph stream: enabled after the configured late-start step

Exact parameter count is produced by `scripts/generate_toricblm_mup_config.py` rather than guessed.

## Implementation Design

### Base shapes

`scripts/generate_toricblm_mup_config.py` builds three CPU models:

- base model at width 512,
- delta model at width 768,
- target model at width 1536.

It then calls:

```python
from mup import set_base_shapes
set_base_shapes(target, base, delta=delta, savefile=..., rescale_params=False)
```

The saved base-shape file is used by the training script when `TORICBLM_MUP=1`.

### Trainer hooks

The adapted trainer `amelie-iska/parameter-golf/train_gpt.py` now exposes:

- `TORICBLM_MUP`
- `MUP_BASE_SHAPES`
- `MUP_WIDTH_MULT`
- `MUP_OUTPUT_MULT`
- `MUP_ATTENTION_SCALE`
- `MUP_ATTENTION_BASE_HEAD_DIM`
- `MUP_RESCALE_PARAMS`
- `MUP_MATRIX_LR_SCALE_POWER`
- `MUP_SCALAR_LR_SCALE_POWER`
- `MUP_TOKEN_LR_SCALE_POWER`
- `MUP_AUX_LR_SCALE_POWER`

When enabled, the trainer:

1. applies the base-shape file to the model;
2. switches attention to explicit muP scaling;
3. scales tied readout logits by `output_mult / width_mult`;
4. scales optimizer LRs by parameter family;
5. logs all effective LRs and muP state to W&B.

### LR transfer policy

Initial values are conservative because the 25K run drifted upward after an excellent early BPB valley.

- Matrix/Muon LR: scaled by `1 / width_mult`
- Token embedding LR: initially unscaled, because tied embeddings are the main language-model scoring table and the previous short runs were highly sensitive to embedding LR
- Scalar/control LR: initially unscaled
- Auxiliary LR: scaled by `1 / sqrt(width_mult)` to avoid over-driving graph/geometry heads at larger width
- FoT loss weight: kept nonzero but reduced relative to short-run exploratory settings
- GFlowNet loss weight: kept nonzero and modest
- BGG/derived: kept as small regularizers and full metrics, not large losses

This is not the final hyperparameter theory.  It is the safest first mu-transfer point given the observed FoT collapse and auxiliary saturation.

## Late-Stage codex5.5_ToT Integration

The new dataset `AmelieSchreiber/codex5.5_ToT` is train-only.  It contains 3,890 graph records with columns for task family, reward, tropical margin, active support nodes, safety metadata, nodes, edges, targets, and full record JSON.

Preparation script:

```bash
python3 scripts/prepare_codex55_tot_late_stage.py \
  --output-dir /home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/data/toricgt/codex55_tot_late_stage \
  --passes 3 \
  --shard-size 1000
```

The script writes:

- `train/*.parquet` with the original train-only records,
- `train_3pass/*.parquet` as a deterministic three-pass late-stage view,
- `manifest.json` documenting source SHA, row count, and output paths.

The training script uses `ScheduledGraphParquetTokenStream` to switch graph LM and sidecar batches toward this stream after `LATE_GRAPH_START_STEP`.

## Planned Validation and Tests

Before a full scaled run:

1. `py_compile` the changed trainer and scripts.
2. Build a small codex5.5_ToT smoke dataset and read one batch through `GraphParquetTokenStream`.
3. Generate the base-shape file and env config.
4. Verify the env config reports target parameter count and expected data paths.
5. Run a minimal trainer import/model-instantiation smoke test on CPU.
6. If the full GPU is available and not running another job, run a tiny low-step smoke test with reduced dimensions before launching the actual 170M run.

## Full-Run Launch Policy

The user requested the next scaled run, but also stated that training should wait until the scaled version is ready.  Therefore the implementation should produce a ready-to-run config and launch script, but the long run should not be started until the setup passes tests and the restart decision is explicit.

The eventual full command should:

1. source `configs/toricblm_mup_170m_codex55.env`;
2. run from the ToricGT project root;
3. use the full ConvexTok-2048 FineWeb data path already present under `/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/data`;
4. use the three-pass codex5.5_ToT late-stage graph stream;
5. validate and checkpoint every 1K steps;
6. keep FoT, GFlowNet, TokenGT graphification, graph-output flattening, GraphCG, BGG, Koszul, vector-bundle/one-dimensional-cone, toric, tropical, persistence, and derived-category metrics active.

## Risks and Mitigations

- **FoT collapse repeats:** reduce FoT weight, keep reward/diversity gates, and use codex5.5 late-stage records as harder structured supervision.
- **Graph auxiliary stream saturates:** late-stage stream begins after the model has learned the main BPB distribution; graph stream remains train-only.
- **muP implementation mismatch:** run coordinate checks and use explicit LR/readout/attention scaling even though the local baseline uses custom Muon rather than MuAdam.
- **Memory pressure:** target width may need batch-token reduction.  Keep sequence length fixed first; adjust batch tokens before reducing model width.
- **Artifact cap:** ToricBLM scaled runs are not competition-size artifacts.  Parameter Golf exports should continue to use the smaller best checkpoint path.

## Implementation Checklist

- [x] Clone or verify `external/mup`.
- [x] Generate train-only codex5.5_ToT graph Parquet output.
- [x] Generate muP base shapes.
- [x] Generate ToricBLM env config.
- [x] Add a launch script that sources the config without hardcoding secrets.
- [x] Run compile and stream smoke tests.
- [x] Verify that the target model accepts the generated base-shape file after the trainer dtype path.
- [x] Record final paths and readiness state.
- [ ] Start full training only when the ready state is confirmed.

## Final Generated Paths

- Base-shape file: `configs/mup/toricblm_base512_delta768_layers9.bsh`
- Scale-up env config: `configs/toricblm_mup_170m_codex55.env`
- Launch script: `scripts/launch_toricblm_mup_full_codex55.sh`
- Late-stage train-only manifest: `/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/data/toricgt/codex55_tot_late_stage/manifest.json`
- Late-stage 3-pass graph glob: `/home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/data/toricgt/codex55_tot_late_stage/train_3pass/*.parquet`

The generated target model has `160,261,743` parameters.  It is therefore slightly under the verbal 170M target but close to the requested 10x scale and preserves a width-first muP transfer path.
