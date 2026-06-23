# ToricBLM Status Summary - 2026-06-23

Generated: 2026-06-23T21:41:44Z

## Current state

The ToricBLM data, documentation, paper updates, and full-run configuration have been implemented and pushed on branch `toricblm-data`.

Latest pushed commit:

```text
3ac0c75 Add ToricBLM bio FoT data and full-run config
```

An active 25K-step training run is currently running in tmux:

```text
tmux session: toricblm_fot_structure_20260623T211600Z
run id: toricblm-mup-fot-structure-20260623T211600Z
wandb: https://wandb.ai/amelie-iska-math/toricgt-parameter-golf/runs/toricblm-mup-fot-structure-20260623T211600Z
log: runs/oai_sidecar/toricblm-mup-fot-structure-20260623T211600Z/train.log
checkpoint dir: checkpoints/toricblm-mup-fot-structure-20260623T211600Z
```

Latest checked training snapshot:

```text
step:400/25000 train_loss:1.1864 train_bpb:0.5564 train_bpt:1.7116 train_time:1408267ms step_avg:3520.67ms aux_stage:0.0200
```

The initial validation at step 0 was high, as expected for the fresh scaled configuration and ConvexTok-8192 setup:

```text
step:0/25000 val_loss:9.0120 val_bpb:4.2186
```

Early training BPB dropped quickly from roughly `4.0876` at step 1 to `0.5564` at step 400. This is a useful early sign that the run is learning the primary BPB objective and that the scaled configuration is not obviously broken.

## What was implemented

### 1. Leakage-aware ToricBLM FoT dataset splits

I created leakage-aware train, validation, and test splits for the graphified ToricBLM FoT biological dataset:

```text
data/uniprot_fot/splits/leakage_v1/toricblm_fot_leakage_v1_train.parquet
data/uniprot_fot/splits/leakage_v1/toricblm_fot_leakage_v1_validation.parquet
data/uniprot_fot/splits/leakage_v1/toricblm_fot_leakage_v1_test.parquet
data/uniprot_fot/manifests/toricblm_fot_split_shards_report_v1.json
```

Split counts:

```text
train: 2788
validation: 156
test: 128
clusters: 2640
```

The splitter is cluster-aware so related entries stay together. This reduces leakage across related sequence/function/structure/FoT examples and makes validation/test more meaningful for the eventual biomedical reasoning objective.

Scripts added:

```text
scripts/split_toricblm_fot_parquet.py
scripts/apply_toricblm_fot_leakage_splits.py
scripts/extract_uniprot_fot_anchors.py
```

Why this was done: the new biological FoT dataset should not be split by random row when many rows can share sequence, structure, annotation, or reasoning-graph neighborhoods. Clustered splitting is the conservative baseline before later ProTrek/Foldseek similarity splits.

### 2. Tokenizer audit for ConvexTok-8192

I audited the current ConvexTok-8192 tokenizer on the new biological FoT fields and compared it against the older ConvexTok-2048 path.

Documents and scripts:

```text
planning/TORICBLM-FOT-TOKENIZER-AUDIT.md
scripts/analyze_toricblm_fot_tokenizer.py
```

Main result:

```text
ConvexTok8192 bytes/token: 1.416
ConvexTok8192 priced-token fraction: 0.261
ConvexTok8192 byte fallback fraction: 0.739
8192 vs 2048 token count reduction: about 7.21 percent on audited graph/FoT fields
```

Decision: keep ConvexTok-8192 for the current run rather than retraining the tokenizer immediately.

Why this was done: the 8192 tokenizer is already better than the 2048 tokenizer on the audited bio/FoT fields and does not require a disruptive re-tokenization pass before this run. The audit also shows that byte fallback is still high, so a future motif-aware tokenizer is justified, especially for amino-acid motifs, residue/atom/chain tags, GO/EC strings, coordinates, and dynamics annotations.

### 3. Structure-readiness and coordinate-native loss implementation

I added coordinate-native structure loss support in:

```text
src/toricgt/structure_flow_matching.py
tests/test_structure_flow_matching.py
scripts/analyze_toricblm_structure_readiness.py
```

Implemented functions include:

```text
pairwise distance computation
contact map construction
contact BCE loss
distogram cross-entropy
rectified-flow velocity loss
centered RMSD
aggregate structure_flow_loss
```

The current readiness report is:

```text
data/uniprot_fot/manifests/toricblm_fot_structure_readiness_train_v1.json
```

Current readiness summary:

```text
records: 2788
UniProt records: 916
chemistry records: 505
structure-association records: 917
coordinate-bearing records: 0
average graph nodes: 15.36
average graph edges: 20.61
average FoT nodes: 26.47
average FoT edges: 44.75
```

Important caveat: the structure losses are implemented and configured, but coordinate losses are dormant in this run because the current curated records contain structure associations and hooks, not actual coordinate tensors. This run trains structure-aware graph/FoT associations; coordinate-native flow/contact/distogram losses become active when coordinate-bearing PDB/AFDB/dynamics data is added.

Why this was done: the model needs a forward-compatible path into de novo biomolecular design and dynamics training without forcing a restart from scratch later. Adding the loss interface now lets the trainer log readiness and activate true coordinate losses once coordinate fields are present.

### 4. Late-stage biological FoT graph stream

I created a late mixed graph stream using symlinks rather than duplicate data:

```text
data/toricblm_late_mixed_fot_structure/train/*.parquet
data/toricblm_late_mixed_fot_structure/train_manifest.txt
```

It includes:

```text
Codex 5.5 ToT/FoT train_3pass shards
toricblm_fot_leakage_v1_train.parquet
```

Why this was done: the primary BPB objective should remain stable early in training, so the richer graph/FoT biological examples are introduced as a late-stage stream instead of overwhelming the initial BPB optimization. Symlinks also avoid wasting disk space.

### 5. ProTrek staged for future trimodal leakage control

I added `external/ProTrek` as a tracked submodule:

```text
external/ProTrek -> https://github.com/amelie-iska/ProTrek.git
```

It is currently a lightweight source checkout. No weights or inference were run in this pass.

Why this was done: ProTrek is the right future mechanism for multimodal split control because it jointly embeds protein sequence, structure, and function. The current split is cluster-aware and ProTrek-ready; the next stronger split should use ProTrek representations plus optional Foldseek/sequence identity clustering to prevent leakage across all three modalities.

### 6. Nested OpenAI baseline adaptation update

I updated the nested `amelie-iska/parameter-golf` trainer so the run logs ToricBLM structure-readiness metadata and exposes the structure training configuration to W&B.

Nested commit:

```text
d7db0ac Log ToricBLM structure readiness in trainer
```

The trainer now logs:

```text
toricblm_structure/flow_weight
toricblm_structure/contact_weight
toricblm_structure/distogram_weight
toricblm_structure/frame_weight
toricblm_structure/start_step
toricblm_structure/coordinate_records
toricblm_structure/structure_association_records
toricblm_structure/coordinate_losses_active
```

Why this was done: the training loop should distinguish between "structure associations are present" and "coordinate-native losses are active." Without these logs, reports could incorrectly claim that structure flow training is active when coordinate tensors are not yet present.

### 7. New 169.7M muP/ConvexTok/FoT full-run config

I added:

```text
configs/toricblm_mup_170m_convextok8192_biomed_fot_structure.env
```

Key settings:

```text
MODEL_DIM=1536
NUM_LAYERS=9
NUM_HEADS=12
NUM_KV_HEADS=6
VOCAB_SIZE=8192
TOKENIZER_KIND=convextok
TRAIN_SEQ_LEN=1024
TRAIN_BATCH_TOKENS=196608
ITERATIONS=25000
GRAPH_LM_PRIMARY=1
TORICGT_SIDECAR=1
OAI_GFLOWNET=1
OAI_EMBEDDING_FOT=1
OAI_MTP=1
LATE_GRAPH_START_STEP=12000
LATE_GRAPH_MIX_RATIO=0.80
```

Startup confirmed:

```text
model_params:169662063
fineweb_graphification:enabled:1
graph_lm_primary:enabled:1
toricgt_sidecar:enabled:1
oai_transfer_heads:gflownet_enabled:1 fot_enabled:1 mtp_enabled:1
toricgt_sidecar_heads:graphcg_full_rank:1536
```

Why this was done: the prior small-run optimization had strong early BPB behavior, but the longer run was underperforming relative to the best short-run BPB. This configuration scales the model while preserving the OAI baseline adaptation, ConvexTok, graphification, FoT/GFlowNet, GraphCG full-rank basis training, and advanced ToricGT sidecar metrics.

## Documentation and paper updates

Updated:

```text
README.md
docs/CURRENT_OAI_TORICGT_STATE.md
data/uniprot_fot/README.md
planning/TORICBLM-BIO-FOT-DATASET-PLAN.md
planning/TORICBLM-FOT-TOKENIZER-AUDIT.md
planning/TORICBLM-FOT-STRUCTURE-FULL-RUN-20260623.md
assets/toricgt_neurips_condensed.tex
assets/toricgt_toric_bgg_rewrite_amelie_schreiber.tex
```

The papers now mention:

```text
ConvexTok-8192 scale-up
bio/FoT graph data
leakage-aware splitting
late-stage ToT/FoT graph stream
coordinate-native structure losses
the current limitation that coordinate losses are dormant until coordinate-bearing records are added
```

LaTeX compile note: I attempted to compile the condensed paper, but the local TeX install is missing `fontspec` / `luaotfload`. This is a system TeX dependency issue, not a confirmed source-level paper error.

## Tests and validation

Executed:

```text
PYTHONPATH=src /home/iska/miniconda3/envs/iska-net-2/bin/python -m pytest \
  tests/test_structure_flow_matching.py \
  tests/test_dataset_split_policy.py \
  tests/test_convextok_integration.py -q
```

Result:

```text
7 passed
```

Also ran Python compile checks on the modified scripts/modules and the nested trainer.

## Why these decisions were made

The central constraint was to improve the ToricBLM/ToricGT path without breaking the OAI Parameter Golf BPB objective. The choices follow that priority:

1. Keep BPB primary by preserving the OAI baseline adaptation, ConvexTok scoring path, graph-output flattening, and FineWeb stream.
2. Add biological FoT data as late-stage graph training rather than immediately mixing it into every early BPB update.
3. Keep ConvexTok-8192 because the audit showed a real token-count improvement over 2048, while avoiding a costly and risky tokenizer retrain before this run.
4. Implement structure losses now, but accurately log that they are dormant until coordinate-bearing records exist.
5. Stage ProTrek for future trimodal leakage control instead of pretending that trimodal leakage splitting has already been performed.
6. Push documentation and paper updates before relying on the configuration, so the implementation state is reproducible.

## Current risks and caveats

1. Coordinate-native structure losses are not active yet because coordinate-bearing records are currently zero.
2. The current leakage split is cluster-aware, but not yet ProTrek/Foldseek-derived.
3. The biological FoT dataset is still small relative to the intended UniProt/PDB/AFDB/trajectory scale.
4. ConvexTok-8192 improves tokenization versus 2048, but byte fallback is still high on bio/FoT fields.
5. The current training run is early. The step-400 train BPB is promising, but validation and later-stage behavior are what matter.

## Next recommended actions

1. Let the current 25K-step run continue through at least the first validation checkpoints before making conclusions.
2. Watch whether train BPB improvements transfer to validation BPB after the early fast drop.
3. At late-stage activation around step 12000, inspect whether the biological FoT stream destabilizes BPB or improves auxiliary reasoning metrics.
4. Add coordinate-bearing PDB/AFDB examples next so `structure_flow_loss`, contact, distogram, and RMSD-style metrics become genuinely active.
5. Run ProTrek/Foldseek-based split refinement once the required model weights and structure tooling are staged.
6. Design a motif-aware ConvexTok follow-up if tokenizer regret remains high on biological symbols and structure/dynamics annotations.

## Local untracked files intentionally not pushed

The repo remains clean except for these local artifacts:

```text
final_model.pt
final_model.int8.ptz
```

They were not committed or pushed because they are generated model artifacts, not source/config/documentation changes for this status update.

