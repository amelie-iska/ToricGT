# Current OAI ToricGT State

Updated: 2026-06-23

## Goal

The active OAI/Parameter-Golf campaign keeps FineWeb BPB as the primary score
while promoting graph structure into the main model.  The latest short-run
campaign used a best-of-10 gate:

```text
target: BPB < 1.19
gate:   1000 optimizer steps
policy: run all 10 attempts, review full analyses/visualizations, pick the
        best configuration, and launch a full-data continuation
```

The current hardware target is roughly 20GB VRAM by using large token batches
with `TRAIN_SEQ_LEN=1024`.

## Previous Best Short Run

The best completed short-run configuration came from:

```text
run:      tg-bpb-bestof10-convextok2048-det-20260621T040717Z-r008-gate2500_experimental_family_selective-20260621T154110Z
profile:  gate2500_experimental_family_selective
step:     1000
train BPB: 0.5036
val BPB:   0.4924
int8+zlib round-trip val BPB: 0.49533264
artifact estimate: 12,202,796 bytes
```

That run remains the best small Parameter-Golf-style checkpoint snapshot, but
the active ToricBLM scale-up path has moved to a larger fresh ConvexTok
vocabulary for biomedical/universal-modality pretraining.

## Active ToricBLM ConvexTok-8192 Restart

```text
export tmux:    toricblm_convextok8192_export
wait tmux:      toricblm8192_wait_launch
train tmux:     toricblm_mup_full_8192 once export manifest exists
export log:     runs/convextok8192_biomed_export.log
wait log:       runs/toricblm8192_wait_and_launch.log
tokenizer:      /home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/data/toricgt/parameter_golf_convextok8192_biomed_det_full/tokenizers/fineweb_convextok_8192_biomed_det.convextok.json
FineWeb shards: /home/iska/Documents/amelie/bio/TropicalGT/TropicalGT-I/data/toricgt/parameter_golf_convextok8192_biomed_det_full/datasets/fineweb10B_convextok8192_biomed_det
config:         configs/toricblm_mup_170m_convextok8192_biomed_fot_structure.env
vocab size:     8192
target params:  169,698,927
layers:         9
width:          1536
heads / KV:     12 / 6
batch tokens:   196,608 initial setting
late graph:     data/toricblm_late_mixed_fot_structure/train/*.parquet
late start:     step 12,000
late mix:       0.80
```

The previous 2048 `.bin` token shards were removed to make room for the fresh
8192 export.  The 2048 tokenizer metadata remains for provenance and tokenizer
comparison.  The new tokenizer is fresh ConvexTok, not BPE and not a reused
2048 vocabulary.  It uses the standard byte fallback plus a 512-token target
reserve for biomedical/control syntax covering UniProt/PDB/AFDB/ESMFold,
molecular graph, UMA MLIP energy/force, dynamics trajectory, cell phenotype,
tropical-toric, persistence, category-theoretic, and Forest-of-Thought tags.

Tokenizer comparison against ConvexTok-2048:

```text
curated train sample:       993,945 -> 790,185 tokens (-20.50%)
biomed/control seed sample:   3,010 ->     669 tokens (-77.77%)
```

The new ToricBLM bio/FoT curation slice is published at
`AmelieSchreiber/toricblm_fot` and stored locally under `data/uniprot_fot`.
It contains graph/FoT records from UniProt function text, UniRef50, DNA coding
regions, Rfam, RNAcentral, and PubChem SELFIES.  The leakage-aware split uses
sequence sketches, function shingles, structure lookup hooks, source graph
histograms, and FoT graph topology:

```text
train:      2,788
validation:   156
test:         128
clusters:   2,640
```

The current tokenizer audit on the bio/FoT fields keeps ConvexTok-8192 for the
immediate full run: it reduces audited graph/FoT token count by 7.21% relative
to ConvexTok-2048, but raw biological sequence strings still use byte fallback
often enough that a later motif-aware vocabulary extension is recommended.

Structure readiness for the balanced graph/FoT train shard:

```text
structure-association records: 917
coordinate-bearing records:    0
```

That larger shard trains sequence/function/chemistry/structure-hook association
through graph LM, FoT, GFlowNet, and ToricGT sidecar losses.  Coordinate-native
training is now supplied by the AFDB v6 coordinate shard:

```text
data/uniprot_fot/structures/afdb_v6/toricblm_afdb_structure_fot_train.parquet
coordinate-bearing train records: 239
validation records: 7
test records: 10
readiness status: ready_for_coordinate_losses
```

The active run
`toricblm-mup-fot-afdb-structure-20260623T220856Z` uses this shard through
`TORICBLM_STRUCTURE_TRAIN_GLOB`; the adapted OAI baseline logs real
`toricblm_structure/*` metrics including flow-matching loss, contact BCE,
distogram CE, centered RMSD, coordinate count, and batch pLDDT.

The public page in `docs/index.html` was regenerated from the best checkpoint
with a 512-token hidden-state extraction and includes fresh interactive
trajectory, GUDHI, CAS, Toric BGG, ConvexTok, Forest-of-Thought, toric
embedding, vector-bundle, and theory-gallery reports.

The active run uses the updated first-class graph path:

```text
FINEWEB_GRAPHIFY=1
TOKENGT_FIRST_CLASS=1
CONVEXTOK_DAG_FEATURES=1
CONVEXTOK_TORIC_REG_WEIGHT>0
OAI_FINEWEB_OUTPUT_FLATTENING=1
GRAPH_OUTPUT_FLATTENING=1
GRAPH_OUTPUT_VIRTUAL_EDGE_TOKENS=1
GRAPH_OUTPUT_SCORE_CORRECTION=1
GRAPH_LM_PRIMARY=1
TORICGT_SIDECAR_COMPUTE_ALL_METRICS=1
```

It also uses the patched adaptive graph-radius and flattening calibration
policy.  FineWeb graphification starts in a local radius-2 or radius-3 regime.
The 1K-step analysis may widen the TokenGT and
graph-output-flattening radii to 4--6 only when the evidence says the graph path
is helping BPB: improving train-BPB slope, nonconflicting TokenGT graph loss,
positive graph-output flattening lift, and no strong W&B correlation indicating
that wider graph features are making BPB worse.

## June 19 BPB-Focused Controls

The current campaign includes four additional controls from
`planning/BPB-IDEAS-1-4-IMPLEMENTATION-20260619.md`:

- `BPB_FIRST_AUX_STAGING=1`: auxiliary graph-LM, sidecar, FoT, GFlowNet, and
  MTP pressure starts at a small multiplier, ramps only after the primary
  FineWeb BPB objective has early descent momentum, and is damped if BPB slope
  or curvature indicates a rebound.
- `AUX_CONFLICT_CONTROLLER=1`: the trainer keeps family-specific gradient route
  scales instead of treating all advanced techniques as one bin.  Conflicting
  auxiliary gradients are damped; aligned families can receive a bounded boost.
- `OAI_FOT_REWARD_MODE=bpb_delta`: the embedding-space Forest-of-Thought reward
  is grounded in the per-byte likelihood delta from its correction path through
  the tied LM head, with graph/consensus/complexity terms kept as bounded
  modifiers.
- `OAI_FOT_ADAPTIVE_CONTROL=1`: FoT temperature, UCB exploration, sparse
  pressure, and effective loss multiplier adapt from observed reward,
  tree-diversity, and activation-entropy metrics during each run.

## Autoregressive Graph Decoding Contract

The active OAI baseline remains causal autoregressive.  Graphification changes
the representation, not the BPB score contract.

```text
FineWeb BPB                 -> left-to-right autoregressive active-tokenizer sequence
FineWeb graphification      -> causal path graph over the same tokens
directed acyclic graph data -> topological autoregressive reveal ranks
cyclic/non-causal graphs    -> deterministic random-order autoregressive ranks
OAI FineWeb output          -> graph hidden states flattened back to sequence order
```

Edge-token and endpoint features are legal only when their reveal ranks are
prefix-valid.  FineWeb output flattening is OAI-only and optional; general
graph records remain graph structured and are not flattened by default.

## BPB-Safe FineWeb Graphification

FineWeb is graphified inside the model without adding extra scored tokens to
the official sequence.  Each active-tokenizer token is a graph node.  The model
adds:

- deterministic low-rank TokenGT-style node identifiers;
- endpoint-pair features for local causal one-dimensional edges;
- local edge-token features;
- toric phase features;
- virtual causal edge-token states folded back into node hidden states.

For OAI FineWeb only, graph-valued hidden states are flattened back to the
active tokenizer order with a gated sequence score-correction adapter.
This preserves graph-in/graph-out hidden computation while keeping BPB a terse
sequence score.  General graph records remain graph structured and should not
use the OAI-only flattening path unless explicitly configured.

## ConvexTok Tropical/Toric Tokenisation Path

ConvexTok training constructs a byte-boundary DAG and solves a sparse LP
relaxation on a configured tokenizer-training sample.  Det/Bias/Int rounding
selects priced substring colours; all byte fallback edges remain available.
Encoding is exact min-plus dynamic programming, so active token paths,
top-two margins, and path entropy are tropical diagnostics.  Candidate token
incidence and LP scores define a toric vocabulary polytope shadow: selected
priced tokens are points/faces, and changes in shortest paths correspond to
normal-fan wall crossings.

The OAI baseline consumes this structure through:

```text
CONVEXTOK_DAG_FEATURES=1          # LP score, rank, byte length, priced/free flag
CONVEXTOK_DAG_FEATURE_WEIGHT=...
CONVEXTOK_TORIC_REG_WEIGHT=...    # embedding geometry follows LP/rank/length face coordinates
tokenizer_regret/*
tokenizer_tropical/*
tokenizer_toric/*
```

The review loop treats tokenizer regret as its own evidence family.  A high LP
gap suggests a tokenizer-bound BPB problem; low regret with high BPB suggests
model or auxiliary-optimization bottlenecks.

## Disk And Artifact Hygiene

Large generated artifacts are not the canonical record of a run.  The canonical
record is the train log, campaign state, markdown report, and cleanup ledger.
The June 19 cleanup keeps five high-priority checkpoint binaries, compact
selected int8 artifacts, and all markdown reports while deleting redundant
full-precision checkpoints, redundant `final_model.pt` files, generated
`*-full-analysis*` payload directories under `training_notes`, noisy Codex
stderr logs, and Python caches.  The ledger is:

```text
docs/CLEANUP-LEDGER-20260619.md
```

The active ConvexTok export directory is explicitly excluded from cleanup until
the campaign is running and has confirmed usable shards.

## Separately Tuned Graph Knobs

These values are logged to W&B and adjusted independently by the adaptive
controller:

```text
TOKENGT_IDENTIFIER_DIM
TOKENGT_IDENTIFIER_WEIGHT
TOKENGT_ENDPOINT_WEIGHT
TOKENGT_EDGE_TOKEN_WEIGHT
GRAPH_OUTPUT_EDGE_TOKEN_WEIGHT
GRAPH_OUTPUT_SCORE_CORRECTION_WEIGHT
GRAPH_OUTPUT_CALIBRATION_LOSS_WEIGHT
GRAPH_OUTPUT_CALIBRATION_EVERY
```

The reason for separating them is empirical discipline: a bad BPB correlation
for one graph component is not evidence that all graph features are harmful.
The calibration loss is regression-only: it compares graph-output-flattened
FineWeb CE against the raw graph hidden-state CE on a small subset and penalizes
the flattening adapter only when flattening worsens the score.  This gives the
controller an explicit `graph_output_flattening/ce_lift` signal for deciding
whether score correction should be strengthened, held, or damped.

The controller now treats loss correlations with the correct sign.  For
loss-like metrics, positive correlation with BPB means the loss is high when BPB
is high, so pressure that reduces that loss may be useful if gradient routing is
clean.  Negative correlation is the suspicious case for a loss: the auxiliary
loss may already be low while BPB remains high, indicating conflict,
over-regularization, or a metric that should stay diagnostic-only.

## Advanced Loss Families

The current OAI baseline transfer also enables three BPB-facing heads/paths:

- `oai_gflownet/*`: a training-only embedding-space graph-of-thought
  GFlowNet head over hidden trajectories, with trajectory-balance residual,
  reward, entropy, action-diversity, score-gap, and `logZ` metrics.
- `oai_mtp/*`: FineWeb-only multi-token prediction at configured future
  offsets using the existing hidden states and tied LM head.
- `score_first_tta/*`: non-destructive score-first validation adaptation.
  Validation examples are scored first; adaptation is restored afterward while
  `SCORE_FIRST_TTA_COMMIT=0`.

The 1000-step review analyzes train BPB, validation BPB when available,
graph-LM BPB, artifact bytes, W&B metrics, screenshots, and the full
`toricgt_sidecar` metric family.  It considers these families separately:

- OAI embedding-space GFlowNet graph-of-thought pressure;
- OAI multi-token prediction;
- score-first test-time adaptation;
- GraphCG full-rank chart pressure;
- TokenGT graph loss;
- graph-LM primary stream;
- analogy and trajectory-memory retrieval;
- tropical/toric geometry;
- Toric BGG and Koszul persistence;
- vector-bundle/sheaf one-dimensional-cone metrics;
- combinatorial commutative algebra and derived-category diagnostics;
- Slepian/Pollak phase concentration and topology metrics.

The optimizer keeps the FineWeb cross-entropy/BPB objective dominant.  Advanced
families remain useful only when they improve BPB, graph-reasoning quality, or
diagnostic reliability without violating score-before-update causality.
All advanced families remain observable from step 0, but the aggregate sidecar
gradient multiplier is now warm-started: it starts small and ramps only after a
short hold.  This preserves rigorous metrics while reducing early competition
with the primary BPB objective.
