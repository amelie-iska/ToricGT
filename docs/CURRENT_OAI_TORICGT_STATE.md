# Current OAI ToricGT State

Updated: 2026-06-17

## Goal

The active OAI/Parameter-Golf campaign keeps FineWeb BPB as the primary score
while promoting graph structure into the main model.  The working gate is:

```text
target: BPB < 1.19
gate:   1500 optimizer steps
policy: if the gate is missed, run full analysis, write a report, adjust
        hyperparameters, and restart from step 0
```

The current hardware target is roughly 20GB VRAM by using large token batches
with `TRAIN_SEQ_LEN=1024`.

## Active Run Versus Next Restart

The run that was already active when this document was updated is pre-patch.
It should be allowed to reach its 1500-step gate.  The takeover watcher then
runs the full analysis suite and starts the next run from step 0 using the
patched code.

The next restart uses the updated first-class graph path:

```text
FINEWEB_GRAPHIFY=1
TOKENGT_FIRST_CLASS=1
OAI_FINEWEB_OUTPUT_FLATTENING=1
GRAPH_OUTPUT_FLATTENING=1
GRAPH_OUTPUT_VIRTUAL_EDGE_TOKENS=1
GRAPH_OUTPUT_SCORE_CORRECTION=1
GRAPH_LM_PRIMARY=1
TORICGT_SIDECAR_COMPUTE_ALL_METRICS=1
```

## BPB-Safe FineWeb Graphification

FineWeb is graphified inside the model without adding extra scored tokens to
the official sequence.  Each SentencePiece token is a graph node.  The model
adds:

- deterministic low-rank TokenGT-style node identifiers;
- endpoint-pair features for local causal one-dimensional edges;
- local edge-token features;
- toric phase features;
- virtual causal edge-token states folded back into node hidden states.

For OAI FineWeb only, graph-valued hidden states are flattened back to the
original SentencePiece order with a gated sequence score-correction adapter.
This preserves graph-in/graph-out hidden computation while keeping BPB a terse
sequence score.  General graph records remain graph structured and should not
use the OAI-only flattening path unless explicitly configured.

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
```

The reason for separating them is empirical discipline: a bad BPB correlation
for one graph component is not evidence that all graph features are harmful.

## Advanced Loss Families

The 1500-step review analyzes train BPB, validation BPB when available,
graph-LM BPB, artifact bytes, W&B metrics, screenshots, and the full
`toricgt_sidecar` metric family.  It considers these families separately:

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
