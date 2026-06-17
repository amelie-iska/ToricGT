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

The next restart also uses the patched adaptive graph-radius and flattening
calibration policy.  FineWeb graphification starts in a local radius-2 or
radius-3 regime.  The 1.5K-step analysis may widen the TokenGT and
graph-output-flattening radii to 4--6 only when the evidence says the graph path
is helping BPB: improving train-BPB slope, nonconflicting TokenGT graph loss,
positive graph-output flattening lift, and no strong W&B correlation indicating
that wider graph features are making BPB worse.

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
All advanced families remain observable from step 0, but the aggregate sidecar
gradient multiplier is now warm-started: it starts small and ramps only after a
short hold.  This preserves rigorous metrics while reducing early competition
with the primary BPB objective.
