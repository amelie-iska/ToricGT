# Hybrid Byte-BPB TokenGT Training

This path keeps the OpenAI Parameter-Golf objective byte-level while adding
TokenGT-style graph supervision inside the random-order model.

## Why The Hybrid Exists

The official BPB metric is computed from byte log probabilities. A pure graph
token model can report useful graph-token bits-per-token proxies, but those are
not the same object as byte-level BPB. The hybrid path therefore uses
`DenseRandomOrderToricLM` for byte logits and official BPB, then applies graph,
topological, toric, BGG, Koszul, Slepian-Pollak, GFlowNet, memory, and analogy
losses to the hidden trajectory that produces those logits.

The current OAI-baseline adaptation applies the same rule to the SP1024
baseline script.  FineWeb is graphified in the primary model, but the scored
object is still the original SentencePiece sequence.  The model receives
TokenGT-style node identifiers, endpoint-pair features, local
one-dimensional-edge features, toric phase features, and virtual causal
edge-token states.  These are folded into graph hidden states and then
flattened back to sequence order through an OAI-FineWeb-only score-correction
adapter before BPB is computed.  This lets the model use edge-token structure
without making the official scorer pay for verbose graph serialization.

The active adaptive policy starts graph neighborhoods conservatively and widens
them only when evidence supports doing so.  Radius 2--3 is the default early
BPB regime.  Later restarts may use radius 4--6 only when train BPB is falling,
the graph/edge losses are not conflicting with BPB, and graph-output flattening
has positive CE lift.  This keeps local sequence evidence dominant early while
still allowing more generous graph context when the model demonstrates that it
can use it.

## Graph Construction

For a byte sequence `x[0:L]`, random-order autoregression samples a permutation
`pi`. At reveal step `k`, the model scores byte `x[pi[k]]` using only bytes
revealed at steps `< k`.

The internal TokenGT graph is:

```text
vertex v_k      = hidden state for reveal step k
position p_k    = pi[k]
local edge      = |p_i - p_j| <= r
causal edge     = local edge and i < j
```

When a causal order is meaningful, the graph objective uses directed acyclic
edges. When a source graph is cyclic or has no trusted causal interpretation,
`tokengt_graph_noncausal_policy: undirected_regularizer` disables direction and
cycle terms and keeps only undirected structural matching.

For OAI FineWeb, the graph is the causal path graph induced by the original
sequence. For general graph records, the graph remains a graph: directed
acyclic records use their causal/topological order, and noncausal records use a
deterministic content-independent reveal order only for autoregressive scoring
or regularization. Only the OAI FineWeb path should set
`OAI_FINEWEB_OUTPUT_FLATTENING=1`.

## Native TokenGT Route

The native TokenGT trainer also has a score-valid FineWeb route.  It is not a
full bidirectional graph encoder for the LM objective.  When
`use_causal_graph_attention: true` and `use_lm_token_embeddings: true`, every
graph batch carries `node_causal_rank`:

```text
FineWeb chain       -> rank(i) = i
directed acyclic GoT -> rank(i) = topological_rank(i)
cyclic/noncausal GoT -> rank(i) = deterministic_random_rank(sample_id, node_id)
```

Edge-token ranks are the maximum rank of their endpoints.  The TokenGT
attention mask is then:

```text
can_attend(query, key) =
    valid(query) and valid(key) and causal_rank(key) <= causal_rank(query)
```

This gives left-to-right autoregression on FineWeb, topological
autoregression on DAG-like reasoning graphs, and random-order autoregression
on cyclic or non-causal graphs without inventing a false direction.  Only this
causal-token route is allowed to emit the OAI-style restricted FineWeb aliases
`03_validation/oai_parameter_golf_restricted_fineweb_bpb`,
`oai_competition/bpb`, `competition/oai_bpb`, and `bpb/oai_competition`.
Non-causal or legacy TokenGT runs must keep their graph-node BPB estimates
under `tokengt/*` proxy namespaces.

The route-(1) TokenGT config ports the OAI baseline efficiency stack in a
parameter-conscious way:

```text
tied SP1024 LM head        -> fewer stored weights and better embedding/logit alignment
recurrent block passes     -> more compute per stored parameter
target-position embedding  -> OAI-style target-position conditioning
toric position phase       -> compact sinusoidal/cocycle position features
strict-prefix hash context -> n-gram-like memory without a dense 1024x1024 table
revealed-neighbor hash     -> legal graph-adjacent prior from earlier reveal ranks
SP operator features       -> CaseOps-style token class features for SentencePiece
SmearGate temperature      -> input-dependent logit sharpness
```

Every term is computed from already revealed input ids, public target
positions, reveal ranks, or graph structure.  The dense SP1024 bigram-bias
table remains available as an ablation, but the default causal route disables
it because the hash-context and revealed-neighbor embeddings are smaller
versions of the same idea and leave more room under the 16 MB artifact cap.  A
neighbor contributes only when its reveal rank is strictly less than the query
rank: FineWeb therefore uses the left-to-right chain, DAGs use their
topological predecessors, and cyclic/noncausal graphs use the deterministic
random-order reveal.

## Differentiable Objective

The training loss remains:

```text
L = L_byte_ce
  + w_gfn L_gflownet
  + w_tg L_tokengt_graph
  + w_gcg L_graphcg
  + w_top L_analogy_topology
  + w_toric L_toric
  + w_bgg L_bgg
  + w_koszul L_koszul
  + w_slepian L_slepian_pollak
  + w_mem L_trajectory_memory
  + ...
```

The TokenGT graph term is:

```text
L_tokengt_graph =
    a_edge BCE(sim(h_i,h_j)/tau, edge_ij)
  + a_dir  E_edges [relu(m - cos(h_j-h_i, phi(p_j)-phi(p_i)))^2]
  + a_pos  SmoothL1(d_h(i,j), log(1+|p_i-p_j|))
  + a_cls  BCE(sim(h_i,h_j)/tau, same_byte_class_ij)
  + a_cyc  E_backward_local_edges sigmoid(sim(h_i,h_j)/tau)
```

Here `phi(p)` is the model's toric phase feature map. The direction and cycle
terms are active only under causal policies.

Graph-output flattening has an explicit BPB calibration guard:

```text
L_flat_cal = max(0, CE(flattened graph output) - CE(raw graph hidden output) + margin)
```

The raw graph hidden state is detached before applying the flattening adapter,
so this loss mainly trains the FineWeb-only flattening and score-correction
path. It is intentionally one-sided: helpful flattening is logged as positive
CE lift, while harmful flattening receives a small correction penalty. The
1.5K controller uses this lift/regression signal separately from GraphCG,
topology, toric, BGG, Koszul, memory, and graph-LM signals.

## Config

Use:

```bash
PYTHONPATH=src python scripts/train_parameter_golf_random_order.py \
  --config config/train.parameter_golf_random_order_hybrid_tokengt.yaml \
  --checkpoint-dir checkpoints/$RUN_NAME
```

The config starts from step 0, keeps `oai_competition` validation enabled, and
logs the official restricted FineWeb BPB under:

```text
03_validation/oai_parameter_golf_restricted_fineweb_bpb
```

The new objective logs under:

```text
tokengt_graph/loss
tokengt_graph/edge_bce
tokengt_graph/direction_loss
tokengt_graph/position_loss
tokengt_graph/byte_class_loss
tokengt_graph/cycle_loss
tokengt_graph/edge_density
tokengt_graph/causal_edge_fraction
```

`00_primary/tokengt_graph_loss` is also emitted for quick dashboard review.

The OAI-baseline route additionally logs:

```text
fineweb_graphify/identifier_weight
fineweb_graphify/endpoint_weight
fineweb_graphify/edge_token_weight
graph_output_flattening/virtual_edge_tokens
graph_output_flattening/edge_token_weight
graph_output_flattening/score_correction
graph_output_flattening/score_correction_weight
graph_output_flattening/ce_lift
graph_output_flattening/ce_regression
graph_output_flattening/calibration_loss
```

The adaptive 1.5K-step controller adjusts these separately from GraphCG,
analogy, memory, toric, BGG, Koszul, vector-bundle/sheaf, and combinatorial
toric loss weights.  A poor correlation for one family is not treated as a
blanket rejection of graphification.

The OAI-baseline route now adds two BPB-facing training auxiliaries to this
hybrid objective:

```text
oai_gflownet/loss       embedding-space graph-of-thought trajectory balance
oai_gflownet/entropy    policy entropy for the graph-of-thought action buckets
oai_gflownet/reward     detached likelihood-derived terminal reward
oai_mtp/loss            FineWeb-only multi-token prediction CE at future offsets
```

These auxiliaries use the same graphified hidden states that produce the BPB
logits.  They do not change the official scoring order, and they are routed
through auxiliary-gradient controls before touching the backbone.  The review
loop should interpret them separately: GFlowNet is a hidden-trajectory
organization and exploration signal, while MTP is a local likelihood
acceleration signal.

Advanced sidecar pressure is also scheduled globally. Metrics and examples are
computed from step 0, but the aggregate sidecar gradient multiplier starts
small, holds briefly, and ramps. This keeps the mathematical audit always on
without forcing all algebraic/topological objectives to compete with early
FineWeb cross-entropy at full strength.

## Artifact Size

The current hybrid config is `d_model=448`, `num_heads=8`, 7 stored dense
blocks, and 2 recurrent passes. A local exporter probe using 6-bit row
quantization and LZMA produced:

```text
artifact bytes: 15,192,560
limit:          16,000,000
margin:            807,440
```

The d480 variant exceeded the cap in the same probe, so d448 is the largest
tested width under the current counted artifact boundary. If a future
submission package must also count additional code bytes, d416 is the safer
fallback width.
