# Full TokenGT GoT/FineWeb Curriculum

This plan is for the full graph-structured TokenGT run after the OAI Parameter-Golf BPB recovery gate is stable.  The graph-of-thought trajectory is a directed acyclic graph with branching and merging vertices, not a linear chain.  Plain text rows, FineWeb-derived rows, and older linear `graph_json` rows are converted into branch/merge DAG cells before they reach the TokenGT model.

## Objectives

- Keep the OAI Parameter-Golf BPB gate visible and decisive: validation BPB remains the early score metric, and the run must continue only when OAI BPB does not regress.
- Use full curated GoT, ToT, CoT, graph-reasoning, analogy, safety-reasoning, and memory-retrieval datasets through `data/curated_hf_shards/train/*.parquet`.
- Train the TokenGT base model on graph objects where reasoning steps are vertices and dependencies, branches, merges, memory links, and analogy links are directed edges.
- Train memory retrieval from branch/merge trajectory summaries, including DAG topology, toric phase features, GraphCG chart coordinates, and derived-category chain-complex features.
- Keep artifact size pressure explicit with the 16 MB target.  Model width should remain small enough for int8/zlib or PolarQuant-packed artifacts to satisfy the competition limit.

## Phase 0: Data and Graphification

Every row is converted to a graph before tokenization.

1. Existing `graph_json` rows are preserved when they already contain branch and merge structure.
2. Linear `graph_json` rows receive conservative diamond cells:
   \[
   v_i \to \{v_{i+1}, v_{i+2}\} \to v_{i+3}.
   \]
3. Raw text rows become a document root plus reasoning-step spans and branch/merge edges.
4. The dataset loader expands directories and globs, so `data/curated_hf_shards/train/*.parquet` uses the full shard set.

The graphification invariant is:
\[
G_\tau=(V_\tau,E_\tau),\qquad E_\tau \subset V_\tau \times V_\tau,
\]
where \(G_\tau\) is acyclic in intended reasoning time but may contain undirected simplices after forgetting orientation.

## Phase 1: BPB-Preserving Warmup

Use OAI Parameter-Golf validation BPB as the early gate while TokenGT learns the full graph domain.

Recommended starting settings:

- `got_dag_loss_weight = 5e-4`
- `trajectory_memory_loss_weight = 5e-5`
- `derived_category_loss_weight = 1e-5`
- `gflownet_loss_weight = 1e-2`
- `batch_size = 2`
- `grad_accum_steps = 32`
- `d_model = 192`, `num_layers = 6`, `num_heads = 6`

The losses are deliberately small.  The supervised graph reconstruction objective remains the main signal, and advanced losses shape the geometry without overwhelming BPB.

## Phase 2: Branch/Merge DAG Geometry

For hidden states \(h_v\) on reasoning vertices \(v\), the DAG loss measures:

- branch count and merge count;
- back-edge fraction;
- branch diversity;
- merge scatter;
- simplex edge and triangle densities.

The key branch/merge pressure is:
\[
\mathcal L_{\mathrm{DAG}}
= \lambda_a \rho_{\mathrm{back}}
+ \lambda_b [\sigma_b-\operatorname{Var}_{\mathrm{branch}}]_+
+ \lambda_m [\operatorname{Scatter}_{\mathrm{merge}}-\sigma_m]_+
+ \lambda_c |\#\mathrm{branch}-\#\mathrm{merge}|.
\]

This makes the graph-of-thought trajectory fork into alternatives, merge compatible continuations, and keep a directed noncommutative order.

## Phase 3: Derived-Category and CCA Trajectory Comparison

Each trajectory DAG induces a finite simplicial chain complex after forgetting orientation:
\[
C_2 \xrightarrow{\partial_2} C_1 \xrightarrow{\partial_1} C_0.
\]

The trainable analogical comparison between two trajectories \(X\) and \(Y\) learns soft transports
\[
P_0:C_0(X)\to C_0(Y),\quad
P_1:C_1(X)\to C_1(Y),\quad P_2:C_2(X)\to C_2(Y)
\]
from hidden-state similarities and penalizes finite chain-map failure:
\[
\mathcal L_{\mathrm{chain}}
= \|\partial^Y_1P_1-P_0\partial^X_1\|_F^2
+ \|\partial^Y_2P_2-P_1\partial^X_2\|_F^2.
\]

The mapping-cone residual is the same obstruction viewed as the failure of the cone differential to square to zero:
\[
\mathcal L_{\mathrm{cone}}
= \|\partial_{\mathrm{Cone}(P)}^2\|.
\]

Exact symbolic objects are emitted for audit:

- simplicial chain complexes and Betti numbers;
- Stanley-Reisner ideals over multigraded polynomial rings;
- Hochster Betti rows;
- Taylor free-resolution ranks;
- compressed DG differential entries;
- Fitting-entry ideal summaries;
- graded-commutative DG product summaries.

These exact objects are written periodically as JSON examples and can be rendered into the same analysis directories as reasoning trajectories, energy landscapes, triangles, and toric plots.

## Phase 4: Memory and Analogical Retrieval

The trajectory-memory teacher now compares candidate memories by:

- GraphCG chart similarity;
- toric phase similarity;
- local topology and speed/curvature statistics;
- branch/merge DAG similarity;
- derived-category feature similarity;
- observed local quality.

The intended behavior is:
\[
\operatorname{retrieve}(\tau)
= \arg\max_{\mu\in\mathcal M}
\langle q(\tau),k(\mu)\rangle
\alpha\,s_{\mathrm{DAG}}(\tau,\mu)
\beta\,s_{\mathrm{D^b}}(\tau,\mu)
\gamma\,s_{\mathrm{toric}}(\tau,\mu).
\]

This supports analogical reasoning by selecting prior graph trajectories with compatible algebraic and topological structure rather than merely similar surface text.

## Phase 5: GFlowNet and Test-Time Scaling

The GFlowNet head should be used to learn high-reward branches in embedding space.  During later phases, inference should sample multiple branch/merge trajectories, score them by BPB or task quality, and merge high-value branches.

Recommended scaling protocol:

1. sample \(K\) candidate graph trajectories;
2. score each by loss, BPB, memory compatibility, and derived-category consistency;
3. merge compatible high-score branches;
4. rerank final trajectories by validation BPB and reasoning diagnostics.

## Phase 6: Long Context and Tropical Ring Attention

Long-context training should expand only after the BPB gate is stable.  Tropical ring attention is the preferred long-context mechanism because it preserves structured path composition and bounded memory use.

The ramp should be:

- stabilize at current graph window;
- increase graph span and node count;
- increase memory retrieval context;
- enable broader test-time trajectory sampling.

## Gates

- Early OAI BPB gate: do not accept a run that regresses OAI validation BPB.
- Stability gate: no non-finite loss, gradient, or parameter events.
- Artifact gate: quantized competition artifact stays below 16 MB.
- Analysis gate: periodic outputs include chain complexes, symbolic resolutions, DAG plots, memory retrieval diagnostics, and 3D trajectory/energy visualizations.

## Launch Config

Use `config/train.full_tokengt_got_fineweb_derived.yaml` after the focused tests pass.  It enables the full shard glob, conservative DAG/memory/derived losses, W&B logging, checkpointing, and periodic derived-category example JSON output.
