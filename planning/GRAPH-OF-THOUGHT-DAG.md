# Graph-of-Thought Branch/Merge DAG Contract

Status: implementation contract for TokenGT, trajectory memory, W&B metrics,
and periodic geometry analyses.

## Core Object

A graph-of-thought trajectory is not a linear sequence. It is a finite directed
acyclic graph

\[
G_\tau=(V_\tau,E_\tau),\qquad E_\tau\subset V_\tau\times V_\tau,
\]

where each vertex \(v\in V_\tau\) carries a reasoning-state vector
\(h_v\in\mathbb R^d\), a local filtered simplicial complex
\(K_v(\rho)\), a toric/tropical chamber label, local energy or BPB proxy, and
optional memory/analogical metadata. Branching is represented by vertices with
\(\operatorname{outdeg}(v)>1\). Merging is represented by vertices with
\(\operatorname{indeg}(v)>1\). A valid nontrivial GoT training example should
contain both phenomena unless it is explicitly marked as a chain-of-thought
ablation.

The preferred local cell is a diamond

\[
a\longrightarrow \{b,c\}\longrightarrow d,
\]

where \(a\) is a partial state, \(b,c\) are alternative reasoning
continuations, and \(d\) is a join state that reconciles or selects between the
branches. Larger trajectories are DAGs obtained by gluing these cells along
vertices and faces.

## Training Metrics

For hidden states \(H=(h_1,\ldots,h_n)\) and directed edges \(E\), the live
training helper computes:

- branch count: \(\sum_v \sigma(3(\operatorname{outdeg}(v)-1))\);
- merge count: \(\sum_v \sigma(3(\operatorname{indeg}(v)-1))\);
- back-edge fraction: fraction of edges that violate the topological order;
- branch diversity: variance of outgoing successor embeddings at branch
  vertices;
- merge scatter: variance of predecessor embeddings at merge vertices;
- branch/merge balance residual;
- induced 1-simplex edge density and 2-simplex triangle density.

The small structural loss is

\[
\mathcal L_{\mathrm{GoT\text{-}DAG}}
=\lambda_b[\nu_b-\mathrm{div}_{\mathrm{branch}}]_+
+\lambda_m[\mathrm{scatter}_{\mathrm{merge}}-\nu_m]_+
+\lambda_a\mathrm{backedge}
+\lambda_q
\frac{|B-M|}{B+M+1}.
\]

It is intended as a medium-conservative advanced-phase loss: low enough not to
fight BPB, but strong enough to prevent the graph-of-thought mechanism from
degenerating into a single chain.

## Data Contract

Curated graph rows use their explicit `graph_json` edges. Older linear
`graph_json` records are conservatively upgraded with branch/merge diamond
edges when they have at least four vertices. Raw text rows, including FineWeb
style rows without graph metadata, are converted into branch/merge DAG records:
root to early alternatives, local diamond windows, and join edges that merge
nearby reasoning spans.

This keeps full-dataset TokenGT training graph-structured even when the source
dataset is plain text.

## Memory Contract

Trajectory-memory keys now include DAG topology scalars in addition to pooled
state, endpoint displacement, speed/curvature, local Vietoris-Rips density,
toric phase moments, and quality. The in-batch teacher score adds a DAG
similarity term:

\[
\tilde r(s,i)=
\lambda_G\langle g_s,g_i\rangle
\lambda_\Theta\langle \phi_s,\phi_i\rangle
-\lambda_T d_{\mathrm{top}}(s,i)
+\lambda_D\langle d_s,d_i\rangle
+q_i,
\]

where \(d_s\) contains branch count, merge count, back-edge fraction, branch
diversity, merge scatter, and branch/merge balance. Retrieval therefore
prefers analogical memories with compatible branching and joining structure,
not merely similar endpoints.

## Visualization Contract

3D reasoning plots must render DAG edges, not a polyline. Recommended colors:

- magenta: branch fan-out edges;
- green: merge fan-in edges;
- cyan: ordinary directed edges;
- energy/reward color scale on vertices.

The same convention applies to toric phase plots, PCA hidden-space plots,
interactive Plotly outputs, and paper figures. A line-only trajectory plot is a
chain-of-thought ablation and should be labeled as such.
