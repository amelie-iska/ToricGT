# Directed Persistent Topology Implementation Plan

This earlier implementation checklist is retained for engineering continuity.
The corrected conceptual separation is in
`planning/GRAPHCG-ANALOGY-TOPOLOGY-PLAN.md`: GraphCG learns a steerable concept
basis, while the analogical-reasoning paper supplies the functor/Dirichlet
energy mechanism.  ToricGT composes them by building directed persistent
topology over reasoning trajectories expressed in the GraphCG chart.  The maps
are still functor-like, but they carry additional filtered topological
structure: they should preserve inclusion, commute with chain/boundary maps up
to residual, and induce approximate persistence-module morphisms.

## Correct Target Object

For each branch of an embedding-space graph-of-thought trajectory
\[
\gamma=(h_0,h_1,\ldots,h_T),\qquad h_t\in\mathbb R^d,
\]
sample local windows
\[
W_s=\{h_s,h_{s+1},\ldots,h_{s+w-1}\}.
\]
Each window is a reasoning-step point cloud.  For radius levels
\(\rho_1<\cdots<\rho_L\), build a nested flag/Vietoris--Rips hierarchy
\[
K_s(\rho_1)\hookrightarrow K_s(\rho_2)\hookrightarrow\cdots\hookrightarrow K_s(\rho_L).
\]
The simplex tree at level \(\rho\) contains:

- vertices: hidden reasoning states in the local window;
- 1-simplices: pairs with normalized distance below \(\rho\);
- 2-simplices: cliques of three edges, used as a cheap higher-order closure
  proxy;
- directed 1-simplices and directed 2-simplices: the same hierarchy biased by
  time orientation and antisymmetric toric skew.

This differs from the previous relation-arrow-only implementation.  Relation
arrows remain useful for GraphCG basis training, but the persistent object now
lives over actual reasoning states.

## Directionality And Noncommutativity

For hidden states \(h_i,h_j\), define normalized distance \(d_{ij}\), a temporal
orientation term
\[
\tau_{ij}=\eta\,\operatorname{sign}(j-i),
\]
and an antisymmetric toric form
\[
\Omega_{ij}=\frac{x_i^\top y_j-y_i^\top x_j}{\operatorname{median}|\Omega|+\epsilon},
\]
where \(h_i=(x_i,y_i,\ldots)\) is a split-channel view.  The directed soft edge
is
\[
A^\rightarrow_{ij}(\rho)=
\sigma\left(\frac{\rho-d_{ij}+\lambda_\Omega\Omega_{ij}+\tau_{ij}}{T}\right).
\]
Since \(A^\rightarrow_{ij}\neq A^\rightarrow_{ji}\) in general, the resulting
directed filtration detects noncommutative reasoning flow.  The analysis logs:

- directed asymmetry \(\|A-A^\top\|_1\);
- cycle flux \(A_{ij}A_{jk}A_{ki}-A_{ji}A_{kj}A_{ik}\);
- chain-map commutators across radii;
- transitive closure residuals.

## HDBSCAN-Style Density Stability

Exact HDBSCAN is not differentiable and is too heavy for every microbatch, so
training uses a mutual-reachability surrogate:
\[
d_{\rm mr}(i,j)=\max\{c_i,c_j,d_{ij}\},
\]
where \(c_i\) is the \(k\)-nearest-neighbor core radius.  Across the same radii,
the persistent density affinity is
\[
H_{ij}=\frac{1}{L}\sum_{\ell=1}^{L}
\sigma\left(\frac{\rho_\ell-d_{\rm mr}(i,j)}{T}\right).
\]
Only high-stability pairs receive pullback weight.  This prevents the topology
regularizer from collapsing outliers or rare but valid analogies.

## Algebraic, Combinatorial, Geometric, And Differential Statistics

The implementation should compute:

- algebraic topology: Betti-0 proxy, cycle-rank proxy, boundary residual
  \(\partial_1\partial_2\approx0\), persistence/inclusion residuals;
- combinatorial topology: simplex-tree counts for vertices, edges, triangles,
  and clique growth;
- geometric topology: scale-normalized distances, Dirichlet energy
  \(\sum_{ij}A_{ij}\|h_i-h_j\|^2/\sum_{ij}A_{ij}\);
- differential topology: velocity/acceleration from trajectory flow, plus
  curvature already reported in the geometry suite;
- density topology: HDBSCAN stability, noise fraction, core radius, and
  persistent edge density.

## Implementation Completed In This Pass

- `src/toricgt/topological_reasoning.py`
  - Adds `ReasoningTopologyConfig`.
  - Adds differentiable `reasoning_step_topology_loss`.
  - Adds NumPy audit function `directed_step_filtration_stats_np`.
- `src/toricgt/random_order_lm.py`
  - Adds step-local topology config fields.
  - Folds the step topology loss into the existing analogy lattice loss.
  - Exposes W&B metrics under `train/analogy_step_*`.
- `scripts/evaluate_reasoning_geometry_suite.py`
  - Replaces relation-arrow topology diagnostics with step-local directed
    simplex hierarchies.
  - Adds step-radius heatmaps for edge density, Betti-0, and cycle rank.
- `tests/test_random_order_lm.py`
  - Adds finite-loss and nestedness checks for the new topology utilities.

## Training Policy

The base BPB objective remains dominant.  In the early 1250--3000 recovery
window, topology is diagnostic only or almost zero because recent runs showed
BPB ricochet behavior before topology was the limiting factor.  After the byte
model is stable, the topology weights ramp in through the existing phase
curriculum:

- step 1250--2000: likelihood capture, no GFlowNet/topology pressure;
- step 2000--3000: very light medium stream, still no topology pressure;
- step 3000--6000: tiny GraphCG/topology pressure for monitoring;
- step 6000+: GFlowNet, GraphCG, directed topology, and hard graph records ramp
  in together.

## Acceptance Criteria

1. Unit tests pass for random-order LM and topology.
2. Training logs finite `train/analogy_step_*` metrics when analogy is enabled.
3. Periodic geometry suite writes:
   - `*_directed_filtration.png`;
   - `*_noncommutative_heatmaps.png`;
   - `*_step_radius_hierarchy.png`.
4. Inclusion residual stays near zero; if it grows, radius scheduling or
   temperature is wrong.
5. Directed asymmetry is nonzero but bounded; if it vanishes, noncommutative
   flow is not being used.
6. HDBSCAN noise is neither zero nor one for reasoning-heavy records; saturated
   density means the radius range is too large or too small.
