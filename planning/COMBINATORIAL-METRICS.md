# Combinatorial Commutative-Algebra and Topology Metrics

Status: active implementation plan for BPB-first Seq4096 training and later
reasoning phases.

## Purpose

ToricGT should not use toric geometry as a decorative label. The train-time
metrics must expose finite algebraic and topological structure that can be
computed, bounded, ablated, and tied back to BPB. The current implementation
therefore treats a sampled local hidden-state window as a finite combinatorial
object:

1. A chart fan, approximated by a small cyclic chamber system.
2. A monomial exponent table and a finite toric ideal shadow.
3. A Stanley-Reisner complex over chamber coactivation.
4. A multiparameter persistence/Koszul module over local chart actions.
5. A directed Vietoris-Rips flag filtration over reasoning trajectory states.

The training objective remains primarily next-token cross entropy and byte-level
BPB. These metrics are auxiliary controls: they should improve the geometry of
hidden reasoning without dominating the language-model objective or violating
the 16MB Parameter-Golf artifact constraint.

## Mathematical Objects

Let a local hidden-state window be

```text
H = (h_1, ..., h_n),    h_i in R^d.
```

The implementation samples at most `max_points` vectors from a training
sequence, normalizes them, and computes all algebraic/topological quantities on
that small finite set.

### Toric Monomial Data

Choose a deterministic exponent table

```text
A = {a_1, ..., a_m} subset Z^r or R^r,
```

where `m = num_chambers` is small, usually 6 to 8 inside training. The exponent
table represents a compact Newton-polytope shadow. It is not learned and is not
stored in the exported model.

The associated monomial map is

```text
phi_A : k[x_1, ..., x_m] -> k[t_1^{+-1}, ..., t_r^{+-1}],
phi_A(x_i) = t^{a_i}.
```

The toric ideal is

```text
I_A = ker(phi_A)
    = < x^u - x^v : A u = A v, u,v in N^m >.
```

In training we use short degree-two binomial shadows:

```text
a_i + a_j approx a_k + a_l.
```

For chamber logits `z_i(h)`, the differentiable residual is

```text
L_binom =
  mean_{(i,j,k,l) in R} (z_i + z_j - z_k - z_l)^2
  / mean_i z_i^2.
```

This is a combinatorial commutative-algebra metric: it asks the hidden chart to
respect low-degree binomial relations of the finite toric monomial
configuration. It is intentionally a shadow, not a symbolic Groebner-basis
computation in the training loop.

### Stanley-Reisner Complex

Let `Delta` be the cyclic fan complex on chamber labels. Vertices are chambers,
edges connect adjacent fan rays, and nonfaces are chamber pairs that should not
coactivate in the same local toric chart.

The Stanley-Reisner ideal is

```text
I_Delta = < x_F = prod_{i in F} x_i : F notin Delta >.
```

Given soft chamber probabilities

```text
p_i(h) = softmax(z(h) / tau)_i,
```

define coactivation

```text
C_ij = (1/n) sum_t p_i(h_t) p_j(h_t).
```

The training residual is

```text
L_SR = average_{(i,j) notin Delta} C_ij.
```

This is the differentiable Stanley-Reisner nonface mass. It discourages
collapsed hidden charts where mutually incompatible toric chambers fire
together. The companion diagnostics are chamber entropy, chamber coverage,
allowed edge mass, an Euler-characteristic proxy, and Betti proxies from the
thresholded chamber graph.

### Koszul and Buchsbaum-Eisenbud Shadows

The module `src/toricgt/koszul_persistence.py` builds local row-stochastic
operators

```text
T_1, T_2, T_3 : M -> M
```

from time shift, metric-neighborhood flow, and toric phase/chart flow. These
operators are a finite proxy for commuting parameters in a multiparameter
persistence module or affine-toric module.

For two operators the Koszul differential is

```text
d_1 = [T_1  T_2],
d_2 = [-T_2 ; T_1].
```

For three operators, the usual Koszul signs give

```text
d_1 = [T_1 T_2 T_3],
d_2 =
  [ -T_2  -T_3   0
     T_1    0   -T_3
      0    T_1   T_2 ],
d_3 = [T_3 ; -T_2 ; T_1].
```

The core exactness residual is

```text
L_exact = ||d_1 d_2||_F^2 + ||d_2 d_3||_F^2.
```

The syzygy/commutativity residual is

```text
L_comm = average_{i<j} ||T_i T_j - T_j T_i||_F^2.
```

Rank diagnostics approximate Buchsbaum-Eisenbud conditions and Fitting-ideal
behavior. These rank terms are detached diagnostics during BPB-first training so
SVDs cannot become the main gradient source. The differentiable path is
primarily exactness plus commutator/syzygy residuals.

### Directed Persistent Topology

The module `src/toricgt/topological_reasoning.py` treats a hidden window as a
point cloud with time direction. A radius sweep gives nested Vietoris-Rips/flag
complexes:

```text
K_1 subset K_2 subset ... subset K_L.
```

At radius `rho_l`, soft undirected and directed adjacencies are

```text
S_l(i,j) = sigmoid((rho_l - ||h_i-h_j||) / tau),
D_l(i,j) = sigmoid((rho_l - ||h_i-h_j|| + skew_ij + time_ij) / tau).
```

The loss uses differentiable proxies for:

1. Filtration inclusion, penalizing edges that disappear across the sweep.
2. Flag/simplex closure, penalizing two-step paths not supported by edges.
3. Boundary residuals for oriented 2-simplices.
4. Directed transitivity and directed cycle flux.
5. DEC-inspired conservation: divergence, vorticity drift, kinetic-energy
   drift, Hodge balance, and wedge/interior-product consistency.
6. HDBSCAN-style mutual-reachability stability.
7. Analogical transport between consecutive local complexes.

This is a combinatorial-topology metric because the underlying hard object is a
finite filtered simplicial complex and a directed graph-of-reasoning trajectory.

## Combined Train-Time Loss

The new bridge in `src/toricgt/combinatorial_toric_metrics.py` computes

```text
L_CCA_top =
    w_binom L_binom
  + w_SR L_SR
  + w_entropy (1 - H(chamber_mass))
  + w_balance L_balance
  + w_euler |chi_proxy| / m
  + w_koszul L_koszul
  + w_top L_topology.
```

The bridge returns the differentiable scalar
`toric_cca_topology_loss` and detached metrics:

```text
toric_cca_binomial_residual
toric_cca_stanley_reisner_nonface_mass
toric_cca_chart_entropy
toric_cca_chamber_coverage
toric_cca_fan_balance_loss
toric_cca_euler_characteristic_proxy
toric_cca_betti0_proxy
toric_cca_betti1_proxy
toric_cca_allowed_edge_mass
toric_cca_koszul_loss
toric_cca_topology_loss_component
```

The Seq4096 Parameter-Golf trainer receives this loss only through the existing
advanced auxiliary-loss channel. That channel has:

```text
ADVANCED_LOSS_SCALE
ADVANCED_LOSS_EVERY
ADVANCED_LOSS_WARMUP_STEPS
ADVANCED_LOSS_MAX_CE_RATIO
GRAD_CLIP_NORM
```

so the algebra/topology term is bounded relative to the current cross entropy.

## Wise Utilization Policy

For the BPB-first phase:

1. Keep cross entropy and OpenAI FineWeb validation BPB as the primary objective.
2. Enable CCA/topology metrics from step 0, but with micro weights and a long
   ramp.
3. Cap total auxiliary loss at a tiny ratio of CE, initially `1e-5` to `2e-5`.
4. Log every component to W&B under `advanced/toric_cca_*`.
5. Treat a rising nonfinite flag, sudden BPB stall, or train-BPB spike as a
   reason to reduce the advanced scale, not as a reason to trust the math more.

Recommended scratch-run starting point after the step-260 NaN:

```text
ADVANCED_LOSS_SCALE=0.00025
ADVANCED_LOSS_EVERY=8
ADVANCED_LOSS_WARMUP_STEPS=20000
ADVANCED_LOSS_MAX_CE_RATIO=0.00001
KOSZUL_BGG_LOSS_WEIGHT=0.000001
TORIC_TROPICAL_LOSS_WEIGHT=0.0005
GRAPHCG_LOSS_WEIGHT=0.001
SLEPIAN_LOSS_WEIGHT=0.001
ANALOGY_LOSS_WEIGHT=0.000005
GRAD_CLIP_NORM=0.25
POLARQUANT_TRAIN_START_STEP=750
```

This still means the advanced metrics are active from step 0 for logging and
for the global scheduler, but the train-time force is small enough that the
base BPB descent is protected.

For the post-`<=1.2` reasoning phase:

1. Increase CCA/topology frequency before increasing raw weight.
2. Let the topology component influence graph-of-thought trajectory quality,
   memory retrieval keys, and analogical transport.
3. Add sidecar exact audits: F2 boundary ranks, persistent diagrams, toric ideal
   residual histograms, and Stanley-Reisner nonface heatmaps.
4. Use GraphCG disentanglement to make the chart basis better conditioned
   before asking toric fan metrics to become stronger.
5. Keep a natural-language BPB slice and an OAI competition BPB slice separate
   from reasoning-slice BPB.

## Pseudocode

```python
def combinatorial_toric_loss(hidden, positions):
    windows = sample_windows(hidden, max_windows=2, max_points=16)
    toric_terms = []
    for points in windows:
        directions = deterministic_chart_directions(d_model, num_chambers)
        logits = normalize(points) @ directions.T
        probs = softmax(logits / temperature)

        # Toric ideal shadow: x_i x_j - x_k x_l.
        relations = approximate_degree_two_binomials(exponent_table)
        binom = mean_square(logits[i] + logits[j] - logits[k] - logits[l])

        # Stanley-Reisner shadow for cyclic fan.
        coactivation = probs.T @ probs / len(points)
        nonface = mean(coactivation[pairs_not_in_fan])

        # f-vector and Betti diagnostics.
        chamber_graph = threshold(coactivation)
        euler = vertices - edges + triangles
        betti0, betti1 = graph_betti_proxies(chamber_graph)

        toric_terms.append((binom, nonface, euler, betti0, betti1))

    koszul = koszul_persistence_loss(hidden, positions)
    topology = reasoning_step_topology_loss(hidden)
    return weighted_sum(toric_terms, koszul, topology).clamp(0, max_loss)
```

## Visualization Contract

Periodic analyses should render:

1. A dark-mode Stanley-Reisner heatmap: allowed fan edges vs nonface mass.
2. A toric ideal residual histogram over binomial relations.
3. A chamber-entropy and chamber-coverage timeline.
4. A 3D PCA/UMAP reasoning trajectory with directed simplices overlaid.
5. A triangle/tetrahedron panel with correct filled heatmaps for BPB,
   topology, GraphCG coherence, and CCA residuals.
6. A persistence panel with Betti0/Betti1 proxies and exact F2 audits where
   sidecar runtime allows.
7. A W&B summary panel with `openai_parameter_golf/bpb` first, then
   `advanced/toric_cca_*`, then other auxiliary categories.

## Current Gaps and Next Updates

Implemented now:

1. Parameter-free CCA/topology bridge in `src/toricgt/combinatorial_toric_metrics.py`.
2. Finite, differentiable unit test coverage.
3. Seq4096 patcher hook so the live trainer can use the bridge without adding
   parameters to the exported artifact.

Next recommended updates:

1. Add exact F2 homology audits to periodic sidecar analysis for the same
   chamber complexes used in training.
2. Plot Stanley-Reisner heatmaps and binomial residual histograms in
   `evaluate_reasoning_geometry_suite.py`.
3. Add W&B organization entries for `advanced/toric_cca_*` so these metrics sit
   under a single visible advanced algebra/topology group.
4. Add a controller rule: if `toric_cca_topology_loss` rises while BPB improves,
   hold; if it rises with BPB degradation, reduce `ADVANCED_LOSS_SCALE`; if it
   falls while BPB improves, consider increasing frequency before increasing
   weight.
