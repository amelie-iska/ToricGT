# Toric Vector Bundles, Sheaves, and Tropical-Toric Training

Date: 2026-06-14

## Purpose

This note extends the tropical-into-toric implementation plan with toric vector bundles and equivariant sheaves.  The project convention is now:

1. Tropical attention supplies the max-plus and normal-fan computation actually used by the Transformer.
2. The relevant tropical varieties or tropical decision fans are embedded into toric varieties by choosing the ambient fan, toric strata, and orbit-boundary compactification.
3. Vector bundles and sheaves over that toric ambient space provide a controlled way to talk about hidden-state fibers, boundary behavior, chartwise splitting, and gluing.

The implementation added in this pass is finite and training-oriented.  It does not claim that a Transformer is a toric variety or that a neural hidden state is literally a section of an algebraic vector bundle.  It attaches exact small Klyachko/Cech certificates to hidden states and measures whether those hidden states behave like local sections over toric charts.

## Mathematical Background

### Tropical varieties embedded in toric varieties

For a closed subvariety \(Y \subset T\) of an algebraic torus, Tevelev's tropical compactification puts the closure \(\overline Y\) inside a toric variety \(X(\Sigma)\) whose fan is supported on \(\operatorname{Trop}(Y)\).  The good compactification condition is expressed by the multiplication map \(T \times \overline Y \to X(\Sigma)\), and in the schön case the boundary and orbit intersections behave especially well.  This is the right ambient model for ToricGT: tropical active faces and Newton normal cones are not standalone decorations; they should determine the toric strata in which hidden computations are audited.

The scheme-theoretic viewpoint strengthens this.  Giansiracusa and Giansiracusa define a tropicalization functor from closed subschemes of toric varieties to closed subschemes of tropical toric varieties; the set of tropical points recovers extended tropicalization, while the scheme structure preserves multiplicities and, for projective subschemes, Hilbert polynomials.  Lorscheid's ordered-blueprint formulation further interprets scheme-theoretic tropicalization as a moduli/base-change construction with bend relations.  In practical ToricGT terms: bend relations, binomial relations, and CAS certificates are not merely numeric penalties; they are finite shadows of a scheme-theoretic tropical object embedded chartwise in a toric ambient variety.

### Toric vector bundles

An equivariant vector bundle on a toric variety is a locally free sheaf with a compatible torus action.  Klyachko's classification says that a toric vector bundle can be described by a finite-dimensional vector space \(E\) and a decreasing filtration \(E^\rho(i)\) for every one-dimensional cone \(\rho\) of the fan, subject to a cone compatibility condition: on each cone \(\sigma\), the filtrations for one-dimensional cones in \(\sigma\) must come from a common weight decomposition of \(E\).  Sam Payne's moduli paper gives a useful formulation for computation: fixed Chern data leads to rank conditions inside products of partial flag varieties.  Kaveh and Manon repackage Klyachko data as piecewise-linear maps to valuations, valuations with values in piecewise-linear functions, and points in tropical linear ideals.  This last viewpoint is especially aligned with ToricGT because it places vector bundles directly in tropical geometry.

This suggests a training interpretation:

- the hidden state at a token is a vector in a learned fiber;
- a tropical active one-dimensional cone or toric boundary stratum chooses a Klyachko filtration;
- membership in a filtration says which fiber coordinates should remain available near that boundary;
- compatibility over cones means the hidden fiber should split in a stable chartwise basis;
- chart transitions and Cech gluing measure whether local hidden sections agree on overlaps.

### Equivariant sheaves and Cech gluing

Perling constructs global resolutions for equivariant coherent sheaves over toric varieties using sheaves over posets and gluing of posets.  For training we only need the finite skeleton: affine toric charts, overlaps, local section coordinates, transition maps, and a Cech-style cocycle.  The Macaulay2 `ToricVectorBundles` package implements Klyachko and Kaneyama descriptions and cohomology computations, including vector-bundle validity checks.  On the current machine this package loads, so `--all-exact-cas` now emits a real rank-2 Klyachko vector-bundle certificate on `P^2`.  The GPU training path still uses bounded finite certificates in PyTorch, while the M2 certificate provides exact audit coverage and a future source of richer Klyachko/Kaneyama data.

## Implemented Training Applications

### 1. Klyachko Filtration Membership

File: `src/toricgt/toric_vector_bundles.py`

The new `ToricVectorBundleProbe` builds a deterministic finite Klyachko certificate:

- cyclic fan one-dimensional cones in a two-dimensional lattice shadow,
- adjacent one-dimensional-cone spans,
- nested coordinate-subspace filtrations for every one-dimensional cone,
- orthogonal chart frames,
- adjacent chart overlaps.

For each hidden state, the probe maps the model hidden vector to a learned rank-\(r\) fiber and predicts a toric boundary one-dimensional cone and a filtration level.  A target one-dimensional cone is derived score-safely from the public random-order target position; a target level is derived from the target token or position.  The filtration residual penalizes fiber mass outside the selected exact subspace.

Metric/loss keys:

- `toric_vector_bundle/loss`
- `toric_vector_bundle/ray_ce`
- `toric_vector_bundle/level_ce`
- `toric_vector_bundle/filtration_residual`
- `toric_vector_bundle/ray_entropy`
- `toric_vector_bundle/active_ray_mass`

### 2. Cone Splitting Regularizer

Klyachko compatibility requires a common splitting over each cone.  The implementation approximates this finite condition by measuring whether the weighted hidden-fiber covariance is block-diagonal in the chart frame associated with the cone.  Large off-diagonal covariance means the representation is mixing chart coordinates that the toric bundle certificate expects to split.

Metric key:

- `toric_vector_bundle/cone_splitting_residual`

### 3. Cech Sheaf Gluing

The chart frames define exact transition matrices between affine charts.  The probe computes local weighted means of hidden fibers in each chart and penalizes mismatch after transporting a local mean to an adjacent chart.  This is a small Cech-style gluing loss: local hidden sections should agree on overlaps after the prescribed transition map.

Metric keys:

- `toric_vector_bundle/cech_gluing_residual`
- `toric_sheaf/chart_gluing_residual`

### 4. Exact Certificate Audits

The module exposes two finite algebraic checks:

- `klyachko_nesting_residual`: verifies that each one-dimensional-cone filtration is decreasing.
- `cech_cocycle_residual`: verifies that chart transitions satisfy the cocycle identity \(T_{ac}=T_{bc}T_{ab}\).

These are exact properties of the fixed certificate.  They are logged as metrics and tested.  If a future CAS-generated certificate is loaded, these checks become the hard gate before training uses it.

Metric keys:

- `toric_vector_bundle/klyachko_nesting_residual`
- `toric_sheaf/cocycle_residual`

## Integration

The random-order Parameter-Golf model now has:

- `use_toric_vector_bundle`
- `toric_vector_bundle_rank`
- `toric_vector_bundle_num_rays`
- `toric_vector_bundle_num_cones`
- `toric_vector_bundle_filtration_levels`
- `toric_vector_bundle_max_positions`
- per-loss internal weights

The trainer now has:

- `toric_vector_bundle_loss_weight`
- phase-control support for that key
- W&B metrics under `train/*`, `toric_vector_bundle/*`, and `toric_sheaf/*`
- dashboard routing under `08_toric_tropical_bgg/*`
- a primary alias `00_primary/toric_vector_bundle_loss`

The all-phases run instantiates the probe from step 0, so metrics are always visible.  The optimization weight remains zero in the BPB warmup and ramps only after toric/GraphCG probes are stable:

- warmup: metrics on, weight 0
- graphcg/toric phase: `0.00002`
- topology/Koszul phase: `0.00004`
- memory and BGG phases: `0.00005`
- QAT/export phase: `0.00004`

The probe is excluded from Parameter-Golf artifact export through:

- `toric_vector_bundle_probe.`

This keeps it training-only by default.

## CAS Plan

Implemented now:

- Continue using Sage/Macaulay2 for exact normal-fan, toric-ideal, Stanley-Reisner, and binomial certificates.
- CAS discovery reports `ToricVectorBundles` availability.
- `--all-exact-cas` emits `macaulay2_toric_vector_bundle_certificate`, constructed by M2 from `projectiveSpaceFan 2` and `toricVectorBundle(2,F)`, with `isVectorBundle`, chart count, Euler characteristic, Klyachko filtration text, and basis text.
- Keep finite Klyachko/Cech certificates in PyTorch for cheap per-step GPU losses.

Next CAS upgrade:

1. Generate nontrivial Klyachko or Kaneyama vector-bundle certificates in M2, not only the trivial rank-2 smoke object.
2. Export parsed one-dimensional-cone filtrations, chart weights, transition matrices, `isVectorBundle`, Euler characteristic, and cohomology summaries.
3. Load those JSON certificates into `ToricVectorBundleProbe` instead of the deterministic built-in certificate.
4. Add hard tests that compare PyTorch certificate residuals with M2 `isVectorBundle` and available transition/cocycle checks.

## Why This Is Relevant to Training

Poor BPB is ultimately a language-modeling issue, so vector-bundle losses must not dominate early.  Their purpose is representation organization: hidden states should have stable fiber coordinates attached to toric/tropical boundary strata.  If this works, the model should generalize better on graph-structured reasoning, algebraic continuations, and long-context analogy because chart-local computations can transfer across cones and one-dimensional cones instead of being relearned as unrelated dense directions.

The most important guardrail is empirical: the vector-bundle loss earns nonzero weight only if BPB does not regress and if the new metrics correlate with better reasoning slices.  The metrics are still visible from step 0 so the watcher can detect whether the hidden representation is already organizing around toric boundary strata before any optimization pressure is applied.

## References

- Jenia Tevelev, "Compactifications of subvarieties of tori", arXiv:math/0412329.  <https://arxiv.org/abs/math/0412329>
- Jeffrey Giansiracusa and Noah Giansiracusa, "Equations of tropical varieties", arXiv:1308.0042.  <https://arxiv.org/abs/1308.0042>
- Oliver Lorscheid, "A unifying approach to tropicalization", arXiv:1508.07949.  <https://arxiv.org/abs/1508.07949>
- Sam Payne, "Moduli of toric vector bundles", arXiv:0705.0410.  <https://arxiv.org/abs/0705.0410>
- Kiumars Kaveh and Christopher Manon, "Toric vector bundles, valuations and tropical geometry", arXiv:2304.11211.  <https://arxiv.org/abs/2304.11211>
- Markus Perling, "Resolutions for Equivariant Sheaves over Toric Varieties", arXiv:math/0503501.  <https://arxiv.org/abs/math/0503501>
- Rene Birkner, Nathan Ilten, and Lars Petersen, `ToricVectorBundles` Macaulay2 package documentation.  <https://macaulay2.com/doc/Macaulay2/share/doc/Macaulay2/ToricVectorBundles/html/index.html>
- David Hering, Milena Mustata, and Sam Payne, "Positivity properties of toric vector bundles", Annales de l'Institut Fourier.  <https://www.numdam.org/item/10.5802/aif.2534.pdf>
