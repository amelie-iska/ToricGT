# Toric Affine Charts, Koszul Complexes, and Multiparameter Persistence

This plan integrates the multiparameter-persistence paper
`assets/2210.11433v1.pdf` into the ToricGT geometry stack.  The goal is not to
add another decorative mathematical vocabulary.  The goal is to turn the
current toric fan, directed topology, and graph-of-thought diagnostics into
commutative-algebra objects with falsifiable residuals.

## Core Chain

The useful mathematical chain is:

```text
tropical active faces / toric fan cells
  -> affine toric charts
  -> semigroup coordinate rings
  -> multigraded persistence modules
  -> free/Koszul resolutions
  -> syzygy, Betti, and exactness diagnostics
```

For a rational polyhedral cone `sigma subset N_R`, the affine toric chart is

```math
U_\sigma = \operatorname{Spec} k[\sigma^\vee \cap M].
```

The coordinate ring `R_sigma = k[sigma^vee cap M]` is a semigroup algebra.  In
ToricGT, `sigma` is supplied by an active tropical face, Newton normal cone, or
empirical fan cell.  A reasoning window then becomes a finite sample over a
local affine chart instead of an unstructured cloud of embeddings.

## Multiparameter Persistence Interface

Schreiber's multiparameter-persistence paper uses the standard equivalence:
finite `n`-parameter persistence modules are finitely generated
`N^n`-graded modules over

```math
R = k[x_1,\ldots,x_n].
```

ToricGT already has natural parameters:

- reasoning time / prefix depth,
- Vietoris--Rips radius,
- density or HDBSCAN mutual-reachability threshold,
- tropical margin threshold,
- noncommutative phase distance,
- graph distance along the directed graph-of-thought.

For ordinary persistence diagnostics we use a polynomial ring.  When a toric
fan cell is active, we refine the local ring to the affine toric chart ring

```math
R_\sigma = k[\sigma^\vee \cap M].
```

This makes the persistence module aware of the same semigroup directions used
by the tropical and toric probes.

## Koszul Complex Layer

Given semigroup or polynomial parameters `a_1,...,a_r` acting on a finite
module `M`, attach the Koszul complex

```math
K_q(a;M) = M \otimes \bigwedge^q k^r,
```

with differential

```math
d(m \otimes e_{i_1}\wedge\cdots\wedge e_{i_q})
 =
\sum_{s=1}^q (-1)^{s-1} a_{i_s}m
 \otimes e_{i_1}\wedge\cdots\widehat{e_{i_s}}\cdots\wedge e_{i_q}.
```

The homology satisfies

```math
H_q(K(a;M)) \cong \operatorname{Tor}^{R}_q(R/(a),M).
```

In model terms, nonzero higher Koszul homology means the selected toric or
persistence coordinates are not behaving like clean local parameters for the
observed reasoning module.  This is a useful warning: the model may still lower
BPB while using an incoherent geometry.

## Diagnostics

Add these as analysis metrics before making any of them training losses:

- `toric_affine_chart_id`
- `toric_semigroup_generator_coverage`
- `koszul_exactness_residual`
- `koszul_homology_rank_by_chart`
- `toric_syzygy_residual`
- `fitting_minor_rank_residual`
- `buchsbaum_eisenbud_rank_residual`
- `buchsbaum_eisenbud_multiplier_residual`
- `chart_transition_resolution_shift`
- `multigraded_betti_mass`

The first implementation should compute exact small-window audits over
`F_2`/`F_p` and differentiable float surrogates separately.  Exact audits are
for trust.  Float surrogates are for training only after they match the audit
on small cases.

The Buchsbaum--Eisenbud multiplier audit is the strongest small-window
commutative-algebra check.  For a two-step complex

```math
C_2 \xrightarrow{d_2} C_1 \xrightarrow{d_1} C_0
```

exactness at `C_1` forces `rank(d_1)+rank(d_2)=dim(C_1)`.  The multiplier
relations compare maximal minors of `d_1` with complementary maximal minors of
`d_2`, up to scalar multipliers.  The implemented audit samples disjoint
edge-index sets over `F_2`, tests whether complementary maximal minors can be
nonzero, and reports a mismatch fraction.  This is a tractable finite shadow of
the Buchsbaum--Eisenbud criterion, suitable for reasoning-window diagnostics
and small auxiliary training signals.

## Losses for a Later Training Phase

The safe auxiliary objective is:

```math
L_{\mathrm{Koszul}}
 =
\lambda_{\mathrm{dga}}\|d_K^2\|_F^2
+\lambda_{\mathrm{Tor}}\sum_q
  \|\widehat\beta_{q,\bullet}-\beta^\star_{q,\bullet}\|_1
+\lambda_{\mathrm{Fitt}}L_{\mathrm{minor}}
+\lambda_{\mathrm{BE}}L_{\mathrm{BE}}.
```

Use it only after BPB has entered a stable decreasing regime.  For the
Parameter-Golf artifact, keep these probes training-only unless byte accounting
shows a clear BPB win per byte.

## Plots

The analysis suite should eventually add:

- affine chart occupancy over reasoning trajectories,
- multigraded Betti heatmaps,
- Koszul homology rank by chart and radius,
- Fitting-rank stratum plots,
- free-resolution DAGs for small reasoning windows,
- toric chart overlays on the existing simplex/trajectory plots.

## Skeptic Gate

This plan explicitly calms the "there be dragons" concern by making every
advanced object pass three checks before it can influence training:

1. a small exact algebraic audit,
2. a differentiable surrogate agreement test,
3. an ablation showing no BPB or validation-regression damage.

If any of these fail, the object stays as a diagnostic plot and does not enter
the optimization objective.
