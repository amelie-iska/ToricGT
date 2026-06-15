# Toric Embedding Research Instructions For A Sub-Agent

Date: 2026-06-15

This document is a research and implementation-orientation brief for a sub-agent working on ToricGT's tropical-to-toric embedding program.  The goal is to understand, formalize, and extend the construction that embeds tropical ring attention into toric geometry, so that ToricGT can use toric methods, commutative algebra, vector bundles, sheaves, intersection theory, and exact CAS-backed audits.

The central principle is:

> Tropical ring attention produces finite max-plus candidate systems.  After rationalization, each such system can be represented as a finite Laurent-polynomial or toric-ideal sidecar in an algebraic torus.  Its tropicalization and initial degenerations are controlled by fans, and a rational fan refining the relevant active/Groebner decomposition gives an ambient toric variety.  Toric tools are applied to that finite sidecar, not to the whole Transformer.

Do not claim that arbitrary Transformer layers are toric varieties.  The mathematically sound claim is local and finite: a chosen checkpoint, layer, head, token window, and tropical probe define a finite algebraic sidecar.  That sidecar has toric geometry.

Use the terminology **one-dimensional cones** rather than “rays” in new prose unless quoting a source.

## Immediate Local Files To Read

Read these first.  They contain the current project statements and the new tropical-to-toric embedding material.

1. `./assets/toricgt_neurips_condensed.tex`
   - Primary target section:
     - `Tropical-to-toric embedding`
   - Purpose:
     - Condensed NeurIPS-style statement of the construction.
     - Useful for understanding what must fit in a short paper.

2. `./assets/toricgt_toric_bgg_rewrite_amelie_schreiber.tex`
   - Primary target section:
     - `Embedding tropical ring attention into toric varieties`
   - Secondary sections:
     - `Toric geometry of neural-network charts`
     - `Toric auxiliary losses for training`
     - `Toric BGG structure as trainable supervision`
     - `Affine toric charts and Koszul persistence modules`
     - `Combinatorial commutative-algebra bridge for training`
   - Purpose:
     - Full mathematical exposition with definitions, propositions, proof sketches, losses, and CAS audit contract.

3. `./planning/TROPICAL-TORIC-EMBEDDING-PAPER-UPDATE-20260615.md`
   - Purpose:
     - Notes from the prior research pass.
     - Lists reviewed sources, distilled themes, integration decisions, and validation status.

4. Related planning files:
   - `./planning/TROPICAL-TORIC-CAS-IMPLEMENTATION.md`
   - `./planning/TORIC-VECTOR-BUNDLES-SHEAVES-TRAINING.md`
   - `./planning/GUDHI-M2-TWO-PARAMETER-PERSISTENCE.md`
   - `./planning/COMBINATORIAL-METRICS.md`

5. Relevant local assets:
   - `./assets/combinatorial-commutative-algebra.pdf`
   - `./assets/9403002v1.pdf`
   - `./assets/2509.05894v1.pdf`
   - `./assets/2505.17190v2.pdf`
   - `./assets/2207.02505v2.pdf`
   - `./assets/2310.01889v4.pdf`
   - `./assets/2210.11433v1.pdf`
   - `./assets/1508.01166v2.pdf`

## Core Research Questions

The sub-agent should organize research around these questions.

1. How exactly does max-plus tropical ring attention become an initial degeneration?
2. What is the right algebraic torus and character lattice for a rationalized attention head?
3. What fan should be used to construct the toric variety controlling the tropical sidecar?
4. When is the resulting closure a genuine tropical compactification, and when is it only a finite audit sidecar?
5. Which toric tools are mathematically valid for which sidecar hypotheses?
6. Which metrics and losses can be computed exactly with Sage Math, Macaulay2, Gfan, Polymake, Normaliz, or Python?
7. How should the implementation distinguish exact CAS-certified quantities from differentiable surrogates?
8. How do toric vector bundles, tropical vector bundles, sheaves, Cox modules, and Klyachko filtrations provide new training or audit signals?
9. How do Miller-Sturmfels monomial staircases, toric ideals, Hilbert series, Betti tables, and free resolutions interact with two-parameter persistence and reasoning trajectories?
10. How do these tools improve ToricGT without damaging BPB or introducing unverified auxiliary losses?

## Mathematical Construction To Understand

The current construction is as follows.

Let

```tex
N \simeq \mathbb Z^r,
M = \operatorname{Hom}(N,\mathbb Z),
T_N = \operatorname{Spec} K[M]
```

where `K` is a nonarchimedean field, typically Puiseux series for exposition, with uniformizer `tau` and `val(tau)=1`.

A rationalized tropical attention head or low-rank audit probe gives:

```tex
a_1,\ldots,a_R \in M
```

and affine candidate scores

```tex
\ell_r(u)=\langle a_r,u\rangle+b_r,
\qquad u=P h\in N_{\mathbb R}.
```

Here:

- `h` is a hidden state.
- `P` is a learned or fixed projection into the toric chart.
- `a_r` are quantized candidate exponents.
- `b_r` are rationalized biases.
- `r` indexes candidate keys, affine candidates, or probe terms.

Define a Laurent polynomial sidecar

```tex
f=\sum_{r=1}^R \tau^{-b_r}\chi^{a_r}\in K[M].
```

With the max-plus convention, use initial forms with weight `-u`:

```tex
\operatorname{in}_{-u}(f)
```

The terms retained by this initial form are exactly those satisfying

```tex
r \in \argmax_s \ell_s(u).
```

This is the key proposition: **attention provenance is an initial degeneration**.  The tie locus where at least two terms are active is a tropical hypersurface.  Unique-max chambers are not part of the hypersurface, but they are part of the attention decision stratification and should be retained as normal-fan chambers.

For multiple heads, output coordinates, or graph tokens, form a finite family:

```tex
\mathcal F_H=\{f_\alpha:\alpha\in\mathcal A_H\}\subset K[M]
```

and an ideal:

```tex
I_H=\langle \mathcal F_H\rangle.
```

Then:

```tex
\operatorname{Trop}(Y_H)
=
\{u\in N_{\mathbb R}:\operatorname{in}_{-u}(I_H)\text{ contains no monomial}\},
\qquad
Y_H=V(I_H)\cap T_N.
```

Choose a rational fan `Sigma` refining:

- the normal fans of the lifted Newton polytopes,
- the observed active candidate decompositions,
- and, when ideal sidecars are used, the relevant finite Groebner fan.

This fan defines a toric variety:

```tex
X_\Sigma.
```

The fan gives an orbit-cone stratification.  Each cone labels a constant initial-form/provenance regime after sufficient refinement.

If:

```tex
|\Sigma|=\operatorname{Trop}(Y_H)
```

and the multiplication map

```tex
T_N\times \overline Y_H \to X_\Sigma
```

is flat and surjective, then the closure is a tropical compactification.  If the multiplication map is smooth and surjective, it is a schön compactification.  In most training batches we should not assume this.  Instead, we should mark whether the hypotheses are certified and otherwise treat `X_Sigma` as a finite toric audit envelope.

## Central Themes To Focus On

### 1. Initial Degenerations And Attention Provenance

Focus on the exact sign convention.  Macaulay2 Tropical and many tropical algebraic geometry sources use the min convention:

```tex
\operatorname{trop}(f)(w)=\min_u(\operatorname{val}(c_u)+w\cdot u).
```

ToricGT tropical attention uses max-plus:

```tex
\max_r(\langle a_r,u\rangle+b_r).
```

The bridge is:

```tex
\operatorname{val}(\tau^{-b_r})+\langle -u,a_r\rangle
=
-b_r-\langle a_r,u\rangle
=
-\ell_r(u).
```

So min over the nonarchimedean expression equals max over attention scores.  Every implementation must store and display this convention explicitly.

### 2. Normal Fans, Lifted Newton Polytopes, And Decision Chambers

For a single tropical head:

```tex
\psi(u)=\max_r(\langle a_r,u\rangle+b_r)
```

The active set is the exposed face of:

```tex
\operatorname{Conv}\{(a_r,b_r)\}.
```

The normal fan of the lifted Newton polytope partitions hidden chart space into chambers where the active set is constant.  This is the finite polyhedral object that should drive:

- active-face metrics,
- fan-cell entropy,
- top-two margins,
- bend audits,
- chart occupancy,
- orbit-stratum labels in the toric variety.

### 3. Toric Compactifications

The Maclagan/Tevelev viewpoint is essential:

- Start with a subvariety `Y` of a torus `T`.
- Choose a fan `Sigma`.
- Take the closure `Ybar` in `X_Sigma`.
- If the closure is proper and the multiplication map is flat and surjective, then this is a tropical compactification.
- If the multiplication map is smooth and surjective, it is schön.

For ToricGT:

- Most head sidecars will be finite audit objects, not globally certified compactifications.
- When the hypotheses can be verified on a synthetic or small CAS example, record that separately.
- Do not silently use compactification theorems when the support/equidimensionality/flatness/smoothness hypotheses are unknown.

### 4. Tropical Ideals And Tropical Schemes

Maclagan-Rincón tropical ideals give a scheme-theoretic foundation for tropical subschemes of tropical toric varieties.  Important points:

- Tropical ideals strictly include tropicalizations of classical ideals.
- Varieties of tropical ideals are finite polyhedral complexes.
- Homogeneous tropical ideals have Hilbert polynomials.
- Top-dimensional varieties of tropical ideals are balanced.
- Balanced varieties define Chow classes in the toric variety.

For ToricGT:

- Use tropical ideals when the sidecar is semiring-native.
- Use classical ideals over a valued field when the sidecar is built from Laurent polynomials.
- Use CAS checks to decide which category the sidecar belongs to.

### 5. Toric Ideals And Combinatorial Commutative Algebra

Given exponent matrix:

```tex
A=[a_1,\ldots,a_R],
```

define:

```tex
\varphi_A:K[x_1,\ldots,x_R]\to K[M],
\qquad x_r\mapsto \chi^{a_r}
```

and:

```tex
I_A=\ker \varphi_A
=
\langle x^\alpha-x^\beta:A\alpha=A\beta\rangle.
```

This is the exact algebra behind binomial consistency losses.  Research:

- Markov bases,
- Groebner bases,
- initial ideals,
- Hilbert series,
- Betti tables,
- free resolutions,
- syzygies,
- toric degenerations.

For two variables, Miller-Sturmfels staircases are especially important.  A monomial ideal:

```tex
J=\langle x^{a_1}y^{b_1},\ldots,x^{a_s}y^{b_s}\rangle,
\quad
a_1>\cdots>a_s,\quad b_1<\cdots<b_s
```

has a staircase diagram.  Lattice points outside the shaded ideal region are a `K`-basis for `K[x,y]/J`.  Adjacent generators control the efficient first syzygies.  This is the correct visual model for two-parameter grid modules; do not use a Pareto frontier visualization.

### 6. Divisors, Minkowski Weights, Chow Classes, And Intersection Theory

An integral piecewise-linear support function on a fan defines a torus-invariant Cartier divisor.  For ToricGT:

```tex
\psi(u)=\max_r(\langle a_r,u\rangle+b_r)
```

is such a support function after rational scaling and fan refinement, when Cartier compatibility holds.

Across adjacent maximal cones sharing a codimension-one cone, the slope difference gives a bend.  This bend is the neural analogue of a toric divisor intersection number with the corresponding invariant curve.

Balanced tropical cycles define Minkowski weights.  Fulton-Sturmfels identify operational Chow cohomology of complete toric varieties with Minkowski weights.  Katz connects tropical intersection theory with toric varieties.

For implementation:

- Compute balancing residuals on codimension-one cones.
- Store multiplicities.
- Compare predicted and certified Minkowski weights.
- Use Chow-class equality as a high-level certificate where valid.

### 7. Vector Bundles, Tropical Vector Bundles, And Klyachko Filtrations

Toric vector bundles on `X_Sigma` are classified by Klyachko data:

- a finite-dimensional fiber `E`,
- decreasing filtrations `E^rho(i)` for every one-dimensional cone `rho`,
- compatibility with a weight decomposition on every cone.

Compatibility condition:

```tex
E=\bigoplus_{m\in M}E_m^\sigma,
```

and for each one-dimensional cone `rho <= sigma`:

```tex
E^\rho(i)=
\bigoplus_{\langle m,u_\rho\rangle\ge i}E_m^\sigma.
```

For ToricGT:

- Treat hidden channels or GraphCG axes as a fiber.
- Treat gates, subspaces, or attention channel groups as filtrations.
- Penalize failure of nestedness and local compatibility.
- Use tropical vector bundle work for valuated-matroid and flat-based alternatives.
- Use Kaveh-Manon for toric vector bundles as valuation/tropical objects and positivity as convexity.

Potential losses:

```tex
\mathcal L_{\mathrm{Kly}}
=
\sum_{\sigma,\rho,i}
\left\|
\Pi^\rho_i -
\sum_{\langle m,u_\rho\rangle\ge i}\Pi_m^\sigma
\right\|_F^2
+
\lambda_{\mathrm{flag}}
\sum_{\rho,i}
\|\Pi^\rho_{i+1}\Pi^\rho_i-\Pi^\rho_{i+1}\|_F^2.
```

Only activate this after exact small-window sidecar checks pass.

### 8. Sheaves, Cox Rings, And Resolutions

For a fan `Sigma`, the Cox ring is:

```tex
S_\Sigma=K[x_\rho:\rho\in\Sigma(1)].
```

The irrelevant ideal is:

```tex
B_\Sigma=
\left\langle
\prod_{\rho\not\preceq\sigma}x_\rho:\sigma\in\Sigma
\right\rangle.
```

Equivariant sheaves can be represented as graded Cox modules modulo the irrelevant locus.  For ToricGT, this connects:

- tropical attention sidecars,
- BGG/Tate supervision,
- free resolutions,
- local cohomology,
- Fitting ideals,
- Buchsbaum-Eisenbud rank/minor checks,
- multigraded Betti tables.

Use Macaulay2 for these computations where possible.

## Required Source List

The sub-agent should read or at least inspect the following, in roughly this order.

### Project Papers And Plans

- `./assets/toricgt_neurips_condensed.tex`
- `./assets/toricgt_toric_bgg_rewrite_amelie_schreiber.tex`
- `./planning/TROPICAL-TORIC-EMBEDDING-PAPER-UPDATE-20260615.md`
- `./planning/TROPICAL-TORIC-CAS-IMPLEMENTATION.md`
- `./planning/TORIC-VECTOR-BUNDLES-SHEAVES-TRAINING.md`

### Tropical And Toric Geometry

- Diane Maclagan, *Introduction to tropical algebraic geometry*  
  https://arxiv.org/abs/1207.1925

- Diane Maclagan and Felipe Rincón, *Tropical ideals*  
  https://arxiv.org/abs/1609.03838

- Diane Maclagan and Felipe Rincón, *Varieties of Tropical Ideals are Balanced*  
  https://arxiv.org/abs/2009.14557

- Nolan Schock, *Quasilinear tropical compactifications*  
  https://arxiv.org/abs/2112.02062

- Eric Katz, *Tropical Intersection Theory from Toric Varieties*  
  https://arxiv.org/abs/0907.2488

- William Fulton and Bernd Sturmfels, *Intersection theory on toric varieties*  
  Local: `./assets/9403002v1.pdf`  
  arXiv: https://arxiv.org/abs/alg-geom/9403002

- David Cox, John Little, and Henry Schenck, *Toric Varieties*  
  Local: `./assets/toric-varieties.pdf` if present.

- William Fulton, *Introduction to Toric Varieties*  
  Local: `./assets/introduction-to-toric-varieties.pdf` if present.

### CAS And Computational Tropical Geometry

- Carlos Améndola, Kathlén Kohn, Sara Lamboglia, Diane Maclagan, Ben Smith, Jeff Sommars, Paolo Tripoli, Magdalena Zajaczkowska, *Computing Tropical Varieties in Macaulay2*  
  https://arxiv.org/abs/1710.10651

- Macaulay2 documentation:
  - Main: https://macaulay2.com/
  - Package docs should be checked locally through `M2 --help` or the online docs.
  - Focus packages: `Tropical`, `NormalToricVarieties`, `Polyhedra`, `SimplicialComplexes`, `ToricVectorBundles` if available, and free-resolution tooling.

- Sage Math documentation:
  - Main: https://doc.sagemath.org/
  - Focus areas:
    - polyhedra,
    - cones and fans,
    - toric varieties,
    - lattice polytopes,
    - ideals and Groebner bases where relevant.

- Gfan:
  - https://users-math.au.dk/jensen/software/gfan/gfan.html

- Polymake:
  - https://polymake.org/

- Normaliz:
  - https://www.normaliz.uni-osnabrueck.de/

### Vector Bundles, Sheaves, And Tropical Schemes

- Bivas Khan and Diane Maclagan, *Tropical Vector Bundles*  
  https://arxiv.org/abs/2405.03505

- Jaiung Jun, Kalina Mincheva, and Jeffrey Tolliver, *Vector bundles on tropical schemes*  
  https://arxiv.org/abs/2009.03030

- Kiumars Kaveh and Christopher Manon, *Toric vector bundles, valuations and tropical geometry*  
  https://arxiv.org/abs/2304.11211

### Combinatorial Commutative Algebra

- Ezra Miller and Bernd Sturmfels, *Combinatorial Commutative Algebra*  
  Local: `./assets/combinatorial-commutative-algebra.pdf`
  - Read Chapter 3 on monomial ideals in two variables and staircases.
  - Read Part II on toric algebra.
  - Focus on:
    - affine semigroup rings,
    - toric ideals,
    - initial ideals,
    - Hilbert series,
    - free resolutions,
    - cellular resolutions,
    - multigraded modules,
    - normal fans and quotient constructions.

### Neural/Toric/Tropical Model Geometry

- Yaoying Fu, *Toric geometry of ReLU neural networks*  
  Local: `./assets/2509.05894v1.pdf`

- Liwen Zhang, Gregory Naitzat, Lek-Heng Lim, *Tropical geometry of deep neural networks*  
  arXiv: https://arxiv.org/abs/1805.07091

- Marie-Charlotte Brandenburg, Georg Loho, Guido Montúfar, *The real tropical geometry of neural networks*  
  arXiv: https://arxiv.org/abs/2403.11871

- Baran Hashemi, Kurt Pasque, Chris Teska, Ruriko Yoshida, *Tropical Attention: Neural algorithmic reasoning for combinatorial algorithms*  
  Local: `./assets/2505.17190v2.pdf`

- TokenGT:
  - Local: `./assets/2207.02505v2.pdf`

- Ring Attention:
  - Local: `./assets/2310.01889v4.pdf`

## CAS Implementation Targets

The sub-agent should design all exact computations as sidecar jobs.  GPU training should receive compact cached tensors only.

### Sage Math Targets

Use Sage for:

- rational exponent matrices,
- lattice polytopes,
- cones and fans,
- normal fans,
- orbit incidence,
- affine toric charts,
- semigroup algebra sanity checks,
- visualizable polyhedral data,
- one-dimensional-cone primitive generators,
- divisor support functions where available.

Expected sidecar JSON fields:

```json
{
  "lattice_rank": 3,
  "exponents": [[0,0,0], [1,0,0], [0,1,0]],
  "biases": [0, 1, -1],
  "fan_cones": [[[1,0,0],[0,1,0]]],
  "one_dimensional_cones": [[1,0,0], [0,1,0]],
  "orbit_poset": [],
  "support_function_slopes": [],
  "divisor_bends": []
}
```

### Macaulay2 Targets

Use Macaulay2 for:

- toric ideals,
- Groebner bases,
- initial ideals,
- tropical varieties through `Tropical`,
- multiplicities,
- min/max convention checks,
- Hilbert series,
- Betti tables,
- free resolutions,
- Fitting ideals,
- Buchsbaum-Eisenbud rank/minor checks,
- Cox-ring module computations.

Expected sidecar JSON fields:

```json
{
  "toric_ideal_generators": [],
  "groebner_basis": [],
  "initial_ideals": {},
  "tropical_cones": [],
  "multiplicities": [],
  "balance_residual": 0.0,
  "hilbert_series": "",
  "betti_table": {},
  "resolution_differentials": [],
  "fitting_ideals": []
}
```

### Python Targets

Use Python for:

- extraction of hidden states and attention candidates,
- rational quantization,
- calling Sage/Macaulay2 subprocess sidecars,
- caching by stable hash,
- converting CAS outputs to tensors,
- computing differentiable residuals only after exact targets exist,
- rendering Plotly/HTML reports,
- generating dark-mode visualization pages.

Useful modules in this repo:

- `src/toricgt/cas_oracles.py`
- `src/toricgt/cas_certificates.py`
- `src/toricgt/toric_bgg.py`
- `src/toricgt/toric_vector_bundles.py`
- `src/toricgt/gudhi_persistence.py`
- `src/toricgt/toric_embedding_visualization.py`
- `scripts/run_embedding_cas_sidecar.py`
- `scripts/render_toric_embedding_report.py`
- `scripts/evaluate_tokengt_reasoning_geometry_suite.py`
- `scripts/run_gudhi_persistence_audit.py`

The visualization target is now implemented by
`src/toricgt/toric_embedding_visualization.py` and
`scripts/render_toric_embedding_report.py`.  The renderer consumes exact
SageMath/Macaulay2 sidecar records, writes one dark-mode HTML report page per
embedding record, and optionally builds a Macaulay2 `ToricVectorBundles`
certificate for the Klyachko vector-bundle page.  It must continue to mark
balance, Chow, and multiplicity panels as unavailable unless the exact
certificate fields are present.

## Metrics And Losses To Develop

Every metric must declare whether it is exact, CAS-certified, or differentiable-after-certification.

### Initial-Form Alignment

Purpose:

- Align neural active supports with CAS-certified initial forms.

Inputs:

- active candidate set from attention,
- `in_{-u}(f)` support from CAS or exact rational arithmetic.

Metric:

```tex
\mathcal L_{\mathrm{initial}}
=
\operatorname{CE}(\widehat A(u), A^\star(u))
```

or set-distance loss on supports.

### Fan-Margin And Orbit-Stratum Stability

Purpose:

- Keep hidden states away from unintended chamber walls.
- Make orbit-stratum labels stable under graph relabeling and small perturbations.

Metrics:

- active fan cell id,
- top-two margin,
- nearest wall distance,
- orbit stratum id,
- relabeling consistency.

### Balance And Chow Metrics

Purpose:

- Audit whether tropical cycles are balanced.
- Compare predicted multiplicities with certified Minkowski weights.

Metric:

```tex
\mathcal L_{\mathrm{bal}}
=
\sum_\tau
\left\|
\sum_{\sigma\supset\tau}\omega(\sigma)u_{\sigma/\tau}
\right\|_2^2.
```

### Divisor Bend Metrics

Purpose:

- Interpret changes across chamber walls as toric divisor intersection data.

Metric:

```tex
B_\psi(\tau)=
\langle m_\sigma-m_{\sigma'},u_{\sigma/\tau}\rangle.
```

Compare bends across equivalent graph situations.

### Toric Ideal And Resolution Metrics

Purpose:

- Use binomial relations and free resolutions as exact algebraic certificates.

Metrics:

- binomial consistency,
- Hilbert series,
- Betti table distance,
- free-resolution differential residual,
- Buchsbaum-Eisenbud rank/minor residuals,
- Fitting ideal rank changes.

### Miller-Sturmfels Staircase Metrics

Purpose:

- Correctly represent two-parameter modules and monomial initial ideals.

Metrics:

- staircase area,
- boundary length,
- adjacent-generator syzygy count,
- Hilbert numerator,
- Betti table,
- layer-to-layer staircase shift.

Visualizations:

- grid of monomials,
- shaded ideal region,
- unshaded basis of quotient,
- stacked layers at small height differences for reasoning level/radius/internal degree.

### Klyachko And Vector Bundle Metrics

Purpose:

- Treat hidden channel families as vector-bundle fibers over the toric sidecar.

Metrics:

- filtration nestedness,
- cone-wise compatibility,
- convexity/global-generation proxy,
- valuated-matroid flat consistency where using tropical vector bundles,
- Chern-class or parliament-of-polytopes summary where available.

### Sheaf/Cox Module Metrics

Purpose:

- Connect toric sidecars to BGG/Tate supervision and derived comparisons.

Metrics:

- Cox module resolution,
- local cohomology,
- Fitting ideals,
- sheaf cohomology dimensions,
- maps between resolutions,
- derived-category comparison residuals on finite complexes.

## Visualization Expectations

Every major sidecar should have an optional HTML report.

Minimum visualizations:

1. Newton polytope and lifted Newton polytope.
2. Normal fan with one-dimensional cones.
3. Active chamber plot with hidden states.
4. Toric orbit-stratum incidence graph.
5. Tropical hypersurface/tie locus.
6. Divisor bend plot across codimension-one cones.
7. Balance residual report.
8. Miller-Sturmfels staircase grid for two-variable monomial ideals.
9. Free-resolution diagram with readable differential labels.
10. Klyachko filtration diagram for one-dimensional cones and cones.
11. Cox module and sheaf/cohomology summary.

These reports should be dark-mode, browser-readable, and generated by analysis scripts without requiring training.

## Implementation Rules

1. No silent proxies.
   - If a CAS computation is required but unavailable, mark the metric unavailable.
   - Do not substitute a heuristic while naming it as exact.

2. Exact sidecar first, differentiable loss second.
   - Only train on a differentiable residual after exact small-window checks validate the target.

3. Store conventions.
   - Record max/min convention, sign convention, lattice basis, exponent scaling, fan orientation, and coefficient valuation convention in every sidecar.

4. Cache aggressively.
   - Cache by hash of:

```text
(A, b, Sigma, I_H, convention metadata, software versions)
```

5. Keep GPU payload compact.
   - CAS stays on CPU or subprocesses.
   - GPU receives tensors such as active labels, multiplicities, Betti vectors, and small sparse matrices.

6. BPB remains primary for language-model runs.
   - Toric losses are late-phase auxiliary losses.
   - A toric sidecar earns deployable bytes only if it improves exported BPB or a declared reasoning metric without BPB regression.

7. Use graph equivariance.
   - Sidecar labels attached to graph-token rows must transform equivariantly under graph relabeling.

8. Prefer exact rational arithmetic for sidecars.
   - Quantize exponents and biases before CAS export.
   - Store denominator-clearing scale factors.

## Expected Deliverables From The Sub-Agent

The sub-agent should produce:

1. A source survey memo:
   - what was read,
   - the central theorem or construction from each source,
   - how it maps to ToricGT.

2. A construction memo:
   - precise definition of the tropical attention sidecar,
   - nonarchimedean sign convention,
   - fan/toric variety construction,
   - compactification conditions,
   - valid/invalid claims.

3. An implementation design:
   - Sage tasks,
   - Macaulay2 tasks,
   - Python orchestration,
   - sidecar JSON schema,
   - cache strategy,
   - exactness/error handling.

4. A metrics/losses table:
   - metric name,
   - exact source,
   - CAS required,
   - differentiable training form,
   - activation phase,
   - failure mode.

5. A visualization plan:
   - HTML pages,
   - screenshots,
   - expected graphical elements,
   - interpretation notes.

   The current implementation path for this deliverable is:

```bash
PYTHONPATH=src /home/iska/miniconda3/envs/tokengt/bin/python scripts/render_toric_embedding_report.py \
  --sidecar-dir outputs/<run>/embedding_cas_sidecar \
  --output-dir outputs/toric_embedding_visual_report_manual \
  --build-vector-bundle-certificate

PYTHONPATH=src /home/iska/miniconda3/envs/tokengt/bin/python scripts/render_html_screenshots.py \
  --source-dir outputs/toric_embedding_visual_report_manual \
  --output-dir outputs/toric_embedding_visual_report_manual/html_screenshots
```

   A generated example is available locally at
   `outputs/latest_toric_embedding_visual_report/index.html`, with screenshots
   under `outputs/latest_toric_embedding_visual_report/html_screenshots/`.
   The report includes Newton and lifted Newton polytopes,
   initial-degeneration chambers, normal fans with one-dimensional cones,
   orbit-stratum incidence, toric ideal relations, free-resolution diagrams,
   Miller-Sturmfels staircase layers, and Macaulay2-backed Klyachko filtration
   visualizations when the certificate is present.

6. Patch recommendations:
   - paper updates,
   - docs updates,
   - code modules,
   - tests.

## Suggested Work Order

1. Read the two ToricGT TeX papers and this plan.
2. Read Maclagan `1207.1925` for embedded tropical geometry and toric compactifications.
3. Read Maclagan-Rincón tropical ideals and balancing papers.
4. Read Macaulay2 Tropical paper and inspect local M2 package availability.
5. Read Miller-Sturmfels CCA Chapter 3 and toric algebra sections.
6. Read Fulton-Sturmfels and Katz for Chow/Minkowski/intersection theory.
7. Read Khan-Maclagan, Jun-Mincheva-Tolliver, and Kaveh-Manon for vector bundles/sheaves.
8. Map every theorem to either:
   - exact CAS sidecar,
   - differentiable loss after certification,
   - visualization only,
   - or future/unimplemented.
9. Write implementation tasks.
10. Add tests before integrating into training.

## Open Problems To Investigate

1. What is the smallest fan refinement that preserves all active attention regimes without exploding CAS cost?
2. Can we certify tropical compactification hypotheses for synthetic tropical attention families?
3. Which tropical ideal conditions can be checked cheaply for sidecars from neural heads?
4. How stable are toric Chow/Minkowski weights under rational quantization of learned exponents?
5. Can Klyachko filtrations be learned from GraphCG axes without destabilizing BPB?
6. Which vector-bundle positivity notions are useful for reasoning-memory retrieval?
7. Can monomial staircase layers provide a reliable two-parameter persistence diagnostic for reasoning level and radius?
8. How should maps between sidecar resolutions be used for analogical memory retrieval?
9. What is the right serialization format for Macaulay2/Sage objects so reports are reproducible?
10. Which exact sidecar metrics are cheap enough to run every 250 training steps?

## Final Warning

The research goal is not to decorate a Transformer with toric vocabulary.  The goal is to build exact, finite, reproducible toric sidecars for tropical ring attention and use toric geometry only where the sidecar hypotheses justify it.  Every theorem used in training should have:

- a finite object,
- a stated convention,
- an exact computation path,
- a failure mode,
- and a clear relationship to BPB or a declared reasoning metric.
