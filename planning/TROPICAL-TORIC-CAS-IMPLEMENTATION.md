# Tropical Attention Inside Toric Varieties: CAS-Backed Implementation Plan

This plan records how ToricGT should use tropical geometry inside toric
geometry, and how exact computer algebra systems should certify the resulting
metrics, losses, objectives, and audits. The key rule is simple:

```text
SageMath and Macaulay2 produce exact finite certificates.
PyTorch consumes those certificates through differentiable surrogate losses.
The training loop must never report a Torch heuristic as an exact algebraic
metric unless it was checked against a CAS certificate.
```

On the current machine, `sage` and `M2` are not on `PATH`. The implementation
must therefore support optional CAS backends, deterministic cache keys, and
explicit provenance fields. If an exact metric is requested and the required
CAS backend is unavailable, the job should fail that metric or mark it
`cas_unavailable`; it should not silently substitute a surrogate.

## Research Grounding

The recent transformer expressivity paper
[Expressivity of Transformers: A Tropical Geometry Perspective](https://arxiv.org/abs/2604.14727)
models self-attention as a vector-valued tropical rational map. Its main
operational consequences for ToricGT are:

- zero-temperature attention induces power Voronoi / polyhedral partitions;
- multi-head attention combines Newton polytopes, increasing polyhedral
  complexity through Minkowski sums;
- finite-temperature attention can be treated as a stable smoothing of this
  polyhedral skeleton.

This fits the existing ToricGT stack: tropical attention exposes active
max-plus faces, Newton-polytope moments, fan cells, margins, and bend points.
However, a transformer as a whole is not automatically a toric variety. The
correct statement is narrower:

```text
The tropicalizable part of the computation defines a polyhedral shadow.
That shadow can be embedded into an ambient toric variety by choosing a fan,
semigroup algebra, toric ideal, or tropical compactification compatible with
the sampled active regions.
```

The algebraic-geometry side is classical. Tevelev's
[Compactifications of subvarieties of tori](https://arxiv.org/abs/math/0412329)
constructs tropical compactifications by imposing sufficiently fine
polyhedral structure on non-archimedean amoebas of subvarieties of algebraic
tori. Payne's
[Analytification is the limit of all tropicalizations](https://arxiv.org/abs/0805.1916)
shows that extended tropicalizations over all toric embeddings recover
analytic information by inverse limit. Foster and Payne's
[Adic tropicalizations and cofinality of Gubler models](https://arxiv.org/html/2303.16441v2)
strengthens this viewpoint through Gubler models and initial degenerations.
Schock's [Quasilinear tropical compactifications](https://arxiv.org/abs/2112.02062)
is relevant because it identifies cases where intersection theory is
controlled by the tropical fan. This is the mathematical reason to use toric
geometry as the ambient control structure for tropical attention diagnostics.

The computational side should use established CAS packages. SageMath has
native support for rational polyhedral fans and normal toric varieties:
[Sage toric varieties](https://doc.sagemath.org/html/en/reference/schemes/sage/schemes/toric/variety.html).
Macaulay2 has the `Tropical`, `TropicalToric`, `NormalToricVarieties`, and
`gfanInterface` packages. The Macaulay2 `Tropical` package computes tropical
varieties, multiplicities, balancedness, tropical bases, prevarieties, and
stable intersections; see the package paper
[Computing Tropical Varieties in Macaulay2](https://arxiv.org/abs/1710.10651).
`TropicalToric` computes toric intersection-theory classes from tropical data;
see
[Tropical computations for toric intersection theory in Macaulay2](https://arxiv.org/abs/2209.13414).

## Mathematical Contract

Let a ToricGT hidden window produce affine tropical candidates

```math
\ell_r(h)=\langle a_r,h\rangle+b_r,\qquad r=1,\ldots,R.
```

The learned or fixed exponent table `A={a_r}` defines a Newton polytope

```math
P_A=\operatorname{Conv}(a_1,\ldots,a_R).
```

The active set

```math
\operatorname{argmax}_r \ell_r(h)
```

is a face of the lifted Newton polytope and a cell in the corresponding
normal fan. This part is differentiable only through soft relaxations, but the
cell structure is rational after exponent quantization.

To embed this tropical object into toric geometry, choose one of three finite
ambient models:

1. **Normal-fan ambient toric variety.**
   Use the normal fan of `P_A` or a refinement of the empirical active-face
   fan. The ambient toric variety is `X_Sigma` for a rational fan `Sigma`.

2. **Toric-ideal ambient variety.**
   Let `A` define a monomial map
   `phi_A: k[y_1,...,y_R] -> k[t_1^{+-1},...,t_d^{+-1}]`.
   Its kernel `I_A` is the toric ideal. The tropical variety `Trop(I_A)` and
   Groebner fan certify the algebraic relations among candidate monomials.

3. **Tropical compactification ambient variety.**
   For a sampled subvariety or very affine proxy `Y subset T`, choose a fan
   `Sigma` supported on or refining `Trop(Y)` and take the closure
   `bar Y subset X_Sigma`. Orbit intersections and initial degenerations
   become exact audit objects.

ToricGT should normally use model 1 for fast online diagnostics, model 2 for
exact binomial and Groebner certificates, and model 3 for periodic deep audits
and paper-quality evidence.

## CAS Responsibilities

### SageMath

Sage is the preferred oracle for rational polyhedral and toric-combinatorial
objects:

- construct cones and rational polyhedral fans;
- compute face lattices, ray incidences, star/subfan relations, and
  refinements;
- construct normal toric varieties from fans;
- check whether active-face vectors lie in expected cones;
- compute normal fans of Newton polytopes;
- compute support functions and Cartier divisor data when available;
- compute cone containment, lattice indices, Hilbert bases, and orbit
  incidence metadata;
- provide exact rational arithmetic for quantized exponent tables.

Sage outputs are certificates, not gradients. The trainer receives only finite
tensors: cone ids, incidence matrices, slopes, wall normals, bend targets,
lattice indices, and expected orbit codimensions.

### Macaulay2

Macaulay2 is the preferred oracle for ideals, resolutions, tropical varieties,
and toric intersection theory:

- build toric ideals from exponent matrices;
- compute tropical varieties and multiplicities with `Tropical`;
- check balancedness and tropical-basis status;
- compute Groebner cones, initial ideals, and tropical bases with
  `gfanInterface`;
- compute normal toric varieties, divisor classes, Chow classes, and toric
  maps with `NormalToricVarieties`;
- compute toric intersection classes from tropical cycles with
  `TropicalToric`;
- compute Betti tables, free resolutions, Tor, Ext, Fitting ideals, and
  chain-complex exactness for small certificates;
- compute Stanley-Reisner ideals and simplicial toric boundary data for active
  chamber complexes.

Macaulay2 outputs should include enough provenance to reproduce the command:
package names, versions, field, ideal presentation, exponent matrix, tropical
min/max convention, and hash of the input.

### Optional Backends

The CAS layer can also use:

- `gfan` for Groebner fans and tropical varieties;
- `polymake` for polyhedral fans, cone containment, subdivisions, and tropical
  compactification combinatorics;
- `Normaliz` for Hilbert bases, normalization, and affine semigroup data;
- Singular through Sage or Macaulay2 for Groebner and standard-basis tasks.

These should be treated as backend implementations behind the same certificate
schema.

## Certificate Schema

All exact CAS outputs should be cached in `data/cas_certificates/` or
`outputs/cas_certificates/`, never inside the deployable Parameter-Golf
artifact. A certificate is keyed by:

```text
kind
input_hash
cas_backend
cas_backend_version
tropical_convention
coefficient_field
exponent_matrix_hash
fan_hash
ideal_hash
source_checkpoint
source_step
created_at_utc
```

The payload should be JSON for metadata plus `.npz` or `.pt` for dense/sparse
tensors. A minimal schema is:

```json
{
  "kind": "toric_tropical_embedding_certificate",
  "input_hash": "...",
  "source": {
    "checkpoint": "...",
    "step": 0,
    "layer": 0,
    "head": 0,
    "window_id": "..."
  },
  "cas": {
    "backend": "Macaulay2",
    "version": "...",
    "packages": ["Tropical", "TropicalToric", "NormalToricVarieties"],
    "tropical_convention": "max"
  },
  "toric": {
    "fan_rays": [[0, 1], [1, 0]],
    "maximal_cones": [[0, 1]],
    "orbit_codimensions": [0, 1],
    "cartier_slopes": [[0, 0]],
    "wall_normals": [[1, -1]]
  },
  "tropical": {
    "active_face_ids": [0],
    "multiplicities": [1],
    "balanced": true,
    "initial_ideal_labels": [0],
    "tropical_basis_generators": []
  },
  "commutative_algebra": {
    "toric_ideal_generators": [],
    "betti_table": [[1]],
    "resolution_boundary_matrices": ["boundary_0.npz"],
    "fitting_ranks": [1],
    "tor_ext_summary": {}
  }
}
```

The training loader must attach a `provenance/source` field to every metric:

```text
exact_cas/macaulay2
exact_cas/sage
exact_cas/gfan
surrogate_torch
surrogate_torch_validated_against_cas
cas_unavailable
```

## Exact Metrics

The following metrics should be exact CAS metrics whenever they are reported
as algebraic facts.

| metric | exact backend | training use |
| --- | --- | --- |
| `toric_tropical/fan_refinement_valid` | Sage/polymake | gate only |
| `toric_tropical/cone_containment_error` | Sage | target for cone CE/hinge |
| `toric_tropical/orbit_expected_dimension_error` | Sage/M2 | analysis gate |
| `toric_tropical/tropical_basis_valid` | M2 Tropical/gfan | gate and curriculum label |
| `toric_tropical/initial_ideal_degeneracy` | M2/gfan | degeneracy penalty target |
| `toric_tropical/multiplicity_residual` | M2 Tropical | weight target for balanced fan |
| `toric_tropical/balancing_residual_exact` | M2 Tropical | audit of Torch balancing loss |
| `toric_tropical/stable_intersection_degree` | M2 Tropical/TropicalToric | analysis and paper evidence |
| `toric_tropical/chow_class_match` | M2 TropicalToric | gate only |
| `toric_tropical/cartier_bend_target` | Sage/M2 NormalToricVarieties | bend-loss target |
| `toric_ideal/binomial_generator_count` | M2 | regularizer target |
| `toric_ideal/groebner_cone_id` | M2 gfanInterface | chamber classifier target |
| `koszul/betti_table` | M2 | Betti-profile loss |
| `koszul/free_resolution_exactness` | M2 | audit of `d^2` residual |
| `koszul/fitting_rank_profile` | M2 | Fitting-rank loss target |
| `category_o/euler_koszul_homology_rank` | M2 | late-phase BGG/Euler-Koszul target |

The existing Torch metrics in `src/toricgt/combinatorial_toric_metrics.py`
remain useful, but after this plan they must be described as surrogates unless
they are paired with a matching CAS certificate.

## Losses

CAS computations should not run inside the hot training step. They are too
slow, hard to parallelize on GPUs, and not differentiable. The correct design
is:

```text
offline or periodic CAS certificate
  -> cached exact tensors
  -> PyTorch target loader
  -> differentiable surrogate loss
  -> periodic CAS re-audit
```

### Fan and Cone Losses

Exact source: Sage fan/refinement certificate.

Torch loss:

```math
L_{\mathrm{cone}}
= \operatorname{CE}(\widehat c(h), c^\star)
  + \lambda_{\mathrm{wall}}
    \max(0,\gamma-\operatorname{dist}(h,\partial \sigma_{c^\star})).
```

Use this when the CAS certificate says the quantized active point lies in a
known cone of the normal fan or a verified refinement.

### Tropical Basis and Initial-Ideal Losses

Exact source: Macaulay2 `Tropical` / `gfanInterface`.

Torch loss:

```math
L_{\mathrm{init}}
= \operatorname{CE}(\widehat g(h),g^\star)
  + \lambda_{\mathrm{deg}}\|\widehat d_{\mathrm{init}}-d^\star_{\mathrm{init}}\|_1.
```

Here `g_star` is a Groebner cone or initial-ideal label, and
`d_init_star` is a small vector of CAS-derived invariants such as dimension,
monomial-free flag, generator count, and multiplicity.

### Balanced Tropical Cycle Loss

Exact source: Macaulay2 `Tropical` multiplicities and `isBalanced`.

Torch loss:

```math
L_{\mathrm{bal}}
= \sum_{\tau}
 \left\|
   \sum_{\sigma \supset \tau}
   \widehat m_\sigma u_{\sigma/\tau}
 \right\|_2^2
 + \lambda_m\|\widehat m-m^\star\|_1.
```

The incidence relation `sigma superset tau`, primitive normals
`u_{sigma/tau}`, and multiplicities `m_star` come from CAS. The model may
predict soft multiplicities or use active-face occupancy as a proxy.

### Cartier Bend and Toric Divisor Loss

Exact source: Sage support functions or M2 `NormalToricVarieties`.

Torch loss:

```math
L_{\mathrm{Cartier}}
= \sum_{\tau=\sigma\cap\sigma'}
 \left|
   \langle \widehat m_\sigma-\widehat m_{\sigma'},u_\tau\rangle
   - B^\star_\tau
 \right|^2.
```

This upgrades the existing bend loss from "second differences look small" to
"bends match a verified toric support-function certificate".

### Toric Ideal / Binomial Loss

Exact source: M2 toric ideal kernel or gfan tropical basis.

Torch loss:

```math
L_{\mathrm{binom}}
=
\sum_{\alpha-\beta \in \ker_\mathbb Z A}
\left\|
  \sum_r \alpha_r \ell_r(h)
  - \sum_r \beta_r \ell_r(h)
\right\|_2^2.
```

The integer relation basis should come from CAS for nontrivial exponent
matrices. The current `make_binomial_relations` helper may remain a cheap
fallback for toy cyclic cases, but it should not be called exact in reports.

### Free-Resolution and Koszul Losses

Exact source: M2 `res`, `betti`, `tor`, `ext`, Fitting ideals, and
Euler-Koszul scripts.

Torch loss:

```math
L_{\mathrm{res}}
= \lambda_{\partial^2}\sum_k\|\widehat D_{k-1}\widehat D_k\|_F^2
 + \lambda_D\sum_k\|\widehat D_k-D^\star_k\|_F^2
 + \lambda_\beta\|\widehat \beta-\beta^\star\|_1.
```

The exact boundary matrices and Betti tables must be generated by CAS for
general ideals or modules. The hand-coded cyclic Stanley-Reisner certificates
are acceptable only as deterministic toy certificates.

### Toric BGG / Euler-Koszul Losses

Exact source: M2 chain complexes and finite module calculations, with Sage
used for arrangement/fan combinatorics.

Torch loss:

```math
L_{\mathrm{TBGG-CAS}}
= \lambda_{\mathrm{EK}}\|H_\bullet(K(E-\beta;M))-\widehat H_\bullet\|_1
 + \lambda_{\mathrm{std}}\operatorname{Leak}_{\mathrm{std}}
 + \lambda_{\mathrm{Gale}}\|G_\theta s_A-s_B\|_2^2.
```

This remains late-phase only. CAS should produce exact small certificates:
sign-vector posets, Gale-dual maps, Euler-Koszul differentials, and homology
ranks. The model learns to match them; it does not invent them.

## Proposed Code Structure

Add these modules and scripts in a later implementation pass:

```text
src/toricgt/cas_oracles.py
  CASBackendStatus
  SageToricOracle
  Macaulay2TropicalOracle
  CertificateCache
  exact_or_surrogate_provenance()

src/toricgt/cas_certificates.py
  ToricTropicalEmbeddingCertificate
  TropicalCycleCertificate
  ToricIdealCertificate
  ResolutionCertificate
  certificate_hash()

src/toricgt/cas_backed_losses.py
  load_cas_targets()
  fan_cone_loss()
  balanced_cycle_loss()
  initial_ideal_loss()
  cartier_bend_loss()
  cas_betti_resolution_loss()

scripts/build_toric_tropical_certificates.py
  Offline certificate generator for selected checkpoints, layers, heads,
  synthetic tasks, and reasoning windows.

scripts/validate_cas_certificates.py
  Replays exact CAS commands and verifies hash/provenance.

scripts/run_periodic_cas_audit.py
  Samples small analysis windows, runs CAS if available, and logs surrogate
  agreement metrics to W&B.
```

The periodic watcher should call `run_periodic_cas_audit.py` only on a small
sample. Full CAS generation belongs in a separate CPU job.

## Macaulay2 Command Template

The M2 script emitted by `Macaulay2TropicalOracle` should be generated from a
safe template, never by interpolating untrusted strings. A typical toric ideal
audit is:

```m2
loadPackage "Tropical"
loadPackage "gfanInterface"
loadPackage "NormalToricVarieties"
loadPackage "TropicalToric"

QQ[x_0..x_(R-1)]
-- Build toric map from exponent matrix A.
-- Compute kernel ideal I_A.
-- tropicalVariety(I_A, Prime=>false)
-- multiplicities(...)
-- isBalanced(...)
-- gfanTropicalBasis(I_A)
-- betti res I_A
```

The wrapper must write the exact matrix and script to the output directory,
capture stdout/stderr, parse structured JSON-like output, and retain the raw
M2 transcript for audit.

## Sage Command Template

The Sage script emitted by `SageToricOracle` should:

```python
from sage.all import *

# Read rational rays, cones, and exponent matrix from JSON.
# Construct Polyhedron(vertices=...) for Newton polytope.
# Compute or ingest the normal fan.
# Construct Fan(cones, rays=...)
# Check cone containment and refinement relations.
# Build ToricVariety(fan) when needed.
# Emit rays, maximal cones, face incidence, wall normals,
# support-function slope data, lattice indices, and orbit codimensions.
```

For performance, Sage should run on quantized rational exponents and small
window samples, not full training batches.

## Integration With Existing Metrics

The existing metrics should be reclassified:

- `train/toric_binomial_residual`: surrogate unless its relations are loaded
  from a CAS toric-ideal certificate.
- `train/toric_bend_loss`: surrogate unless compared to a Sage/M2 Cartier
  divisor certificate.
- `toric_cca_koszul_*`: surrogate or toy-exact unless backed by an M2
  free-resolution certificate.
- `toric_cca_stanley_reisner_*`: exact only for the hand-coded cyclic fan toy
  family; otherwise CAS-generated.
- `bgg_category_o/d2_residual`: algebraically meaningful, but exact source
  depends on whether the differentials came from CAS or synthetic generator.
- `tropical/active_face_margin`: exact max-plus margin for the given logits,
  but not an exact tropical-variety metric unless tied to an ideal/fan
  certificate.

Add W&B companion fields:

```text
metric_provenance/<metric_name>
cas/backend_available_sage
cas/backend_available_macaulay2
cas/certificate_cache_hit_rate
cas/surrogate_agreement_error
cas/exact_audit_windows
cas/exact_audit_failures
```

## Training Phases

### Phase 0: Environment Probe

Every launch should log:

```bash
which sage || true
which M2 || true
sage --version || true
M2 --version || true
```

If exact CAS metrics are enabled and the backend is missing, the run should
fail before training or disable only the exact metric with an explicit config
entry.

### Phase 1: Toy CAS Reproduction

Before attaching losses to real checkpoints:

- reproduce cyclic Stanley-Reisner examples with Macaulay2;
- reproduce normal fans of small Newton polytopes with Sage;
- verify Torch toy certificates match CAS Betti tables, cone incidences, and
  balancedness where applicable.

### Phase 2: Offline Certificate Generation

Generate certificates for:

- synthetic tropical shortest-path and transitive-closure records;
- toric ideal examples from fixed exponent matrices;
- small hidden-state windows from frozen checkpoints;
- BGG/Koszul/Euler-Koszul toy modules.

No gradients are involved in this phase.

### Phase 3: Surrogate Agreement Gate

Run the current Torch diagnostics and compare them with exact CAS outputs.
Only promote a surrogate into training if:

```text
relative_error <= tolerance
rank/cell labels agree above threshold
CAS balancedness or exactness failures are rare and explained
BPB does not regress in a matched short run
```

### Phase 4: Training With Cached CAS Targets

Enable tiny weights only:

```yaml
cas_backed_losses:
  enabled: true
  fan_cone_weight: 1.0e-5
  balanced_cycle_weight: 5.0e-6
  cartier_bend_weight: 5.0e-6
  initial_ideal_weight: 0.0
  resolution_betti_weight: 0.0
  toric_bgg_euler_koszul_weight: 0.0
```

Initial-ideal, resolution, and Toric BGG losses remain off until BPB is already
healthy and periodic CAS audits show good agreement.

### Phase 5: Late Algebraic Reasoning Phase

After BPB is stable, introduce:

- initial-ideal labels;
- resolution/Betti targets;
- Euler-Koszul homology ranks;
- Gale-dual consistency targets;
- toric intersection and Chow-class diagnostics as analysis gates.

These should be late-phase reasoning losses, not primary FineWeb BPB losses.

## Accuracy Rules

1. A metric called `exact` must come from Sage, Macaulay2, gfan, polymake, or a
   proven closed-form toy certificate.
2. A metric called `surrogate` may come from Torch, but its report must name
   the exact metric it approximates.
3. A training loss may use a surrogate, but the target must be generated by
   CAS whenever the target is algebraic rather than synthetic.
4. CAS outputs must be cached with input hashes and backend versions.
5. Periodic analyses must report surrogate-vs-CAS disagreement.
6. Missing CAS backends must be visible in W&B and local reports.
7. The Parameter-Golf artifact must not include CAS binaries, CAS outputs, or
   certificate caches unless a separate artifact-budget decision explicitly
   includes them. By default, CAS is training/analysis only.

## Practical Next Steps

Implemented foundation:

- `src/toricgt/cas_certificates.py` defines certificate provenance, stable
  hashes, validation, and cache storage.
- `src/toricgt/cas_oracles.py` discovers SageMath and Macaulay2, refuses
  silent fallback, wraps exact Sage/M2 smoke computations, and emits an exact
  closed-form cyclic Stanley-Reisner certificate.
- `src/toricgt/cas_backed_losses.py` contains differentiable consumers for
  exact binomial, balancing, Cartier-bend, and cone-label targets. These losses
  do not compute algebraic targets themselves.
- `scripts/build_toric_tropical_certificates.py`,
  `scripts/validate_cas_certificates.py`, and
  `scripts/run_periodic_cas_audit.py` provide the first CLI entrypoints.
- `tests/test_cas_certificates.py` checks exact closed-form certificates,
  cache validation, script round-trips, backend unavailability behavior, and
  exact-backend tests that run only when `sage` or `M2` are installed.

Remaining next steps:

1. Add a small certificate generator for:
   - one toric ideal;
   - one Newton polytope normal fan;
   - one Stanley-Reisner ideal;
   - one Koszul/free-resolution certificate.
2. Extend `scripts/watch_training_analysis.py` with an optional
   `--cas-audit` flag and summary section.
3. Extend W&B metric organization with `cas/*` and
   `toric_tropical_exact/*`.
4. Update the paper and condensed NeurIPS version to state that exact
   algebraic metrics are produced by CAS-backed finite certificates, while
   differentiable training losses are cached-target surrogates.

This keeps the mathematical claim honest: tropical attention gives a
polyhedral computation, toric varieties give the ambient algebraic control
space, and CAS backends certify the finite algebraic objects that we use for
metrics, losses, and ablations.
