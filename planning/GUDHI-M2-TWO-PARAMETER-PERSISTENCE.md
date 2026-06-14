# GUDHI and Macaulay2 Two-Parameter Persistence Audits

This plan is the implementation contract for the new exact persistent-homology
audit layer.  It makes the reasoning trajectory filtration explicit as a finite
two-parameter module over

```text
F2[x_level, y_radius]
```

where `x_level` is reasoning-prefix level and `y_radius` is Vietoris-Rips radius
in the embedding/GraphCG space.  The goal is not to replace the training loss
with a symbolic algebra system.  The goal is to run an exact audit beside
training, render it as browser-readable evidence, and expose only small
differentiable vectorizations to the GPU objective.

## External Systems Used

The implementation requires real external systems for the exact audit.

- GUDHI builds Rips complexes and simplex trees, computes persistence diagrams,
  and provides vectorized diagram representations.  The implementation uses
  `RipsComplex`, `SimplexTree.compute_persistence`, and the representation
  classes `Landscape`, `PersistenceImage`, `Silhouette`, and `Entropy`.
  Official docs: <https://gudhi.inria.fr/python/latest/>,
  <https://gudhi.inria.fr/python/latest/rips_complex_ref.html>, and
  <https://gudhi.inria.fr/python/latest/representations.html>.
- Macaulay2 constructs the bigraded chain complex over
  `GF(2)[x_level,y_radius, Degrees=>{{1,0},{0,1}}]`, computes homology modules
  with `HH_i`, and prints presentations, free resolutions, and Betti tables
  with `presentation`, `res`, and `betti`.  Official docs:
  <https://macaulay2.com/doc/Macaulay2-1.22/share/doc/Macaulay2/Macaulay2Doc/html/___Chain__Complex.html>.
- SageMath remains part of the broader CAS toolchain for toric/fan/ideal
  certificates.  This specific two-parameter persistence audit uses Macaulay2
  for the free-resolution stage because multigraded modules and resolutions are
  first-class in the current M2 workflow.

Missing GUDHI or missing Macaulay2 is an environment failure for exact periodic
audits.  The code does not silently replace those exact computations with Torch
heuristics.

## Exact Finite Construction

For a sampled hidden trajectory point cloud `p_0,...,p_{n-1}`, define two
filtration parameters.

1. `x_level`: the prefix size, i.e. only vertices with index at most the current
   reasoning level are available.
2. `y_radius`: the Rips radius in the standardized embedding space.

For each grid point `(i,j)`, GUDHI builds the Rips simplex tree on the prefix
cloud at radius `r_j`.  The audit records:

- simplex counts in dimensions 0, 1, and 2;
- exact `F2` boundary matrices;
- Betti numbers and `d_1 d_2 = 0`;
- exact persistence diagrams from GUDHI;
- vectorized PH features: landscape, persistence image, silhouette, entropy.

The structure maps in the two-parameter module are inclusions:

```text
(i,j) --x_level--> (i+1,j)
(i,j) --y_radius--> (i,j+1)
```

The implementation explicitly checks every square

```text
(i,j)       -> (i+1,j)
  |              |
  v              v
(i,j+1)     -> (i+1,j+1)
```

by multiplying GF(2) chain inclusion matrices for both paths in each chain
degree.  The reported `commutative_square_residual` is the actual XOR residual
of those path matrices, not a placeholder.

## Bigraded Chain Presentation

Each simplex gets a birth bidegree:

```text
deg(simplex) = (max vertex index, first radius index where diameter <= radius)
```

The free modules `C_0`, `C_1`, and `C_2` are then generated over
`F2[x_level,y_radius]` by vertices, edges, and triangles with those shifts.
Boundary entries are monomials

```text
x_level^a y_radius^b
```

where `(a,b)` is the difference between the source simplex bidegree and its
face bidegree.  This makes the boundary maps homogeneous.  Macaulay2 receives a
script of the form:

```m2
needsPackage "JSON"
R = GF(2)[x_level,y_radius, Degrees=>{{1,0},{0,1}}]
C0 = R^{{...}}
C1 = R^{{...}}
C2 = R^{{...}}
d1 = map(C0,C1,matrix{...})
d2 = map(C1,C2,matrix{...})
C = chainComplex({d1,d2})
H0 = HH_0 C
H1 = HH_1 C
H2 = HH_2 C
betti res H0
betti res H1
betti res H2
```

The HTML pages display the M2 script, homology module strings, presentations,
free resolutions, Betti tables, homogeneity checks, and `d1*d2 == 0`.

## Training-Time Differentiable Metrics

GUDHI persistence is an exact audit path, not an autograd path.  For training,
`src/toricgt/topological_reasoning.py` uses differentiable vectorizers on small
diagram-like 0D birth/death tensors derived from local nearest-neighbor death
times:

- `torch_persistence_landscape`;
- `torch_persistence_image`;
- smoothness and energy regularizers over those tensors.

These are marked as differentiable vectorized PH companions.  They do not claim
to differentiate through GUDHI's simplex-tree persistence algorithm.

## Periodic Training Integration

The all-phases config contains:

```yaml
analysis:
  periodic_interval_steps: 250
  gudhi_persistence:
    enabled: true
    records: 3
    max_points: 18
    num_radii: 5
    num_levels: 5
    radius_quantile: 0.62
    macaulay2_timeout_seconds: 180
```

The launcher reads `analysis.periodic_interval_steps`, so the checkpoint and
analysis cadence stays configurable from YAML.  The supervisor passes the
GUDHI/Macaulay2 flags into `scripts/watch_training_analysis.py`; each periodic
analysis writes:

```text
outputs/post_resume_analysis/<run>/step-*/index.html
outputs/post_resume_analysis/<run>/step-*/SYNOPSIS.md
outputs/post_resume_analysis/<run>/step-*/gudhi_persistence/index.html
outputs/post_resume_analysis/<run>/step-*/gudhi_persistence/records/*.html
outputs/post_resume_analysis/<run>/step-*/gudhi_persistence/macaulay2/*.m2
```

The top-level `index.html` is the dark-mode browser landing page linking the
GUDHI/M2 pages, CAS audit, geometry summary, OAI BPB summary, derived-category
report, memory trace report, and test-time-scaling report when present.

## Optional Inference Output

Inference can emit the same audit without changing model prediction:

```bash
conda run --no-capture-output -n tokengt env PYTHONPATH=src \
  python scripts/infer_tokengt_with_geometry.py \
  --checkpoint <checkpoint.pt> \
  --output-dir outputs/inference_with_gudhi \
  --emit-gudhi-persistence \
  --gudhi-records 2 \
  --gudhi-max-points 18
```

The manifest records `gudhi_persistence_index_html` and the audit summary.

## Validation Rules

- Exact audit pages must use GUDHI and Macaulay2 when enabled.
- The `F2[x_level,y_radius]` square residual must be computed from chain maps.
- Macaulay2 must report homogeneous `d1`, homogeneous `d2`, and `d1*d2 == 0`.
- Training metrics must distinguish exact audit values from differentiable
  vectorized companions.
- The audit is CPU-oriented and point-sampled to avoid disrupting GPU training.
