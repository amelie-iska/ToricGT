# Analysis Fixture Data Review - 2026-06-14

## Purpose

This fixture exists so ToricGT analysis, inference, GUDHI persistence, and
Macaulay2 resolution paths can be exercised even when the full curated
validation shards are not present locally.  It is intentionally small,
deterministic, and leakage-safe.  It is not a replacement for FineWeb/OAI BPB
or the full curated reasoning corpus; it is a smoke-test and visualization
input for the geometry/CAS sidecars.

## Generated Inputs

Command:

```bash
conda run --no-capture-output -n tokengt env PYTHONPATH=src \
  python scripts/create_analysis_fixture_data.py \
  --output-dir data/analysis_fixtures
```

Generated files:

- `data/analysis_fixtures/toricgt_analysis_fixture.parquet`
- `data/analysis_fixtures/toricgt_analysis_fixture.jsonl`
- `data/analysis_fixtures/toricgt_analysis_points.json`
- `data/analysis_fixtures/manifest.json`

The Parquet/JSONL fixture contains six branch/merge graph records:

- `toric_tropical_graph_reasoning`
- `gudhi_persistence_simplicial_maps`
- `toric_bgg_category_o`
- `hebrew_morphology_graph`
- `noncommutative_torus_phase_memory`
- `parameter_golf_bpb_control`

The point-cloud fixture contains:

- `fixture_circle_loop`
- `fixture_spiral_phase_leaf`
- `fixture_branch_merge_points`

## Loader Validation

`CuratedGraphIterableDataset` successfully streams the Parquet fixture.  The
first three records each produced a six-node, eight-edge graph with
`topological_dag` causal ranks and a target tensor shaped `(64, 16)` under a
small validation `ModelConfig`.

## Exact GUDHI/Macaulay2 Audit

Command:

```bash
conda run --no-capture-output -n tokengt env PYTHONPATH=src \
  python scripts/run_gudhi_persistence_audit.py \
  --points-json data/analysis_fixtures/toricgt_analysis_points.json \
  --output-dir outputs/analysis_fixture_gudhi_points \
  --records 3 \
  --max-points 18 \
  --num-radii 4 \
  --num-levels 4 \
  --landscape-resolution 24 \
  --image-resolution 8 \
  --macaulay2-timeout-seconds 120
```

Main output:

- `outputs/analysis_fixture_gudhi_points/index.html`
- `outputs/analysis_fixture_gudhi_points/summary.json`
- `outputs/analysis_fixture_gudhi_points/records/*.html`
- `outputs/analysis_fixture_gudhi_points/macaulay2/*.m2`

Summary:

- records: `3`
- mean Betti-0: `1.0`
- mean Betti-1: `0.3333333333333333`
- mean GUDHI simplices: `247.66666666666666`
- mean H0 landscape norm: `4.327367869154354`
- mean H1 landscape norm: `0.10610994584054141`
- mean H1 persistence-image norm: `137.77533296744718`
- mean Macaulay2 homogeneous `d1`: `1.0`
- mean Macaulay2 homogeneous `d2`: `1.0`
- mean Macaulay2 `d^2=0`: `1.0`
- mean two-parameter commutative-square residual: `0.0`

The circle fixture correctly exposes one H1 loop in the bounded audit.  The
two-parameter module reports zero square residual, so the `F2[x_level,
y_radius]` inclusion squares are commuting in this fixture run.

## TokenGT Inference Geometry Run

Command:

```bash
conda run --no-capture-output -n tokengt env PYTHONPATH=src \
  python scripts/infer_tokengt_with_geometry.py \
  --checkpoint checkpoints/toricgt_route1_neighbor_metricfix_max16mb_d320l7_b16ga4_20260607T221415Z/toricgt_step_00023250.pt \
  --config config/train.full_tokengt_got_fineweb_derived.yaml \
  --data-glob data/analysis_fixtures/toricgt_analysis_fixture.parquet \
  --output-dir outputs/analysis_fixture_tokengt_inference_23250 \
  --records 2 \
  --batch-size 1 \
  --parquet-batch-size 2 \
  --device cpu \
  --precision fp32 \
  --derived-category-max-vertices 8 \
  --topology-max-points 16 \
  --topology-max-windows 3 \
  --emit-gudhi-persistence \
  --gudhi-records 2 \
  --gudhi-max-points 12 \
  --gudhi-num-radii 4 \
  --gudhi-num-levels 4
```

Main output:

- `outputs/analysis_fixture_tokengt_inference_23250/inference_output.json`
- `outputs/analysis_fixture_tokengt_inference_23250/geometry/reasoning_geometry_summary.json`
- `outputs/analysis_fixture_tokengt_inference_23250/geometry/plot_family_manifest.json`
- `outputs/analysis_fixture_tokengt_inference_23250/gudhi_persistence/index.html`

Observed geometry summary:

- records: `2`
- mean graph reconstruction MSE: `4.736220034828875e-05`
- best graph reconstruction MSE: `4.671409260481596e-05`
- mean node NLL: `3.9136401414871216`
- max node NLL: `12.81796932220459`
- mean topology Betti-0: `4.875`
- mean topology cycle rank: `0.25`
- mean topology HDBSCAN cluster count: `0.75`
- mean topology variety-complex residual: `0.0`
- mean topology multigraded Betti mass: `5.125`
- mean topology Buchsbaum-Eisenbud multiplier residual:
  `0.008207869433198381`
- mean topology fitting-minor rank residual: `0.10937553130335326`

The plot-family manifest reports:

- interactive 3D/4D complete: `true`
- missing required interactive plots: `[]`
- missing static plots: `[]`

The inference bundle generated 117 geometry files, including trajectory,
energy-landscape, PCA/NLL, projected simplicial toric geometry, toric phase,
directed filtration, exact persistence morphism, commutative algebra, toric
shadow, Slepian, GraphCG, analogical, tropical chamber, triangle, tetrahedron,
and symbolic-resolution artifacts.

## Review

The fixture now covers the main analysis paths that previously failed when no
curated Parquet validation shards were available.  It exercises real graph
loader code, exact GUDHI simplex-tree persistence, vectorized persistence
metrics, Macaulay2 bigraded resolution scripts, TokenGT model inference, and
the rich geometry plot-family writer.  This gives a stable local input for
debugging periodic analysis without touching a training process or requiring a
large dataset download.

Important limitation: these fixture metrics are smoke-test metrics, not model
quality metrics.  BPB on the OAI/FineWeb competition data still requires the
actual FineWeb token shards and the dedicated BPB evaluator.  The fixture is
useful for checking that toric/tropical/topological/BGG/CCA instrumentation is
wired correctly and producing browser-viewable artifacts.

## Next Use

Use this fixture when:

- a periodic analyzer fails because validation Parquet shards are missing;
- a code change touches `CuratedGraphIterableDataset`, TokenGT geometry, GUDHI
  persistence, Macaulay2 resolution generation, or plot-family manifests;
- an inference-only analysis path needs a small deterministic source;
- a browser-visible visualization bundle needs to be generated without GPU
  pressure.
