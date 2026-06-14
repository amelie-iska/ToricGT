# Rendered HTML Audit And Implementation Plan - 2026-06-14

## Scope

This review covers the 29 rendered HTML pages captured as PNGs in
`outputs/rendered_html_screenshots_20260614T215639Z`.  The screenshots verify
that the HTML reports are nonblank and renderable under Playwright/Chromium.
They also expose several implementation issues: raw JSON dominates many pages,
legends and colorbars collide with the plot canvas, dense simplicial chords
obscure trajectories, and the commutative-algebra data is currently textual
where it should be geometric.

The immediate artifacts are:

- `outputs/rendered_html_screenshots_20260614T215639Z/index.html`
- `outputs/rendered_html_screenshots_20260614T215639Z/contact_sheet.png`
- `outputs/rendered_html_screenshots_20260614T215639Z/manifest.json`
- `outputs/rendered_html_screenshots_20260614T215639Z/screenshots/*.png`

All 29 pages rendered successfully.  The review below treats the screenshots
as a design and mathematical audit, not as a model-quality report.

## Implementation Status - 2026-06-14 Update

The first implementation pass is now complete for the exact audit path.
`scripts/run_gudhi_persistence_audit.py` renders pass/fail badges for
homogeneous Macaulay2 boundary maps, `d1*d2 == 0`, and two-parameter
commutative-square residuals.  Each per-record page now includes an
`F2[x_level,y_radius]` xy-grid module view: cell fill is the actual H0 rank,
and overlaid markers show C0, C1, and C2 generator counts at their true
bidegrees.  Hilbert and chain-generator matrices are displayed as readable
tables, while raw Macaulay2 payloads are moved into collapsible sections.

The inference path now persists exact embedding payloads.  Running
`scripts/evaluate_tokengt_reasoning_geometry_suite.py` or
`scripts/infer_tokengt_with_geometry.py` writes `geometry/embeddings/*.npz`
and metadata JSON files containing the hidden states, PCA projections, energy,
NLL, graph edges, local complex coordinates, and complex edges used by the
plots.  `scripts/run_embedding_cas_sidecar.py` consumes that manifest and runs
strict SageMath normal-fan and Macaulay2 toric-ideal computations from a finite
nonnegative integer exponent set derived from the actual saved embedding
vectors.  If SageMath or Macaulay2 is unavailable, or if the exact computation
fails, the sidecar fails rather than emitting a substitute result.

## Implementation Status - 2026-06-14 Second Pass

The CAS-backed training target path now includes a strict Macaulay2
Koszul/free-resolution certificate for `QQ[x,y,z]/(x,y,z)`.  The certificate
is produced by `Macaulay2TropicalOracle.koszul_resolution_certificate`,
included in `scripts/build_toric_tropical_certificates.py --all-exact-cas`,
and tested in `tests/test_cas_certificates.py`.  Macaulay2 verifies the
explicit boundary products `d1*d2 == 0` and `d2*d3 == 0`, computes the minimal
free resolution, and records free ranks `[1,3,3,1]`, projective dimension,
regularity, and Betti rows.  `src/toricgt/cas_backed_losses.py` now has
strict cached-certificate loading plus `koszul_betti_loss_from_certificate`,
so GPU losses can consume the cached exact target without running CAS in the
hot loop.

The GUDHI/Macaulay2 record pages now render the emitted two-variable module
data with a stronger Miller-Sturmfels-style view.  The xy-grid still shows H0
rank as cell fill and C0/C1/C2 chain generators by true bidegree, and now also
overlays the actual C1 Pareto frontier and adjacent lcm corners computed from
the emitted chain-generator degrees.  This makes the visible page look like a
two-variable monomial-module diagram rather than only a heatmap and raw
Macaulay2 text.

The inference wrapper now writes a root dark-mode `index.html` in the output
directory.  That page links the run manifest, geometry bundle, saved embedding
payloads, exact GUDHI/Macaulay2 persistence audit, and embedding CAS sidecar
when those optional outputs are enabled.  Existing fixture output was refreshed
at `outputs/analysis_fixture_tokengt_inference_latest/index.html`.

New fixture outputs were generated:

- `outputs/analysis_fixture_gudhi_points_latest/index.html`
- `outputs/analysis_fixture_tokengt_inference_latest/index.html`
- `outputs/analysis_fixture_tokengt_inference_latest/geometry/embeddings/manifest.json`
- `outputs/analysis_fixture_tokengt_inference_latest/embedding_cas_sidecar/index.html`
- `outputs/analysis_fixture_tokengt_inference_latest/gudhi_persistence/index.html`

## Mathematical Anchors

### Miller-Sturmfels, Page 42

The relevant local source is `assets/combinatorial-commutative-algebra.pdf`,
PDF page 52, book page 42, Section 3.1, "Monomial ideals in two variables."
The page describes a bivariate monomial ideal in `S = k[x,y]` by exponent
vectors `(a_i,b_i)` ordered along a staircase.  Lattice points below/outside
the shaded ideal region form a basis of `S/I`; inner corners are the minimal
monomial generators; outer corners come from adjacent least common multiples.
The efficient Hilbert numerator is `1 - inner corners + outer corners`, and
the minimal first syzygies are determined by adjacent minimal generators.

This is directly relevant to the ToricGT two-parameter persistence module
`F2[x_level,y_radius]`.  Our current GUDHI/Macaulay2 pages correctly build
bigraded modules and report homogeneous maps, but the visualization is not yet
using the xy-grid as the primary representation.  The next implementation
should render a two-dimensional lattice/grid view for each module: generator
bidegrees, occupied Hilbert-function cells, staircase boundary, adjacent
generator syzygies, and outer-corner lcms.  This would make the `x_level` and
`y_radius` parameters visible in the same language as Miller-Sturmfels rather
than burying them in JSON and Macaulay2 matrices.

### GUDHI Persistence Landscapes

The current code uses GUDHI's official vectorizers in
`src/toricgt/gudhi_persistence.py`: `Landscape`, `PersistenceImage`,
`Silhouette`, and `Entropy`.  It also includes differentiable PyTorch helpers
that vectorize already-supplied birth/death pairs; those helpers do not
differentiate through the GUDHI persistence algorithm.  That separation is
mathematically correct.  The remaining work is to validate normalization,
grid ranges, finite-diagram filtering, and loss scaling against GUDHI's
conventions, then expose the vectorized features as optional training
regularizers with explicit detachment rules.

## Per-Page Review

### 01. `outputs/analysis_fixture_gudhi_points/index.html`

The index page is a useful entry point because it exposes the global summary
and links to each record's JSON and Macaulay2 script.  The main improvement is
layout: several numeric values overflow the summary card, and the record cards
are visually sparse.  This page should become a dashboard with compact metric
cards, pass/fail badges for `d^2=0`, homogeneous `d1/d2`, and square residual,
plus small Betti/Hilbert sparklines for each record.  Long floats should be
formatted with fixed precision, and the record cards should show enough
content to guide the user before opening a detail page.

### 02. `outputs/analysis_fixture_gudhi_points/records/fixture_branch_merge_points.html`

This page is one of the strongest current reports: it has core metrics, module
metadata, persistence diagrams, H1 landscape, persistence image, Hilbert grid,
chain-generator grid, map ranks, and Macaulay2 output.  The improvement is to
separate human-readable mathematics from raw CAS output.  The Macaulay2 JSON
should move into a collapsible panel or linked file, while the visible page
should show a Miller-Sturmfels-style `F2[x_level,y_radius]` grid with
generators, boundary arrows, and adjacent syzygies.  The colorbar currently
sits too close to multiple plots; each subplot should get a scoped color scale
or the page should use separate chart rows with explanatory captions.

### 03. `outputs/analysis_fixture_gudhi_points/records/fixture_circle_loop.html`

The circle-loop page is extremely tall, which indicates that the report is
serializing too much detail directly into the page.  The mathematical content
is valuable because the circle fixture should visibly expose a persistent H1
class, but the current page makes that hard to inspect without scrolling
through a long raw section.  It should prioritize the H1 barcode/diagram,
landscape layers, persistence image, representative cycle, and module grid at
the top, with CAS matrices and full JSON available behind toggles.  The page
should also explicitly label the expected loop class and compare observed H1
persistence against that expected fixture property.

### 04. `outputs/analysis_fixture_gudhi_points/records/fixture_spiral_phase_leaf.html`

The spiral phase-leaf page is also far too tall.  It should be redesigned
around the phase-leaf interpretation: show the point cloud, phase order, Rips
filtration evolution, and persistence vectorizations first.  The current page
does not make the relationship between spiral geometry, phase recurrence, and
the resulting module visually obvious.  Add a phase-leaf panel with colored
time order, a radius slider or small multiples by radius, and a compact
summary of which homology classes survive across the two parameters.  CAS data
should remain exact but should not dominate the main reading path.

### 05. `geometry/tetrahedra/cca_topology_bpb.html`

The tetrahedron plot itself is useful, but the page below it is dominated by
a long raw record dump in tiny text.  This makes the report difficult to read
and makes browser screenshots waste most of their height on unstructured JSON.
The tetrahedron pages should share a common component: plot at the top,
metric-card summary below, record table with selected columns, and collapsible
raw JSON.  The CCA/topology/BPB page should also clarify that BPB is proxy
or unavailable for TokenGT geometry unless a true FineWeb byte evaluator was
run.

### 06. `geometry/tetrahedra/complexity_geometry_solution.html`

This page should explain how complexity, geometry, and solution quality are
being projected to tetrahedral coordinates.  Right now the plot is visually
clean, but the raw record dump hides the semantics of the axes and makes it
hard to distinguish measured values from normalized display coordinates.  Add
axis definitions, normalization formulas, and a small table of the two fixture
records with source metric, normalized coordinate, and interpretation.  The
raw JSON should be downloadable rather than the primary content.

### 07. `geometry/tetrahedra/energy_control.html`

The energy-control tetrahedron needs stronger diagnostic labeling.  The image
suggests a single point plotted against a simplex scaffold, but the report
does not say whether high or low energy is good, which controls are active,
or how this relates to training decisions.  Add a "decision use" block that
states whether the plotted point would pass or fail gating thresholds.  Also
include confidence/coverage warnings when the page is generated from two
fixture records rather than a real validation slice.

### 08. `geometry/tetrahedra/reasoning_k_bpb_mst.html`

This page has the right conceptual ingredients for comparing reasoning
complexity, BPB, and MST structure, but it currently inherits the same raw
dump problem as the other tetrahedra.  It should show MST efficiency, path
smoothness, graph reconstruction loss, and BPB/LM-NLL as a compact table
alongside the tetrahedron.  The implementation should distinguish true
byte-normalized BPB from `lm_token_nll` proxy values, since conflating those
metrics has already caused confusion in training reviews.

### 09. `geometry/tetrahedra/solution_bpb_smooth_diversity.html`

This page should become a checkpoint-triage view.  The current tetrahedron is
usable, but the user cannot quickly see whether a run is smooth, diverse, and
low-BPB or whether it is merely plotted in a decorative coordinate system.
Add threshold bands, record-level labels, and a short verdict.  When generated
from fixture data, the verdict should explicitly say "instrumentation smoke
test only" and avoid implying real solution quality.

### 10. `geometry/tetrahedra/toric_gfn_bpb.html`

The toric/GFlowNet/BPB tetrahedron is important for the project, but the
current page does not surface enough toric or GFlowNet-specific information.
It should include toric phase recurrence, fan/chamber metrics, GFlowNet action
entropy or trajectory-balance residual, and true BPB availability status.  A
good improvement would be a side panel showing which toric/GFlowNet metrics
are measured, which are missing, and which are proxies.  Again, raw record
dumps should be collapsed.

### 11. `trajectories/record_000_complex_pca_nll_3d.html`

This complex trajectory view fits in one viewport and is visually usable, but
it needs more semantic scaffolding.  The page should label what a vertex means
in the filtered reasoning-step complex, what the color encodes, and which
edges are original trajectory edges versus filtration/simplicial edges.  Add
controls for edge opacity and maximum displayed edges so dense complexes do
not obscure the signal.  A record-level summary card should list Betti counts,
cycle rank, and mean NLL.

### 12. `trajectories/record_000_energy_landscape.html`

The energy landscape is one of the most interpretable views, with a surface
and trajectory overlay that make local basins visible.  It should be improved
by adding the exact energy definition, whether energy is graph reconstruction
MSE or LM NLL, and a small panel with min/mean/max energy.  The page should
also add optional contour projections and annotate outlier points.  If the
landscape is interpolated from sparse points, the interpolation method and
confidence limitations should be visible.

### 13. `trajectories/record_000_pca_nll_4d.html`

This page uses PCA coordinates with NLL as a fourth coordinate, which is a
reasonable diagnostic, but it needs explained variance and axis provenance
visible in the chart area.  The implementation should add a PCA summary panel
showing variance explained by PC1/PC2/PC3 and whether the NLL source is true
byte BPB, token NLL, or reconstruction energy.  Hover text should include
node id, node type, causal rank, and graph edge membership so users can relate
geometry back to graph tokens.

### 14. `trajectories/record_000_tokengt_node_embedding_trajectory_energy_landscape.html`

This richer energy landscape is useful because it names the TokenGT embedding
source, but it repeats the issue of dumping raw metadata below the chart.  It
should become the canonical model-side embedding page: save the actual
embedding tensor or compact `.npz`, provide a link to the source record, and
show PCA/UMAP/t-SNE or toric-chart projections as tabs.  Add a clear export
contract so inference runs always include embedding payload paths when the
user requests analysis output.

### 15. `trajectories/record_000_tokengt_node_embedding_trajectory_projected_simplicial_toric_geometry.html`

This page demonstrates the intended toric/tropical visualization but is too
dense.  The cyan nearest-neighbor chords obscure the trajectory, and the
legend/colorbar collide at the top right.  The next version should default to
a sparse view with trajectory skeleton, active-face vertices, and selected
simplicial edges, then provide toggles for all chords, filtered complex
vertices, toric fan cells, and chamber walls.  It should also replace raw JSON
with a compact metrics panel and provide links to Sage/Macaulay2 artifacts.

### 16. `trajectories/record_000_tokengt_node_embedding_trajectory_toric_phase_simplicial_trajectory.html`

The toric phase simplicial trajectory is conceptually important, but it also
suffers from dense chord clutter.  The main plotted object should be the
phase-leaf path, with simplicial edges as an optional overlay.  Add recurrence
markers, expected rotation increments, phase residual summaries, and
noncommutative cocycle labels where available.  This page should be able to
answer whether phase channels are coherent or merely decorative.

### 17. `trajectories/record_000_tokengt_node_embedding_trajectory_trajectory_3d.html`

This page shows the TokenGT node embedding trajectory with graph edges, and
it renders cleanly in one viewport.  The improvement is to reduce overplotting
and make the causal directed graph visible.  Use directed arrows for causal
edges where possible, differentiate graph-token types by marker shape, and
allow the user to isolate node tokens, edge tokens, branch nodes, merge nodes,
and certificate nodes.  The legend should move outside the plotting canvas.

### 18. `trajectories/record_000_trajectory_3d.html`

The generic branch/merge DAG trajectory is readable but still dense relative
to the fixture size.  It should show the actual six-node fixture graph more
directly rather than letting model-internal max-node padding dominate the
view.  Add a "valid graph tokens only" default and a separate optional layer
for padded/latent tokens if needed.  The page should link back to the fixture
record and display causal rank/topological order.

### 19. `trajectories/record_001_complex_pca_nll_3d.html`

The second complex trajectory page has the same strengths and weaknesses as
record 000.  Since this record corresponds to a different fixture family, the
page should show family-specific metadata in the title or subtitle.  The
current generic naming makes it hard to know whether the page is a persistence
fixture, BGG fixture, or tropical fixture.  Add record id, dataset, task
family, and selected target labels to the chart header.

### 20. `trajectories/record_001_energy_landscape.html`

This energy landscape renders cleanly and should be kept, but the report
needs comparative context.  Add a small "record comparison" strip showing how
record 001 differs from record 000 in mean NLL, path smoothness, Betti count,
and reconstruction MSE.  That would make the page useful for checkpoint
triage rather than only single-record inspection.

### 21. `trajectories/record_001_pca_nll_4d.html`

The PCA/NLL view for record 001 should receive the same improvements as record
000: explained variance, hover metadata, source metric clarification, and
valid-token masking.  The page should also show whether PCA components are fit
per record or globally across the batch, because per-record PCA can make two
pages visually incomparable even when the model behavior differs meaningfully.

### 22. `trajectories/record_001_tokengt_node_embedding_trajectory_energy_landscape.html`

This model-side energy page should be part of a multi-record embedding report
instead of a standalone chart with raw metadata.  The page should export the
embedding payload and include links to exact topology and CAS analyses derived
from the same embedding.  A viewer should be able to move from embedding page
to simplicial map page, module grid page, and retrieval-memory candidate page
without reconstructing paths manually.

### 23. `trajectories/record_001_tokengt_node_embedding_trajectory_projected_simplicial_toric_geometry.html`

This page again shows the core need for edge filtering and better toric
semantics.  It should include actual toric-variety embedding metadata when
available: rays, cones, semigroup generators, Newton polytope face id, and
active tropical face.  The current "projected simplicial toric geometry" view
is mostly a dense hidden-state chord plot; the next version must separate
tropical active-face structure from generic nearest-neighbor topology.

### 24. `trajectories/record_001_tokengt_node_embedding_trajectory_toric_phase_simplicial_trajectory.html`

The phase trajectory for record 001 should show whether phase dynamics differ
from record 000.  Add shared axes or comparable normalization across records,
and include a small recurrence/leaf residual table.  This page should also
make noncommutative structure visible through cocycle increments or expected
clock-shift phase transitions rather than relying only on sinusoidal phase
coordinates.

### 25. `trajectories/record_001_tokengt_node_embedding_trajectory_trajectory_3d.html`

This TokenGT trajectory page should make graph structure explicit.  Since the
fixture input is a branch/merge DAG, the page should default to showing those
six graph nodes and eight edges before any high-dimensional latent expansion.
TokenGT-style graph tokenization means node tokens, edge tokens, endpoint
features, and causal masks should be visible in the report.  Add a small table
showing token type counts and causal mask validity.

### 26. `trajectories/record_001_trajectory_3d.html`

The generic record 001 trajectory page has the same overplotting issue as
record 000.  It should include branch/merge labels and topological ranks, and
it should distinguish observed graph edges from inferred/simplicial edges.
For inference use, this page should be safe to emit by default because it is
small and useful, but the dense overlays should remain optional.

### 27. `outputs/analysis_fixture_tokengt_inference_23250/gudhi_persistence/index.html`

The checkpoint-tensor GUDHI index is valuable because it audits learned
weights or embeddings directly.  It needs stronger labeling: users should see
which tensors were sampled, why they were selected, and whether the point
cloud came from weights, embeddings, hidden states, or graph tokens.  Add a
source-type badge for each record and a warning that weight-space persistence
is not the same as trajectory persistence.

### 28. `gudhi_persistence/records/lm_token_emb_weight.html`

This record page correctly applies the GUDHI/Macaulay2 machinery to the LM
token embedding weight cloud.  The improvement is interpretability.  The page
should show token labels or token classes for sampled points when available,
and it should report whether persistent features are stable under resampling.
For a training metric, this should not be used directly as a loss unless the
sampling rule is deterministic and causality-safe; it is better as an audit
or compression diagnostic.

### 29. `gudhi_persistence/records/tokenizer_edge_id_weight.html`

The edge-id embedding audit is useful but needs explicit graph-token context.
It should explain how edge id embeddings relate to TokenGT endpoint features
and causal graph attention.  If persistence features are reported for this
tensor, the page should separate "embedding table topology" from "reasoning
trajectory topology."  Add a comparison panel between edge-id topology and
actual edge-token hidden-state topology from inference batches.

## Implementation Plan

### Phase 1 - Report Layout And Browser Robustness

Refactor the HTML report templates so every page has a consistent dark-mode
layout: title, source metadata, metric cards, primary chart, secondary charts,
and collapsible raw JSON/CAS output.  Move long JSON and Macaulay2 text into
linked artifacts or `<details>` sections.  Fix legend and colorbar placement
by reserving right-side layout space or placing legends below the chart.  Add
precision formatting for long floats and badges for pass/fail checks.  The
test target is a Playwright screenshot test that renders every fixture HTML
page, verifies nonblank canvas/image regions, and checks that no visible legend
overlaps the colorbar bounding box.

### Phase 2 - Miller-Sturmfels xy-Grid Module Views

Implement an `F2[x_level,y_radius]` xy-grid visualization for bigraded modules
and monomial data.  For each two-parameter persistence module, render lattice
points for generator bidegrees, Hilbert-function values, boundary arrows, and
adjacent syzygies.  The visual language should follow the bivariate staircase
model from Miller-Sturmfels page 42: inner corners as minimal generators,
outer corners as adjacent lcm data, and unshaded lattice points as basis
elements or surviving quotient/module data.  For higher-dimensional or
non-artinian cases, show a bounded window with explicit truncation metadata.
Back this with tests that compare plotted grid cells against the generator and
boundary matrices in `bigraded_chain_presentation`.

### Phase 3 - GUDHI Vectorized PH Correctness And Training Use

Strengthen the vectorized persistence implementation.  Keep GUDHI as the
ground-truth audit backend for diagrams, landscapes, silhouettes, entropy, and
persistence images.  Add tests comparing the PyTorch landscape helper against
GUDHI `Landscape` on fixed finite diagrams after matching grid conventions.
Expose vectorized PH outputs as optional training regularizers only after
clear detachment rules: GUDHI-derived vectors are audit/teacher targets, while
PyTorch vectorizers can receive gradients only through predicted birth/death
pairs or differentiable surrogate distances.  Log landscape norms, silhouette
norms, entropy, image norms, and stability under resampling.

### Phase 4 - Embedding Export During Inference

Add an inference option such as `--emit-embedding-payloads` that writes hidden
states, graph token metadata, causal ranks, node/edge masks, PCA projections,
and energy/NLL arrays to `.npz` plus JSON metadata.  The existing HTML pages
should link to these payloads.  This will make the visualizations reproducible
without rerunning the model.  Tests should run a tiny fixture inference, load
the emitted `.npz`, and verify that shapes match the graph masks and that the
HTML manifest references the payload.

### Phase 5 - Optional Sage/Macaulay2 CAS Inference Sidecar

Add `--emit-cas-sidecar` for inference and periodic analysis.  The sidecar
should call Sage for toric cones, fans, semigroup data, and tropical/toric
embeddings when available, and Macaulay2 for monomial modules, free
resolutions, Betti tables, Fitting ideals, Buchsbaum-Eisenbud rank/minor
checks, and maps between chain complexes.  For Buchsbaum-Eisenbud-style
checks, implement explicit rank and complementary-minor/multiplier audits in
Macaulay2 scripts generated from the finite complexes, then parse JSON-marked
outputs back into the report.  Every CAS result must have a source script
stored beside it so the computation is auditable.

### Phase 6 - Maps Between Resolutions And Derived Category Maps

Represent each reasoning trajectory and each reasoning step as a finite chain
complex with labels back to graph tokens.  Build chain maps for analogical
transfer, verify `d'F = Fd`, and when resolutions are available, compute or
audit maps between free resolutions.  The report should distinguish exact
finite-field maps, CAS-verified maps, and learned approximate maps.  Derived
category pages should show objects, morphisms, cones, homology ranks, and
failure residuals rather than only scalar summaries.

### Phase 7 - GUDHI Simplex Trees And Analogical Retrieval

Use GUDHI simplex trees for full reasoning trajectories and individual
reasoning-step windows.  Build simplicial maps between source and target
simplex trees using nearest-neighbor or learned transport maps, then verify
simplex validity by radius shift.  Feed vectorized topological signatures,
chain-map residuals, and CAS resolution signatures into the retrieval head as
optional memory keys.  Evaluation should report whether topological/CAS-aware
retrieval improves useful-helper rank without hurting BPB.

### Phase 8 - TokenGT Graph Tokenization For OAI Baseline

Ensure every OAI baseline record is graph structured before TokenGT ingestion.
Byte/FineWeb examples should become causal directed token-chain graphs; curated
reasoning examples should retain branch/merge DAG structure; Hebrew morphology
records should expose root/template/binyan/feature nodes; and algebraic
examples should expose generators, relations, certificates, and maps as typed
nodes/edges.  The tokenizer report should show node-token counts, edge-token
counts, endpoint features, causal ranks, and mask validity.  Tests should
verify that no graph-token attention path can read future bytes in FineWeb
mode and that noncausal graphs receive deterministic content-independent
random reveal ranks.

### Phase 9 - Visualization Updates From Screenshot Review

Implement density controls for every 3D trajectory page: valid-token-only
default, max displayed edges, edge opacity, separate toggles for graph edges,
simplicial edges, nearest-neighbor chords, active-face vertices, and toric
walls.  Add consistent record headers with dataset, task family, checkpoint,
source metric, and BPB availability.  Add cross-record comparison panels where
multiple records are rendered.  Add a master HTML index linking original HTML,
screenshots, JSON, CAS scripts, embedding payloads, and plot-family manifests.
The acceptance test is that a user can open one index and navigate to every
rendered artifact without knowing the output directory structure.

### Phase 10 - Training Integration And Gates

Do not turn every new metric into a loss immediately.  First log them as
periodic audits every configured interval.  Then enable training losses in
this order: differentiable PH vectorizer alignment, xy-grid module consistency
for synthetic fixtures, chain-map residuals, CAS teacher distillation for
small exact certificates, and finally retrieval-head signature distillation.
Each loss needs an ablation and a BPB guard.  OAI/FineWeb BPB remains the
separate byte-normalized score from the dedicated evaluator, not the TokenGT
geometry proxy.

## Test Plan

The minimum test suite for the next implementation pass is:

- fixture generation and `CuratedGraphIterableDataset` loading;
- GUDHI/Macaulay2 audit on the loop/spiral/branch fixture;
- GUDHI vectorizer parity tests for landscapes and persistence images;
- xy-grid module rendering tests against known bivariate monomial ideals;
- Playwright render/screenshot tests for every generated HTML page;
- inference embedding payload shape and manifest-reference tests;
- CAS sidecar smoke tests for Sage and Macaulay2 scripts;
- TokenGT causal-mask tests for FineWeb graph-token chains;
- analogical simplex-map validity tests on tiny source/target simplex trees;
- dedicated BPB evaluator tests using legal score-before-update logic.

## Priority Order

The highest-value immediate updates are layout refactoring, xy-grid module
views, embedding payload export, and vectorized PH validation.  Those changes
will make the reports readable and mathematically grounded.  The larger CAS
sidecar, maps between resolutions, derived-category maps, and retrieval-head
integration should follow once the fixture reports are clean and the output
contracts are stable.
