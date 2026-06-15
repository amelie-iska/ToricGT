# Exactness, Persistent-Homology Visualization, and ToricGT Completion Plan - 2026-06-15

## Execution Rule

This document is the working checklist for the current implementation pass.
The standing goal is: **leave nothing in this plan unimplemented**.  Each time
the implementation is updated, this file must be revisited and checklist
statuses must be advanced or left explicitly open.  Do not mark this plan
complete until every item below is implemented, tested, and documented.

This pass is **analysis and implementation only**.  It must not start,
restart, kill, or modify any training run.  Training entry points remain
guarded by explicit operator flags.

## Standing Goal: Eliminate Proxies And Fallbacks

All outputs that are meant to be exact must be backed by the corresponding
exact engine or must fail loudly with an `unavailable` status.  No silent
substitute values, no implicit torch-only topology stand-ins for exact GUDHI
audits, and no fake CAS payloads.  Diagnostic-only quantities may still exist,
but they must be labeled as such and must not be reported as exact.

## Scope

The immediate scope is the ToricGT analysis, inference, and training-ready
instrumentation path:

- exact GUDHI persistent homology and vectorized PH features;
- optional inference emission of those PH artifacts;
- exact Sage/Macaulay2 toric, tropical, and module certificates;
- two-parameter `F2[x_level,y_radius]` persistence modules;
- analogical-memory retrieval features from vectorized PH;
- TokenGT graph-structured analysis outputs;
- browser-visible dark-mode HTML reports and screenshots;
- process/status tooling that makes background training or analysis explicit.

## Checklist

### 1. Planning And Exactness Policy

- [x] Write this planning document.
- [x] Add a machine-readable exactness/provenance validator for analysis
  manifests.
- [x] Ensure required exact analysis commands fail if GUDHI, SageMath, or
  Macaulay2 are missing.
- [x] Add tests proving exact-mode commands do not silently emit fallback
  values.

### 2. Persistent-Homology Metrics And Visualizations

- [x] Add first-class visualizations for every vectorized PH feature:
  persistence landscapes, persistence images, silhouettes, entropy vectors,
  Betti curves, interval lifetime histograms, and vector-norm summaries.
- [x] Add a PH feature dashboard to every GUDHI record page.
- [x] Add a PH feature summary to every GUDHI index page.
- [x] Persist vectorized PH features in JSON/NPZ form for downstream training
  and retrieval.
- [x] Make PH visualization output optional for inference with explicit CLI
  flags.
- [x] Add tests that generated PH visualizations are nonempty and linked from
  HTML.

### 3. Two-Parameter Persistence And CAS Modules

- [x] Build finite `F2[x_level,y_radius]` chain complexes from reasoning level
  and Rips radius.
- [x] Emit Macaulay2 free resolutions, homology modules, Ext/Tor modules,
  identity chain maps, and mapping cones.
- [x] Render Miller-Sturmfels-style xy-grid module views.
- [x] Add richer browser visualization of maps between grid modules and
  derived identity-cone acyclicity.
- [x] Add strict tests for commutative squares, `d^2=0`, exactness at `C1`,
  and identity mapping-cone acyclicity.

### 4. Tropical-To-Toric Exact CAS Path

- [x] Use one-dimensional-cone terminology in user-facing toric fan outputs.
- [x] Emit Sage normal-fan one-dimensional-cone aliases while preserving
  compatibility keys.
- [x] Add exact Sage checks for cone containment, fan refinement, face
  incidence, orbit strata, and normal fan data.
- [x] Add exact Sage/Macaulay2 certificate validation for tropical active
  faces embedded into ambient toric varieties.
- [x] Add tests for the exact certificate fields.

### 5. Analogical Memory Retrieval

- [x] Use GUDHI-derived vectorized PH signatures in trajectory-memory records.
- [x] Add explicit PH-feature similarity panels for analogical retrieval.
- [x] Add exact simplicial-map validity to memory candidate scoring when
  simplex-tree data is available.
- [x] Add tests that retrieval changes when PH landscape/image/silhouette
  signatures differ.

### 6. TokenGT Graph-Structured Data Path

- [x] TokenGT-style graph tokenization exists for graph records.
- [x] The inference geometry path can consume graph fixture data and emit
  graph-token embedding payloads.
- [x] Add a validation report proving OAI baseline records have graph-structured
  causal directed records where possible.
- [x] Add W&B/manifest metrics for graph-token mask validity, causal edge
  counts, and graph-token utilization.

### 7. BGG / Category O

- [x] Finite Toric BGG certificate scaffolding exists.
- [x] Ensure BGG losses remain late-gated/off unless explicitly enabled.
- [x] Add exact certificate provenance to every BGG metric.
- [x] Add browser panels for standard leakage, `d^2`, Koszul residual, Gale
  consistency, Ext/Tor summaries, and unavailable statuses.

### 8. Vector Bundles And Sheaves

- [x] Finite Klyachko/vector-bundle probe exists.
- [x] User-facing terminology now uses one-dimensional cones.
- [x] Parse richer Macaulay2 `ToricVectorBundles` data when available:
  one-dimensional-cone filtrations, chart weights, transition matrices,
  Cech cocycles, cohomology.
- [x] Add exact sheaf gluing and chart-overlap browser panels.
- [x] Add tests comparing finite PyTorch certificates with exact M2 bundle
  validity fields.

### 9. Inference Output Contract

- [x] Optional inference output can emit geometry, embedding payloads, GUDHI
  persistence, and CAS sidecars.
- [x] Add `--emit-ph-feature-visualizations` and
  `--no-emit-ph-feature-visualizations` inference flags.
- [x] Include PH feature output paths in `inference_output.json`.
- [x] Ensure inference never starts training and cannot call training launchers.

### 10. Visualization Bundle Contract

- [x] Analysis bundles write a dark-mode root `index.html`.
- [x] HTML pages can be screenshot-rendered into `outputs`.
- [x] Static PNG contact sheets can be generated.
- [x] Add a reusable screenshot CLI so this does not depend on inline scripts.
- [x] Add nonblank screenshot tests for representative pages.
- [x] Add contact-sheet links to root inference indexes.

### 11. Training Metrics And W&B

- [x] GUDHI and CAS summaries can be mirrored to W&B by the watcher.
- [x] Add W&B namespaces for every vectorized PH feature norm and count.
- [x] Add missing-metric alerts for all exact PH/CAS/BGG/vector-bundle
  namespaces.
- [x] Keep restart/start guarded by explicit flags only.

### 12. Documentation And Tests

- [x] Update README with the exactness policy, PH feature visualization CLI,
  and analysis-only screenshot command.
- [x] Add planning cross-links from the previous GUDHI/M2 and rendered-HTML
  planning docs.
- [x] Run focused tests after each group.
- [x] Run the full test suite after the final implementation group.

## Progress Log

- 2026-06-15: Plan created.  Existing code audit shows exact GUDHI,
  Macaulay2 two-parameter modules, Sage/Macaulay2 embedding CAS sidecars,
  vectorized PH signatures, and inference optional sidecars already exist.
  Remaining immediate gap: first-class visualizations for every vectorized PH
  feature and exactness/provenance validation/reporting around those outputs.
- 2026-06-15: Implementation pass completed.  Added strict
  `scripts/validate_analysis_exactness.py` validation for GUDHI, embedding-CAS,
  inference, and TokenGT graph-data reports; added per-record PH feature
  dashboards and JSON/NPZ artifacts; added optional inference PH visualization
  flags; added PH-signature retrieval and exact simplicial-map scoring panels;
  added richer Sage normal-fan checks; added BGG/Category O and
  vector-bundle/sheaf browser reports; added `scripts/validate_tokengt_graph_data.py`;
  added reusable screenshot/status CLIs; updated docs and prior planning
  cross-links.  Focused tests passed with 51 passed / 2 skipped, and the full
  suite passed with 265 passed / 2 skipped.
