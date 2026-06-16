# Visualization Correctness Implementation Plan

Date: 2026-06-15

This plan is the active checklist for correcting the ToricGT visualization and
analysis artifacts.  This is an analysis/rendering implementation pass only; it
must not start, restart, or supervise training.

## Non-Negotiable Requirements

- Do not start training.
- Do not label a visualization or metric exact unless it is computed from the
  actual stored finite object or an exact CAS/GUDHI computation.
- Embedding-space comparisons must use the original high-dimensional embedding
  vectors.  PCA is display-only.
- If an analogical memory candidate fails the required tests, the page must say
  no analogy was emitted and must not render a misleading accepted map.
- Every full trajectory simplicial object must have a radius slider and a
  reasoning-level slider.
- Every per-step simplicial object must have a radius slider and a decoding-order
  slider.
- Miller-Sturmfels staircases must be represented as monomial-ideal staircases in
  the xy lattice.  3D layers may be used only as close z-height module layers;
  the overhead view must look like the xy-grid staircase.

## Checklist

### A. Audit

- [x] Inspect the current branching trajectory renderer and payload schema.
- [x] Inspect current GUDHI helper functions for diagrams, vectorizers, simplex
      sets, and filtration values.
- [x] Inspect current toric embedding staircase and resolution renderers.

### B. Analogical Memory Retrieval Visualization

- [x] Build explicit source and memory simplex-tree payloads using GUDHI Rips
      complexes over original embeddings.
- [x] Store vertices, edges, triangles, filtration/radius values, and PCA display
      coordinates for source and memory trees.
- [x] Compute the vertex map from source to memory in original embedding space.
- [x] Compute edge and triangle image validity, including collapsed simplex
      cases.
- [x] Render the source and memory simplex trees side by side in 3D PCA.
- [x] Render map arrows only as candidate-map arrows and label whether the
      analogy was accepted or rejected.
- [x] Add a validity table for vertex, edge, triangle, and full-trajectory map
      status.
- [x] Keep the no-analogy state visible when any acceptance criterion fails.

### C. GUDHI Vectorized PH Panels

- [x] Compute and store actual GUDHI H0/H1/H2 persistence diagrams for source
      and memory trajectories.
- [x] Compute and store persistence landscapes, persistence images, silhouettes,
      entropy vectors, Betti curves, and lifetime histograms for both sides.
- [x] Render persistence diagrams side by side.
- [x] Render landscape curves side by side.
- [x] Render persistence images as heatmaps side by side.
- [x] Render silhouettes, entropy vectors, Betti curves, and lifetime histograms.
- [x] Render similarity scores for every vectorized feature family.

### D. Miller-Sturmfels Staircases

- [x] Replace the current stacked staircase with actual 2D monomial ideal
      staircase logic for exponent generators in two variables.
- [x] Render an overhead xy-grid view with ideal region, quotient basis points,
      minimal generators, and staircase boundary.
- [x] Render close z-height module layers in 3D for `S/I`, first syzygy-like
      lcm layer, and optional higher/module comparison layers.
- [x] Ensure z-heights are close together and the overhead projection preserves
      the Miller-Sturmfels xy-grid shape.
- [x] Include exact generator and lcm-layer metadata in the record summary.

### E. Resolutions And Differentials

- [x] Replace token-frequency differential heatmap with a cleaner differential
      matrix/string panel that shows source/target module strips.
- [x] Render differential cards with degree/module labels and raw exact strings
      in collapsible details.
- [x] Add compact `d^2=0`, mapping-cone, Ext, and Tor status cards.

### F. Tests

- [x] Test analogical report contains source/memory simplex-tree payloads,
      map-validity tables, no-analogy gate, and dual sliders.
- [x] Test PH panels contain diagrams, landscapes, persistence images,
      silhouettes, entropy vectors, Betti curves, and lifetime histograms.
- [x] Test staircase report contains overhead xy-grid, close-height 3D module
      layers, quotient basis, ideal region, and lcm layer metadata.
- [x] Test resolution report contains module strips, differential cards, and
      compact derived-status cards.

### G. Docs And Artifacts

- [x] Update README/docs with corrected visualization semantics and commands.
- [x] Regenerate branching trajectory HTML.
- [x] Regenerate toric embedding HTML.
- [x] Render screenshots and contact sheets for every regenerated HTML bundle.
- [x] Inspect screenshots visually and record findings.
- [x] Run focused tests and full `tests/` suite.

## Completed Outputs

- Branching report:
  `outputs/latest_branching_reasoning_trajectory_report/branching_reasoning_trajectory.html`
- Branching screenshots:
  `outputs/latest_branching_reasoning_trajectory_report/html_screenshots/contact_sheet.png`
- Toric embedding report:
  `outputs/latest_toric_embedding_visual_report/records/record_000_toric_embedding.html`
- Toric embedding screenshots:
  `outputs/latest_toric_embedding_visual_report/html_screenshots/contact_sheet.png`

## Validation

- Focused visualization tests:
  `6 passed`
- Full suite:
  `279 passed, 2 skipped, 2 warnings`

## Screenshot Review

- Branching report: full graph-of-thought filtered complex, per-step simplex
  tree, no-analogy banner, side-by-side source/memory simplex trees, validity
  table, and GUDHI persistence/vectorized panels render correctly.
- Toric embedding report: resolution module strip, exact differential cards,
  status cards, Miller-Sturmfels overhead xy staircase, and close z-height
  module layers render correctly.  The regenerated old checkpoint sidecar has a
  simple two-generator monomial ideal, so the staircase is correspondingly
  simple but no longer a shifted Pareto/frontier plot.

## Completion Criteria

This plan is complete only when every checklist item above is checked, the
focused and full tests pass, regenerated HTML artifacts exist, screenshots exist
under `outputs/`, and the final response reports the exact paths.
