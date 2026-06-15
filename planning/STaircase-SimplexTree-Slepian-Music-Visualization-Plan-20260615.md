# Staircase, Simplex-Tree, Slepian, and Music Visualization Plan - 2026-06-15

## Execution Rule

This is the implementation ledger for the current visualization pass.  The
instruction is to leave nothing in this plan unimplemented.  Every update to
code must be reflected here, and this file is not complete until every item is
implemented, tested, and documented.

This is analysis/inference functionality only.  Do not start, restart, pause,
kill, or otherwise modify training runs while implementing this plan.

## Mathematical And Visualization Contract

The bivariate module views should follow the Miller-Sturmfels staircase picture
for monomial ideals in two variables: lattice points in the quotient region,
staircase boundary, shifted positive orthants for monomial generators, adjacent
lcm corners, and syzygy segments.  The report must not call this a Pareto
frontier in user-facing text.

Reasoning-trajectory topology must be attached to actual TokenGT hidden
embeddings.  Each reasoning step is a sub-collection of token embedding vectors;
each step has a filtered simplex tree over radius; the full reasoning
trajectory is a graph whose vertices are reasoning steps, with directed decode
order and step-to-step filtered simplicial-complex maps.

Slepian/Pollak output must be both visual and audible: the report should render
the finite prolate/DPSS concentration signal as a colormap on a foliated torus,
and optional inference should export a WAV plus JSON metadata.

## Checklist

### 1. Miller-Sturmfels Staircase Modules

- [x] Replace user-facing "Pareto frontier" language with "staircase".
- [x] Render bivariate module views as staircase diagrams with shifted
  orthants for monomial generators.
- [x] Add small layered height offsets for C0/C1/C2 generators, lcm corners,
  and syzygy edges so the module data is visually separated.
- [x] Persist enough staircase metadata in record JSON for testing and
  downstream review.
- [x] Add tests that generated GUDHI pages contain staircase terminology and
  no user-facing "Pareto frontier".

### 2. Cleaner Resolution And Differential Visuals

- [x] Add a compact resolution strip diagram: C2 -> C1 -> C0 with ranks,
  Betti/Hilbert summaries, and pass/fail `d^2`/exactness badges.
- [x] Add differential matrix heatmaps for `d1` and `d2` over `GF(2)`.
- [x] Keep raw Macaulay2 text in collapsible details, but make the first view
  graphical and readable.
- [x] Add tests that record pages include resolution-strip and differential
  heatmap sections.

### 3. Reasoning Step Simplex Trees And Full Trajectory Maps

- [x] Store/reconstruct per-step subcollections of token embeddings from the
  TokenGT reasoning trajectory.
- [x] Build a filtered simplex tree summary for each reasoning step over a
  fixed radius schedule.
- [x] Render an interactive 3D PCA full reasoning trajectory with reasoning
  steps as nodes, node color as NLL, and slider-controlled radius.
- [x] Add hover/click text showing the filtered simplicial complex for each
  reasoning step at the current radius.
- [x] Add faint dotted directed decode-order arrows that appear only as the
  slider advances.  Edge arrows are half-arrows that become visible at the
  relevant reasoning level.
- [x] Render analogical maps between reasoning-step simplex trees and between
  full trajectory filtered complexes, including vectorized persistence
  comparison bars.
- [x] Add these outputs to the plot-family manifest.
- [x] Add tests proving the interactive HTML contains slider steps, simplex-tree
  payload, NLL coloring, and analogical map traces.

### 4. 3D PCA Coverage

- [x] Ensure every new reasoning-step, full trajectory, and analogical map
  visualization has a 3D PCA representation.
- [x] Use NLL coloring for token/step nodes where NLL is available; otherwise
  mark the value as unavailable rather than substituting an unlabeled proxy.

### 5. Slepian/Pollak Foliated Torus Visualization

- [x] Render a 3D torus surface whose color is the finite Slepian/Pollak
  reconstruction/envelope over the irrational phase foliation.
- [x] Add hover text containing phase coordinates, Slepian value, envelope,
  local NLL/energy, and trajectory index.
- [x] Add click-ready Plotly customdata for downstream browser inspection.
- [x] Add the Slepian torus output to the plot-family manifest.
- [x] Add tests proving the HTML contains torus surface, Slepian customdata,
  and dark-mode Plotly payload.

### 6. Slepian/Pollak Music Export

- [x] Add optional inference flags for exporting Slepian/Pollak music.
- [x] Export WAV and JSON metadata through the existing deterministic music
  generator.
- [x] Link the WAV/metadata from the inference root index and manifest.
- [x] Add tests that the inference CLI exposes music flags and the music file
  generation path writes WAV/JSON.

### 7. Documentation And Verification

- [x] Update README with staircase modules, reasoning-step simplex-tree
  trajectory views, Slepian torus surface, and optional music export.
- [x] Run focused tests for GUDHI, geometry, music, and inference CLI.
- [x] Run the full test suite.
- [x] Regenerate a small visualization bundle from an old checkpoint and render
  screenshots into `outputs`.

## Progress Log

- 2026-06-15: Plan created after reviewing the current GUDHI, geometry,
  Slepian, and music paths.
- 2026-06-15: Implemented staircase module views, resolution-strip and
  differential heatmaps, reasoning-step simplex-tree HTML, analogical
  simplex-map HTML, Slepian/Pollak torus HTML, and optional Slepian music
  inference export.  Focused GUDHI/geometry/music tests pass.
- 2026-06-15: Full suite passed (`266 passed, 2 skipped`).  Generated
  `outputs/staircase_simplextree_slepian_audit_20260615T174121Z`, rendered 12
  strict Playwright screenshots with zero errors, validated the inference
  manifest with `validate_analysis_exactness.py`, and linked
  `outputs/latest_staircase_simplextree_slepian_audit`.
