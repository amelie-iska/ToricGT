# Analogy And Simplex Visualization Repair Plan - 2026-06-16

This document is the active implementation checklist for repairing the
branching reasoning trajectory, simplex-tree, analogical retrieval, and
vectorized persistent-homology visualizations.

Hard rule for this pass: visible one-dimensional simplex edges must not be
capped.  If a visualization becomes visually dense, the UI may use sliders,
toggles, camera controls, opacity, or default radius values, but it must not
silently drop rendered edges from the active filtered complex.

## Checklist

- [x] Inspect the current visualization generator, render script, tests, and
  README references.
- [x] Convert this document into the live implementation record and keep it
  updated as items are completed.
- [x] Remove visible edge caps from simplex-tree payloads and mark edge
  rendering policy as `all_visible_edges`.
- [x] Keep full high-dimensional embedding coordinates as the comparison space;
  retain PCA only for display coordinates.
- [x] Relax binary analogy gating into confidence tiers:
  `strong_analogy`, `weak_analogy`, `candidate_analogy`, and `no_analogy`.
- [x] Emit analogical map arrows for strong, weak, and candidate analogies;
  show no arrows only for true `no_analogy`.
- [x] Replace the hard rejection banner with a confidence-aware decision
  banner that explains which map and PH checks passed or failed.
- [x] Add default-off triangle toggles for the full trajectory, selected
  reasoning-step simplex, and analogical map views so vertices and
  one-dimensional edges are visible first.
- [x] Improve the selected reasoning-step view so it reads as a filtered
  simplex tree rather than a filled 3D hull.
- [x] Add a selected-step simplex-tree filtration table with vertices, edges,
  triangles, birth radius, simplex dimension, and token labels.
- [x] Add map diagnostics that expose full trajectory map score, stepwise map
  score, PH feature similarities, tier thresholds, and exact source edge and
  triangle validity counts.
- [x] Preserve token hover/click metadata wherever token points appear:
  token text, token type, source step, decode order, NLL, logprob, entropy,
  rank, and global token id.
- [x] Generate a longer, more branching graph-of-thought fixture with more
  levels, more lanes, and longer side branches than the current sample.
- [x] Regenerate the HTML report and screenshots.
- [x] Inspect the regenerated screenshots for:
  clear per-step simplex tree edges,
  visible analogy map arrows when structures match,
  usable PH panels,
  readable legends/colorbars,
  and token detail panels.
- [x] Run focused visualization tests.
- [x] Run the full test suite.
- [x] Update README/docs with the new controls and no-edge-cap contract.

## Final Output Record

Generated report:

- `outputs/latest_branching_reasoning_trajectory_report/branching_reasoning_trajectory.html`
- `outputs/latest_branching_reasoning_trajectory_report/branching_reasoning_payload.json`
- `outputs/latest_branching_reasoning_trajectory_report/index.html`

Generated screenshots:

- `outputs/latest_branching_reasoning_trajectory_report/html_screenshots/contact_sheet.png`
- `outputs/latest_branching_reasoning_trajectory_report/html_screenshots/screenshots/branching_reasoning_trajectory__slice_00.png`
- `outputs/latest_branching_reasoning_trajectory_report/html_screenshots/screenshots/branching_reasoning_trajectory__slice_01.png`
- `outputs/latest_branching_reasoning_trajectory_report/html_screenshots/screenshots/branching_reasoning_trajectory__slice_02.png`
- `outputs/latest_branching_reasoning_trajectory_report/html_screenshots/screenshots/branching_reasoning_trajectory__slice_03.png`
- `outputs/latest_branching_reasoning_trajectory_report/html_screenshots/screenshots/branching_reasoning_trajectory__slice_04.png`
- `outputs/latest_branching_reasoning_trajectory_report/html_screenshots/screenshots/branching_reasoning_trajectory__slice_05.png`
- `outputs/latest_branching_reasoning_trajectory_report/html_screenshots/screenshots/branching_reasoning_trajectory__slice_06.png`
- `outputs/latest_branching_reasoning_trajectory_report/html_screenshots/screenshots/branching_reasoning_trajectory__slice_07.png`
- `outputs/latest_branching_reasoning_trajectory_report/html_screenshots/screenshots/branching_reasoning_trajectory__slice_08.png`
- `outputs/latest_branching_reasoning_trajectory_report/html_screenshots/screenshots/branching_reasoning_trajectory__slice_09.png`

Final fixture:

- reasoning nodes: 218
- DAG edges: 264
- analogy status: `weak_analogy`
- analogy confidence: 0.7983
- full simplex-map valid fraction: 0.9880
- vectorized PH mean: 0.9437
- vectorized PH gate score: 0.9650
- source simplex-tree edges: 16,557 exact and 16,557 rendered
- memory simplex-tree edges: 16,534 exact and 16,534 rendered

Validation:

- Focused visualization tests: `4 passed`
- Full test suite: `282 passed, 2 skipped, 2 warnings`

## Notes From Screenshot Review

The previous selected-step view overemphasized filled two-simplices and produced
a small tangled hull.  The repaired default should emphasize vertices, radius
edges, and decode-order arrows, with filled two-simplices available only when
the user explicitly enables them.

The previous analogical map showed two visually similar structures but emitted
`No analogy`.  The repaired gate should use a tiered decision, because a high
valid simplicial map with moderate PH agreement is still a useful weak or
candidate analogy for retrieval diagnostics.
