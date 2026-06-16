# Analogy threshold and long-trajectory visualization repair - 2026-06-16

This is the active checklist for repairing the analogy-memory visualization
after screenshot review.  It is intentionally analysis-only: do not start,
restart, or alter training during this pass.

## Checklist

- [x] Inspect the current branch/simplex report HTML generator, screenshot
      tooling, and report tests before editing.
- [x] Fix all visible text so threshold labels match the actual analogy tier
      decision rule in the payload.
- [x] Add an explicit tier explanation for strong, weak, and candidate gates,
      including why a weak analogy was emitted when map/PH pass but step-map
      fails the strong threshold.
- [x] Add pass/fail threshold status badges for full trajectory simplex-map, step simplex-map,
      vectorized PH mean, PH gate, strong tier, weak tier, and candidate tier.
- [x] Make the analogy decision table use semantically correct column headers
      rather than mixing numeric scores with "valid" counts.
- [x] Ensure "Vectorized PH" spelling is consistent everywhere.
- [x] Make filled 2-simplex toggles semantically clear: off should render only
      one-dimensional simplex edges; on should add mesh faces.  Text labels
      must not imply meshes when only dense edges are visible.
- [x] Default dense full-trajectory and analogy labels to hidden for long
      reports, with clear label state text; keep hover/click metadata fully
      available.
- [x] Improve map readability without capping visible one-dimensional edges:
      encode map-arrow status and distance with color/opacity and add a compact
      map summary panel.
- [x] Add slider-state captions showing active radius, reasoning level,
      visible vertices, visible one-dimensional edges, visible 2-simplices,
      and mapped arrows.
- [x] Move/duplicate PH comparison summary closer to the analogy plot so the
      vectorized-PH evidence is visible before scrolling to the full PH panels.
- [x] Add regression tests for threshold text, status explanation, spelling,
      label toggles, slider captions, and map summary fields.
- [x] Regenerate a long branch/merge report with many reasoning branches and
      longer branches, preserving hover/click token metadata and no edge caps.
- [x] Render interaction screenshots, inspect them, and iterate if the rendered
      output still has obvious threshold/toggle/label problems.
- [x] Update README/docs with the repaired decision explanation and screenshot
      workflow.
- [x] Run focused tests and the project-local test suite.

## Completion notes

- Regenerated report: `outputs/branching_reasoning_trajectory_long_repaired_20260616T051840Z`.
- Latest symlink: `outputs/latest_branching_reasoning_trajectory_report`.
- Report manifest: 353 reasoning nodes, 419 graph-of-thought DAG edges,
  analogy emitted.
- Screenshot audit: 2 HTML pages, 6 interaction states, 0 errors.
- Screenshot review confirmed the weak-analogy banner now states the exact
  strong-threshold failure: `step simplex-map mean 0.4636 < 0.7000`.
- Focused tests: 10 passed.
- Project-local suite: 286 passed, 4 skipped, 2 warnings.

## Non-goals

- Do not start training.
- Do not push the mixed worktree without a separate scope review.
- Do not reintroduce edge caps inside the selected point set.
- Do not replace exact GUDHI computations with proxies.
