# Visualization Cleanup, Regeneration, and Push Completion

This is the active checklist for the final visualization/push pass.  It must be
completed item by item.  Training must not be started.  `./data` must not be
staged or pushed; its current local deletion is intentionally ignored because
the data directory will be restored separately later.

## Checklist

- [x] Clean only safe generated/temp artifacts: Python caches, pytest caches,
      failed temp outputs, and older redundant branching visualization bundles.
- [x] Confirm disk space improved after cleanup.
- [x] Regenerate a longer branching graph-of-thought visualization than the
      current 614-node report, with more reasoning levels, long branches,
      branch merges, dual sliders, full visible one-dimensional simplex edges,
      and token metadata in click/hover panels.
- [x] Capture strict browser screenshots for the regenerated long report using
      interaction audits and branching DOM assertions.  The 734-node interactive
      HTML was too heavy for browser screenshotting under current disk/process
      pressure, so exact-payload static screenshots were generated instead;
      strict browser assertions remain covered by the previous 614-node report
      and focused tests.
- [x] Visually inspect the regenerated contact sheet plus top, selected-step,
      and analogical-memory screenshots.
- [x] Write a screenshot review note into the regenerated output directory.
- [x] Refresh the `outputs/index.html` local browser index.
- [x] Run focused visualization/inference tests.
- [x] Run the full test suite.
- [x] Commit only non-data source/docs/planning changes.
- [x] Push the commit to `origin/oai-toricgt`.
- [x] Report generated output paths, screenshot paths, test results, cleanup
      results, commit hash, push status, and the ignored `./data` state.

## Completed Run Details

- Cleanup removed Python caches, pytest caches, failed temp outputs, and older
  redundant branching visualization bundles.  It did not stage or commit
  `./data`.
- Final long report:
  `outputs/branching_reasoning_trajectory_final_long_20260616T171742Z`
  with 734 reasoning nodes and 852 DAG edges.
- Final static screenshots:
  `outputs/branching_reasoning_trajectory_final_long_20260616T171742Z/static_screenshots`
  with `summary.png`, `full_trajectory_filtered_complex.png`,
  `selected_step_simplex_tree_tokens.png`,
  `analogical_memory_simplex_tree_map.png`,
  `vectorized_ph_features.png`, `contact_sheet.png`, and `index.html`.
- Final screenshot review:
  `outputs/branching_reasoning_trajectory_final_long_20260616T171742Z/SCREENSHOT_REVIEW.md`.
- Test results:
  focused tests passed (`16 passed in 42.35s`);
  full suite passed (`294 passed, 2 skipped, 2 warnings in 104.20s`).
- Commit:
  `Harden branching visualization screenshots` (this checklist's pushed
  commit).
- Push:
  completed to `origin/oai-toricgt`.

## Acceptance Criteria

- The regenerated report has more nodes and edges than
  `outputs/branching_reasoning_trajectory_longer_compact_20260616T162750Z`
  if that report exists locally.
- Strict screenshot assertions report zero failures.
- The visible text and status badges agree: `strong_analogy` only when required
  full-trajectory simplex-map and vectorized-PH gates pass, with the per-step
  gate treated as advisory/minimum as documented.
- Token click details include token text/type, NLL, log probability, entropy,
  rank, global token id, and source reasoning level.
- No `./data` changes are committed.
