# Branching Reasoning Visualization Completion Pass

This checklist is the active completion contract for the current implementation
pass.  Do not start training during this pass.  Do not stop with unchecked items
unless an item is technically blocked by missing local dependencies or corrupted
inputs.  Update this file as each item is completed.

## Scope

The goal is to make the branching graph-of-thought, simplex-tree, analogical
memory, vectorized persistent-homology, and token-hover visualization pipeline
ready for ordinary use from old checkpoint embedding payloads and from optional
inference output.  The pass must use original high-dimensional embedding
coordinates for comparisons and maps; 3D PCA coordinates are display-only.

## Checklist

- [x] Verify the current repository state and preserve unrelated `./data`
      changes.
- [x] Make inference-generated branching report screenshots use the strict
      branching DOM assertion by default, with an explicit debugging opt-out.
- [x] Add tests proving the inference wrapper includes the strict branching
      screenshot assertion by default and omits it only when requested.
- [x] Regenerate a real old-checkpoint embedding-payload branching report from
      an existing saved NPZ/JSON payload.
- [x] Regenerate a longer synthetic branching trajectory with more levels,
      branches, and side-branches than the current latest report, keeping GPU
      memory untouched and avoiding any training process.
- [x] Capture browser screenshots for both generated reports using interaction
      audits, dual-slider coverage, token-detail click coverage, analogy-detail
      click coverage, and strict branching assertions.
- [x] View the generated screenshots locally and record a screenshot review note
      for the latest long report.
- [x] Refresh the outputs index so the new reports are linked from
      `outputs/index.html`.
- [x] Update README/docs if the inference screenshot assertion or latest
      generated report instructions changed.
- [x] Run focused visualization/inference tests.
- [x] Run the full test suite.
- [x] Report exact output paths, screenshot paths, test results, and any
      remaining unrelated dirty state.

## Completed Run Details

- Real old-checkpoint embedding-payload report:
  `outputs/branching_reasoning_real_embedding_payload_20260616T161728Z`
  with 180 reasoning nodes, 179 DAG edges, `checkpoint_embedding_payload`
  source mode, and `strong_analogy` emitted.
- Longer synthetic report:
  `outputs/branching_reasoning_trajectory_longer_compact_20260616T162750Z`
  with 614 reasoning nodes, 720 DAG edges, full visible one-dimensional
  radius-filtered simplex edges, and `strong_analogy` emitted.
- Latest symlink:
  `outputs/latest_branching_reasoning_trajectory_report` now points to the
  longer compact report.
- Screenshot reviews:
  both generated reports have `html_screenshots/manifest.json` with
  `assert_branching_report=true`, 12 assertions, and 0 assertion failures.
- Visual review:
  `outputs/branching_reasoning_trajectory_longer_compact_20260616T162750Z/SCREENSHOT_REVIEW.md`.
- Tests:
  focused visualization/inference tests passed (`15 passed in 42.60s`);
  full suite passed (`293 passed, 2 skipped, 2 warnings in 104.12s`).

## Acceptance Criteria

- The latest long report has substantially more reasoning nodes than the prior
  latest long report and includes token metadata in hover/click panels.
- Every simplicial visualization has both a radius slider and a reasoning-level
  or decoding-order slider where applicable.
- The analogical report exposes the full-trajectory simplex-map gate,
  vectorized PH gates, and the advisory per-step simplex-map score without
  contradicting the rendered analogy status.
- The screenshot harness reports zero strict branching assertion failures.
- No training is started or restarted.
