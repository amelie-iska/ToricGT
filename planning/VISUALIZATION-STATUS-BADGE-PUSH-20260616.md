# Visualization status-badge cleanup and push plan - 2026-06-16

This checklist governs the current pass.  The scope is visualization naming,
long branch/merge report regeneration, tests, docs, and a relevant-code push.
Do not start or restart training.  Do not stage or push `./data` changes.

## Checklist

- [x] Replace confusing "chip" terminology in generated branching-reasoning
      HTML, JavaScript ids, tests, docs, and planning notes with threshold or
      gate status badge terminology.
- [x] Preserve the visual pass/fail/status UI behavior while changing only the
      names and visible/exposed identifiers.
- [x] Defer heavy Plotly PH-panel initialization until after browser load so
      long reports can be screenshotted without changing simplex-edge
      visibility.
- [x] Regenerate a longer branch/merge reasoning-trajectory report with many
      reasoning branches, longer branches, all visible one-dimensional simplex
      edges inside the selected window, and token hover/click metadata.
- [x] Render interaction screenshots for the regenerated report.
- [x] Inspect representative screenshots, especially the analogy map/status
      badge panel and token-detail panel.
- [x] Run focused visualization tests.
- [x] Run the project-local test suite.

## Generated artifacts

- Latest report symlink:
  `outputs/latest_branching_reasoning_trajectory_report`
- Long screenshotable report:
  `outputs/branching_reasoning_trajectory_long_status_badges_20260616T010000Z`
- Report manifest: 392 reasoning nodes, 470 graph-of-thought DAG edges,
  analogy emitted.
- Screenshot manifest: 2 HTML pages, 6 interaction states, 0 errors.
- Screenshot review: contact sheet, analogy-detail panel, and selected-step
  token-detail panel inspected.

## Validation

- Focused tests:
  `conda run -n tokengt env PYTHONPATH=src:. pytest -q tests/test_branching_reasoning_visualization.py tests/test_render_html_screenshots.py tests/test_outputs_index.py`
  passed with 9 tests.
- Project-local suite:
  `conda run -n tokengt env PYTHONPATH=src:. pytest -q tests`
  passed with 288 tests, 2 skipped, 2 warnings.
- [x] Stage only relevant source, scripts, tests, docs, and planning files;
      explicitly exclude `./data` and unrelated generated bulk artifacts.
- [x] Commit and push the relevant branch.

## Notes

- "Status badge" here means the compact UI indicator for the actual threshold
  decision rule: strong, weak, candidate, map score, step-map score, vectorized
  PH mean, and PH gate.
- The UI must keep rejected or weak maps inspectable; no visible
  one-dimensional simplex edge cap should be introduced.
