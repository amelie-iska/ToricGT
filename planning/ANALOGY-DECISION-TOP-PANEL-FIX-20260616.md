# Analogy decision top-panel fix - 2026-06-16

The first viewport previously showed `weak_analogy` next to high full-map and
vectorized-PH scores, while the failed strong gate was visible only by reading
the raw JSON or scrolling to the lower analogy panel.  This made the decision
look inconsistent even though `step simplex-map mean = 0.4636 < 0.7000` was
correctly blocking `strong_analogy`.

## Checklist

- [x] Put the threshold status badges in the top `Analogy Decision` card.
- [x] Put the strong/weak/candidate threshold table in the top card.
- [x] Keep the raw decision payload available but move it behind a collapsible
      details block.
- [x] Add regression-test coverage for the top status badges and threshold
      table.
- [x] Regenerate the long visualization report.
- [x] Render screenshots and inspect the first viewport.
- [x] Run focused tests.
- [ ] Commit and push this bugfix.

## Validation

- First viewport screenshot inspected:
  `outputs/latest_branching_reasoning_trajectory_report/html_screenshots/screenshots/branching_reasoning_trajectory__slice_00.png`.
  The top decision panel now states that `weak_analogy` is emitted because the
  strong step-map gate failed: `step simplex-map mean 0.4444 < 0.7000`.
- Focused tests:
  `conda run -n tokengt env PYTHONPATH=src:. pytest -q tests/test_branching_reasoning_visualization.py tests/test_render_html_screenshots.py tests/test_outputs_index.py`
  passed with 9 tests.
