# Analogy step-gate relaxation - 2026-06-16

The per-step simplex-map mean should not be as strict as the full trajectory
simplex-map and vectorized persistent-homology evidence.  It is a local sanity
check for stepwise consistency, not the main analogy criterion.

## Checklist

- [x] Lower the default strong per-step simplex-map threshold from `0.70` to
      `0.40`.
- [x] Reduce the per-step contribution to analogy confidence from `0.30` to
      `0.15`.
- [x] Increase the full-trajectory and vectorized-PH confidence weights to
      `0.50` and `0.35`.
- [x] Add regression coverage that the strong step threshold stays loose.
- [x] Update README and Parameter-Golf docs to state that per-step consistency
      is a loose sanity check.
- [x] Regenerate the current long report and screenshots.
- [x] Run focused tests.
- [x] Commit and push the update.

## Validation

- Focused tests:
  `conda run -n tokengt env PYTHONPATH=src:. pytest -q tests/test_branching_reasoning_visualization.py tests/test_render_html_screenshots.py tests/test_outputs_index.py`
  passed with 9 tests.
- Project-local suite:
  `conda run -n tokengt env PYTHONPATH=src:. pytest -q tests`
  passed with 288 tests, 2 skipped, 2 warnings.

## Generated report check

- Current long report now emits `strong_analogy`.
- Scores: full map `0.9905`, vectorized PH mean `0.9719`, PH gate `0.9875`,
  step simplex-map mean `0.4444`.
- Strong thresholds: full map `0.8000`, vectorized PH mean `0.5500`, step
  simplex-map mean `0.4000`.
- First viewport screenshot inspected:
  `outputs/latest_branching_reasoning_trajectory_report/html_screenshots/screenshots/branching_reasoning_trajectory__slice_00.png`.
