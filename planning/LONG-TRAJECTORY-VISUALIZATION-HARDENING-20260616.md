# Long-Trajectory Visualization Hardening Plan

This pass must finish with a regenerated long branching graph-of-thought report, verified browser screenshots, and tests that assert the visual contract.  Do not start training.  Do not treat dense payload blowups as acceptable; the report must remain exact for visible one-dimensional simplex edges while scaling to longer trajectories.

## Checklist

- [x] Add compact exact edge-birth payloads for full reasoning-trajectory filtered simplicial complexes and per-step token simplex trees.
- [x] Optimize 2-simplex selection so capped face rendering does not materialize every possible triangle for long reports.
- [x] Update browser rendering to use edge-birth records for radius sliders while preserving all visible one-dimensional simplex edges.
- [x] Keep token hover/click metadata available in all token-level simplicial views and analogy views.
- [x] Add strict screenshot DOM assertions for analogy threshold tables, status badges, slider captions, and token/analogy detail panels.
- [x] Add or update automated tests covering compact payloads, all-visible edge semantics, optimized triangle handling, and screenshot assertions.
- [x] Update README/docs with the compact edge-birth visualization contract and the screenshot assertion workflow.
- [x] Regenerate a longer branching trajectory visualization with many reasoning branches and longer branches.
- [x] Render screenshots, interaction screenshots, and a contact sheet into `outputs/`.
- [x] Inspect the generated screenshots and report exact output paths.

## Acceptance Criteria

- Full trajectory and per-step simplex filtering are exact for all one-dimensional edges visible at the configured radius levels.
- No visible one-dimensional simplex edges are capped.
- 2-simplex rendering remains optional and bounded by the existing face cap.
- Screenshot automation fails loudly if the main branching report loses threshold evidence, status badges, or token/analogy detail panels.
- The regenerated report is longer than the previous 392-node audit and remains browser-renderable.

## Completion Notes

- Regenerated report: `outputs/branching_reasoning_trajectory_long_edgebirth_20260616T144202Z`.
- Latest report link: `outputs/latest_branching_reasoning_trajectory_report`.
- Report size: 432 reasoning nodes, 510 DAG edges, 67,029 radius-grid edge births, zero dense trajectory-distance rows in the browser payload.
- Screenshot audit: 2 HTML pages, 6 interaction states, 7 branching-report DOM assertions, 0 assertion failures.
- Screenshot contact sheet: `outputs/latest_branching_reasoning_trajectory_report/html_screenshots/contact_sheet.png`.
- Project test suite: `conda run -n tokengt env PYTHONPATH=src:. pytest -q tests` returned `289 passed, 2 skipped, 2 warnings`.
- Raw `pytest -q` was intentionally not used for final verification because it collects an unrelated vendored `amelie-iska/soft-mixture-of-experts/tests/conftest.py` that expects a nonlocal `--slow` option and aborts collection.
