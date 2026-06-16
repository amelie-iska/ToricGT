# Branching Analogy Scalability Repair Plan

This pass must complete the remaining branch/merge reasoning visualization and audit gaps without starting training.  The goal is a report that scales to longer trajectories, explains analogy decisions without status/threshold mismatches, keeps token metadata available on hover/click, and is usable from the periodic analysis watcher.

## Checklist

- [x] Make strong analogy depend on required full-trajectory simplex-map and vectorized-PH gates, with per-step simplex-map evidence treated as advisory plus a low minimum sanity floor.
- [x] Render threshold rows with an explicit role (`required`, `advisory`, or `minimum`) so a nonblocking advisory warning cannot look like a status contradiction.
- [x] Compact source/memory simplex-tree edge and triangle records into array records while keeping all visible one-dimensional simplex-tree edges exact at each radius level.
- [x] Compact candidate simplicial-map image records into array records and update browser tables/summaries to decode them.
- [x] Preserve token and reasoning-vertex hover/click metadata in full trajectory, selected-step, and analogy views after compaction.
- [x] Strengthen screenshot assertions so they verify status/table consistency, compact edge-birth mode, token detail updates, and analogy detail updates.
- [x] Wire strict branch/merge report screenshot assertions into `scripts/watch_training_analysis.py` so periodic training analysis can fail loudly when the report contract breaks.
- [x] Prevent raw `pytest` from collecting unrelated vendored external tests while keeping project tests available.
- [x] Add tests for the nonblocking strong-step advisory rule, compact simplex-tree records, compact map-image records, strict screenshot assertions, and watcher screenshot assertion wiring.
- [x] Update README/docs with the analogy decision semantics, compact payload format, and periodic watcher assertion behavior.
- [x] Regenerate a longer branching trajectory report with more levels/branches than the previous 432-node report, or document the largest exact report that remains practical locally.
- [x] Render screenshot and interaction screenshot artifacts, inspect them, and write a short screenshot review note.
- [x] Refresh `outputs/index.html`.

## Completion Notes

- Regenerated report: `outputs/branching_reasoning_trajectory_long_compact_20260616T151500Z`, symlinked by `outputs/latest_branching_reasoning_trajectory_report`.
- Report scale: 492 reasoning steps, 580 graph-of-thought DAG edges, 7 radius levels, compact source/memory simplex-tree payloads, and compact candidate map-image records.
- Screenshot bundle: `outputs/latest_branching_reasoning_trajectory_report/html_screenshots`.
- Screenshot assertions: 10 checked, 0 failed.
- Screenshot review note: `outputs/latest_branching_reasoning_trajectory_report/SCREENSHOT_REVIEW.md`.
- Output index: `outputs/index.html` refreshed.

## Acceptance Criteria

- Strong analogy can be emitted when full trajectory map and vectorized PH are strong, even if the per-step mean is below the advisory target but above the minimum sanity floor.
- The visual threshold table states which rows are required and which are advisory.
- Full and analogy simplex-tree one-dimensional edges remain uncapped within the selected point set.
- The browser payload is smaller than the previous dense/verbose 432-node payload for the same fixture.
- `conda run -n tokengt env PYTHONPATH=src:. pytest -q tests` passes.
- The regenerated report screenshot manifest has zero assertion failures.
