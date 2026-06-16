# Visualization hardening completion plan - 2026-06-16

This checklist is the active implementation contract for this pass.  It is
scoped to analysis and visualization hardening only: no training run is started
by this work.  The implementation should preserve the current policy that
visible one-dimensional simplices are not capped inside the selected node
window.

## Checklist

- [x] Inspect the current branch/simplex report, screenshot renderer, periodic
      watcher, inference sidecar, and output-index paths before editing.
- [x] Add a strict browser interaction-audit mode to the screenshot renderer so
      screenshots cover slider movement, triangle toggles, and token/detail
      panels rather than only default page state.
- [x] Expose a report-level screenshot audit hook from the branch/simplex HTML
      so automated screenshots can reveal reasoning-node, token, and analogy
      metadata without relying on brittle pixel-coordinate clicks.
- [x] Add label-density controls to the branch/simplex report so long dense
      trajectories remain readable without dropping vertices or edges.
- [x] Wire interaction screenshots into periodic checkpoint analysis and
      optional inference screenshot generation.
- [x] Add tests for interaction screenshots, report audit hook wiring, and
      label controls.
- [x] Regenerate a substantially longer synthetic branch/merge reasoning
      trajectory with many levels, more branches, longer side branches, no edge
      capping, and full token metadata in hover/click panels.
- [x] Render browser screenshots, including interaction-audit states, for the
      regenerated report and refresh the outputs index.
- [x] Inspect the generated screenshots and iterate on visible issues found in
      the rendered output.
- [x] Update README/docs with the new screenshot-audit and long-trajectory
      generation commands.
- [x] Run focused tests, then the full test suite if the focused tests pass.

## Completion notes

- Verified long synthetic report: `outputs/branching_reasoning_trajectory_long_verified_20260616T045134Z`.
- Latest symlink: `outputs/latest_branching_reasoning_trajectory_report`.
- Report manifest: 353 reasoning nodes, 419 DAG edges, analogy emitted.
- Interaction screenshots: six branch/simplex audit states captured with no screenshot errors.
- Focused tests: 26 passed.
- ToricGT test suite: 286 passed, 4 skipped, 2 warnings.
- Root-level `pytest` was intentionally not used because the workspace contains
  an external repository whose own test plugin expects a non-ToricGT `--slow`
  option; `pytest tests` is the project-local suite.

## Non-goals for this pass

- Do not start or restart training.
- Do not push the mixed worktree without a separate commit-scope review.
- Do not delete large assets or external repositories without explicit review.
- Do not reintroduce edge caps in the visible simplex views.
