# Full Visualization, Analogical Retrieval, And PH Implementation Plan

Date: 2026-06-15

This is the active checklist for the current implementation pass.  This pass is
analysis, rendering, inference/audit plumbing, and tests only.  It must not
start, restart, or supervise training.

## Non-Negotiable Requirements

- Do not start training.
- Do not call a visualization exact unless it is computed from actual GUDHI,
  SageMath, Macaulay2, or explicitly stored finite data.
- Embedding-space comparisons use original high-dimensional embeddings.  PCA is
  display-only.
- If analogical retrieval does not pass the required map and PH checks, the
  report must visibly say no analogy was emitted.
- Full reasoning trajectories must have radius and reasoning-level sliders.
- Per-step simplex views must have radius and decoding-order sliders.
- Token-bearing plots must expose token information in hover text and in a
  click-details panel.
- Regenerate HTML reports, render screenshots, inspect screenshots, and record
  findings before this plan is marked complete.

## Checklist

### A. Long Branching Trajectory Generator

- [x] Add configurable long-trajectory generation with many reasoning levels.
- [x] Ensure branches have long internal chains and occasional merges.
- [x] Ensure every reasoning step has token embeddings and token metadata.
- [x] Ensure full graph-of-thought plots remain readable at larger size.

### B. Token Hover/Click Mechanics

- [x] Add richer token metadata: text, token type, decode rank, NLL, logprob,
      entropy, and source reasoning step.
- [x] Add hover templates showing token metadata in per-step simplex plots.
- [x] Add click-details panels for full reasoning nodes and per-step tokens.
- [x] Ensure hover/click details do not overlap or crowd the plots.

### C. Analogical Memory Retrieval Visualization

- [x] Keep source and memory simplex trees built by GUDHI over original
      embeddings.
- [x] Render accepted map arrows only if full trajectory map, per-step maps, and
      vectorized PH checks pass.
- [x] Render rejected candidates as no-analogy states with a validity table.
- [x] Add more explicit source-memory tree labels and map status text.
- [x] Keep all vectorized PH panels visible: diagrams, landscapes, persistence
      images, silhouettes, entropy vectors, Betti curves, and lifetime
      histograms.

### D. Rich Miller-Sturmfels Staircase Demonstration

- [x] Add a deterministic richer sidecar generator for staircase visualization
      with at least four adjacent minimal monomial generators.
- [x] Render the old-checkpoint exact sidecar and the richer demonstration
      sidecar in the same report bundle.
- [x] Keep the old sidecar exact and label the richer one as a deterministic
      visualization sidecar, not as a checkpoint-derived measurement.
- [x] Ensure the richer staircase overhead resembles the Miller-Sturmfels
      two-variable staircase picture.
- [x] Ensure close z-height module layers are visually separated but not
      exaggerated.

### E. Resolution And Differential Visualization

- [x] Parse simple Macaulay2 matrix strings into structured rows where possible.
- [x] Render source/target differential dimensions when parseable.
- [x] Keep raw exact differential strings available in collapsible details.
- [x] Keep d^2, Ext, Tor, and mapping-cone status cards adjacent to the maps.

### F. Tests

- [x] Test long trajectory generation has more nodes, branches, merges, and
      token metadata.
- [x] Test generated HTML contains node and token detail panels plus hover
      metadata keys.
- [x] Test analogical report still contains no-analogy gating and PH panels.
- [x] Test rich staircase demo sidecar has at least four minimal generators and
      adjacent lcm layer metadata.
- [x] Test differential parser/display handles simple Macaulay2 matrix strings.

### G. Documentation And Artifacts

- [x] Update README with the long branching report and rich staircase demo
      commands.
- [x] Regenerate branching report with a much longer trajectory.
- [x] Regenerate toric embedding report with old sidecar plus richer staircase
      demo sidecar.
- [x] Render screenshots/contact sheets for all regenerated reports.
- [x] Inspect screenshots and record findings.
- [x] Run focused tests and the full test suite.

## Generated Outputs

- Long branching report:
  `outputs/latest_branching_reasoning_trajectory_report/branching_reasoning_trajectory.html`
- Long branching screenshot contact sheet:
  `outputs/latest_branching_reasoning_trajectory_report/html_screenshots/contact_sheet.png`
- Long branching viewport slices:
  `outputs/latest_branching_reasoning_trajectory_report/html_screenshots/screenshots/branching_reasoning_trajectory__slice_00.png`
  through
  `outputs/latest_branching_reasoning_trajectory_report/html_screenshots/screenshots/branching_reasoning_trajectory__slice_08.png`
- Toric report index:
  `outputs/latest_toric_embedding_visual_report/index.html`
- Rich staircase demo:
  `outputs/latest_toric_embedding_visual_report/records/rich_miller_sturmfels_staircase_demo_toric_embedding.html`
- Toric screenshot contact sheet:
  `outputs/latest_toric_embedding_visual_report/html_screenshots/contact_sheet.png`

## Screenshot Review

- Long branching report: regenerated with 146 reasoning nodes, 186 DAG edges,
  long lane chains, side branches, cross-lane merges, token metadata, node
  click-detail panel, and token click-detail panel.  Viewport slice screenshots
  confirm the top metrics, per-step simplex tree, no-analogy gate, map validity
  table, and vectorized PH panels render correctly.
- Toric rich staircase report: regenerated with the old exact checkpoint sidecar
  plus a deterministic non-checkpoint Miller-Sturmfels demo sidecar.  The rich
  demo shows a legible two-variable staircase overhead, close z-height module
  layers, adjacent lcm layer, and bounded raw panels.

## Validation

- Focused tests:
  `10 passed`
- Full suite:
  `282 passed, 2 skipped, 2 warnings`

## Completion Criteria

This plan is complete only when every checklist item is checked, regenerated
HTML artifacts and screenshots exist, screenshots have been inspected, focused
and full tests pass, and the final response reports the exact output paths.
