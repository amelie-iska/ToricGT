# Branching Periodic/Inference Integration Plan - 2026-06-16

This is the live checklist for the current pass.  The goal is to complete the
branching graph-of-thought simplex visualization as a reusable audit artifact:
standalone long fixture, old-checkpoint embedding payload renderer, periodic
analysis sidecar, inference optional output, and output index.  This pass does
not start training.

Non-negotiable rendering rule: visible one-dimensional simplex edges must not
be capped or silently dropped inside a selected point set.  For large real
checkpoint payloads, the renderer may expose an explicit node-window selection
before building the visible complex; within that selected window, every active
visible one-dimensional simplex edge must be rendered.

## Checklist

- [x] Inspect branch/simplex renderer, TokenGT geometry embedding payloads,
  periodic watcher, inference wrapper, and existing screenshot/index tools.
- [x] Write this live checklist before implementation.
- [x] Add explicit checkpoint embedding-payload node-window controls:
  `--embedding-max-nodes` and `--embedding-node-offset`.
- [x] Preserve no-edge-cap behavior inside the selected node window and record
  original/selected node metadata in the report payload.
- [x] Add tests for bounded checkpoint-payload rendering and CLI propagation.
- [x] Add periodic watcher CLI switches and a sidecar function that renders the
  branch/simplex report from the geometry-suite embedding manifest.
- [x] Include branch/simplex report links and availability in periodic
  `index.html`, `SYNOPSIS.md`, and `analysis_status.json`.
- [x] Add inference-wrapper switches to optionally emit the branch/simplex
  report from saved embedding payloads and link it from inference `index.html`.
- [x] Add a lightweight `outputs/index.html` renderer that links the latest
  branch/simplex, checkpoint-payload, toric embedding, GUDHI, and screenshot
  artifacts when present.
- [x] Run focused tests for branch/simplex, watcher, inference, and output
  index behavior.
- [x] Regenerate a checkpoint-payload branch/simplex report using an existing
  old checkpoint embedding payload and the bounded node-window path.
- [x] Generate screenshots/contact sheets for the current synthetic and
  checkpoint-payload reports.
- [x] Patch the analogy visualization so it clearly shows a candidate
  source-to-memory simplex-tree map even when the final analogy tier is not
  emitted.
- [x] Make weak/candidate analogy thresholds visibly lenient and configurable,
  while still reporting exact GUDHI PH feature similarities and exact
  simplicial-map validity.
- [x] Render a longer synthetic branch/merge report than the current 218-node
  fixture without starting training or filling the disk.  The run may reduce PH
  display resolution for browser tractability, but it must not cap visible
  one-dimensional simplex edges inside the rendered point set.
- [x] Generate a fresh `outputs/index.html` after the latest render pass.
- [x] Inspect representative screenshots and iterate if controls, PH panels,
  simplex maps, token metadata, or source-mode metadata are missing or
  misleading.
- [x] Update README/docs with the new periodic, inference, node-window, and
  output-index commands.
- [x] Run the full test suite.
- [x] Keep disk usage bounded: remove interrupted reports, pytest temp files,
  and generated CAS caches after large render/test passes.  Do not delete
  source files, current checkpoints, or current complete report outputs.

## Acceptance Criteria

- Long synthetic report remains at least 18 levels, 8 lanes, and side branches
  of length at least 5 by default.
- Checkpoint embedding-payload reports can be explicitly windowed for browser
  tractability while preserving all visible one-dimensional edges inside the
  selected window.
- Periodic analysis can render and screenshot the branch/simplex report from
  geometry embedding payloads without blocking training checkpoint analysis if
  a checkpoint family lacks embeddings.
- Inference can optionally emit the same report and link it from its main HTML
  bundle.
- The central outputs index links the latest generated HTML reports and
  screenshot contact sheets.
- Focused and full tests pass before this checklist is complete.
