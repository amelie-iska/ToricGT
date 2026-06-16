# Branching Real Embedding Visual Completion Plan - 2026-06-16

This is the live checklist for the current implementation pass.  The goal is
to make the graph-of-thought/simplex-tree/analogical-memory report usable both
as a long stress-test fixture and as a renderer for saved checkpoint inference
embedding payloads.  This pass does not start training.

Non-negotiable rendering rule: visible one-dimensional simplex edges must not
be capped or silently dropped.  Dense views may use radius sliders, reasoning
level sliders, decoding-order sliders, opacity, default-off two-simplex
surfaces, tables, and diagnostics, but every active visible one-dimensional
simplex edge must be present in the rendered payload.

## Checklist

- [x] Inspect current branch/merge visualization generator, CLI, tests, README,
  and the TokenGT geometry-suite embedding payload format.
- [x] Write this live implementation checklist before patching.
- [x] Add a first-class builder for saved `toricgt.embedding_payload.v1` NPZ/JSON
  outputs from checkpoint inference, using original hidden embeddings for all
  topology and analogy comparisons.
- [x] Extend the CLI with `--embedding-payload-npz` and
  `--embedding-payload-json` so the same HTML report can render old checkpoint
  inference payloads without inventing a second format.
- [x] Make the dataclass defaults long and branchy enough to stress the UI
  without needing CLI overrides.
- [x] Add source-mode metadata to payloads so screenshots/docs distinguish
  synthetic stress fixtures from checkpoint embedding payload renders.
- [x] Strengthen token hover/click metadata and simplex-tree tables so token
  text, type, decode order, source step, NLL, log probability, entropy, rank,
  and global token id are available in every token-bearing view.
- [x] Preserve and test all GUDHI vectorized PH panels: diagrams, landscapes,
  persistence images, silhouettes, entropy vectors, Betti curves, lifetime
  histograms, PH similarity bars, PH difference plots, exact bottleneck, and
  optional GUDHI Wasserstein.
- [x] Preserve and test analogical-memory gates based on full-trajectory
  simplex-tree maps, per-step simplex maps, and vectorized PH similarity, with
  strong/weak/candidate/no-analogy tiers.
- [x] Regenerate a much longer branch/merge report with more reasoning levels,
  branches, and side-branch length than the earlier fixture.
- [x] Generate screenshots and a contact sheet under `./outputs`.
- [x] Inspect representative screenshot slices and iterate if the report is
  visually misleading, missing controls, missing token metadata, or missing PH
  panels.
- [x] Run focused visualization tests.
- [x] Run the full test suite.
- [x] Update README/docs with the new embedding-payload input path, long
  defaults, generated output paths, and no-edge-cap contract.

## Rendered Output Record

Synthetic long fixture:

- `outputs/branching_reasoning_trajectory_long_20260616T021855Z/branching_reasoning_trajectory.html`
- `outputs/branching_reasoning_trajectory_long_20260616T021855Z/branching_reasoning_payload.json`
- `outputs/branching_reasoning_trajectory_long_20260616T021855Z/html_screenshots/contact_sheet.png`
- `outputs/latest_branching_reasoning_trajectory_report` now points to this timestamped bundle.

Checkpoint embedding payload example:

- `outputs/branching_reasoning_embedding_payload_20260616T021855Z/branching_reasoning_trajectory.html`
- `outputs/branching_reasoning_embedding_payload_20260616T021855Z/branching_reasoning_payload.json`
- `outputs/branching_reasoning_embedding_payload_20260616T021855Z/html_screenshots/contact_sheet.png`

Screenshot review notes:

- The synthetic report shows the source mode, 218 reasoning steps, 264 DAG
  edges, a weak analogy, full/step/analogy sliders, selected-step token
  subcomplexes, and PH panels.
- The selected-step simplex view renders vertices, one-dimensional radius
  edges, dotted decoding-order arrows, token NLL colors, and a simplex
  filtration table with token labels.
- The analogical-memory view emits map arrows for the weak analogy and includes
  the vertex correspondence table, map validity table, vectorized PH bar chart,
  exact bottleneck/Wasserstein table, diagrams, landscapes, images,
  silhouettes, entropy vectors, Betti curves, and lifetime histograms.
- The checkpoint payload report renders with source mode
  `checkpoint_embedding_payload`, 384 nodes, 383 graph edges, and a strong
  analogy from a saved old-checkpoint embedding payload.

## Acceptance Criteria

- The default synthetic fixture uses at least 18 reasoning levels, 8 branch
  lanes, and side branches of length at least 5.
- The generated stress report contains hundreds of reasoning nodes, explicit
  branch and merge behavior, dual sliders for full/step/analogy views, default
  off two-simplex surfaces, all visible one-dimensional simplex edges, and
  token hover/click details.
- The saved-embedding builder accepts the NPZ/JSON produced by
  `scripts/evaluate_tokengt_reasoning_geometry_suite.py --emit-embedding-payloads`
  and emits the same report schema.
- PCA coordinates are marked display-only; all maps and PH metrics use the
  original hidden embedding space.
- Focused tests and the full test suite pass before the checklist is marked
  complete.

## Validation

- Focused visualization tests:
  `conda run -n tokengt env PYTHONPATH=src:. pytest -q tests/test_branching_reasoning_visualization.py`
  returned `5 passed`.
- Full test suite:
  `conda run -n tokengt env PYTHONPATH=src:. pytest -q tests`
  returned `283 passed, 2 skipped, 2 warnings`.
