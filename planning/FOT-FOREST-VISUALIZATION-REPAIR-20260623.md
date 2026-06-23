# FoT Forest Visualization Repair Plan

This pass fixes a specific visualization bug: the public "reasoning trajectories
and reasoning-step filtered simplicial complexes" page currently shows a single
linear trajectory even when the model was trained with the embedding-space
Forest-of-Thought head.  That is misleading.  Forest-of-Thought should be shown
as several parallel thought trees grown under a reasoning budget, with expansion,
self-correction, consensus, and per-node local filtered simplicial complexes.

## Diagnosis

- `docs/page_interactive/branching_reasoning/branching_reasoning_interactive.json`
  currently has 512 nodes and 511 `dag_edges`, beginning `0 -> 1 -> 2`.  That is
  a path, not a forest.
- `src/toricgt/branching_reasoning_visualization.py` falls back to index-order
  checkpoint edges when the extracted OAI hidden-state payload has no explicit
  graph/FoT edges.
- `scripts/render_embedding_fot_report.py` does load the real checkpoint FoT head
  and the real hidden-state embedding payload, but the existing
  `trace_payload()` emits only interleaved per-tree chains.
- The public GitHub Pages branch is built through `scripts/build_toricgt_pages.py`,
  whose low-resource report compacts `branching_reasoning_payload.json` and does
  not currently ingest the sibling `embedding_fot_report/embedding_fot_trace.json`.

## Implementation Checklist

- [x] Extend `scripts/render_embedding_fot_report.py` to materialize an explicit
  FoT forest from the real checkpoint FoT head:
  - use real selected hidden states, target ids, NLL, activation logits, value
    head, forward-policy logits, correction vectors, flow values, and consensus
    head;
  - organize selected hidden states into parallel trees with BFS-style branch
    expansion using the configured branching factor;
  - attach expansion edges with policy probability and action id;
  - attach self-correction edges only when the real value/NLL evidence supports
    an improvement from one node to a later node in the same tree;
  - attach consensus edges from tree leaves to a consensus node;
  - attach budget/depth levels for an iterative growth slider;
  - attach original hidden-space distance births for forest-level radius edges;
  - attach each forest node to the corresponding reasoning-step simplex payload
    by original selected position.
- [x] Modify `scripts/build_toricgt_pages.py` so the branching-reasoning public
  page copies `embedding_fot_report/embedding_fot_trace.json` when present and
  includes a compact `forest` block in `branching_reasoning_interactive.json`.
- [x] Add a native Canvas 3D FoT section to the public branching-reasoning page:
  - controls: FoT budget level, radius, yaw, pitch, zoom, expansion/correction/
    consensus toggles;
  - render all parallel trees with visible branching and consensus;
  - clicking/hovering a FoT node reveals tree id, depth, branch action, policy
    probability, NLL, activation, value, correction norm, and source position;
  - clicking a FoT node also selects its local reasoning-step filtered
    simplicial complex in the existing step viewer.
- [x] Regenerate the best-checkpoint page artifacts from an existing checkpoint
  output bundle.
- [x] Capture screenshots of the repaired FoT forest and selected local complex.
- [x] Verify the generated JSON is compact enough for low-resource browsers and
  that no browser-side expensive algebra or topology computation is introduced.

## Completed Verification

- Regenerated `outputs/github_pages_best_checkpoint_long_manual_20260621T182232Z/embedding_fot_report/embedding_fot_trace.json`
  from the real checkpoint FoT head and hidden-state embedding payload.
- Rebuilt `docs/page_interactive/branching_reasoning/index.html` and
  `docs/page_interactive/branching_reasoning/branching_reasoning_interactive.json`.
- Verified the public compact forest contains 61 nodes, 112 edges, 4 parallel
  trees, branching factor 3, and edge-kind counts
  `{"expansion": 56, "self_correction": 16, "consensus": 40}`.
- Fixed the compact exporter so integer zero values are preserved; tree `0`
  renders as `T0`, not `T-1`.
- Captured the browser screenshot
  `outputs/fot_forest_repair/branching_reasoning_forest_page_fixed.png`.
- Ran `git diff --check` and Python compile checks for
  `scripts/render_embedding_fot_report.py` and `scripts/build_toricgt_pages.py`.

## Expected Result

The trajectory section will no longer imply that FoT is a single chain.  It will
show:

- multiple parallel thought trees;
- branch expansion through a budget slider;
- self-correction edges;
- consensus edges;
- per-node links to local filtered simplicial complexes with radius and decoding
  sliders;
- the original full trajectory and analogy views as separate evidence, not as a
  substitute for the FoT forest.
