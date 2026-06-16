# Branching Visualization PH/Analogy Completion Plan

This implementation pass must be completed without starting training.  The goal is to make the branch/merge graph-of-thought visualization more rigorous as an analogy audit: longer trajectories, clearer simplex-tree map evidence, explicit vectorized persistent-homology feature panels, and token metadata available wherever token or reasoning vertices are shown.

## Checklist

- [x] Add explicit vectorized PH feature-family summary payloads for landscape, persistence image, silhouette, and entropy vectors, including per-dimension source/memory norms and cosine similarities.
- [x] Render a compact PH feature-family table in the HTML so the evidence behind the analogy decision is visible without opening the raw payload.
- [x] Add a map-confidence summary to the payload and visual table, separating vertex-map distance quality from simplex-image validity.
- [x] Make analogy-map labels less ambiguous by naming source/memory simplex-tree objects and nearest-neighbor map arrows consistently in captions and tables.
- [x] Add fixture-size metadata and tests for a much longer branch/merge graph-of-thought trajectory with longer side branches and more merge behavior.
- [x] Strengthen tests for token hover/click metadata fields in every token-bearing visualization payload.
- [x] Update screenshot assertions to require the new PH feature-family table and map-confidence summary.
- [x] Update README/docs with the new PH and map-confidence semantics.
- [x] Regenerate a larger long-branch report and screenshots.
- [x] Visually inspect the screenshots and update the local screenshot review note.
- [x] Refresh `outputs/index.html`.
- [x] Run the focused and full project test suites.

## Acceptance Criteria

- The analogy section shows all vectorized PH families as first-class evidence, not only an aggregate cosine.
- The analogy section separately reports simplex-map validity, vertex-map distance quality, and PH similarity.
- The generated long fixture has strictly more reasoning steps and graph-of-thought DAG edges than the previous 492-step report.
- Token metadata includes text, type, NLL, logprob, entropy, rank, source step, decode order, and global token id in the payload and clicked-detail path.
- Screenshot assertions pass with zero failures.
- `conda run -n tokengt env PYTHONPATH=src:. pytest -q tests` passes.

## Completion Notes

- Larger regenerated report: `outputs/branching_reasoning_trajectory_longer_ph_20260616T154110Z`, symlinked by `outputs/latest_branching_reasoning_trajectory_report`.
- Report scale: `566` reasoning steps and `672` graph-of-thought DAG edges.
- Screenshot bundle: `outputs/latest_branching_reasoning_trajectory_report/html_screenshots`.
- Screenshot assertions: 12 checked, 0 failed.
- Screenshot review note: `outputs/latest_branching_reasoning_trajectory_report/SCREENSHOT_REVIEW.md`.
- Output index: `outputs/index.html` refreshed.
- Focused tests: `11 passed`.
- Full project tests: `291 passed, 2 skipped`.
