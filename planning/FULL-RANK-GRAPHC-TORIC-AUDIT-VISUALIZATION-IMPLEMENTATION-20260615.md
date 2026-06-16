# Full-Rank GraphCG, Exact Toric Audits, And Branching Trajectory Visualization Plan

Date: 2026-06-15

This plan tracks the requested implementation pass.  The work is code, config,
analysis, and visualization only.  It must not start or restart training.

## Requested Items

1. Make GraphCG training full rank.
2. Fully implement periodic exact sidecar report generation from bounded
   checkpoint/inference analysis samples.
3. Fully implement exact tropical multiplicity, balancing, and Chow/Minkowski
   audit certificates for the tropical attention hypersurface embedded into
   the toric fan.
4. Fully implement CAS availability preflight checks.
5. Improve the resolution/differential visual grammar.
6. Regenerate all visualizations, including new ones.
7. Generate a long reasoning trajectory with multiple reasoning levels showing
   graph-of-thought branching and merging behavior.
8. Ensure reasoning trajectories, simplicial objects, filtered simplicial
   complexes, and per-step simplex panels have two controls:
   - radius;
   - decoding order for step-level objects, or reasoning level for the full
     trajectory complex.
9. Ensure hover-click on reasoning nodes reveals the corresponding filtered
   simplicial complex for that reasoning step.

## Exactness Policy

- No heuristic replacement may be labeled exact.
- If a CAS package is unavailable, the preflight must fail or mark the exact
  certificate unavailable with a concrete reason.
- Tropical multiplicities must come from integer lattice computations or exact
  CAS output.
- Balance residuals must be computed over exact integer normal vectors and
  exact integer weights.
- Chow/Minkowski panels must be marked certified only when balancing data are
  present and pass the exact balance audit.
- Visualization may use PCA only for display; comparison, retrieval, and
  certificate computations must use the original embedding vectors.

## Implementation Checklist

- [x] Audit the current GraphCG config and code paths.
- [x] Change the active all-phases Parameter-Golf config to full-rank GraphCG
      for `d_model=384`.
- [x] Add validation that full-rank GraphCG means
      `graphcg_num_directions == d_model`.
- [x] Audit current periodic analysis code and add exact toric report rendering
      hooks.
- [x] Add bounded periodic report controls for CAS rendering and screenshots.
- [x] Add report paths to periodic `analysis_status.json` and artifact
      inventories.
- [x] Implement exact tropical hypersurface multiplicity certificates for
      pairwise facets from exponent differences.
- [x] Implement exact balancing audits for codimension-two tie vertices where
      the finite support has at least three active monomials.
- [x] Implement Chow/Minkowski certification state from exact balancing data.
- [x] Add sidecar JSON fields for multiplicity, balance, and Chow audit data.
- [x] Add CAS preflight script output for M2, Sage, GUDHI, Normaliz, gfan, and
      relevant M2 packages.
- [x] Improve the toric report resolution diagram with module strips, sparse
      differential heatmaps, and collapsible raw CAS payloads.
- [x] Implement long branching reasoning trajectory data generation from an
      existing checkpoint/fixture-compatible synthetic trajectory.
- [x] Implement HTML trajectory visualization with:
      - full trajectory graph;
      - branch/merge levels;
      - node NLL coloring;
      - hover-click step simplex details;
      - radius slider;
      - reasoning-level slider for full trajectory complex;
      - decoding-order slider for per-step simplex panels;
      - radius-controlled simplices;
      - dotted decode/order arrows that appear with slider progress.
- [x] Add tests for the new exact certificate fields.
- [x] Add tests for generated HTML containing both controls and step panels.
- [x] Regenerate toric embedding reports from an existing sidecar.
- [x] Regenerate branching trajectory reports.
- [x] Generate screenshots for all generated HTML pages.
- [x] Visually inspect at least the contact sheet and main report screenshots.
- [x] Update README/docs with commands and locations.
- [x] Update this plan as items are completed.

## Generated Artifacts

- `outputs/latest_toric_embedding_visual_report/index.html`
- `outputs/latest_toric_embedding_visual_report/records/record_000_toric_embedding.html`
- `outputs/latest_toric_embedding_visual_report/html_screenshots/contact_sheet.png`
- `outputs/latest_branching_reasoning_trajectory_report/index.html`
- `outputs/latest_branching_reasoning_trajectory_report/branching_reasoning_trajectory.html`
- `outputs/latest_branching_reasoning_trajectory_report/html_screenshots/contact_sheet.png`

## Verification

- `conda run -n tokengt env PYTHONPATH=src pytest -q tests/test_branching_reasoning_visualization.py tests/test_tropical_toric_certificates.py tests/test_toric_embedding_visualization.py tests/test_validate_cas_certificates.py tests/test_embedding_cas_sidecar.py tests/test_tokengt_geometry_analysis.py`
- Result: `12 passed`.
- `conda run -n tokengt env PYTHONPATH=src:. pytest -q tests`
- Result: `279 passed, 2 skipped, 2 warnings`.

## Expected Output Paths

Generated reports should be written under:

```text
outputs/latest_toric_embedding_visual_report/
outputs/latest_branching_reasoning_trajectory_report/
```

Screenshots should be written under each report's `html_screenshots/`
subdirectory.

## Work Order

1. Audit and patch config-level full-rank GraphCG first.
2. Implement exact certificate computation in the sidecar layer.
3. Update report rendering to consume the new certificates.
4. Integrate report rendering into periodic analysis.
5. Add the branching trajectory HTML renderer and fixture generator.
6. Run focused tests after each major item.
7. Regenerate outputs and screenshots.
8. Update docs.
