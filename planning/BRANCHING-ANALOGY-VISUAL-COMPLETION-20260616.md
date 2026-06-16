# Branching Analogy Visual Completion Plan - 2026-06-16

This is the active completion checklist for the second branch/merge reasoning
trajectory visualization pass.  The goal is to make the report useful as an
audit artifact rather than merely a dense 3D rendering.

Non-negotiable rendering rule: visible one-dimensional simplex edges must not
be capped or silently dropped.  Dense views may use radius sliders, opacity,
default-off two-simplex surfaces, legends, diagnostics, and scrollable tables,
but all active one-dimensional simplex edges sent to a visible complex plot must
be present.

## Checklist

- [x] Inspect current visualization generator, render script, tests, and
  rendered outputs.
- [x] Write this live checklist before implementation and update it after each
  implementation pass.
- [x] Make the CLI long-trajectory defaults substantially longer and branchier
  than the previous fixture.
- [x] Add a dedicated analogical correspondence table with source vertex,
  target memory vertex, source/memory labels, source/memory levels, NLL values,
  and original-embedding nearest-neighbor distance.
- [x] Add a vectorized PH similarity bar chart and a source-minus-memory
  difference panel for persistence landscapes, silhouettes, entropy vectors,
  Betti curves, and persistence images.
- [x] Add bottleneck/Wasserstein-style PH distance metrics when GUDHI exposes
  them locally; otherwise add deterministic finite-diagram distance summaries
  without pretending they are exact GUDHI bottleneck/Wasserstein.
- [x] Add click handlers for analogy source/memory vertices that populate a
  detail panel with reasoning-step metadata.
- [x] Add a selected analogy vertex-map detail panel and make the map arrows
  visually inspectable without relying on labels in the 3D scene.
- [x] Keep token hover/click metadata available for every token-bearing
  simplicial view.
- [x] Regenerate the long report and screenshots.
- [x] Inspect representative screenshot slices and iterate if the report is
  visually misleading or broken.
- [x] Run focused visualization tests.
- [x] Run the full test suite.
- [x] Update README with the new panels, defaults, and generated output paths.

## Acceptance Criteria

- The generated report uses at least 18 reasoning levels, 8 branch lanes, and
  side branches of length at least 5 by default.
- The long generated report includes hundreds of reasoning steps, visible
  branch/merge structure, and an emitted analogy tier when the source and memory
  structures are similar.
- The source and memory simplex-tree payloads report exact edge counts equal to
  rendered edge counts.
- The report contains both a 3D analogy map and tabular/2D diagnostics that make
  the map and vectorized PH comparisons readable.
- Screenshot generation completes without browser errors.
- Focused and full tests pass.

## Final Output Record

Generated report:

- `outputs/latest_branching_reasoning_trajectory_report/branching_reasoning_trajectory.html`
- `outputs/latest_branching_reasoning_trajectory_report/branching_reasoning_payload.json`
- `outputs/latest_branching_reasoning_trajectory_report/index.html`

Generated screenshots:

- `outputs/latest_branching_reasoning_trajectory_report/html_screenshots/contact_sheet.png`
- `outputs/latest_branching_reasoning_trajectory_report/html_screenshots/screenshots/branching_reasoning_trajectory__slice_00.png`
- `outputs/latest_branching_reasoning_trajectory_report/html_screenshots/screenshots/branching_reasoning_trajectory__slice_01.png`
- `outputs/latest_branching_reasoning_trajectory_report/html_screenshots/screenshots/branching_reasoning_trajectory__slice_02.png`
- `outputs/latest_branching_reasoning_trajectory_report/html_screenshots/screenshots/branching_reasoning_trajectory__slice_03.png`
- `outputs/latest_branching_reasoning_trajectory_report/html_screenshots/screenshots/branching_reasoning_trajectory__slice_04.png`
- `outputs/latest_branching_reasoning_trajectory_report/html_screenshots/screenshots/branching_reasoning_trajectory__slice_05.png`
- `outputs/latest_branching_reasoning_trajectory_report/html_screenshots/screenshots/branching_reasoning_trajectory__slice_06.png`
- `outputs/latest_branching_reasoning_trajectory_report/html_screenshots/screenshots/branching_reasoning_trajectory__slice_07.png`
- `outputs/latest_branching_reasoning_trajectory_report/html_screenshots/screenshots/branching_reasoning_trajectory__slice_08.png`
- `outputs/latest_branching_reasoning_trajectory_report/html_screenshots/screenshots/branching_reasoning_trajectory__slice_09.png`
- `outputs/latest_branching_reasoning_trajectory_report/html_screenshots/screenshots/branching_reasoning_trajectory__slice_10.png`
- `outputs/latest_branching_reasoning_trajectory_report/html_screenshots/screenshots/branching_reasoning_trajectory__slice_11.png`

Final fixture:

- reasoning nodes: 218
- DAG edges: 264
- analogy status: `weak_analogy`
- analogy confidence: 0.7983
- full simplex-map valid fraction: 0.9880
- vectorized PH mean: 0.9437
- vectorized PH gate score: 0.9650
- source simplex-tree edges: 16,557 exact and 16,557 rendered
- memory simplex-tree edges: 16,534 exact and 16,534 rendered
- exact GUDHI H0 bottleneck: 0.2074
- exact GUDHI H1 bottleneck: 0.2190
- GUDHI H0 Wasserstein: 2.7512
- GUDHI H1 Wasserstein: 3.0915

Validation:

- Focused visualization tests: `4 passed`
- Full test suite: `282 passed, 2 skipped, 2 warnings`
