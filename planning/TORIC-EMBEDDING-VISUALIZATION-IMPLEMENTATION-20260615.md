# Toric Embedding Visualization Implementation Plan

Date: 2026-06-15

This plan tracks the implementation of visualizations for the tropical-ring-attention-to-toric-geometry material added to the ToricGT papers.  The work must be exact where exact data are claimed.  The renderer must consume certified sidecar records produced by SageMath/Macaulay2 or exact finite closed-form certificates.  It must not silently substitute heuristic proxies when a CAS certificate is unavailable.

## Scope

Implement a visualization/report pipeline for the new material:

1. tropical attention as an initial degeneration;
2. Newton polytope and lifted Newton polytope;
3. normal fan and one-dimensional cones;
4. toric orbit-stratum incidence;
5. tropical hypersurface / tie locus / unique-max chamber decomposition;
6. Cartier divisor bends and balance/Chow audit state;
7. toric ideal relations and free-resolution diagrams;
8. Miller-Sturmfels staircase diagrams for two-variable monomial ideals;
9. Klyachko vector-bundle filtrations;
10. Cox/sheaf/free-resolution/Ext/Tor/derived-map summaries;
11. exact CAS provenance, convention, and failure-mode visibility.

This work is analysis/report generation only.  It must not start training.

## Required Implementation Items

- [x] Add a reusable visualization module under `src/toricgt/`.
- [x] Add a CLI entrypoint under `scripts/` that renders a complete HTML report from one or more exact embedding CAS sidecar records.
- [x] The CLI must accept:
  - one or more `*_cas_sidecar.json` files;
  - a directory containing sidecar records;
  - optional output directory;
  - optional exact Macaulay2 vector-bundle certificate JSON;
  - an option to build the exact Macaulay2 `ToricVectorBundles` certificate when available.
- [x] The report must include an `index.html`.
- [x] The report must include one page per sidecar record.
- [x] The report must include a machine-readable `manifest.json`.
- [x] Every report page must state provenance and convention metadata.
- [x] Use dark-mode styling consistent with the existing ToricGT analysis reports.
- [x] Use Plotly for interactive plots where useful, embedded directly so screenshots work offline.
- [x] Include all required visualization panels:
  - [x] exponent/Newton polytope plot;
  - [x] lifted Newton polytope plot;
  - [x] active chamber / initial degeneration heatmap;
  - [x] normal fan with one-dimensional cones;
  - [x] orbit-stratum incidence graph;
  - [x] divisor/balance/Chow audit panel;
  - [x] toric ideal relation graph;
  - [x] free-resolution diagram;
  - [x] Miller-Sturmfels staircase and stacked staircase layers;
  - [x] Klyachko filtration diagram;
  - [x] Cox/sheaf/Ext/Tor/derived-map summary.
- [x] If data for a panel are not present, the page must say exactly which certificate field is missing and mark the panel unavailable.  It must not invent a replacement.
- [x] Add tests for the renderer using exact fixture sidecar JSON data.
- [x] Tests must check:
  - [x] HTML files are generated;
  - [x] manifest is generated;
  - [x] all required panel headings are present;
  - [x] no deprecated one-dimensional-cone terminology appears in rendered new report text;
  - [x] unavailable panels are explicit when data are absent;
  - [x] the script works from CLI.
- [x] Generate a real report under `./outputs` using an existing exact sidecar from an old checkpoint/inference bundle.
- [x] Generate screenshots for every generated HTML page using `scripts/render_html_screenshots.py`.
- [x] Review generated screenshot manifest and ensure `error_count == 0`.
- [x] Update `README.md` with the new command, output location, and interpretation.
- [x] Update `planning/TORIC-EMBEDDING.md` with a pointer to the visualization renderer.
- [x] Commit and push the implementation.

## Exactness Rules

1. A Sage normal-fan panel is exact only when the sidecar record has `sage_normal_fan.provenance == "exact_cas/sage"`.
2. A Macaulay2 toric-ideal/free-resolution panel is exact only when the sidecar record has `macaulay2_toric_ideal.provenance == "exact_cas/macaulay2"`.
3. A Klyachko/vector-bundle panel is exact only when it consumes a valid `macaulay2_toric_vector_bundle_certificate` or an explicit finite closed-form Klyachko certificate.
4. A Miller-Sturmfels staircase built from integer exponent generators is an exact finite monomial-ideal visualization of those generators.  It must not be described as an initial ideal unless an initial ideal certificate is present.
5. Balance/Chow metrics require multiplicities and codimension-one adjacency data.  If the sidecar only has a normal fan and no tropical-cycle multiplicities, the report must say that the balance/Chow class is not certified.
6. The max/min convention must be visible:
   - ToricGT attention uses max-plus.
   - CAS tropical geometry often uses min-plus.
   - The bridge uses `in_{-u}(f)` and coefficients `tau^{-b_r}`.

## Implementation Design

### Module

Create `src/toricgt/toric_embedding_visualization.py`.

The module should expose:

```python
render_toric_embedding_report(
    sidecar_records: list[Path],
    output_dir: Path,
    vector_bundle_certificate: Path | None = None,
    build_vector_bundle_certificate: bool = False,
) -> dict
```

Internal helpers should:

- load and validate sidecar record structure;
- extract exponent points and optional biases;
- compute exact finite active-chamber grids from the stored exponent/bias data;
- build Plotly HTML fragments;
- build a Miller-Sturmfels staircase from two-dimensional exponent generators;
- render resolution and relation diagrams using deterministic HTML/SVG;
- load or build a vector-bundle certificate through `Macaulay2TropicalOracle`;
- write `index.html`, per-record HTML pages, JSON summaries, and `manifest.json`.

### Script

Create `scripts/render_toric_embedding_report.py`.

CLI example:

```bash
python scripts/render_toric_embedding_report.py \
  --sidecar-dir outputs/old_checkpoint_full_visual_audit_20260615T165947Z/embedding_cas_sidecar \
  --output-dir outputs/toric_embedding_visual_report_manual \
  --build-vector-bundle-certificate
```

### Generated Output

Expected structure:

```text
outputs/<run>/
  index.html
  manifest.json
  records/
    record_000_toric_embedding.html
    record_000_toric_embedding_summary.json
  vector_bundle/
    toric_vector_bundle_certificate.json
    klyachko_vector_bundle.html
  html_screenshots/
    index.html
    manifest.json
    contact_sheet.png
    screenshots/*.png
```

## Review Checklist

After implementation:

- [x] Run targeted tests.
  - `PYTHONPATH=src /home/iska/miniconda3/envs/tokengt/bin/python -m pytest tests/test_toric_embedding_visualization.py tests/test_render_html_screenshots.py -q`
  - `4 passed`
- [x] Generate the report.
  - `outputs/toric_embedding_visual_report_20260615T193343Z`
  - `outputs/latest_toric_embedding_visual_report`
- [x] Generate screenshots.
  - `outputs/latest_toric_embedding_visual_report/html_screenshots`
- [x] Open or inspect screenshot manifest.
  - Screenshot manifest reports `error_count == 0`.
  - Contact sheet and record page screenshot were visually inspected.
- [x] Record output paths in final response.
- [x] Confirm no training was started.
