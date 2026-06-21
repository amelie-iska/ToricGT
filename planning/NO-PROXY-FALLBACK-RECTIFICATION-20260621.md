# No-Proxy / No-Fallback Rectification Plan

Date: 2026-06-21

Scope: audit the public GitHub Pages artifacts, with emphasis on `docs/page_interactive/toric_embedding/index.html`, and the generator paths that create those pages.

## Operating Rule

Public scientific pages must not substitute proxy, fallback, demo, smoke, or stand-in data for exact generated analysis artifacts. If an exact artifact was not generated, the page should either:

1. omit that panel, or
2. state that the exact artifact was not generated and name the missing source file or certificate.

It should not replace the missing object with a toy object, deterministic demo, smoke-run artifact, or visually nicer stand-in.

## Findings

### 1. Rich Miller-Sturmfels demo sidecar

Files:

- `src/toricgt/toric_embedding_visualization.py`
- `scripts/render_toric_embedding_report.py`

Problem:

- `rich_staircase_demo_record()` creates deterministic monomial data with several generators.
- `scripts/render_toric_embedding_report.py --include-rich-staircase-demo` appends this sidecar to reports.
- It is labeled as a demo, but it still creates a path by which a public page can show non-checkpoint-derived staircase geometry.

Required fix:

- Remove the demo sidecar path from the public report flow.
- Keep Miller-Sturmfels staircases only when they are computed from exact sidecar exponent points.
- If a richer staircase is desired, generate more exact sidecar records from model embeddings until the product-order minimal antichain is nontrivial; do not fabricate a demo staircase.

### 2. Smoke-output default report sources

File:

- `scripts/build_toricgt_pages.py`

Problem:

- `IMAGE_SOURCES` and `INTERACTIVE_REPORTS` contain defaults under `outputs/smoke_oai_full_iteration_analysis_2/...`.
- If an explicit analysis output directory is not supplied, public Pages can silently copy smoke-run artifacts.

Required fix:

- Stop using smoke outputs as default public sources.
- Require an explicit analysis output directory for generated analysis reports.
- Static assets under `assets/` are fine; analysis-derived screenshots and reports must come from the selected analysis directory.

### 3. Toric embedding public page says compact/omitted/unavailable

Files:

- `docs/page_interactive/toric_embedding/index.html`
- `scripts/build_toricgt_pages.py`

Problem:

- The page states that the large Plotly record is omitted and that compact summaries are used.
- This is not a fake metric, but it reads like a substitute for the exact report.

Required fix:

- Reword the page as an exact summary mirror of the selected analysis artifact.
- Link to the exact source summary JSON and exact sidecar JSON when available.
- Add a collapse diagnostic explaining why the visible Miller-Sturmfels ideal has only two generators.
- Avoid implying that the compact renderer is a mathematical replacement for missing data.

### 4. Toric embedding page lacks collapse provenance

Files:

- `docs/page_interactive/toric_embedding/index.html`
- `docs/page_interactive/toric_embedding/toric_embedding_interactive.json`
- `scripts/build_toricgt_pages.py`

Problem:

- The public page shows the ideal `<x^6,y^12>` without explaining that the original exponent cloud had eight points and that six were product-order dominated.
- This makes the page look like a toy/fallback even though it is actually the exact minimal ideal for the selected sidecar.

Required fix:

- Include original exponent points in the compact JSON.
- Compute and show the product-order dominance relation for each non-minimal point.
- Add a table that states: original points, minimal generators, dominated points, quotient-basis count, adjacent LCM corners, and exact CAS provenance.

### 5. Generic unavailable panels

Files:

- `src/toricgt/toric_embedding_visualization.py`
- `scripts/build_toricgt_pages.py`

Problem:

- The full report intentionally renders missing exact certificates as "Unavailable" rather than inventing a surrogate. This is mathematically safer than a fallback.
- The public site should still avoid presenting unavailable rows as if they were expected content.

Required fix:

- For public Pages, only list panels whose exact data exists.
- If a missing exact certificate is important, show a short "Exact certificate not generated" diagnostic with the required file/key, not a stand-in visualization.

### 6. ConvexTok page has true byte fallback terminology

File:

- `scripts/build_toricgt_pages.py`

Problem:

- ConvexTok includes free byte fallback edges. This is legitimate tokenizer terminology, not a proxy artifact.

Required fix:

- Do not remove this from ConvexTok; it is part of the exact tokenization DAG method.
- Ensure wording makes clear that these are exact byte edges in the tokenizer graph, not a visualization substitute.

## Immediate Implementation Checklist

- [x] Remove rich staircase demo generation from public report tooling.
- [x] Remove smoke analysis defaults from public Pages builder.
- [x] Preserve only explicit analysis-output-derived report sources.
- [x] Regenerate the toric CAS sidecar with an exact checkpoint-derived product-order antichain so the Miller-Sturmfels staircase is nontrivial.
- [x] Make the toric ideal complete for that selected exact antichain: Sage normal fan and Macaulay2 toric ideal both certify the subconfiguration.
- [x] Extend toric embedding native JSON with exact exponent points, selection provenance, toric ideal relation counts, complete relation polynomials, Miller-Sturmfels generators, quotient basis, adjacent LCM layer, and precomputed resolution rows.
- [x] Add exact staircase/resolution and toric-ideal panels to the toric embedding page.
- [x] Rebuild docs from the selected analysis output.
- [x] Verify no public toric embedding page text contains proxy, fallback, stand-in, demo-only, smoke, or placeholder language except legitimate ConvexTok byte fallback terminology outside this page.
- [ ] Commit and push.

## Verification Notes

- Public toric embedding page now embeds the exact native JSON payload inline and also writes it to `toric_embedding_interactive.json`; the inline payload avoids a local-file `fetch()` dependency and keeps weak browsers from doing unnecessary work.
- Browser work is limited to drawing one SVG staircase and hover/click handlers over precomputed lattice data. Sage, Macaulay2, antichain selection, toric ideal generation, and Miller-Sturmfels resolution rows are all computed before page generation.
- The selected exact artifact uses 435 unique checkpoint-derived exponent points, selects a 7-point product-order antichain, computes 791 quotient-basis monomials, 6 adjacent LCM corners, a 6-row Miller-Sturmfels two-step resolution table, and a complete 96-generator Macaulay2 toric ideal for the selected antichain.
- Local Playwright render in the `tokengt` environment passed for the generated page with `ok_count=1` and `error_count=0`; temporary screenshot outputs were removed after inspection.
