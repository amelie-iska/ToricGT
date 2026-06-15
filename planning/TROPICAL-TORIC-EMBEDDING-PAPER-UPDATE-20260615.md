# Tropical Ring Attention Embedded In Toric Geometry Paper Update Plan

Date: 2026-06-15

This plan tracks the paper-update objective requested after the analogical simplex-tree visualization work.  The instruction for this run is to be methodical: finish the current visualization correction cleanly, then immediately integrate the tropical-to-toric embedding theory into the two paper TeX files.

## Current Item: Analogical Simplex-Tree Visualization Correction

- [x] Stop interrupted background analysis process so no stale render continues consuming CPU.
- [x] Replace fake intra-record analogical maps with gated cross-record analogical retrieval.
- [x] Add two-parameter reasoning-level/radius sliders to reasoning-step simplex-tree pages.
- [x] Compare analogies in actual embedding space, using PCA only for visualization.
- [x] Add GUDHI vectorized PH signature gates and full filtered-complex simplicial-map gates.
- [x] Add a continuous embedding-space GFlowNet trajectory-balance helper and tests.
- [x] Generate corrected screenshots showing the new two-slider simplex-tree pages.
- [x] Generate an accepted-analogy screenshot after lenient thresholds, or explicitly record why the fixture still emits no analogy.

## Current Item: Tropical Varieties Embedded Into Toric Varieties

- [x] Read `./assets/toricgt_neurips_condensed.tex` in full.
- [x] Read `./assets/toricgt_toric_bgg_rewrite_amelie_schreiber.tex` in full.
- [x] Extract and inspect abstracts for every paper-like source in `./assets`.
- [x] Select the most relevant local sources for toric embeddings, toric algebras, vector bundles, sheaves, intersection theory, ReLU/tropical Transformer geometry, and ToricGT's BGG supervision.
- [x] Research Maclagan's work and adjacent work on tropical compactifications, embeddings of tropical varieties into toric varieties, and schön/tropical intersection methods.
- [x] Review the requested sources:
  - `https://arxiv.org/abs/2405.03505`
  - `https://arxiv.org/abs/2009.03030`
  - `https://arxiv.org/abs/1710.10651`
- [x] Review relevant toric algebra material from Miller-Sturmfels, especially semigroup algebras, toric ideals, lattice polytopes, monomial ideals, free resolutions, and multigraded modules.
- [x] Review relevant toric geometry material on fans, orbit stratifications, toric Chow/intersection theory, Cartier divisors, vector bundles, and equivariant sheaves.
- [x] Write a rigorous construction embedding ToricGT tropical ring attention into a toric variety:
  - affine candidate exponent set;
  - Newton polytope and lifted Newton polytope;
  - normal fan and associated toric variety;
  - active tropical face as orbit/stratum data;
  - tropicalization/initial degeneration language;
  - compactification/tropical compactification conditions when applicable;
  - trainable metrics and losses derived from toric divisors, Chow classes, vector bundles, sheaves, and free resolutions.
- [x] Add definitions, equations, theorem/proposition statements, proof sketches, and reproducible implementation details to both TeX papers.
- [x] Ensure terminology uses "one-dimensional cones" rather than "rays" where requested.
- [x] Compile or at least syntax-check the modified TeX where feasible.
- [x] README/docs review: no separate README edit needed for this request because the requested user-facing methodology update is in the two paper TeX files and this planning note.

## Research Notes

- Maclagan-centered theory:
  - Maclagan's embedded tropical geometry viewpoint treats a subvariety of an algebraic torus through initial degenerations, Gröbner fans, and closures in toric varieties.  The relevant ToricGT translation is: a rationalized tropical attention head determines finite Laurent polynomial or ideal data in a torus; the active max-plus candidates are the support of an initial form after a fixed max/min sign convention; a fan refining the finite active/Gröbner decomposition gives the ambient toric variety that organizes orbit strata and compactification data.
  - Tropical compactification gives the clean toric control statement: for \(Y\subset T\), closure \(\overline Y\subset X_\Sigma\) is a tropical compactification when it is proper and the multiplication map \(T\times\overline Y\to X_\Sigma\) is flat and surjective.  The finite training version should not overclaim this globally; it should construct audit compactifications for finite certificate families and mark whether the exact hypotheses are verified.
  - Maclagan-Rincón tropical ideals give an intrinsic scheme-theoretic route: tropical ideals define subschemes of tropical toric varieties, have finite polyhedral varieties, Hilbert polynomials, and balanced top-dimensional parts.  This justifies balance, Hilbert/Betti, Chow-class, and specialization/projection metrics when the sidecar ideal satisfies the tropical-ideal checks.
  - Fulton-Sturmfels and Katz identify toric/tropical intersection data with Minkowski weights and Chow classes.  In ToricGT terms, balanced active fans and tropical cycles can be audited as Minkowski weights on the toric fan, and divisor bends across codimension-one cones give intersection-style diagnostics.
- Requested arXiv sources:
  - `2405.03505` Khan-Maclagan: tropical vector bundles over tropical varieties; over tropical toric varieties these tropicalize toric vector bundles and are described by valuated matroids plus flats.  Useful for replacing scalar active-face metrics by vector-bundle and sheaf diagnostics.
  - `2009.03030` Jun-Mincheva-Tolliver: vector bundles on tropical schemes, semiring algebra background, relation to topological and monoid-scheme vector bundles.  Useful for the optional semiring/tropical-scheme sidecar rather than only classical toric sidecars.
  - `1710.10651` Améndola--Kohn--Lamboglia--Maclagan--Smith--Sommars--Tripoli--Zajaczkowska: Macaulay2 Tropical package wrapping Gfan/Polymake with multiplicities and min/max convention support.  This is the right reproducibility target for exact tropical fan and multiplicity computations.
  - Additional sources reviewed: Maclagan's introduction to tropical algebraic geometry (`1207.1925`), Maclagan-Rincón tropical ideals (`1609.03838`) and balanced varieties (`2009.14557`), Schock on quasilinear tropical compactifications (`2112.02062`), Kaveh-Manon on toric vector bundles as tropical objects (`2304.11211`), Katz on tropical intersection theory from toric varieties (`0907.2488`), Fulton-Sturmfels on Chow/Minkowski weights, and Miller-Sturmfels combinatorial commutative algebra.
- Local assets abstract survey:
  - Most relevant local PDFs: Fu on toric geometry of ReLU neural networks, Zhang--Naitzat--Lim and Brandenburg--Loho--Montúfar on tropical neural-network geometry, Hashemi et al. on Tropical Attention, TokenGT, Ring Attention, GraphCG, continuous GFlowNets, Schreiber multiparameter persistence, Mohamed--Hirani--Samtaney DEC, Fulton--Sturmfels intersection theory, and Miller--Sturmfels combinatorial commutative algebra.
  - The Miller-Sturmfels toric algebra sections supply semigroup rings, toric ideals, quotient/Cox constructions, normal fans of lattice polytopes, irrelevant ideals, and toric spectrum.  The monomial-ideal chapter supplies the two-variable staircase representation requested for grid modules and Hilbert/Betti diagnostics.
- Paper integration targets:
  - Long paper: add a dedicated subsection after toric auxiliary losses and before Toric BGG, because this is the missing bridge between tropical ring attention and the existing BGG/sheaf/CAS material.
  - Condensed paper: add a compact paragraph after the toric diagnostic theorem and add bibliography entries.
  - Implementation stance to state explicitly: Transformers are not toric in general; ToricGT constructs rational finite toric sidecars from tropical attention/probe data and applies toric geometry to those sidecars only.
- Validation notes:
  - Citation/bibliography consistency passes for both TeX files: no missing `\cite{...}` keys and no duplicate `\bibitem{...}` keys.
  - Basic document structure passes for both TeX files: one `\begin{document}` and one `\end{document}`.
  - The local TeX compile attempt with `lualatex -draftmode` fails before reaching document content because the environment is missing `fontspec.sty` and `luaotfload-main`.  This is a local TeX installation issue, not a detected syntax error in the edited sections.
  - The two TeX files now have no standalone `ray` or `rays` terminology under an `rg` word-boundary scan; relevant wording uses `one-dimensional cones`.
