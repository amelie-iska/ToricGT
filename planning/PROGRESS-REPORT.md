# ToricGT Progress Report

Date: 2026-06-15
Branch: `oai-toricgt`
Latest pushed implementation commit reviewed here: `bb9808b Add exact toric embedding visualization audits`

This report summarizes the current state of ToricGT after the tropical-to-toric embedding, exact CAS sidecars, GUDHI persistence audits, BGG/Category O probes, vector-bundle/sheaf probes, and Parameter-Golf BPB infrastructure work.  It separates what is implemented and usable now from what still needs validation, tuning, or a stricter integration pass.

## Executive State

ToricGT is now a single tropical/toric project rather than two separate efforts.  Tropical ring attention supplies the max-plus, piecewise-linear computation; toric geometry supplies the ambient algebraic structure used to audit, regularize, visualize, and eventually compress that computation.  The current codebase has working implementations for the main analysis pathways:

- TokenGT-style causal graph projection for byte and graph records;
- hybrid softmax/tropical-ring attention for the OAI/Parameter-Golf byte model;
- low-weight toric geometry losses from step 0;
- GraphCG-style basis disentanglement and analogy-lattice losses;
- embedding-space GFlowNet auxiliary policy;
- trajectory memory with topology, GraphCG, toric, and vectorized persistence signatures;
- GUDHI persistent-homology audits and vectorized PH features;
- Macaulay2/Sage-style CAS sidecars for toric ideals, normal fans, resolutions, and vector-bundle certificates;
- exact tropical-to-toric HTML reports with screenshots, including exact
  closed-form 2D tropical hypersurface multiplicity/balance certificates;
- branching graph-of-thought trajectory HTML reports with full and per-step
  filtered simplicial complexes, dual radius/order sliders, and GUDHI
  vectorized PH comparisons;
- BGG/Category O probes instantiated from the start but with loss pressure toggled off until late phases.

The current full all-phases Parameter-Golf config is `config/train.parameter_golf_all_phases_medium_conservative.yaml`.  It is BPB-first: likelihood remains dominant, all advanced losses are small, and `toric_bgg_loss_weight` is zero until the late `toric_bgg_category_o` phase.  The config evaluates OAI competition BPB every 250 optimizer steps and keeps the FineWeb calibration stream active at `fineweb_calibration.mix_ratio: 0.60`.

The most important caveat: the advanced metrics and reports are now broad, but not every advanced loss has evidence that it improves BPB after export.  The competition rule remains strict: only compact features that improve official or faithful validation BPB under the 16,000,000 byte artifact cap should survive into the submitted artifact.  CAS, exact PH, BGG, vector-bundle, and derived-category reports are primarily training/audit infrastructure unless distilled into a small probe with demonstrated BPB gain.

## Current Worktree State

The pushed branch is current through `bb9808b`.  Local uncommitted leftovers remain outside the pushed implementation:

- `data/README.md` is deleted locally but not staged;
- several downloaded research PDFs/notebooks/images under `assets/` are untracked;
- `external/` is untracked.

These were intentionally not pushed with the visualization implementation.  They should be reviewed separately before any future commit.

## Implemented And Ready

### OAI Parameter-Golf Byte Model Path

The OAI baseline model we have been training is implemented through `scripts/train_parameter_golf_random_order.py` and the Parameter-Golf configs.  The active all-phases config uses:

- byte-level vocabulary with reserved reasoning/memory/analogy tokens;
- tied byte embeddings/output projection where applicable;
- `d_model: 384`, `num_layers: 7`, `num_heads: 6`;
- hybrid attention with upper tropical-ring attention;
- `max_seq_len: 2048` for the current conservative 24 GB VRAM surface;
- `batch_size: 1`, `grad_accum_steps: 32`;
- OAI competition validation every 250 steps;
- score-first evaluation adaptation;
- 6-bit row quantized export target.

Existing preserved Parameter-Golf checkpoints are already below the 16 MB exported artifact cap after compact export:

| Checkpoint | Step | Validation BPB | Counted Export Bytes |
| --- | ---: | ---: | ---: |
| `parameter_golf_oai_4k_bpb1p20848.pt` | 4,000 | 1.20848 | 15,901,889 |
| `parameter_golf_oai_20k_bpb1p16858.pt` | 20,000 | 1.16858 | 15,894,735 |

The raw `.pt` files remain resume/analysis checkpoints and are not the submitted artifact.  The compact exports are what matter for Parameter Golf.

### TokenGT-Style Graph Projection

The byte baseline now has a TokenGT-style graph path rather than only a flat byte stream.  The core idea is:

- each byte/reveal position becomes a graph vertex;
- sequential or graph-derived relations become typed edges;
- causal reveal ranks determine which nodes/edges are visible;
- cyclic or noncausal sources fall back to deterministic content-independent causal-when-possible handling;
- graph-context fusion contributes to logits through a small fusion weight.

The active config has:

- `use_tokengt_causal_graph: true`;
- `use_tokengt_graph_fusion: true`;
- `tokengt_graph_max_nodes: 128`;
- `tokengt_graph_fusion_weight: 0.06`.

This gives the OAI byte model access to graph-structured inductive bias while preserving score-first causality.  The relevant validation entrypoint is `scripts/validate_tokengt_graph_data.py`.

### Tropical Ring Attention

Tropical ring attention is implemented as a computationally structured max-plus attention path.  Mathematically, each tropical head computes

```tex
Y_{ic}=\max_j(S_{ij}+V_{jc})
```

or its blockwise ring-evaluated equivalent.  The blockwise ring schedule is exact in real arithmetic because max is associative and commutative over a partition of keys.  This is the operational tropical component of the model.

Training uses tropical attention in two ways:

1. It directly changes the hidden computation by providing max-plus heads for active predecessor/active face behavior.
2. It emits active-face, margin, and chamber diagnostics used by toric geometry losses and sidecar reports.

### Tropical Attention Embedded Into A Toric Variety

The current construction embeds the tropical ring attention probe into toric geometry by turning affine tropical candidates into a finite monomial/exponent system.

For a hidden state `h`, the toric probe defines affine candidates:

```tex
\ell_r(h)=\langle a_r,h\rangle+b_r .
```

The tropical attention decision is:

```tex
\psi(h)=\max_r \ell_r(h).
```

The exponent vectors `a_r` define a Newton polytope:

```tex
P=\operatorname{Conv}\{a_r\}.
```

The lifted exponents `(a_r,b_r)` define a lifted Newton polytope.  Its normal fan partitions hidden/query space into chambers.  A chamber corresponds to an active monomial or active face.  A tie locus, where two or more candidates are co-active, is the tropical hypersurface associated to the finite monomial system.  The normal fan gives the toric fan `Σ`; the ambient toric variety is:

```tex
X_\Sigma .
```

Equivalently, the monomial map

```tex
t \mapsto [\chi^{a_1}(t):\cdots:\chi^{a_R}(t)]
```

embeds the algebraic torus into projective space; the closure is a toric variety controlled by the same exponent data.  The tropical attention computation is therefore the piecewise-linear shadow of a toric monomial embedding.  In reports and sidecars, this gives:

- Newton and lifted Newton polytopes;
- normal fan and one-dimensional cones;
- orbit-stratum incidence;
- toric ideal relations among monomials;
- initial degeneration / active chamber labels;
- finite support for later divisor, Chow, sheaf, and vector-bundle audits.

The max-plus/min-plus convention is kept explicit.  ToricGT uses max-plus; CAS tropical geometry often uses min-plus.  The bridge is represented by the sign convention in initial forms such as `in_{-u}(f)` and by storing convention metadata in sidecar records.

### Exact CAS Sidecars

The exact sidecar path is implemented for saved hidden embedding payloads:

- `scripts/run_embedding_cas_sidecar.py`;
- `src/toricgt/cas_oracles.py`;
- `src/toricgt/cas_certificates.py`.

The sidecars compute or store exact certificates for:

- normal fans / one-dimensional cones;
- toric ideals;
- Groebner-style relation data;
- free resolutions where available;
- derived algebra summaries;
- Klyachko vector-bundle certificates through Macaulay2 when available.

The report renderer refuses to invent missing exact algebra.  If multiplicities or codimension-one adjacency data are absent, balance/Chow panels are marked unavailable rather than approximated.

### HTML Visualizations And Screenshots

The exact tropical-to-toric report renderer is implemented:

- `src/toricgt/toric_embedding_visualization.py`;
- `scripts/render_toric_embedding_report.py`;
- `tests/test_toric_embedding_visualization.py`.

Generated local report:

- `outputs/latest_toric_embedding_visual_report/index.html`;
- `outputs/latest_toric_embedding_visual_report/records/record_000_toric_embedding.html`;
- `outputs/latest_toric_embedding_visual_report/vector_bundle/klyachko_vector_bundle.html`;
- screenshots under `outputs/latest_toric_embedding_visual_report/html_screenshots/`.

The renderer includes:

- Newton polytope;
- lifted Newton polytope;
- active chamber / initial degeneration heatmap;
- normal fan with one-dimensional cones;
- orbit-stratum incidence;
- balance/Chow availability audit;
- toric ideal relation graph;
- free-resolution diagram;
- Miller-Sturmfels staircase layers;
- Klyachko vector-bundle filtration page;
- Cox/sheaf/Ext/Tor/derived-map summary.

The last focused validation run for this implementation passed:

```text
47 passed, 4 skipped
```

### GUDHI Persistence And Vectorized PH Features

The persistent-homology path is implemented for audits and optional inference outputs:

- `src/toricgt/gudhi_persistence.py`;
- `scripts/run_gudhi_persistence_audit.py`;
- `scripts/evaluate_tokengt_reasoning_geometry_suite.py`;
- `tests/test_gudhi_persistence.py`.

The implemented feature families include:

- persistence landscapes;
- persistence images;
- silhouettes;
- entropy vectors;
- Betti curves;
- lifetime histograms;
- exact finite-field chain audits;
- simplicial map validity;
- two-parameter level/radius persistence summaries.

These features are used in trajectory-memory signatures and analogical retrieval diagnostics.  Exact GUDHI computations are not differentiable by themselves; the differentiable training side uses vectorized summaries and torch-side surrogate losses after exact audits confirm the finite objects.

### BGG / Category O Infrastructure

The Toric BGG path is implemented as probes and certificates but intentionally off as a loss until late training:

- `use_toric_bgg: true`;
- `toric_bgg_loss_weight: 0.0` during early and middle phases;
- late phase `toric_bgg_category_o` activates a small weight: `0.000025`.

The intended finite targets are:

- sign-vector/poset skeletons;
- standard filtration leakage;
- `d^2=0` boundary consistency;
- Koszul linearity;
- Gale-dual signatures;
- trajectory signatures for memory retrieval.

This is the right default.  BGG probes should be present for metrics and logging from step 0, but they should not pressure the byte model until the base BPB slope is stable.

### GraphCG, Analogical Lattice, And GFlowNet

GraphCG-style basis disentanglement is implemented as an auxiliary coordinate system for hidden trajectories.  The active config uses:

- `use_graphcg: true`;
- `graphcg_require_full_rank: true`;
- `graphcg_num_directions: 384`;
- `graphcg_min_directions: 384`;
- `graphcg_max_directions: 384`.

This is now full rank for `d_model=384` in
`config/train.parameter_golf_all_phases_medium_conservative.yaml`.  The trainer
raises during startup if full-rank mode is requested and the direction count no
longer equals `d_model`.

The analogy lattice is active:

- `use_analogy_lattice: true`;
- directed topology enabled;
- HDBSCAN-style stable-neighborhood gating enabled;
- step topology enabled.

The compact embedding-space GFlowNet is active:

- `use_gflownet_policy: true`;
- `gflownet_num_actions: 16`;
- low early loss weights in the all-phases config.

The GFlowNet does not replace the byte likelihood.  It samples or regularizes graph-of-thought/embedding-space trajectories and contributes a small trajectory-balance-style pressure.

## What Needs Updating, Implementing, Or Improving

### 1. Clarify Live Training Versus Analysis-Only Paths

There are now many scripts and metrics.  The README is improved, but the operational split still needs to be sharper:

- competition training path;
- graph research training path;
- periodic analysis watcher;
- exact sidecar report generation;
- inference-only visualization outputs;
- local fixture/smoke-test path.

Recommended update: add a single `docs/OPERATIONAL_RUNBOOK.md` with exact commands for “train”, “watch”, “analyze old checkpoint”, “render reports”, “evaluate OAI BPB”, and “export artifact”.

### 2. Make BPB Authority Explicit Everywhere

The repo has both FineWeb scaffold metrics and OAI competition metrics.  The report should consistently state:

- `oai_competition/*` is the native competition-style decoded-byte BPB path;
- `fineweb/*` is scaffold/calibration unless the exact official evaluator is being run;
- graph-token SP1024 bits/token is not official byte BPB.

Recommended update: enforce this naming in W&B summaries and report manifests so no plot can be mistaken for official BPB.

### 3. Full-Rank GraphCG Status

Full-rank GraphCG is now configured from step 0 for the active conservative
all-phases run.  The main remaining validation is a BPB/VRAM smoke test before
using it in a long run; the code path itself is guarded so a rank mismatch is a
startup error rather than a silent partial-rank run.

### 4. Strengthen GFlowNet Correctness Validation

The code has an embedding-space GFlowNet policy and continuous-GFlowNet planning, but we should still add stronger tests for:

- trajectory-balance algebra on toy continuous/discrete hybrid trajectories;
- prefix visibility of actions in byte scoring;
- no future-token leakage through graph-of-thought state construction;
- reward normalization and entropy behavior under repeated trajectories;
- memory retrieval reward not dominating likelihood.

Recommended update: add a small deterministic GFlowNet toy environment test and a score-first audit for graph-of-thought action features.

### 5. Periodic Exact Sidecar Reports

`scripts/watch_training_analysis.py` now has a bounded exact tropical-to-toric
report hook.  When enabled, the TokenGT geometry analysis emits embedding
payloads, the watcher attempts a small Sage/Macaulay2 sidecar, renders
`render_toric_embedding_report.py`, screenshots the HTML bundle, and writes a
`toric_embedding_report/status.json` entry that is linked from the analysis
index and synopsis.  CAS failures are recorded as unavailable rather than
replaced by a surrogate.

### 6. Multiplicity / Chow / Balance Status

The sidecar path now stores exact closed-form certificates for two-dimensional
tropical attention hypersurface supports.  Pairwise active walls get integer
lattice multiplicities from `gcd(a_i-a_j)`, multiway vertices get exact
balancing stars, and balanced integer weights certify the finite
Minkowski-weight/Chow audit state.  Higher-dimensional tropical-cycle
certificates still require a future CAS-backed extension rather than a
closed-form 2D routine.

### 7. CAS Availability Checks

`scripts/validate_cas_certificates.py --preflight` is now the standard
toolchain preflight.  It reports Sage, Macaulay2, GUDHI, and toric/tropical
tool availability including gfan, Singular, Normaliz, 4ti2, LattE, lrslib,
TOPCOM, polymake, and nauty, with strict `--require-*` flags for exact jobs.

### 8. Resolution Visual Grammar

The toric embedding report now renders free resolutions as horizontal module
strips, adds an exact raw differential-token incidence heatmap, and moves raw
Macaulay2 resolution and differential payloads into collapsible sections.  This
keeps the exact strings available without making the report unreadable.

### 9. Close The Loop From Advanced Metrics To BPB Decisions

We have controller reports and structural recapture diagnostics, but the policy needs more evidence.  Required ablations:

- base byte model without advanced losses;
- toric geometry only;
- GraphCG only;
- GFlowNet only;
- graph fusion only;
- trajectory memory only;
- topology/vectorized PH only;
- combined all-phases;
- all with identical data order, seed, and evaluation windows.

The right metric is not whether an auxiliary metric looks good; it is whether validation BPB improves after export or whether a declared reasoning metric improves without BPB regression.

## How The Techniques Are Used For Training

### Tropical Ring Attention

Tropical heads provide active-max computations.  They help the model express:

- shortest-path-like recurrence;
- dynamic-programming selection;
- active predecessor tracing;
- sparse decision regions;
- max-margin chamber decisions.

During training, tropical outputs and margins can be regularized through:

- active-face entropy floors;
- top-two margin losses;
- active chamber consistency;
- provenance stability under perturbations.

### Toric Geometry

Toric geometry organizes tropical decisions into a finite algebraic object.  The training-side toric losses currently include:

- fan margin loss;
- bend consistency;
- binomial relation consistency;
- moment-map target loss;
- Coxeter/braid losses;
- toric phase leaf residual;
- toric entropy floor.

For BPB training these weights are intentionally tiny.  They are meant to shape hidden geometry without overwhelming next-byte likelihood.

### CAS-Backed Algebra

Macaulay2/Sage/GUDHI sidecars are not in the gradient path by default.  Their roles:

- generate exact labels for audits;
- validate whether differentiable surrogates correspond to real algebraic/topological objects;
- provide low-dimensional tensors for later supervised probes;
- produce visual and machine-readable evidence for checkpoint triage.

The long-term training pattern should be:

1. exact CAS audit on small sidecar samples;
2. train low-rank differentiable probes to match exact finite certificates;
3. use probes as auxiliary losses only after they are predictive and BPB-safe;
4. distill only compact successful probes into the artifact.

### Persistent Homology

Persistent homology is used to characterize hidden reasoning trajectories and analogical memory candidates.  It supplies:

- connected-component persistence;
- loop/cycle persistence;
- directed asymmetry and cycle flux;
- vectorized PH signatures for retrieval;
- simplicial-map validity for analogy.

The training-safe forms are vectorized PH summaries and differentiable surrogates.  Exact GUDHI simplex-tree computations remain audit/certificate computations.

### BGG / Category O

BGG supervision is a late-phase structural discipline.  It is intended to make hidden trajectories behave like finite complexes:

- local moves resemble differentials;
- `d^2=0` is approximately satisfied;
- standard filtration leakage is small;
- Gale-dual pairs produce compatible signatures;
- memory retrieval can match homological skeletons.

The current config correctly keeps this instantiated but off as a loss until late training.

## Expected Improvements By Area

### 1. Reasoning And Memory Retrieval

The memory head should improve when helper trajectories are retrieved by structure, not only by embedding cosine similarity.  The current trajectory signature includes:

- GraphCG coordinates;
- toric active-face / moment features;
- vectorized PH features;
- topology and persistence summaries;
- trajectory quality score;
- BGG/Koszul signatures when available.

This can improve reasoning because a retrieved memory can match the shape of the reasoning path: same active chambers, same persistence barcode shape, same simplex-map validity, similar resolution skeleton, and similar graph-of-thought branching pattern.

For training, the memory head receives distillation-style signals from those signatures.  For inference, optional visual sidecars can show whether retrieved helpers are topologically and torically compatible.

### 2. Analogical Reasoning

Analogical reasoning is represented as maps between reasoning structures:

- maps between reasoning-step simplex trees;
- maps between full trajectory filtered complexes;
- vectorized PH similarity between source and target;
- toric/chamber consistency of active-face paths;
- GraphCG relation arrows aligned in basis coordinates.

The rule should be strict conceptually: if there is no reasonable simplicial map, no high PH signature similarity, and no good full-trajectory map, the system should not call it a strong analogy.  Thresholds can be lenient for exploration, but the labels should remain honest.

GraphCG helps here by giving relation arrows a reusable coordinate chart.  GFlowNets help by sampling alternative graph-of-thought paths.  Toric/tropical chambers help by telling whether the analogy crosses the same kinds of walls.  PH features help by measuring whether the source and target reasoning paths have the same topological shape.

### 3. BPB Score

The direct route to better BPB is still likelihood optimization on the OAI/FineWeb byte stream.  The advanced methods can help BPB only if they improve predictive distributions.  The plausible mechanisms are:

- graph fusion improves local byte/context structure;
- tropical attention improves sparse active selection and long-context recurrence;
- toric memory gives compact nonrepeating phase features for positions and recurrence;
- GraphCG reduces hidden superposition and makes reusable features easier for a tiny model to exploit;
- GFlowNet samples useful latent reasoning or memory actions at eval/training time;
- trajectory memory retrieves structurally similar contexts;
- vectorized PH and toric signatures provide better retrieval keys;
- QAT/export-aware training reduces post-export BPB regression.

Risks:

- advanced losses can damage BPB if too strong early;
- CAS metrics can look good while BPB worsens;
- graph-heavy data can pull the model away from FineWeb byte statistics;
- full-rank GraphCG from step 0 may increase gradient noise or VRAM pressure;
- random-order objectives can mismatch left-to-right BPB if not configured carefully.

Current mitigation:

- all advanced weights are small;
- OAI BPB is evaluated every 250 steps;
- BGG pressure is zero until late phase;
- FineWeb calibration mix is high;
- shock guards and robust micro-loss guards are active;
- QAT starts late and with tiny weight.

### 4. Behavior And Output Quality Control

Tropical and toric techniques provide control surfaces:

- active-face margins show whether a decision is stable or near a wall;
- normal-fan cells identify which region of reasoning space the model uses;
- toric phase leaf residuals show whether recurrence/position memory is coherent;
- Slepian concentration shows whether phase energy is localized rather than noisy;
- binomial relation losses check whether algebraic paths agree;
- Coxeter/braid losses check consistency of symbolic transformations;
- BGG/Koszul residuals check whether multi-step reasoning forms a complex;
- persistence maps show whether trajectory topology is stable.

These controls can be used to select checkpoints, gate auxiliary losses, and diagnose output failures.  They should not be allowed to override the BPB objective unless an ablation demonstrates benefit.

### 5. Interaction With GraphCG And Embedding-Space GFlowNets

GraphCG provides a learned basis `B` for hidden space.  Toric probes pull back through that basis:

```tex
\ell_r(B\xi)=\langle B^T a_r,\xi\rangle+b_r .
```

Thus GraphCG determines the coordinate system in which tropical/toric walls are seen.  If GraphCG works, the active fan walls align with reusable semantic or reasoning axes rather than dense arbitrary directions.  That makes:

- active-face transitions easier to interpret;
- analogical arrows more stable;
- memory retrieval signatures more compact;
- GFlowNet actions more meaningful.

Embedding-space GFlowNets then search in this structured space.  Instead of sampling arbitrary hidden perturbations, the policy can learn to move along:

- graph-of-thought branch/merge actions;
- GraphCG basis directions;
- toric chamber/wall transitions;
- memory retrieval actions;
- topology-preserving trajectory edits.

This is the intended synthesis:

```text
GraphCG gives coordinates.
Tropical attention gives active piecewise-linear decisions.
Toric geometry gives the ambient fan/variety/monomial algebra.
GUDHI gives topology of trajectories in that space.
BGG/Koszul gives chain-complex discipline.
GFlowNets search over graph-of-thought trajectories using those structures.
The byte likelihood/BPB objective decides whether any of it is useful for the competition.
```

## Recommended Next Decisions

1. Run a short no-training-start smoke analysis or explicit user-approved
   training smoke before relying on full-rank GraphCG in a long BPB run.
2. Add an operational runbook so training, watching, exact analysis, rendering, export, and BPB evaluation are unambiguous.
3. Add exact tropical-cycle multiplicity and balancing certificates so the balance/Chow panels become real, not unavailable.
4. Run controlled BPB ablations with identical seeds/data windows before increasing any advanced loss.
5. Keep BGG loss pressure off until late phase unless probe-only evidence shows the hidden representation already contains the relevant structure.
6. Make periodic exact sidecar rendering asynchronous and bounded so it never kills the training process.
7. Keep CAS reports and generated visualizations out of the competition artifact unless a compact distilled probe earns its bytes.

## Bottom Line

The tropical-to-toric machinery is now implemented as a real analysis and training scaffold.  The OAI Parameter-Golf model embeds tropical ring attention into toric geometry by treating tropical affine candidates as monomial exponent data, building Newton/lifted Newton polytopes and normal fans, and using the resulting toric variety/fan structure for metrics, visualizations, and small auxiliary losses.  GraphCG supplies the coordinate basis, GFlowNets explore graph-of-thought trajectories in that basis, GUDHI and CAS tools certify topological/algebraic structure, and BGG/Category O enters late as chain-complex supervision.

The system is ready for careful BPB-first training and analysis.  The main remaining work is not adding more concepts; it is tightening the operational loop: authoritative BPB measurement, controlled ablations, exact multiplicity/balance certificates, full-rank GraphCG decision, and asynchronous periodic reports that cannot interfere with training.
