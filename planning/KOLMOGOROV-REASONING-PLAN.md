# Kolmogorov Complexity and Reasoning Robustness Plan

## Purpose

Add an explicit metric and training-analysis layer that treats robust reasoning
as reusable compression, while keeping the OpenAI Parameter Golf objective
strictly grounded in tokenizer-agnostic BPB.  The goal is not to pretend that
true Kolmogorov complexity is computable.  The goal is to build a family of
auditable approximations that reveal whether ToricGT is learning short,
invariant programs for hard technical domains, rather than memorizing brittle
surface traces.

The competition objective remains validation BPB under the official artifact
and wallclock constraints.  Complexity metrics are a second axis:

- BPB measures predictive compression of byte streams.
- Kolmogorov-style metrics measure whether solutions, reasoning traces, graph
  trajectories, and intermediate programs are compact, reusable, invariant,
  and robust under transformations.

The desired model should sit on the Pareto frontier: low BPB, high answer
accuracy, low conditional reasoning complexity, high invariance under graph and
serialization perturbations, and high test-time reasoning improvement per unit
of sampled graph-of-thought compute.

## Core Principle

For an input problem `x`, answer `y`, and reasoning trajectory `tau`, robust
reasoning means that the model has learned an internal program `p` such that

```text
U(p, x) = y
```

with `|p|` small relative to memorizing `y` or a full surface trace.  In
Kolmogorov language:

```text
K(y | x, rules) << K(y | x)
```

ToricGT's architectural priors are intended to lower this conditional
complexity:

- permutation equivariance removes arbitrary graph labels;
- tropical attention implements compact max/min-plus programs;
- noncommutative toric memory implements compact algebraic normal forms;
- random-order autoregression reduces dependence on one serialization;
- GFlowNet graph-of-thought rollouts sample compact high-reward trajectories;
- domain graphs expose reusable structure in math, code, chemistry, proteins,
  Hebrew morphology, and medicine-design tasks.

The complexity layer should verify whether these priors actually produce
shorter, more stable reasoning programs.

## Definitions and Approximate Metrics

True `K(.)` is uncomputable.  We use compressor-, grammar-, and model-based
surrogates.  Every reported metric must be named with its estimator, for
example `K_zstd`, `K_lzma`, `K_graph_mdl`, or `K_model`.

### 1. Compressor Complexity

For byte string `s` and compressor `c`:

```text
K_c(s) = bytes(c(s))
```

Use at least `zlib`, `lzma`, and, if installed, `zstd`.  Report all estimators
instead of hiding disagreement.

Recommended W&B metrics:

```text
complexity/input_k_zlib
complexity/input_k_lzma
complexity/trace_k_zlib
complexity/trace_k_lzma
complexity/answer_k_zlib
complexity/answer_k_lzma
```

### 2. Conditional Complexity

For input `x` and trace or answer `y`, approximate conditional complexity by a
two-string code:

```text
K_c(y | x) ~= max(0, K_c(x || sep || y) - K_c(x))
```

This is crude but useful for trends.  A dictionary-compressor variant is
better when available:

```text
K_c_dict(y | x) = bytes(c(y, dictionary=x))
```

Recommended metrics:

```text
complexity/answer_cond_k_lzma
complexity/trace_cond_k_lzma
complexity/got_cond_k_lzma
complexity/trace_cond_per_correct_answer
```

Interpretation:

- low `K(trace | input)` plus correct answer suggests reusable reasoning;
- low `K(trace | input)` plus wrong answer suggests shortcutting or collapse;
- high `K(trace | input)` plus correct answer may be brute-force search;
- high `K(trace | input)` plus wrong answer is pure waste.

### 3. Normalized Compression Distance

For two traces `a,b`:

```text
NCD_c(a,b) = (K_c(a || b) - min(K_c(a), K_c(b))) / max(K_c(a), K_c(b))
```

Use NCD to measure whether multiple successful trajectories are genuinely
different or merely surface variants.  This is important for GFlowNets because
terminal diversity should not mean duplicated chains with cosmetic edits.

Recommended metrics:

```text
complexity/gfn_terminal_ncd_mean
complexity/gfn_terminal_ncd_p10
complexity/gfn_success_ncd_mean
complexity/order_seed_ncd_mean
```

### 4. Graph Minimum Description Length

For graph record `G=(V,E,T,X)`, define a graph MDL code:

```text
L_graph(G) = L_schema + L_node_types + L_edge_types
             + L_attributes + L_payloads + L_targets
```

Use canonical graph serialization so arbitrary node ids do not inflate the
code.  The first implementation should use:

- sorted node type inventory;
- sorted edge type inventory;
- stable canonical node labels by Weisfeiler-Lehman color refinement where
  possible;
- payload compression after removing deterministic ids;
- explicit accounting for source text bytes versus graph structure bytes.

Recommended metrics:

```text
complexity/graph_mdl_total
complexity/graph_mdl_structure
complexity/graph_mdl_payload
complexity/graph_mdl_target
complexity/graph_mdl_per_node
complexity/graph_mdl_per_edge
```

### 5. Algorithmic Mutual Information

Approximate the algorithmic mutual information between input and successful
trace:

```text
I_c(x : tau) = K_c(x) + K_c(tau) - K_c(x || tau)
```

High `I_c(x:tau)` can mean the trace is input-specific.  That is useful only
when paired with correctness and low conditional complexity.  Very low
`I_c(x:tau)` can indicate generic boilerplate reasoning.

Recommended metrics:

```text
complexity/input_trace_mi_lzma
complexity/input_answer_mi_lzma
complexity/boilerplate_trace_ratio
```

### 6. Transformation Robustness Complexity

Let `T` be a set of semantics-preserving transformations:

- graph relabeling;
- edge order permutation;
- random-order byte target permutation;
- equivalent algebra normal form;
- equivalent Hebrew root-template rendering where attested;
- code formatting and variable renaming;
- molecule graph atom order permutation;
- protein residue graph relabeling when topology is preserved.

For each `t in T`, evaluate answer invariance, log-prob stability, and
complexity stability:

```text
R_answer = mean[ answer(f(x)) == answer(f(t(x))) ]
R_logp   = std_t log p(y | t(x))
R_K      = std_t K_c(tau_t | t(x))
```

Recommended metrics:

```text
complexity/transform_answer_invariance
complexity/transform_logp_std
complexity/transform_cond_k_std
complexity/transform_graph_mdl_std
```

The target behavior is high answer invariance, low conditional-K variance, and
low log-prob variance.  This is the main bridge between Kolmogorov complexity
and reasoning robustness.

## Reasoning Quality Scores

Complexity alone is not a quality metric.  A tiny wrong proof is not good.
Every complexity score must be conditioned on correctness, verifier pass rate,
or reward.

### Correctness-Weighted Conditional Complexity

```text
CWCK = E[ 1_correct * K_hat(tau | x) + 1_wrong * penalty_wrong ]
```

Use `penalty_wrong` larger than the 90th percentile of observed conditional
trace complexity, so wrong compact traces are punished.

### Reasoning Compression Gain

```text
RCG = K_c(raw_trace) - K_graph_mdl(program_trace)
```

Large positive `RCG` means the graph/program representation compresses the
surface reasoning trace.  This is desirable only if answer accuracy and
verifier score remain high.

### Robust Reasoning Efficiency

```text
RRE = accuracy / (epsilon + E[K_hat(tau | x)])
```

Report by domain, not only globally:

```text
reasoning_rre/math
reasoning_rre/code
reasoning_rre/biochem
reasoning_rre/biophysics
reasoning_rre/medicine_design
reasoning_rre/hebrew
reasoning_rre/toric_algebra
```

### GFlowNet Complexity-Reward Calibration

For terminal trajectory `tau` with reward `R(tau)`:

```text
calibration_error = | log P_F(tau) - log R(tau) + log Z |
```

Add a complexity-aware diagnostic:

```text
R_K(tau) = R(tau) * exp(-lambda_K * K_hat(tau | x))
```

This should initially be diagnostic-only.  Turn it into a training reward only
after verifying that it improves held-out reasoning without increasing BPB.

## BPB Orthogonality and Two-Objective Accounting

Parameter Golf scoring is byte prediction:

```text
BPB = -sum_t log2 q(b_t | b_<t) / N_bytes
```

The Kolmogorov layer should not replace BPB.  Instead, use a two-part MDL view:

```text
TotalCodeLength ~= artifact_bits + N_bytes * BPB
```

The 16,000,000 byte cap fixes a hard upper bound on `artifact_bits`.  BPB
measures how well the artifact compresses validation bytes.  Reasoning
complexity metrics explain whether low BPB comes from robust reusable programs
or from shallow byte-pattern exploitation.

Training objective should remain:

```text
L_total = L_bpb
          + lambda_gfn L_gfn
          + lambda_contrast L_contrast
          + lambda_q L_qat
          + lambda_K L_K_aux
```

with strict guardrails:

- `lambda_K = 0` by default for the first implementation;
- enable `lambda_K > 0` only after metrics are stable;
- require repeated validation BPB improvement or no statistically meaningful
  BPB regression;
- never let complexity rewards access future validation bytes;
- never reward short traces without correctness or verifier support.

Recommended decision thresholds:

- keep diagnostic-only metrics regardless of BPB;
- keep training-time complexity regularization only if repeated validation
  delta is no worse than `+0.003 BPB`;
- keep complexity-aware GFlowNet rewards only if reasoning accuracy improves
  and validation BPB is no worse than `+0.005`;
- remove any complexity term that reduces terminal diversity without improving
  correctness.

## Integration With Existing ToricGT Ideas

### Random-Order Autoregression

Use random target orders to estimate serialization dependence:

```text
complexity/order_cond_k_mean
complexity/order_cond_k_std
complexity/order_bpb_std
complexity/order_answer_invariance
```

The model is robust when different random orders produce similar answer
distributions and similar conditional complexity.

### Tropical Ring Attention

Tropical heads expose active maximizers and margins.  Treat active-face
sequences as compact programs:

```text
tau_trop = [(query_i, coord_c, active_key_j, margin_delta), ...]
```

Metrics:

```text
complexity/tropical_active_face_k
complexity/tropical_active_face_cond_k
complexity/tropical_margin_per_k
reasoning/tropical_correct_margin_per_k
```

The desired behavior is high margin and correctness with a compact active-face
program, especially on shortest path, dynamic programming, and alignment tasks.

### Noncommutative Toric Memory

For rotation words and higher torus tasks, normal forms are short programs.
Track the compression of raw words into normal forms:

```text
raw:      V U U^{-1} V ...
program: phase=c, exponent=(a,b), convention=VU=omega UV
```

Metrics:

```text
complexity/toric_raw_word_k
complexity/toric_normal_form_k
complexity/toric_compression_gain
reasoning/toric_normal_form_accuracy
reasoning/toric_commutator_error
```

### GFlowNet Graph-of-Thought

Each GFlowNet trajectory should be serialized at three levels:

1. surface text;
2. graph action trace;
3. compressed program trace.

Metrics:

```text
complexity/gfn_surface_trace_k
complexity/gfn_action_trace_k
complexity/gfn_program_trace_k
complexity/gfn_program_gain
complexity/gfn_success_cond_k
complexity/gfn_failed_cond_k
```

The model is improving when successful terminal trajectories have lower
conditional program complexity than failed trajectories while retaining high
NCD diversity across successes.

### 3D Trajectory and Energy/Fitness Plots

Existing visualization plans should add complexity channels:

- 3D trajectory: color by `K_hat(prefix_tau | x)`;
- Ramachandran-style torsion plot: axes are two latent trajectory torsions,
  color is correctness-weighted complexity;
- energy/fitness landscape: height is negative reward or BPB loss, color is
  conditional complexity, minima are high-quality low-complexity solutions.

Output targets:

```text
outputs/visualizations/kolmogorov_trajectory_3d.html
outputs/visualizations/kolmogorov_ramachandran.png
outputs/visualizations/kolmogorov_energy_landscape.html
```

### BPB Projection Script

Extend `scripts/project_pg_loss.py` later to optionally overlay complexity:

```text
--metrics train/loss train/bpb complexity/trace_cond_k_lzma
```

This lets us ask whether BPB improvements coincide with more compact robust
reasoning, or whether they are mostly local byte modeling gains.

## Domain-Specific Plans

### Mathematics

Represent proofs as dependency graphs:

- theorem statement nodes;
- equation nodes;
- transformation nodes;
- justification edges;
- final-answer nodes.

Metrics:

```text
complexity/math_proof_graph_mdl
complexity/math_proof_cond_k
reasoning/math_verifier_pass_per_k
reasoning/math_answer_accuracy
```

Target behavior: shorter valid proof programs on held-out problem families,
not shorter invalid explanations.

### Coding

Use AST and control-flow approximations where practical:

- raw solution bytes;
- AST serialization length;
- execution trace length;
- failing versus passing test trace complexity.

Metrics:

```text
complexity/code_ast_mdl
complexity/code_patch_cond_k
reasoning/code_tests_passed_per_k
reasoning/code_variable_rename_invariance
```

Target behavior: robust code generation under variable renaming and formatting
changes, with low conditional patch complexity.

### Biochemistry, Biophysics, and Medicine Design

Use typed molecular and structural biology graphs when available:

- atoms/residues as nodes;
- bonds, contacts, hydrogen bonds, salt bridges, and proximity edges;
- active-site and ligand nodes;
- energy/fitness annotations as graph targets.

Metrics:

```text
complexity/chem_graph_mdl
complexity/protein_contact_graph_mdl
complexity/ligand_reasoning_cond_k
reasoning/biochem_answer_accuracy_per_k
reasoning/biophysics_energy_rank_per_k
reasoning/medicine_design_constraint_pass_per_k
```

Target behavior: compact reasoning traces that preserve mechanistic constraints
such as binding-site compatibility, sterics, electrostatics, sequence-structure
relationships, and assay-condition caveats.

### Hebrew and Root Graphs

Use root-template graphs:

```text
surface form -> root radicals -> binyan/template -> feature bundle
```

Metrics:

```text
complexity/hebrew_surface_k
complexity/hebrew_root_template_mdl
complexity/hebrew_paradigm_cond_k
reasoning/hebrew_root_accuracy_per_k
reasoning/hebrew_binyan_accuracy_per_k
```

Target behavior: lower conditional complexity for pointed and unpointed forms
that share the same root family, with robust held-out-root generalization.

## Implementation Plan

### Stage 0: Metric-Only Baseline

Add metric code without changing training.

Planned files:

```text
src/toricgt/complexity.py
scripts/evaluate_complexity.py
scripts/plot_complexity.py
tests/test_complexity_metrics.py
```

Initial functions:

```text
compress_len_zlib(bytes) -> int
compress_len_lzma(bytes) -> int
conditional_compress_len(x, y, compressor) -> int
ncd(x, y, compressor) -> float
canonical_graph_bytes(graph_json) -> bytes
graph_mdl(graph_json) -> dict
trajectory_complexity(trace, input) -> dict
```

No training code changes in this stage.

### Stage 1: W&B Diagnostics

Log complexity metrics during validation and test-time scaling:

```text
complexity/answer_cond_k_lzma
complexity/trace_cond_k_lzma
complexity/gfn_success_ncd_mean
complexity/transform_cond_k_std
complexity/graph_mdl_total
```

Do not log every sample.  Use bounded samples per evaluation window:

```text
complexity_eval_samples: 128
complexity_eval_every: 500
complexity_compressors: [zlib, lzma]
```

### Stage 2: Visualization

Add static and interactive plots:

- `K_cond` versus correctness;
- `K_cond` versus BPB;
- BPB/accuracy/complexity Pareto frontier;
- NCD heatmap for successful GFlowNet terminals;
- transformation robustness violin plots;
- 3D trajectory plots with complexity coloring.

### Stage 3: Complexity-Aware GFlowNet Reward

Only after Stages 0-2 are stable, add an optional reward term:

```text
R'(tau) = R(tau) * exp(-lambda_K * normalized_K_cond(tau | x))
```

Default:

```text
lambda_K: 0.0
```

Trial values:

```text
lambda_K: [0.001, 0.003, 0.01]
```

Keep the term only if it improves reasoning accuracy or terminal verifier pass
rate without materially worsening validation BPB.

### Stage 4: Complexity-Aware Data Curriculum

Use complexity buckets for curriculum sampling:

- low-complexity examples for syntax and conventions;
- medium-complexity examples for common reasoning programs;
- high-complexity examples for hard technical reasoning and GFlowNet rollouts.

Sampling should not chase maximum complexity blindly.  Very high compressor
complexity can mean noisy data.  Use verifier or quality flags as a gate.

### Stage 5: Competition-Gated Integration

Before promoting any complexity-aware term to the default Parameter Golf run:

1. run float validation;
2. run packed artifact validation;
3. compare BPB to the current dense random-order baseline;
4. check complexity metrics by domain;
5. check causal audit;
6. check artifact size;
7. keep only if BPB and reasoning diagnostics justify the extra code/compute.

## Config Additions

Proposed YAML section:

```yaml
complexity:
  enabled: true
  train_regularizer_weight: 0.0
  gflownet_reward_weight: 0.0
  eval_every: 500
  eval_samples: 128
  compressors:
    - zlib
    - lzma
  graph_mdl: true
  transformation_robustness: true
  ncd_samples: 32
  wandb_prefix: complexity
```

## Reporting Contract

Every run that enables complexity diagnostics should report:

- official-style BPB;
- packed artifact BPB delta;
- answer accuracy or verifier pass rate;
- conditional trace complexity;
- graph MDL;
- GFlowNet terminal NCD;
- transformation robustness;
- complexity/accuracy Pareto plot;
- complexity/BPB Pareto plot.

Do not report a single "Kolmogorov score" as if it were canonical.  Always
report the estimator and the domain.

## Failure Modes and Controls

### Failure: Short Wrong Traces

Control: condition complexity rewards on correctness or verifier pass.

### Failure: Compressor Rewards Formatting Tricks

Control: canonicalize graph traces and compare multiple compressors.

### Failure: BPB Regression

Control: keep complexity terms diagnostic-only until repeated BPB-safe evidence.

### Failure: GFlowNet Diversity Collapse

Control: monitor NCD among successful terminals and entropy of action traces.

### Failure: Domain Noise Looks Like Complexity

Control: bucket by quality flags and verifier support; do not upsample high-K
examples without quality gates.

### Failure: Medical/Biochemical Overclaiming

Control: treat biomedical and medicine-design tasks as benchmark reasoning and
constraint satisfaction, not as clinical advice or deployable design guidance.

## Success Criteria

Short-term success:

- complexity metrics run on validation without slowing training materially;
- plots reveal meaningful separation between successful and failed traces;
- transformation robustness metrics catch brittle serialization behavior;
- no BPB regression from metric-only instrumentation.

Medium-term success:

- complexity-aware GFlowNet reward improves held-out reasoning pass rate;
- successful trajectories have lower conditional program complexity than failed
  trajectories;
- GFlowNet terminal diversity remains high by NCD;
- random-order BPB variance decreases.

Competition success:

- packed validation BPB approaches the sub-1.2 target band;
- complexity diagnostics show that BPB gains are not only local byte hacks;
- reasoning robustness improves in math, coding, biochemical, biophysical,
  medicine-design, Hebrew, and toric-algebra evaluations;
- the artifact remains below `16,000,000` bytes and passes causal audits.

## Recommended First Implementation

Implement Stage 0 and Stage 1 first.  Keep all complexity metrics diagnostic.
The first useful command should look like:

```bash
conda run --no-capture-output -n tokengt env PYTHONPATH=src \
  python scripts/evaluate_complexity.py \
  --checkpoint checkpoints/parameter_golf_oai_dense/best.pt \
  --data-glob 'data/curated_hf_shards/validation/*.parquet' \
  --samples 512 \
  --output-dir outputs/complexity/oai-validation \
  --wandb-run amelie-iska-math/toricgt-parameter-golf/gbmw7z3a
```

Only after this produces stable diagnostics should we consider adding
`lambda_K` to GFlowNet reward shaping.
