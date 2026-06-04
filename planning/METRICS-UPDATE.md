# Metrics And Losses Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ToricGT's advanced BGG, Koszul, topology, toric, tropical, analogical, and PolarQuant diagnostics actively improve OpenAI Parameter-Golf FineWeb BPB while preserving a <=16MB competition artifact.

**Architecture:** Keep cross-entropy and OpenAI FineWeb BPB as the primary early objective. Treat advanced mathematical losses as evidence-gated sidecars before the <=1.2 BPB checkpoint exists, then promote them into larger post-threshold reasoning and memory phases. Add a BPB-transfer controller that measures whether each advanced metric family historically predicts the next validation BPB drop and emits recommended loss scales, artifact-size policy, and phase controls.

**Tech Stack:** PyTorch, W&B, matplotlib, pandas/numpy, existing ToricGT modules under `src/toricgt`, Seq4096 watcher scripts under `scripts`, and compact Parameter-Golf training script under `amelie-iska/parameter-golf/records/.../train_gpt.py`.

## Current Live Update - 2026-06-04 R60

- Active branch: `oai-advanced` in both ToricGT and the nested
  Parameter-Golf repo.
- Active BPB run: `toricgt_seq4096_4k_recovery_r60_20260604T153456Z`.
- Current recovery origin: best authoritative validation checkpoint at step
  3750 with validation/OpenAI BPB `1.2131`.
- Most recent completed gate result: R59 reached step 4000 with authoritative
  validation BPB `1.2288`, so the <=1.2 threshold was **not** reached. R60 was
  relaunched from R59 step 3750 with optimizer/RNG/loader reset.
- Current controller policy: pre-threshold BPB-clean recovery. Keep
  cross-entropy, validation BPB, train-wave shape, bigram bias, LR, batch-token,
  and Muon-warmup controls on the critical path. Keep BGG/Koszul, topology,
  toric, tropical, Slepian/Pollak, GraphCG, memory, and analogy families as
  sidecar evidence and damping priors until either an authoritative <=1.2
  threshold checkpoint is preserved or a held-out transfer study shows direct
  BPB benefit.
- Artifact policy: under the 16,000,000 byte cap but extremely tight.  The
  latest int8+zlib probes have been within the cap but with little margin, so
  do not add exported probes, code bloat, or larger stored weights before
  preserving the threshold checkpoint.
- PolarQuant/larger-parameter policy: PolarQuant currently quantizes K/V cache
  tensors, not stored model weights.  Larger-parameter retraining is permitted
  only as a separate sidecar branch if a true stored-weight/export compression
  path proves code+weights remain under `16,000,000` bytes and improves held-out
  BPB.  It should not interrupt the active BPB-clean R60 gate attempt.
- Periodic analysis action: the live watcher for R60 must run the W&B metrics,
  historical OAI BPB entrypoint, compact Seq4096 BPB evaluator, reasoning
  simplex, geometry/topology/toric/Slepian/Koszul/BGG/GraphCG/analogical
  visualization suite, final artifact inventory refresh, and Codex/sub-agent
  review hook on the current-run checkpoint outputs. Reviewers must inspect
  every inventoried output family and report early BPB-critical recommendations
  separately from later advanced-reasoning recommendations.

---

## Current Implementation With Pseudocode

### Toric BGG Category O Metrics

Implementation file: `src/toricgt/toric_bgg.py`

Purpose: finite training/audit shadow of category O behavior. Hidden states are projected into a small standard-label poset and finite chain-complex certificate. The probe is excluded from the competition export by default.

Pseudocode:

```python
def toric_bgg_probe(hidden, target_positions, target_tokens):
    x = sample_positions(hidden, max_positions)
    features = tanh(linear_down(x))

    standard_logits = linear_standard(features)
    standard_probs = softmax(standard_logits)
    labels = target_positions % num_standard_labels

    # Category O standard filtration: mass should stay in the order ideal.
    allowed = standard_mask[labels]  # mu <= lambda
    standard_leakage = sum_probability_outside_allowed(standard_probs, allowed)

    # Finite free-resolution sanity: d^2 should vanish.
    summary = mean(features)
    d1 = linear_boundary_1(summary).reshape(labels, labels)
    d2 = linear_boundary_2(summary).reshape(labels, labels)
    d2_residual = mean((d1 @ d2) ** 2)

    # Koszul linearity shadow: mass should lie near internal_degree = homological_degree.
    mass = mean(standard_probs, axis=items)
    koszul_linearity = weighted_abs_degree_error(mass, homological_degree, internal_degree)

    # Gale duality: paired signatures should agree under a learned dual map.
    signatures = normalize(linear_signature(features))
    gale_consistency = mse(signatures[:-1] @ gale_map.T, flip(signatures[1:]))

    signature_smoothness = mean_square_difference(consecutive(signatures))

    loss = (
        d2_weight * d2_residual
        + standard_weight * standard_leakage
        + koszul_weight * koszul_linearity
        + gale_weight * gale_consistency
        + signature_weight * signature_smoothness
    )
    return loss, detached_metrics
```

Utilization:

- Logged as `train/toric_bgg_*`, `bgg_category_o/*`, and `category_o/*`.
- Weighted into training only when `toric_bgg_loss_weight > 0`.
- Useful early as a sidecar pressure: high standard leakage or high `d2` residual means hidden reasoning states are not respecting finite resolution structure.
- Useful after threshold as a larger auxiliary loss for rigorous reasoning trajectories.

### Koszul Metrics

Implementation file: `src/toricgt/koszul_persistence.py`

Purpose: differentiable local commutative-algebra proxy over hidden-state windows. The model builds small row-stochastic chart actions and checks Koszul exactness, syzygies, rank conditions, and multigraded Betti pressure.

Pseudocode:

```python
def koszul_persistence_loss(hidden, positions):
    for sequence in batch:
        for window in sampled_windows(sequence):
            points = normalize(sample_points(window))

            time_action = row_stochastic_next_step_operator(points)
            radius_action = row_stochastic_radius_operator(pairwise_dist(points))
            phase_action = row_stochastic_toric_phase_operator(positions)
            actions = [time_action, radius_action, phase_action]

            d1, d2, d3 = first_koszul_differentials(actions)
            exactness = mean((d1 @ d2) ** 2) + mean((d2 @ d3) ** 2)

            syzygy = mean((Ai @ Aj - Aj @ Ai) ** 2 for i < j)
            soft_ranks = svd_soft_rank(d1), svd_soft_rank(d2), svd_soft_rank(d3)
            fitting_rank_residual = rank_condition_failure(soft_ranks)
            be_rank_residual = buchsbaum_eisenbud_rank_condition_failure(soft_ranks)
            betti_mass = multigraded_betti_proxy(soft_ranks)

            chart_entropy = entropy(mean(chart_probs))
            chart_coverage = fraction_of_active_charts(chart_probs)
            chart_shift = distance_to_previous_chart_id(chart_probs)

    loss = exactness + 0.35 * syzygy + 0.10 * fitting + 0.10 * be_rank
    return loss, detached_metrics
```

Utilization:

- Logged as `train/koszul_*` and `koszul_persistence/*`.
- Weighted into training only when `koszul_persistence_loss_weight > 0`.
- Before <=1.2 BPB, it should generally be a sidecar or very low-weight regularizer because it can consume gradient budget without direct BPB transfer evidence.
- After threshold, it is appropriate for advanced algebraic reasoning phases.

### Topological And Persistent Homology Metrics

Implementation file: `src/toricgt/topological_reasoning.py`

Purpose: differentiable directed persistent topology over local reasoning-step windows. Vertices are hidden states, soft edges come from a radius filtration, directed edges add time orientation and an antisymmetric toric skew form, and HDBSCAN-style persistence scores stable clusters.

Pseudocode:

```python
def reasoning_step_topology_loss(hidden):
    radii = linspace(radius_min, radius_max, levels)
    for sequence in batch:
        for window in sampled_windows(sequence):
            points = normalize(sample_points(window))
            dist = pairwise_distance(points) / median_positive_distance
            skew = antisymmetric_toric_form(points)
            time_bias = sign(j - i) * time_bias_scale

            for radius in radii:
                sym = sigmoid((radius - dist) / temperature)
                directed = sigmoid((radius - dist + skew + time_bias) / temperature)

                closure = mean((sym @ sym) * (1 - sym))
                inclusion = relu(previous_sym - sym) ** 2
                boundary = oriented_triangle_boundary_residual(directed)
                dirichlet = edge_weighted_distance_energy(sym, dist)

                flow = directed - directed.T
                dec_conservation = divergence_loss(flow) + vorticity_drift + hodge_balance

                directed_transitive = mean((directed @ directed) * (1 - directed))
                directed_cycle_flux = abs(forward_3cycles - backward_3cycles)

                hdbscan_density = sigmoid((radius - mutual_reachability) / temperature)

            persistent_edges = mean(hdbscan_density_over_radii)
            hdbscan_loss = stable_pair_distance_loss(persistent_edges, dist)
            betti0_proxy = heat_kernel_trace(laplacian(sym))
            cycle_rank_proxy = edge_count - vertex_count + betti0_proxy

    # Analogical maps between consecutive complexes.
    for source_complex, target_complex in consecutive_complexes:
        transport = softmax(-cdist(source_points, target_points) / temperature)
        analogical_map_loss = mse(transport.T @ source_sym @ transport, target_sym)
        directed_map_loss = mse(transport.T @ source_dir @ transport, target_dir)

    loss = (
        barcode + closure + 0.15 * inclusion + 0.15 * boundary
        + 0.25 * dirichlet + 0.15 * dec_conservation
        + 0.35 * directed_loss + 0.15 * analogical_map
        + 0.15 * directed_map + 0.20 * hdbscan
    )
    return loss, detached_metrics
```

Utilization:

- Logged through analogy-step metrics as `topology/*` and through full diagnostics as `topology/topology_loss`, directed losses, Betti proxies, triangle density, cycle rank, HDBSCAN stability, and DEC residuals.
- Early competition phase: use primarily to decide whether validation stalls are structural or optimizer-driven.
- Advanced phase: train graph-of-thought trajectories as directed noncommutative paths through reasoning-step simplicial complexes.

### Toric Metrics

Implementation file: `src/toricgt/toric_geometry_tasks.py`

Purpose: low-rank synthetic toric geometry probe for active faces, moment maps, fan margins, bends, binomial relations, affine Coxeter walls, braid relations, and toric leaf motion.

Pseudocode:

```python
def toric_geometry_probe(hidden, positions, tokens):
    logits = quantized_low_rank_probe(hidden)
    teacher_features = toric_phase_features(positions, theta, beta)
    teacher_logits = teacher_features @ exponent_table.T
    target_face = argmax(teacher_logits)

    face_ce = cross_entropy(logits, target_face)
    face_margin = target_logit - max_other_logit
    fan_loss = face_ce + relu(margin_target - face_margin)

    pred_probs = softmax(logits / temperature)
    teacher_probs = softmax(teacher_logits / temperature)
    pred_moment = pred_probs @ exponents
    teacher_moment = teacher_probs @ exponents
    moment_loss = mse(pred_moment, teacher_moment)

    bend_loss = mse(second_difference(pred_moment), second_difference(teacher_moment))
    binomial_loss = mse(logits[i] + logits[j] - logits[k] - logits[l], teacher_relation)
    coxeter_loss = mse(reflect(pred_moment), reflect(teacher_moment))
    braid_loss = mse(s1s2s1(pred_moment), s2s1s2(pred_moment))
    leaf_residual = mean(1 - cos(projected_phase_delta - expected_delta))

    loss = weighted_sum(fan, bend, binomial, moment, coxeter, braid, leaf)
    return loss, detached_metrics
```

Utilization:

- Logged as `train/toric_*`, `toric/*`, and some `tropical/*` aliases.
- The probe weights are excluded from competition export unless explicitly included.
- Early phase: high toric bend or negative active-face margin is a structural pressure, not automatically a reason to add large toric loss.
- Later phase: useful for robust embedding geometry and noncommutative trajectory regularity.

### Tropical Metrics

Implementation files: `src/toricgt/tropical_attention.py`, `src/toricgt/metrics.py`

Purpose: tropical attention is part of the architecture. It uses max-plus attention and block/ring max-plus reduction for long contexts. Tropical metrics track active face confidence, plateau pressure, and max-minus-second-max margins.

Pseudocode:

```python
def tropical_attention(q, k, v, mask):
    scores = (q @ k.T) / sqrt(dim)
    scores = mask_invalid(scores, -inf)
    # Max-plus aggregation.
    out = max_over_keys(scores[..., key] + v[..., key, :])
    return out, scores

def tropical_ring_attention(q, k, v, block_size):
    for q_block in query_blocks:
        partial = -inf
        for kv_block in key_value_blocks:
            scores = q_block @ k_block.T
            block_context = max(scores + v_block)
            partial = max(partial, block_context)
        write_output(q_block, partial)

def tropical_margin(scores):
    top1, top2 = topk(scores, 2)
    return mean(top1 - top2)
```

Utilization:

- Architectural long-context path for reasoning trajectories.
- Logged via `tropical/*` aliases and diagnostics.
- Early phase: treat tropical plateau pressure as an indicator that BPB descent has flattened and a restart/control intervention may be needed.
- Later phase: scale context beyond 4096 with tropical ring attention for long graph-of-thought and memory paths.

### Analogical Reasoning Metrics

Implementation files: `src/toricgt/random_order_lm.py`, `src/toricgt/topological_reasoning.py`, `src/toricgt/complexity.py`

Purpose: relation vectors between hidden states are grouped by coarse byte-class arrows. Same arrows should induce similar displacement vectors, compose into parallelograms, map into GraphCG basis directions, and preserve topology under analogical transport.

Pseudocode:

```python
def analogy_lattice_losses(hidden, target_tokens):
    relation = normalize(hidden[t + stride] - hidden[t])
    arrow_key = coarse_byte_class(target[t]) * 16 + coarse_byte_class(target[t + stride])

    for repeated_arrow_group in groups(arrow_key):
        group_mean = normalize(mean(relation[group]))
        functor_loss += mean((relation[group] - group_mean) ** 2)

        group_complex = build_filtration(relation[group])
        topology_loss += simplex_closure + inclusion + chain_map
        directed_topology_loss += directed_transitive + directed_cycle + directed_chain
        hdbscan_loss += stable_cluster_distance

    if same_arrow(t -> t+s, t+s -> t+2s):
        parallelogram_loss = mse(normalize(rel_left), normalize(rel_right))

    if GraphCG_basis_enabled:
        coords = relation @ basis.T
        axis_probs = softmax(abs(coords) / temperature)
        reconstructed = normalize(signed_axis_probs @ basis)
        basis_loss = mean(1 - dot(relation, reconstructed))
        axis_entropy = entropy(axis_probs)
        lattice_margin = top1_abs_coord - top2_abs_coord

    step_topology = reasoning_step_topology_loss(project_to_graphcg_chart(hidden))

    loss = functor_loss + basis_weight*basis_loss + parallelogram_weight*parallelogram
           + topology_weight*(topology_loss + step_topology)
    return loss, detached_metrics
```

Utilization:

- Logged as `train/analogy_*`, `topology/*`, and `graphcg/*`.
- Early phase: helpful if evidence shows validation BPB transfer; otherwise keep low-scale or sidecar because it can optimize representation while validation BPB stalls.
- Later phase: core objective for GoT/ToT/CoT, GFlowNet branch selection, and graph-structured memory retrieval.

### GraphCG Metrics

Implementation file: `src/toricgt/random_order_lm.py`

Purpose: learn a disentangled basis of latent edit directions.

Pseudocode:

```python
def graphcg_losses(hidden):
    codes = normalize(sample_hidden_codes(hidden))
    directions = normalize(learned_direction_basis)

    edited_near = normalize(codes + alpha * directions)
    edited_far = normalize(codes + 2 * alpha * directions)
    code_loss = cross_entropy(sim(edited_near, edited_far), matching_direction_id)

    gram = directions @ directions.T
    orthogonal_loss = offdiag_mse(gram, eye)

    coords = codes @ directions.T
    covariance_loss = offdiag_corr_mse(coords)
    sparsity_loss = mean(abs(directions))

    return code_loss + w1*orthogonal + w2*covariance + w3*sparsity
```

Utilization:

- Logged as `train/graphcg_*` and `graphcg/*`.
- Useful as a basis for analogy and topology charts.
- Should be promoted carefully after a BPB-clean checkpoint or when transfer evidence supports it.

### PolarQuant Methodology

Implementation files: `src/toricgt/polar_cache.py`, `src/toricgt/tropical_attention.py`, compact script `amelie-iska/parameter-golf/.../train_gpt.py`

Paper reference: `assets/2502.02617v1.pdf`.

Purpose: efficient K/V cache quantization via random preconditioning, recursive polar transform, and angle quantization. For the competition script, the most helpful safe use is training/evaluation perturbation of K/V vectors so the model becomes tolerant to low-bit K/V cache behavior without bloating the final artifact.

Pseudocode:

```python
def polarquant_kv(x, random_signs, bits):
    # Randomized Hadamard-style preconditioning spreads outliers.
    signed = x * random_signs
    preconditioned = hadamard(signed)

    # Recursive polar transform.
    radius = abs(preconditioned)
    angles = []
    while radius.dim_last > 1:
        left, right = pair(radius)
        angle = atan2(right, left)
        angles.append(quantize(angle, bits))
        radius = sqrt(left**2 + right**2)

    # Decode quantized angles and undo preconditioning.
    reconstructed = recursive_polar_decode(radius, angles)
    return hadamard(reconstructed) * random_signs
```

Utilization:

- Current compact competition exporter remains int8+zlib+adaptive pruning for the <=16MB final artifact.
- PolarQuant is best used as a K/V robustness method or later inference-efficiency method, not as the sole final weight compression method.
- Training-time PolarQuant must be sampled or otherwise memory-bounded because full-batch float32 K/V transforms caused an OOM in R51.

---

## Recommended Updates

### Priority 1: BPB-Transfer Controller

Add a controller that combines:

- validation BPB velocity required to hit <=1.2 by step 4000;
- train-to-validation transfer efficiency;
- advanced metric family pressure;
- historical next-validation BPB lift for each metric family;
- 16MB artifact size margin.

Recommended policy:

```python
if best_val_bpb <= 1.2 and artifact_under_16mb:
    preserve_competition_checkpoint()
    enable_post_threshold_advanced_phases()
elif artifact_total_bytes > 16_000_000:
    force_export_guard()
    keep_advanced_losses_sidecar_only()
elif family_has_positive_next_val_lift and val_velocity_shortfall_is_high:
    allow_guarded_low_scale_aux_loss(family)
elif family_has_negative_next_val_lift:
    use_family_as_sidecar_or_damping_signal()
else:
    keep_primary_bpb_clean()
```

Expected BPB impact:

- Prevents beautiful advanced losses from stealing gradient budget during the competition phase.
- Makes structural metrics actionable when they actually predict validation BPB improvement.
- Avoids repeating failed 4K branches with no evidence gate.

### Priority 2: Memory-Safe PolarQuant Training Perturbation

Add `POLARQUANT_TRAIN_SAMPLE_TOKENS` to the compact Seq4096 script and `polarquant_train_sample_tokens` to ToricGT tropical attention. During training, perturb only an evenly sampled subset of K/V tokens. During eval, perturb the full K/V tensor if enabled.

Expected BPB impact:

- Lets us test PolarQuant robustness without the R51 OOM failure mode.
- Makes low-bit K/V tolerance possible before long-context reasoning phases.
- Does not increase final artifact size because no new persistent tensors are required.

### Priority 3: Artifact-Size-Aware Metrics Policy

Add explicit controller output:

- `artifact_size_policy`;
- `artifact_size_margin_bytes`;
- `competition_phase_policy`;
- per-family recommended loss scale;
- whether the family is export-included or sidecar-only.

Expected BPB impact:

- Keeps the final OpenAI Parameter-Golf code+weights under the decimal 16,000,000 byte limit.
- Forces final threshold checkpoint preservation before larger reasoning modules are trained.

### Priority 4: Normalize Advanced Loss Pressures By BPB Causality

For every advanced family, track:

- raw pressure;
- normalized pressure;
- next-validation BPB lift when active;
- transfer efficiency;
- recommended loss scale.

Expected BPB impact:

- Avoids one-note “lower all losses” thinking.
- Promotes only metrics that improve compression of the competition distribution.

### Priority 5: Phase-Specific BPB Breakdown

Keep separate BPB surfaces:

- `competition/fineweb_bpb`;
- `reasoning/got_bpb`;
- `reasoning/tot_bpb`;
- `reasoning/cot_bpb`;
- `gflownet/test_time_scaled_bpb`;
- `memory/retrieval_bpb`;
- `analogy/transfer_bpb`;
- `long_context/8192_bpb` and `long_context/16384_bpb`.

Expected BPB impact:

- Prevents post-threshold reasoning improvements from hiding competition regression.
- Makes memory, analogy, and GFlowNet training measurable rather than philosophical.

---

## Implementation Tasks

### Task 1: Add BPB-Transfer Controller

**Files:**

- Create: `src/toricgt/bpb_transfer_controller.py`
- Test: `tests/test_bpb_transfer_controller.py`

- [x] **Step 1: Write failing controller tests**

Test supportive topology pressure before threshold, adverse toric pressure before threshold, artifact-over-limit export guard, and post-threshold advanced phase promotion.

- [x] **Step 2: Implement controller**

Implement `bpb_transfer_control_report(current_report, evidence_report, target_bpb, gate_step, artifact_size_limit_bytes)`.

- [x] **Step 3: Run tests**

Run:

```bash
pytest tests/test_bpb_transfer_controller.py -q
```

Expected: all tests pass.

### Task 2: Integrate Controller Into Seq4096 Watcher

**Files:**

- Modify: `scripts/watch_seq4096_analysis.py`
- Test: `tests/test_seq4096_analysis.py`

- [x] **Step 1: Add watcher assertions**

Assert that periodic analysis writes:

- `bpb/bpb_transfer_controller_report.json`;
- `bpb/bpb_transfer_controller_map.png`;
- controller fields in `analysis_status.json`.

- [x] **Step 2: Add watcher integration**

Call the controller after structural and historical evidence reports are computed. Write JSON, plot family recommended loss scales, and add synopsis lines.

- [x] **Step 3: Run tests**

Run:

```bash
pytest tests/test_seq4096_analysis.py -q
```

Expected: all tests pass.

### Task 3: Add Sampled PolarQuant K/V Perturbation

**Files:**

- Modify: `src/toricgt/tropical_attention.py`
- Modify: `src/toricgt/random_order_lm.py`
- Modify: `amelie-iska/parameter-golf/records/track_10min_16mb/2026-03-19_TrainingOptSeq4096/train_gpt.py`
- Test: `tests/test_tropical_attention.py`

- [x] **Step 1: Write failing tests**

Assert sampled PolarQuant changes only sampled K/V token positions in training mode and preserves output shape/finite gradients.

- [x] **Step 2: Implement sampled perturbation**

Add `polarquant_train_sample_tokens`.

Training behavior:

```python
if training and polarquant_train and sample_tokens > 0:
    idx = evenly_spaced_token_indices(seq_len, sample_tokens)
    perturbed = full_tensor.clone()
    perturbed[idx] = polarquant(full_tensor[idx])
else:
    perturbed = polarquant(full_tensor)
```

Evaluation behavior: full K/V PolarQuant perturbation when bits > 0.

- [x] **Step 3: Run tests and smoke compact script**

Run:

```bash
pytest tests/test_tropical_attention.py -q
python -m py_compile amelie-iska/parameter-golf/records/track_10min_16mb/2026-03-19_TrainingOptSeq4096/train_gpt.py
```

Expected: tests pass, compile passes.

### Live Update: 2026-06-04 R52 to R54 Gate Recovery

R52 reached step 3750 with validation BPB `1.2131`, but the periodic
analysis marked the branch `off_track_unreachable`:

- projected target step: `4161.9497`;
- recent validation drop: `0.00318 BPB / 100 steps`;
- required validation drop to hit the 4K gate: `0.00524 BPB / 100 steps`;
- velocity shortfall: `0.00256 BPB / 100 steps`;
- velocity shortfall pressure: `0.512`;
- artifact export probe: `15,877,701` bytes, margin `122,299` bytes.

Decision: restart before burning the final 250 steps. The live gate was
re-run with one-step projected-miss patience and a corrected high restart
ceiling. It launched `toricgt_seq4096_4k_recovery_r54_20260604T134941Z` from
the R52 step-3500 checkpoint, whose validation BPB was `1.2198`.

R54 recovery controls:

- `TIED_EMBED_LR=0.03808`;
- `BIGRAM_BIAS_LR=0.0162`;
- `MATRIX_LR=0.018`;
- `SCALAR_LR=0.018`;
- `MUON_MOMENTUM=0.985`;
- `TRAIN_BATCH_TOKENS=1048576`;
- reset optimizer, RNG, and data loader on resume;
- keep GraphCG/Slepian/topology/toric/BGG/Koszul losses in sidecar transfer
  until a competition checkpoint <= `1.2` BPB is preserved.

Rationale: the periodic diagnostics identify useful structural pressure, with
dominant live pressure in toric/Slepian-side metrics, but the pre-threshold
competition phase still needs clean BPB velocity. The advanced metrics should
steer branch selection, W&B diagnostics, and sidecar-transfer decisions here,
not add heavy primary loss terms before the <= `1.2` BPB checkpoint exists.

Gate automation update: `scripts/watch_seq4096_4k_recovery.py` now normalizes
stale low positive `max_restarts` values when they are paired with a high
absolute `restart_index`. For example, `restart_index=53` and `max_restarts=8`
becomes an effective ceiling of `61`, so the gate can keep launching recovery
runs instead of silently stopping at the exact moment a recovery is needed.

### Live Update: 2026-06-04 Live Checkpoint Analysis Sidecar

The live analysis scope is now attached to the up-to-date training run, not
only to historical post-resume bundles.  A Seq4096 live supervisor watches tmux
for the newest active `toricgt_seq4096_*` training session and launches one
`scripts/watch_seq4096_analysis.py` watcher per fresh recovery run.  Outputs go
under:

```text
outputs/live_periodic_reviews/<run_id>/step-XXXXXXXX
```

For R55 (`toricgt_seq4096_4k_recovery_r55_20260604T140841Z`), the live sidecar
has produced a step-3500 review and is waiting on the step-3750 checkpoint.
That review reports:

- state: `off_track_unreachable`;
- best validation BPB: `1.2198`;
- target gap: `0.0198`;
- competition-phase policy: `pre_threshold_primary_bpb_clean`;
- sidecar/tiny loss policy for GraphCG, Slepian/Pollak, contrastive, and
  GFlowNet signals; toric/BGG/Koszul/memory/trajectory losses remain held from
  the primary pre-threshold objective.

Compatibility finding: the older native
`scripts/watch_training_analysis.py -> evaluate_oai_competition_bpb.py ->
evaluate_reasoning_simplex.py -> evaluate_reasoning_geometry_suite.py` chain is
not directly safe for live Seq4096 checkpoints. Those scripts load
`RandomOrderLMConfig` / `DenseRandomOrderToricLM`; compact Seq4096 checkpoints
contain GPT-style keys such as `tok_emb.weight` and no compatible config
payload. The correct live suite is therefore `scripts/watch_seq4096_analysis.py`
plus FineWeb diagnostics, W&B metric analysis, BPB descent plots, artifact-size
probes, structural proxy maps, and training-adjustment proposals.

Implemented follow-up: `scripts/evaluate_seq4096_competition_bpb.py` now loads
compact GPT-style Seq4096 checkpoints directly, infers architecture from tensor
shapes, and emits the expected OAI aliases:

- `oai_competition/bpb`;
- `competition/oai_bpb`;
- `bpb/oai_competition`;
- `seq4096/oai_competition_bpb`.

`scripts/watch_seq4096_analysis.py` runs this evaluator as a cheap sampled CPU
checkpoint probe and writes `oai_competition/seq4096_summary.json` for future
live reviews. The full validation BPB from the trainer/gate remains the
authoritative pre-threshold competition metric; sampled sidecar BPB is a load
and aliasing sanity check unless `val_max_sequences=0` is used for a full
evaluation.

Live-analysis correction: the sidecar is intended to review up-to-date training
checkpoints during training, not stale completed analysis folders. The active
R56 live watcher now analyzes
`outputs/live_periodic_reviews/toricgt_seq4096_4k_recovery_r56_20260604T142900Z/step-00003750`
from the current run's own checkpoint and then waits for step 4000. The live
supervisor now also parses `resume_checkpoint:..._step_N.pt` before the trainer
emits `checkpoint_resumed:... step:N`, which prevents resumed runs from
accidentally launching their analysis watcher at step 0 during warmup. This is
the behavior required for sub-agent/sidecar recommendations to track current
checkpoint evidence.

Fresh R56 step-3750 sidecar readout:

- authoritative trainer validation BPB: `1.2131`;
- train BPB in W&B summary: about `1.1846`;
- sampled compact OAI probe BPB: `1.1972` with `eval_scope=sampled`, `seq_len=256`,
  and `val_max_sequences=1`;
- state: `off_track_unreachable` for the step-4000 <=1.2 BPB gate;
- action: keep R56 running to the 4000 checkpoint while the gate remains armed
  to relaunch from the best checkpoint if the authoritative validation BPB misses
  the threshold. Treat the sampled compact probe as an alias/load sanity signal,
  not as the gate metric.

Advanced visualization correction: the live Seq4096 sidecar now includes a
compact-checkpoint-compatible advanced geometry runner:

```text
scripts/evaluate_seq4096_reasoning_geometry_suite.py
```

It writes the old-suite review surface into the current checkpoint's analysis
directory:

- `geometry/reasoning_geometry_summary.json`;
- `geometry/reasoning_geometry_records.{json,csv}`;
- `geometry/selected_records.json`;
- `geometry/trajectories/*trajectory_3d.png`, `*energy_landscape.png`,
  `*phase_energy.png`, `*toric_phase_simplicial_trajectory.png`,
  `*toric_phase_winding_collection.png`, and companion HTML files;
- `geometry/topology/*directed_filtration.png`,
  `*step_radius_hierarchy.png`, `*noncommutative_heatmaps.png`,
  `*exact_persistence_morphisms.png`,
  `*commutative_algebra_audit.png`, `*toric_shadow_audit.png`, and
  `*toric_slepian_audit.png`;
- `geometry/graphcg/*graphcg_basis_disentanglement.png`;
- `geometry/analogical/*analogical_transport_map.png`;
- `geometry/tropical/*tropical_chamber_audit.png`;
- `geometry/triangles/*.png` and `geometry/tetrahedra/*.{png,html}`;
- `simplex/reasoning_simplex_summary.json`;
- `simplex/reasoning_simplex_records.{json,csv}`;
- `simplex/*triangle.png` and `simplex/*tetrahedron.{png,html}`;
- `artifact_inventory.json` and `analysis_review_prompt.md` for Codex/sub-agent
  image-by-image review.

The runner labels its source as `seq4096_checkpoint_embedding_proxy`: it uses
current compact checkpoint tensors, embedding-space trajectories, layer-control
paths, attention-transport statistics, and bigram memory-graph surfaces. It is
not a fake placeholder, but it is not yet a generated hidden-state trace from a
compact GPT forward pass. The next recommended analysis upgrade is a compact
hidden-state adapter that registers hooks on the Seq4096 GPT blocks and feeds
actual validation-token hidden trajectories into the same plotting surface.

Manual current-run R56 step-3750 execution produced 93 current-run images and
120 reviewable output files under:

```text
outputs/live_periodic_reviews/toricgt_seq4096_4k_recovery_r56_20260604T142900Z/step-00003750
```

Summary signals:

- GraphCG disentanglement score mean: `0.7294` (desired);
- Slepian concentration mean: `0.7261`, leakage mean `0.2739` (desired);
- toric fan-cell entropy mean: `0.8195` (desired);
- topology directed asymmetry mean: `0.4842` (useful noncommutative signal);
- tropical chamber crossing rate mean: `0.8981` (high, monitor as sidecar);
- BGG standard leakage mean: `0.1111` and Koszul d2 residual `0.0` (desired);
- analogical map score mean: `0.2181` (desired direction but too weak).

Training policy from this analysis: do not inject advanced losses before the
4K BPB gate. Keep the competition phase BPB-clean and use these structural
metrics as sidecar evidence for restarts/controllers. After a <=1.2 checkpoint
is preserved, prioritize analogical pair and graph-structured memory curricula,
because analogical transport is the weakest current advanced surface.

Compatibility correction: `scripts/evaluate_oai_competition_bpb.py` now detects
compact Seq4096 GPT checkpoints and delegates to the compact evaluator while
preserving the historical metric aliases. This keeps the familiar analysis
script usable for both RandomOrderLM and compact Seq4096 checkpoint families.

Live-analysis inventory correction: `scripts/watch_seq4096_analysis.py` now
refreshes `artifact_inventory.json` and `analysis_review_prompt.md` after the
full periodic sidecar stack has completed. This is important because the compact
advanced-geometry runner writes its inventory before later BPB, W&B metric,
sampled OAI, proposal, status, and hook-log artifacts exist. The final inventory
now scans the current analysis directory and includes every reviewable PNG,
JSON, CSV, Markdown, HTML, text, and log file except the inventory itself. This
prevents the sub-agent review queue from silently missing plots such as
`bpb/bpb_descent_timeseries.png`, `bpb/bpb_transfer_efficiency.png`,
`metrics/core_metric_timeseries.png`, and `metrics/recent_metric_slopes.png`.

Resume-checkpoint live-analysis correction: resumed recovery branches begin
with a saved checkpoint at step 3750, but the old watcher computed the first
target as `start_step + interval_steps`, so preempted branches such as R60 and
R61 could leave empty live-review directories until they survived to step 4000.
`scripts/watch_seq4096_analysis.py` now supports `--analyze-start-step`, and
both the live supervisor and 4K recovery launcher pass it. R62 was restarted
with this flag and now has complete live outputs for both `step-00003750` and
`step-00004000`, each with 177 reviewable files and 112 images. One-shot
backfills were also launched for R60 and R61 step 3750 so the formerly empty
directories become viewable.

R56/R57/R58 gate update: R56 reached the step-4000 gate with authoritative
validation BPB `1.2163`, so the <=1.2 BPB threshold was **not** reached. The
gate correctly relaunched R57 from the R56 step-3750 checkpoint, which remained
the best validation checkpoint at `1.2131`. R57 then reached the step-4000 gate
with authoritative validation BPB `1.2299`, so it also missed. The gate launched
R58 from the R57 step-3750 checkpoint, again preserving the best validation BPB
`1.2131` as the recovery origin. R58 uses optimizer/RNG/loader reset,
`TRAIN_BATCH_TOKENS=983040`, tied embedding LR `0.034`, matrix/scalar LR
`0.018`, Muon momentum warmup to `0.985`, validation/checkpoint interval 250,
and low-rate bigram bias. The policy remains BPB-clean before threshold: use
advanced metrics as sidecar evidence and controller priors, not heavy auxiliary
losses, until an authoritative <=1.2 competition checkpoint is preserved.

R58 replay-diversity correction: early R58 train BPB reproduced R57 almost
exactly (`3800 -> 1.2424`, `3850 -> 1.2101`), so the gate controller now lets
damped low-LR train-wave analogues preempt instead of holding them for another
known-miss validation probe. The controller still gives hot-velocity probes one
validation readout, but a damped branch that matches prior failed train waves
now needs only one completed failed replay to switch to
`failed_train_wave_damped_transfer_probe`. The bigram-bias LR damping was also
corrected so an already-low LR such as `0.01` damps to `0.006` rather than
being raised to the old floor. This is still BPB-clean: it changes
optimizer/transition-bias controls, not auxiliary topology/toric/BGG losses.

R59 launch: after the patch was loaded into the R58 gate process, the controller
preempted R58 at step 3900 because its train-wave profile matched failed R57
with RMSE about `5.8e-05`. R59 launched from the R58 step-3750 checkpoint with
`failed_train_wave_damped_transfer_probe`: tied embedding LR `0.032`, bigram
bias LR `0.006`, matrix/scalar LR `0.018`, train batch tokens `983040`, and
Muon warmup steps `4750`. Early train BPB moved off the exact R57/R58 replay:
R59 has `3800 -> 1.2412` and `3850 -> 1.2087`, versus `1.2424` and `1.2101`
for the repeated branch. R59 still missed the authoritative step-4000 gate with
validation BPB `1.2288`, so the threshold was not reached and the best preserved
origin remained step 3750 at `1.2131`.

R60 launch: the gate relaunched
`toricgt_seq4096_4k_recovery_r60_20260604T153456Z` from the R59 step-3750
checkpoint with optimizer/RNG/loader reset, tied embedding LR `0.032`, bigram
bias LR `0.006`, matrix/scalar LR `0.018`, train batch tokens `983040`, seed
`7391`, and Muon warmup steps `4750`. W&B is enabled for R60, and its live
periodic analysis watcher is attached to the current checkpoint directory.
R60's first post-resume train point (`3800 -> 1.2413`) is already nearly the
same basin as R59 (`3800 -> 1.2412`), so the controller now has a second-stage
`repeated_damped_train_wave_diversity_probe`. If a run at the damped floor
(`tied_embed_lr <= 0.0325`, `bigram_bias_lr <= 0.0065`) matches a prior failed
train wave, the next branch lowers tied/matrix/scalar/bigram LR, lowers batch
tokens to `917504`, increases bigram-bias scale to at least `1.15`, and extends
Muon warmup. This remains BPB-clean: it changes optimizer and transition-bias
controls rather than adding heavy topology/toric/BGG/Koszul/GraphCG losses
before the <=1.2 checkpoint is preserved.

R61/R62 update: the old R60 gate watcher launched R61 with the previous damped
floor controls before the second-stage controller patch was active. After
restarting the R61 gate watcher with the patched code, R61 matched failed R59
exactly at step 3850 (`3800 -> 1.2412`, `3850 -> 1.2087`, train-wave RMSE
`0.0`) and was preempted. R62 is now active:
`toricgt_seq4096_4k_recovery_r62_20260604T155056Z`. It resumes from the R61
step-3750 checkpoint with `train_batch_tokens=917504`,
`tied_embed_lr=0.03008`, `matrix_lr=scalar_lr=0.01656`,
`bigram_bias_lr=0.0042`, `bigram_bias_scale=1.15`, seed `7393`, and Muon
warmup steps `5000`. W&B and the live full periodic analysis watcher are
enabled for R62. Stale R61 sidecars were stopped so the active reviewer queue
follows R62.

R62/R63 update: R62 changed the train trajectory materially (`3800 -> 1.1871`,
`3850 -> 1.2055`, `3900 -> 1.2186`, `3950 -> 1.2153`, `4000 -> 1.2001`) but
still missed the authoritative gate with validation BPB `1.2266` at step 4000.
The advanced full-diagnostics sidecar at the miss reported high structural
pressure: dominant family `toric_slepian`, structural recapture score about
`0.774`, topology loss about `1.22`, directed topology loss about `0.150`,
Slepian leakage `1.0`, toric active-face margin about `-1.965`, BGG standard
leakage about `0.521`, and BGG `d^2` residual about `0.0191`. This is now
stronger evidence that the next branches may need guarded use of the advanced
structural theory rather than only optimizer replay-diversity controls. The
gate launched R63 from the best R62 step-3750 checkpoint with the same
BPB-clean diversity controls while preserving the `1.2131` recovery origin.

Recommended follow-up for the advanced-analysis track: implement compact
Seq4096 analogues of the old simplex/geometry entrypoints so graph-of-thought
trajectory, directed simplicial, toric, Slepian/Pollak, BGG/Koszul, and memory
diagnostics can run directly on compact checkpoints without converting them
into RandomOrderLM payloads.

PolarQuant and larger-parameter policy: R52's int8+zlib export margin is tight.
PolarQuant currently quantizes runtime K/V cache tensors and can reduce memory
pressure for longer contexts, but it does not by itself shrink stored model
weights. Larger-parameter competition variants are allowed only after a
byte-accounted stored-weight export path proves code plus weights stay under
`16,000,000` bytes and the variant improves held-out BPB.

### Task 4: Keep Live Early-Phase Training Optimized

**Files:**

- No code file required unless controller recommends a relaunch.
- Inspect: current R52 logs, W&B, checkpoint exports, periodic analysis outputs.

- [x] **Step 1: Monitor R52**

Check train/val BPB, W&B sync, checkpoint saving, GPU memory, and 16MB export report.

- [x] **Step 2: If R52 misses the required gate velocity**

Use best checkpoint at or before 3250/3500/3750, then launch the controller-recommended branch. Prefer BPB-clean controls before structural losses:

- tied embedding LR 0.034 to 0.037 range;
- matrix/scalar LR 0.018 range;
- full validation every 250 steps;
- bigram bias enabled, low LR;
- PolarQuant sampled only if memory headroom is proven.
- larger-parameter variants only after export byte accounting proves a
  stored-weight compression path, because PolarQuant K/V cache compression does
  not shrink model weights.

Current action update (2026-06-04): R52 missed the projected gate velocity at
step 3750, R54 and R55 did not improve the authoritative validation gate, R56
missed step 4000 at `val_bpb=1.2163`, R57 missed at `1.2299`, R59 missed at
`1.2288`, and R62 missed at `1.2266`. R60/R61/R63/R64/R65 repeatedly replayed
the same step-3750 recovery origin (`val_bpb=1.2131`) and matched failed
train-wave analogues by step 3850. This is now enough evidence to begin guarded
advanced-methodology experimentation before the final <=1.2 checkpoint: the
Seq4096 trainer has compact auxiliary losses, disabled by default, for GraphCG
embedding-basis disentanglement, toric/tropical chamber pressure, Slepian/Pollak
trajectory concentration, Koszul/BGG exactness, and analogical transport
consistency. The recovery controller promotes these only after repeated damped
replay, using `ADVANCED_LOSS_SCALE=0.20` with low per-family weights
(`GRAPHCG=0.05`, `TORIC_TROPICAL=0.03`, `SLEPIAN=0.02`, `KOSZUL_BGG=0.01`,
`ANALOGY=0.01`) so BPB cross-entropy remains dominant. W&B reports
`train/advanced_aux_loss` plus each `advanced/*` component; `train/bpb` remains
computed from the base cross-entropy, not the auxiliary objective.

Live analysis update: recovery-launched live watchers must include
`--analyze-start-step` so the step-3750 resume checkpoint is analyzed before a
short recovery is preempted. The compact geometry suite now renders static
energy-landscape PNGs as real 3D PCA/local-energy plots, keeps the existing HTML
companions, uses dark-mode styling where possible, and restores filled
reasoning-simplex triangle heatmaps via barycentric RBF interpolation instead of
sparse point-only triangles. These outputs should be reviewed at every live
periodic interval and correlated with W&B advanced-loss metrics before
increasing any auxiliary scale.

Advanced-branch launch update (2026-06-04): the gate has now promoted from
sidecar-only structural diagnostics to a live guarded advanced-loss recovery,
`toricgt_seq4096_4k_recovery_r67_20260604T163811Z`, from the best step-3750
checkpoint (`val_bpb=1.2131`). This branch enables GraphCG, toric/tropical,
Slepian/Pollak, Koszul/BGG, and analogy auxiliary losses with
`ADVANCED_LOSS_SCALE=0.20` while preserving the BPB cross-entropy objective as
the primary training loss. Because this branch is within 250 steps of the
competition gate, the active command was tightened to `VAL_LOSS_EVERY=50` and
`CHECKPOINT_EVERY=50`; the recovery controller now applies that 50-step
observation cadence automatically to future advanced-loss restarts while
baseline restarts remain at 250. Review each 50-step validation/checkpoint for
BPB transfer, `train/advanced_aux_loss`, and the individual `advanced/*`
component metrics before increasing auxiliary scale.

- [ ] **Step 3: If <=1.2 BPB is reached**

Save immutable threshold checkpoint, export with int8+zlib+adaptive pruning, verify code+weights <=16,000,000 bytes, then start post-threshold advanced reasoning/memory phases.

### Task 5: Post-Threshold Advanced Training Phases

**Files:**

- Future config updates under `configs/`
- Future README updates after a threshold checkpoint or stable controller configuration.

- [ ] **Step 1: Enable graph-of-thought and memory token curricula**

Use existing reasoning and memory special tokens where tokenizer/model shape permits.

- [x] **Step 2: Enable GraphCG, analogy, topology, BGG, Koszul, toric, tropical, Slepian/Pollak losses by controller scale**

Start from low weights and promote families with positive held-out BPB transfer.
Initial compact Seq4096 implementation is training-module-free and
byte-accounted: it adds no stored parameters, so the competition artifact weight
size is unchanged. Next updates should compare the first advanced-loss branch
against R62-R66 replay baselines and only raise `ADVANCED_LOSS_SCALE` if
validation BPB improves before or at the 4K gate.

- [ ] **Step 3: Add long-context tropical ring phase**

Scale from 4096 to 8192 and beyond only after BPB checkpoint preservation.

---

## Success Criteria

- W&B reports train BPB, validation BPB, artifact size, and controller outputs.
- Periodic analysis creates controller JSON and chart outputs.
- Compact competition export remains <=16,000,000 bytes.
- PolarQuant can be enabled in sampled training mode without OOM.
- Advanced losses are promoted only when they improve validation BPB transfer or after the competition threshold checkpoint is preserved.
