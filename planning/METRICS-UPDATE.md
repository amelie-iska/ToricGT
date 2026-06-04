# Metrics And Losses Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ToricGT's advanced BGG, Koszul, topology, toric, tropical, analogical, and PolarQuant diagnostics actively improve OpenAI Parameter-Golf FineWeb BPB while preserving a <=16MB competition artifact.

**Architecture:** Keep cross-entropy and OpenAI FineWeb BPB as the primary early objective. Treat advanced mathematical losses as evidence-gated sidecars before the <=1.2 BPB checkpoint exists, then promote them into larger post-threshold reasoning and memory phases. Add a BPB-transfer controller that measures whether each advanced metric family historically predicts the next validation BPB drop and emits recommended loss scales, artifact-size policy, and phase controls.

**Tech Stack:** PyTorch, W&B, matplotlib, pandas/numpy, existing ToricGT modules under `src/toricgt`, Seq4096 watcher scripts under `scripts`, and compact Parameter-Golf training script under `amelie-iska/parameter-golf/records/.../train_gpt.py`.

## Current Live Update - 2026-06-04 R52

- Active branch: `oai-advanced` in both ToricGT and the nested
  Parameter-Golf repo.
- Active BPB run: `toricgt_seq4096_warmdown_r52_20260604T125517Z`.
- Latest completed validation: step 3500, validation/OpenAI BPB `1.2198`,
  down from `1.2454` at step 3000 and `1.2290` at step 3250.
- Latest gate projection: target step `3886.71875`, recent drop `0.00512` BPB
  per 100 steps, required drop `0.00396` BPB per 100 steps.
- Controller policy at step 3500: `pre_threshold_primary_bpb_clean`.
  Keep `bpb_gap` at scale `1.0`; keep BGG/Koszul, topology, toric, tropical,
  Slepian/Pollak, GraphCG, memory, and analogy families as sidecar/damping
  evidence until either the <=1.2 threshold checkpoint is preserved or the
  controller sees positive held-out BPB transfer.
- Artifact policy: under the 16,000,000 byte cap but extremely tight.  The
  manual step-3500 export probe reported `15,996,978` total bytes, leaving only
  `3,022` bytes of margin after 8% export pruning, so do not add exported probes,
  code bloat, or larger stored weights before preserving the threshold
  checkpoint.
- PolarQuant/larger-parameter policy: PolarQuant currently quantizes K/V cache
  tensors, not stored model weights.  Larger-parameter retraining is permitted
  only as a separate sidecar branch if a true stored-weight/export compression
  path proves code+weights remain under `16,000,000` bytes and improves held-out
  BPB.  It should not interrupt R52 while R52 remains on-track.
- Training action: do not restart R52 before step-3750 validation unless the
  gate watcher reports a material projected miss.  If step-3750 validation
  misses velocity, restart from the best 3000/3250/3500/3750 checkpoint with
  BPB-clean controls before adding structural loss weight.

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

### Task 4: Keep Live Early-Phase Training Optimized

**Files:**

- No code file required unless controller recommends a relaunch.
- Inspect: current R52 logs, W&B, checkpoint exports, periodic analysis outputs.

- [x] **Step 1: Monitor R52**

Check train/val BPB, W&B sync, checkpoint saving, GPU memory, and 16MB export report.

- [ ] **Step 2: If R52 misses the required gate velocity**

Use best checkpoint at or before 3250/3500/3750, then launch the controller-recommended branch. Prefer BPB-clean controls before structural losses:

- tied embedding LR 0.034 to 0.037 range;
- matrix/scalar LR 0.018 range;
- full validation every 250 steps;
- bigram bias enabled, low LR;
- PolarQuant sampled only if memory headroom is proven.
- larger-parameter variants only after export byte accounting proves a
  stored-weight compression path, because PolarQuant K/V cache compression does
  not shrink model weights.

- [ ] **Step 3: If <=1.2 BPB is reached**

Save immutable threshold checkpoint, export with int8+zlib+adaptive pruning, verify code+weights <=16,000,000 bytes, then start post-threshold advanced reasoning/memory phases.

### Task 5: Post-Threshold Advanced Training Phases

**Files:**

- Future config updates under `configs/`
- Future README updates after a threshold checkpoint or stable controller configuration.

- [ ] **Step 1: Enable graph-of-thought and memory token curricula**

Use existing reasoning and memory special tokens where tokenizer/model shape permits.

- [ ] **Step 2: Enable GraphCG, analogy, topology, BGG, Koszul, toric, tropical, Slepian/Pollak losses by controller scale**

Start from low weights and promote families with positive held-out BPB transfer.

- [ ] **Step 3: Add long-context tropical ring phase**

Scale from 4096 to 8192 and beyond only after BPB checkpoint preservation.

---

## Success Criteria

- W&B reports train BPB, validation BPB, artifact size, and controller outputs.
- Periodic analysis creates controller JSON and chart outputs.
- Compact competition export remains <=16,000,000 bytes.
- PolarQuant can be enabled in sampled training mode without OOM.
- Advanced losses are promoted only when they improve validation BPB transfer or after the competition threshold checkpoint is preserved.
