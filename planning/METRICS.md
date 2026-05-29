# ToricGT OAI Metrics Audit

Date: 2026-05-27 UTC.

Training was paused from tmux session `toricgt_pg_oai`.  No training process is currently running.  The latest retained periodic checkpoint is

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00014750.pt
```

The W&B run analyzed here is

```text
https://wandb.ai/amelie-iska-math/toricgt-parameter-golf/runs/8v4wnovf
```

The run now appears as `crashed` in W&B because it was intentionally interrupted for this audit.  The last W&B history step is `14950`; the latest optimizer checkpoint is step `14750`.

## Output Artifacts

The metric-analysis outputs are in:

```text
outputs/metrics_analysis/oai-8v4wnovf-step14750/
```

Important files:

```text
wandb_history.csv
wandb_history.parquet
metric_stats.csv
metric_stats.json
category_summary.json
automatic_metrics_report.md
core_metric_timeseries.png
metric_category_counts.png
recent_metric_slopes.png
selected_metric_correlations.png
validation_recheck.json
```

The reasoning-geometry outputs are in:

```text
outputs/reasoning_simplex/oai-8v4wnovf-step14750-val-seed10017/
```

Important files:

```text
reasoning_simplex_records.json
reasoning_simplex_records.csv
reasoning_k_bpb_triangle.png
robustness_triangle.png
efficiency_triangle.png
reasoning_k_bpb_mst_tetrahedron.png
reasoning_k_bpb_mst_tetrahedron.html
reasoning_k_bpb_diversity_tetrahedron.png
reasoning_k_bpb_diversity_tetrahedron.html
```

The first simplex run with the default seed is preserved at

```text
outputs/reasoning_simplex/oai-8v4wnovf-step14750/
```

but it should not be used for training decisions because it did not use the trainer's validation seed.

## Statistical Methods

For every numeric W&B metric with at least two observations, the audit computes:

1. Median-window change.  If \(y_t\) is a metric and the window width is \(w=\lceil 0.1n\rceil\), the reported relative change is
   \[
   \Delta_{\mathrm{rel}}
   =
   \frac{\operatorname{median}(y_{n-w:n})-\operatorname{median}(y_{1:w})}
        {\max(|\operatorname{median}(y_{1:w})|,10^{-9})}.
   \]
2. Global and recent OLS slopes.  Steps are measured in thousands:
   \[
   y_t = \alpha+\beta\frac{s_t-s_0}{1000}+\epsilon_t.
   \]
   The table reports \(\beta\) and its usual OLS \(t\)-statistic.  Recent slopes use the final 30 percent of available points.
3. Spearman rank correlation \(\rho_s(s,y)\), to detect monotone behavior that need not be linear.
4. Spike count from median-absolute-deviation normalized first differences:
   \[
   z_t=0.6745\frac{\Delta y_t-\operatorname{median}(\Delta y)}
              {\operatorname{MAD}(\Delta y)}.
   \]
   A spike is \(|z_t|>6\).
5. Coefficient of variation for stable diagnostics:
   \[
   \operatorname{CV}(y)=\frac{\operatorname{std}(y)}{\max(|\operatorname{mean}(y)|,10^{-9})}.
   \]

The automatic classifier labels metrics as:

- `as_desired`
- `as_desired_but_not_strong_or_fast_enough`
- `not_as_desired`

The full metric-by-metric table is `metric_stats.csv`.  The human conclusions below override the automatic labels where the statistic is technically correct but semantically misleading, for example near-constant low-amplitude diagnostic standard deviations.

## Summary Counts

Automatic categories over 210 numeric metrics:

| category | count |
|---|---:|
| `as_desired` | 110 |
| `as_desired_but_not_strong_or_fast_enough` | 53 |
| `not_as_desired` | 47 |

These counts should be read as a screening device, not a final scientific conclusion.  Many `*_std` metrics with tiny denominators are over-sensitive under coefficient-of-variation tests.

## Core Metrics

| metric | category | evidence | interpretation |
|---|---|---|---|
| `artifact/initial_bytes` | as desired | constant at `11,416,300` bytes | Good.  There is substantial headroom under the 16,000,000-byte cap. |
| `artifact/deployment_parameters` | as desired | constant at `14,816,001` deployable parameters | Good.  Model size is close enough to the allowed budget to be competitive without exceeding it. |
| `audit/future_permutation_logit_error` | as desired | exactly `0` in all audit records | Good.  The random-order autoregressive implementation is not using future tokens in the audited setting. |
| `train/bpb` | as desired but not fast enough | median `5.0465 -> 4.8937`, global slope \(-0.01596\)/1k, recent slope \(+0.01043\)/1k | Training improves, but the final segment is flattening or slightly regressing. |
| `train/loss_ema` | as desired but not fast enough | median `3.5090 -> 3.3842`, global \(t=-48.6\), recent \(t=+4.97\) | Strong early learning followed by late-stage upward drift. |
| `train/grad_norm` | as desired | median `0.6566 -> 0.2315` | The automatic stable-metric label flagged movement, but a falling clipped gradient norm here indicates optimization stabilization, not failure. |
| `train/gflownet_loss` | as desired but not fast enough | median `1.7197 -> 1.6816`, only `-2.21%` | It is moving in the right direction, but weakly. |
| `train/gflownet_entropy` | as desired but not useful enough | near \(\log 16=2.77259\) | The policy is non-collapsed, but it is almost uniform.  That protects diversity but weakens directed reasoning. |
| `train/gflownet_action_diversity` | as desired but not useful enough | around `0.9986` | Diversity is excellent; task-conditioned selectivity is likely too weak. |
| `train/toric_memory_entropy` | not as desired | median `0.1692 -> 0.1072`, last `0.0886` | Toric memory usage has partially collapsed.  Recent slope is positive, so it may be recovering, but the level is too low. |
| `train/trajectory_flow_loss` | not as desired | median `0.0552 -> 0.0689`, up `24.7%` | Embedding trajectories became more viscous/irregular across the run.  Recent slope is improving, but the global trend is bad. |
| `train/trajectory_viscous_dissipation` | not as desired | median `0.0543 -> 0.0677`, up `24.6%` | Same diagnosis as flow loss.  The reasoning dynamics are not yet energy-efficient. |
| `train/smear_temperature` | as desired but monitor | median `0.9537 -> 1.7195`, close to configured max `1.75` | Calibration softened aggressively.  This can prevent overconfidence, but a cap-saturated gate can hide weak logits. |
| `train/qat_loss` | not as desired but low-risk | absolute value remains around \(10^{-6}\) to \(10^{-5}\) | It rises, but the weight is \(10^{-6}\).  This is not currently a primary failure mode. |
| `complexity/val/argmax_byte_accuracy` | as desired | `0.1558 -> 0.1655`, up `6.27%` | Byte-level argmax accuracy is improving on the fixed small validation diagnostic. |
| `complexity/val/bpb` | as desired but not fast enough | `4.9180 -> 4.8489`, down `1.4%` | Direction is good, but the diagnostic is a small fixed batch and cannot be the sole gate. |
| `complexity/val/prediction_target_NCD_lzma_mean` | not strong enough | `0.9270 -> 0.9315`, up `0.48%` | LZMA says predictions are slightly less target-aligned late in the run. |
| `complexity/val/prediction_target_NCD_zlib_mean` | as desired but not fast enough | `1.0179 -> 1.0097`, down `0.81%` | Zlib says alignment improves.  The estimator disagreement is a warning to use both and average over more samples. |
| W&B `val/bpb` | not as desired as logged | `7.2546 -> 8.4711`, up `16.8%` | This series is inconsistent with reproducible checkpoint evaluation.  Treat it as a validation-protocol defect until fixed. |

## Validation Recheck

The W&B `val/bpb` history is bad as logged, but a deterministic checkpoint re-evaluation using the trainer validation seed gives a different result:

| check | BPB | loss | notes |
|---|---:|---:|---|
| 4-batch, no score-first bias, one GFlowNet sample | `4.5906` | `3.1820` | checkpoint step 14750 |
| 4-batch, score-first bias `0.025`, two GFlowNet samples | `4.5870` | `3.1795` | small improvement |
| 20-batch trainer-equivalent recheck | `4.5106` | `3.1265` | best local recheck |

The W&B `val/bpb` series therefore should not drive checkpoint promotion until the evaluator is made deterministic and auditable.  The likely causes are not model weights alone: persistent validation iterator state, appended logs from multiple sessions, and a tiny complexity batch are producing mutually inconsistent validation views.

## Mathematical Explanations

### BPB And Cross-Entropy

The Parameter-Golf objective is
\[
\operatorname{BPB}(q)
=-\frac{1}{N_{\mathrm{bytes}}}\sum_t \log_2 q(b_t\mid b_{<t}).
\]
Training loss is the same quantity up to the conversion factor \(\log 2\) when the token units are bytes.  The observed training BPB decrease means the model distribution \(q_\theta\) is moving closer to the empirical byte distribution.  The late positive slope means either the learning rate is too high for the current basin, the curriculum distribution shifted faster than the model could adapt, or auxiliary losses are drawing capacity away from byte prediction.

### Validation Disagreement

Let \(P_{\mathrm{train}}\), \(P_{\mathrm{val,reset}}\), and \(P_{\mathrm{val,stream}}\) denote the training distribution, a deterministic reset validation distribution, and a persistent-stream validation distribution.  W&B `complexity/val/bpb` estimates a small fixed slice of \(P_{\mathrm{val,reset}}\), while W&B `val/bpb` appears to estimate a different stream and is not reproducible from the checkpoint.  The current estimator is therefore high-variance and possibly path-dependent:
\[
\widehat L_{\mathrm{val}}=\frac{1}{m}\sum_{i=1}^m \ell(x_i,\pi_i)
\quad\text{with}\quad
x_i,\pi_i\sim \widehat P_{\mathrm{iterator}}.
\]
Until the iterator, seed, number of batches, and pass ids are frozen and logged, the validation signal is not a reliable promotion gate.

### GFlowNet Entropy

There are 16 GFlowNet actions, so a uniform policy has entropy
\[
H_{\max}=\log 16=2.77259.
\]
The observed entropy is essentially \(H_{\max}\).  This is useful because it prevents mode collapse, but it also means the policy is close to random.  Trajectory balance can only improve reasoning if reward gradients distinguish actions:
\[
\mathcal L_{\mathrm{TB}}
=\left(\log Z+\sum_t\log P_F(a_t\mid s_t)-\log R(x)-\sum_t\log P_B(s_t\mid s_{t+1})\right)^2.
\]
If \(R(x)\) is weakly coupled to byte prediction or graph-quality improvements, entropy remains high and directed reasoning does not emerge.

### Toric Memory Entropy

For toric memory slot weights \(p_j\), the entropy is
\[
H_{\mathrm{toric}}=-\sum_j p_j\log p_j.
\]
The drop from `0.169` to `0.107` indicates slot concentration.  In the noncommutative-torus interpretation, too little entropy means phase memory is not spanning enough projective channels; it is closer to a single preferred character than a useful Weyl-pair memory.  That weakens the intended algebraic inductive bias.

### Trajectory Flow And Viscosity

The embedding-space trajectory terms approximate a discrete action functional.  If \(h_t\) is the hidden trajectory,
\[
E_{\mathrm{kin}}=\sum_t\|h_{t+1}-h_t\|^2,
\qquad
E_{\mathrm{visc}}=\nu\sum_t\|\Delta h_t\|^2.
\]
Rising viscous dissipation means trajectories are becoming less smooth or more curved under the current curriculum.  Some curvature is useful for reasoning, but rising dissipation without BPB improvement means the model is spending compute on movement that is not improving the predictive distribution.

### Kolmogorov Proxies And NCD

True \(K(x)\) is uncomputable, so the audit uses compressor-indexed proxies
\[
K_C(x)=|C(x)|,
\qquad
\operatorname{NCD}_C(x,y)
=\frac{K_C(xy)-\min(K_C(x),K_C(y))}
       {\max(K_C(x),K_C(y))}.
\]
For predictions \(\hat y\) and targets \(y\), lower \(\operatorname{NCD}_C(\hat y,y)\) is better.  Zlib improves while LZMA slightly worsens, meaning the model is learning some local byte regularities but not yet enough longer-range structure.  This is exactly the regime where graph-projected context and reasoning traces should help, but only if their reward is coupled to likelihood.

### Simplex And Tetrahedron Diagnostics

For triangle vertices \(v_r,v_k,v_b\) and normalized nonnegative scores \(r,k,b\), the plotted point is
\[
z_\triangle
=
\frac{(\epsilon+r)v_r+(\epsilon+k)v_k+(\epsilon+b)v_b}
     {3\epsilon+r+k+b}.
\]
The tetrahedron is analogous with a fourth score, usually MST efficiency or GFlowNet diversity.

Trainer-aligned checkpoint simplex results:

| budget | BPB | K proxy | MST efficiency | reasoning score | interpretation |
|---:|---:|---:|---:|---:|---|
| 1 | `4.5903` | `65.0` | `0.5491` | `0.0000` | Best BPB, least reasoning compute. |
| 2 | `4.5932` | `277.2` | `0.5854` | `0.2915` | More complex and geometrically cleaner, but BPB slightly worse. |
| 4 | `4.5951` | `277.1` | `0.5971` | `0.9968` | Highest MST efficiency, but BPB worsens. |
| 8 | `4.5935` | `278.4` | `0.5964` | `1.0000` | Similar to budget 4; no BPB gain. |

The plots show compute moving the model toward the \(K(x)\), reasoning, and MST-efficiency vertices, but not toward the low-BPB vertex.  That is the central inference-time scaling issue: extra reasoning is structured, but not yet likelihood-improving.

### MST Interpretation

For hidden states \(h_1,\dots,h_T\), form a complete weighted graph with
\[
w_{ij}=\|h_i-h_j\|_2.
\]
The minimum spanning tree weight \(W_{\mathrm{MST}}\) is a minimal skeleton length for the trajectory cloud.  The efficiency score used here is
\[
\eta_{\mathrm{MST}}=\frac{1}{1+\bar w_{\mathrm{MST}}},
\]
where \(\bar w_{\mathrm{MST}}\) is mean MST edge length.  Higher budget improves \(\eta_{\mathrm{MST}}\), so the model is organizing hidden trajectories more tightly.  Because BPB does not improve, MST efficiency should become a secondary reward only when paired with log-likelihood improvement.

## Three-Tier Behavioral Diagnosis

### As Desired

The following behaviors are good and should be preserved:

1. Artifact size and deployment parameter count are stable and comfortably under the byte cap.
2. Causal future-token audit is exactly zero.
3. Training BPB and loss improved materially from the resumed step-1000 run.
4. Gradient norms fell, indicating optimization stabilization.
5. Byte argmax accuracy on the fixed validation complexity slice improved.
6. GFlowNet action diversity remains high.
7. Score-first bias adaptation gives a small but legal validation improvement.
8. MST efficiency improves with larger reasoning budgets.

### As Desired But Not Strong Or Fast Enough

The following are directionally useful but too weak:

1. Training BPB has only improved about 3 percent over the analyzed run segment, and the recent slope is positive.
2. Complexity validation BPB improves only about 1.4 percent.
3. GFlowNet loss improves only about 2.2 percent.
4. GFlowNet diversity is high, but policy entropy is too close to uniform for task-directed reasoning.
5. Larger inference budgets increase trajectory organization but do not improve BPB.
6. Prediction-target NCD improves under zlib but not under LZMA.
7. Smear temperature appears useful for calibration, but it is approaching its cap.

### Not As Desired

The following require intervention:

1. W&B `val/bpb` is not reliable as currently logged and contradicts deterministic checkpoint re-evaluation.
2. The best checkpoint remains `best.pt` from step 500, while the current latest checkpoint is step 14750.
3. Toric memory entropy is too low.
4. Trajectory flow loss and viscous dissipation rose significantly.
5. Larger reasoning budgets do not reduce BPB.
6. QAT loss is drifting upward, though its absolute magnitude and weight are small.

## Intervention Plan For Approval

No model or training changes should be made until this plan is approved.

### 1. Fix Validation Before Resuming

Add a deterministic validation protocol:

```text
val/reset_bpb
val/reset_loss
val/reset_bias_bpb
val/hard_bpb
val/complexity_bpb
val/budget1_bpb
val/budget4_bpb
```

The validation loader should be re-created from a logged seed at every evaluation.  The validation batch indices, pass ids, order-sample count, GFlowNet-sample count, and score-first-bias parameters should be logged.  Checkpoint promotion should use this deterministic reset metric, not the current ambiguous `val/bpb`.

Mathematically, the goal is to estimate a fixed quantity
\[
L_{\mathrm{val}}(\theta)
=\mathbb E_{(x,\pi)\sim P_{\mathrm{fixed}}}
[-\log q_\theta(x_\pi\mid x_{\pi,<t})],
\]
not a drifting iterator-dependent quantity.

### 2. Resume From Latest, But Run A Controlled Best-Checkpoint Branch

Use two short comparison runs:

1. `latest-rescue`: resume from step 14750 with lower LR, fixed validation, and softer curriculum.
2. `best-branch`: resume from `best.pt` at step 500 with the same fixed validation and curriculum ramp.

Do not overwrite the current checkpoint directory until one branch beats the deterministic validation gate.

### 3. Replace Hard Complex-Curriculum Switch With A Mixture Schedule

The current run switches fully to complex rows after step 1000.  Replace that with
\[
p_{\mathrm{complex}}(s)
=p_{\max}\sigma\left(\frac{s-s_0}{\tau}\right),
\]
with \(p_{\max}\in[0.25,0.50]\) until validation BPB improves.  Keep base byte rows in every phase so BPB optimization is not starved by long technical composites.

### 4. Add Budget-Monotonic Reasoning Loss

The simplex plots show
\[
\operatorname{BPB}_{4}>\operatorname{BPB}_{1}
\]
even though MST efficiency improves.  Add a budget-consistency penalty:
\[
\mathcal L_{\mathrm{budget}}
=
\sum_{B\in\{2,4,8\}}
\max(0,\operatorname{BPB}_B-\operatorname{BPB}_1+\delta).
\]
This preserves extra reasoning only when it does not harm likelihood.  In inference, select the best of several order/GFlowNet samples by a legal prefix score proxy, not by future target bytes.

### 5. Retarget GFlowNet Rewards

Current GFlowNet entropy is essentially uniform.  Replace the weak reward with
\[
\log R(x)
=
-\lambda_b\operatorname{BPB}(x)
-\lambda_d\operatorname{NCD}(\hat y,y)
+\lambda_m\eta_{\mathrm{MST}}
+\lambda_n\operatorname{Novelty}(x)
-\lambda_c C_{\mathrm{traj}}(x).
\]
Use an entropy target rather than always maximizing entropy:
\[
\mathcal L_H
=\lambda_H(H(P_F)-H_\star)^2,
\qquad H_\star < \log 16.
\]
This should keep exploration but make the policy less uniform.

### 6. Protect Toric Memory

Add a toric-memory entropy floor:
\[
\mathcal L_{\mathrm{toric\_ent}}
=
\lambda_T\max(0,H_{\min}-H_{\mathrm{toric}})^2.
\]
Also log per-slot mass histograms.  This preserves noncommutative phase memory without forcing uniform usage.

### 7. Smooth Trajectory Flow

Use a delayed or adaptive viscosity:
\[
\nu(s)=\nu_0\min(1,s/s_\nu),
\]
and penalize only excess dissipation:
\[
\mathcal L_{\mathrm{visc}}
=\lambda_\nu\max(0,E_{\mathrm{visc}}-\tau_\nu)^2.
\]
The current flow terms are mathematically useful but too active before they are coupled to predictive improvement.

### 8. Delay Or Narrow QAT

Keep QAT enabled only for a small tensor subset until validation BPB is below the best checkpoint.  Then ramp QAT.  This prevents quantization pressure from fighting early likelihood learning.

### 9. Use MST For Pruning, Not As A Standalone Reward

Use MST to extract a minimal reasoning skeleton from hidden trajectories.  Reward lower BPB at fixed or lower MST total weight:
\[
\Delta_{\mathrm{eff}}
=
\frac{\operatorname{BPB}_{\mathrm{base}}-\operatorname{BPB}_{\mathrm{reasoned}}}
     {1+W_{\mathrm{MST}}}.
\]
This avoids rewarding beautiful but useless trajectories.

### 10. Plot On Every Evaluation Cycle

Every 500 or 1000 steps, run a small version of:

```bash
conda run --no-capture-output -n tokengt env PYTHONPATH=src \
  python scripts/evaluate_reasoning_simplex.py \
  --checkpoint checkpoints/parameter_golf_oai_dense/<checkpoint>.pt \
  --data-glob 'data/curated_hf_shards/validation/*.parquet' \
  --seed 10017 \
  --samples 8 \
  --batch-size 2 \
  --seq-len 1024 \
  --budgets 1 2 4 8 \
  --output-dir outputs/reasoning_simplex/<run-step>
```

Approval criterion: extra budget must either reduce BPB or improve a reasoning diagnostic without increasing BPB beyond a small tolerance.

## Approval Recommendation

Do not resume the same training command unchanged.  The safe next step is:

1. implement deterministic validation logging;
2. run a 500-step rescue from `random_order_step_00014750.pt` with lower LR and mixed curriculum;
3. run a parallel 500-step branch from `best.pt`;
4. compare deterministic validation BPB, simplex budget monotonicity, toric entropy, and trajectory dissipation;
5. only then choose the branch for the next long run.

This plan preserves the metrics that are already good: byte budget, causal audit, gradient stabilization, high action diversity, and graph/reasoning geometry.  It directly targets the failing behaviors: validation ambiguity, weak likelihood gains from extra reasoning, toric-memory collapse, and excessive trajectory dissipation.

## Extended Geometry Suite

The first simplex pass was budget-level.  It was useful, but it was too coarse:
it did not show individual graph-of-thought branches, and it did not show
whether branch endpoints passed through actual source solution spans.  I added
and ran a second evaluator:

```bash
conda run --no-capture-output -n tokengt env PYTHONPATH=src \
  python scripts/evaluate_reasoning_geometry_suite.py \
  --checkpoint checkpoints/parameter_golf_oai_dense/random_order_step_00014750.pt \
  --data-glob 'data/curated_hf_shards/validation/*.parquet' \
  --output-dir outputs/reasoning_geometry_suite/oai-8v4wnovf-step14750 \
  --records 4 \
  --branches 6 \
  --seq-len 1024 \
  --seed 10017
```

The script evaluates six randomized-order/GFlowNet branches for each selected
reasoning-heavy validation record, projects the real hidden states with PCA,
marks actual answer/solution spans when present, and writes branch-level
simplex records.  The outputs are:

- `outputs/reasoning_geometry_suite/oai-8v4wnovf-step14750/reasoning_geometry_records.json`
- `outputs/reasoning_geometry_suite/oai-8v4wnovf-step14750/selected_records.json`
- `outputs/reasoning_geometry_suite/oai-8v4wnovf-step14750/triangles/`
- `outputs/reasoning_geometry_suite/oai-8v4wnovf-step14750/tetrahedra/`
- `outputs/reasoning_geometry_suite/oai-8v4wnovf-step14750/trajectories/`

The triangle suite now includes eight heatmap diagnostics:

1. `reasoning_k_bpb.png`
2. `solution_basin.png`
3. `trajectory_flow.png`
4. `exploration_control.png`
5. `compression_reasoning.png`
6. `robustness_generalization.png`
7. `toric_memory_control.png`
8. `energy_landscape.png`

The tetrahedron suite includes five 3-simplex diagnostics, each as static PNG
and interactive HTML:

1. `reasoning_k_bpb_mst`
2. `solution_bpb_smooth_diversity`
3. `complexity_geometry_solution`
4. `energy_control`
5. `toric_gfn_bpb`

For each selected record, the trajectory directory contains:

- `*_trajectory_3d.png` and `*_trajectory_3d.html`: manipulatable 3D
  embedding-space graph-of-thought branches.  Cyan marks starts, gold marks
  actual answer/solution span steps, and the star marks the best-likelihood
  terminal branch.
- `*_phase_energy.png`: a Ramachandran-style plot where
  \[
  \phi_t=\operatorname{atan2}(v_{t,2},v_{t,1}),\qquad
  \psi_t=\operatorname{atan2}(v_{t+1,2},v_{t+1,1})
  \]
  for projected hidden velocity \(v_t=z_{t+1}-z_t\), colored by local
  negative log likelihood.
- `*_energy_landscape.png`: a PC1/PC2 energy landscape colored by local token
  NLL, with branch paths overlaid.

### Selected Records

The suite selected four validation records:

| Record | Dataset | Family | Answer span |
|---:|---|---|---|
| R0 | `Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b` | `frontier_gptoss_reasoning` | `[256, 768]` |
| R1 | `Sefaria/hebrew_library` | `jewish_hebrew_text` | `[87, 977]` |
| R2 | `gss1147/Got_Math_500K` | `got_math` | `[378, 890]` |
| R3 | `AI-MO/NuminaMath-CoT` | `cot_math` | `[254, 771]` |

These are solution-conditioned trajectories: the actual source solution or
answer span is present inside the scored sequence.  This is not yet a claim
that the model freely generates the solution from the problem prompt.  It is a
diagnostic for whether the hidden trajectory assigns coherent likelihood and
geometry to known solution-bearing byte spans.

### Geometry Metric Summary

| Record | Mean BPB | Best BPB | Mean answer BPB | MST efficiency | Path smoothness | Toric entropy |
|---:|---:|---:|---:|---:|---:|---:|
| R0 frontier code reasoning | 5.277 | 5.264 | 5.332 | 0.541 | 0.117 | 0.101 |
| R1 Sefaria Hebrew | 12.603 | 12.584 | 13.197 | 0.707 | 0.050 | 0.121 |
| R2 GoT math | 4.951 | 4.947 | 4.903 | 0.546 | 0.111 | 0.135 |
| R3 NuminaMath | 4.887 | 4.886 | 4.796 | 0.572 | 0.075 | 0.025 |

The branch-level order robustness is high: \(0.981\) to \(0.999\).  This means
different random orders do not wildly change BPB on these windows.  That is
good for score stability but also means the current GFlowNet perturbation is
not yet producing materially better solution branches.

### Three-Tier Categorization For Geometry

**As desired**

- The evaluator now produces actual hidden-state 3D trajectories from the
  checkpoint, not synthetic curves.
- The selected set is domain-diverse: code reasoning, Hebrew/Jewish text,
  graph-of-thought math, and NuminaMath.
- Order robustness is high across six branches per record, so random-order
  decoding is not obviously destabilizing the model.
- GFlowNet action diversity is noncollapsed at the action-support level
  (`1.0` in this run).
- The GoT and NuminaMath answer spans score better than or comparable to their
  full windows, suggesting the model has learned some local solution texture.

**As desired but not strong or fast enough**

- Math BPB around \(4.89\)-\(4.95\) is improving but far from the competition
  target.  The model has useful structure but not enough byte-likelihood
  strength.
- MST efficiency around \(0.54\)-\(0.57\) for math/code windows is acceptable
  but not yet tied to lower BPB.  The geometry is readable, but the trajectory
  reward is not yet likelihood-aligned.
- Path smoothness is positive but low.  The projected trajectories still have
  high curvature, which matches the earlier high viscous-dissipation metrics.
- Branch-to-branch BPB spread is small.  Stability is good, but inference-time
  scaling needs a mechanism that makes some branches meaningfully better.

**Not as desired**

- Sefaria Hebrew BPB is very high (\(\approx 12.6\), answer span
  \(\approx 13.2\)).  This is consistent with byte-level Hebrew being much
  harder under the current tokenizer and with unpointed/pointed Hebrew
  heterogeneity.
- Toric memory entropy remains low, especially on NuminaMath
  (\(\approx 0.025\)).  The phase memory is still too collapsed for a model that
  is meant to exploit noncommutative-toric structure.
- GFlowNet action diversity is maximal, but the policy is likely close to
  uniformly exploratory rather than reward-selective.  High support diversity
  is not the same as useful branch specialization.
- No branch can yet be called a free-form generated solution.  The plots show
  solution-conditioned terminals and likelihood basins, not autonomous solve
  traces.

### Mathematical Interpretation

Let \(h_t\in\mathbb R^d\) be the hidden state along a random-order trajectory.
The PCA projection \(z_t=P(h_t-\bar h)\in\mathbb R^3\) is only a diagnostic, but
the velocity and curvature statistics are meaningful:
\[
v_t=z_{t+1}-z_t,\qquad
a_t=v_{t+1}-v_t,\qquad
\kappa_t=\frac{\|a_t\|}{\|v_t\|^2+\epsilon}.
\]
High curvature with no BPB improvement means the model is spending trajectory
length without a corresponding decrease in energy
\[
E_t=-\log p_\theta(x_{\pi_t}\mid x_{\pi_{<t}}).
\]
In Navier-Stokes language, the trajectory has excess dissipation without useful
transport.  The current viscosity penalty is therefore correctly motivated but
should become adaptive: penalize curvature only above a target threshold and
only when it fails to buy likelihood or answer-span improvement.

The MST statistics measure whether a trajectory has a compact reasoning
skeleton.  If \(W_{\mathrm{MST}}\) is the total MST weight of trajectory points,
a useful inference-time branch should improve
\[
\Delta_{\mathrm{eff}}
=\frac{\operatorname{BPB}_{\mathrm{base}}-\operatorname{BPB}_{\mathrm{branch}}}
       {1+W_{\mathrm{MST}}}.
\]
Current branches have readable geometry but weak \(\Delta_{\mathrm{eff}}\).  So
MST should be used as a pruning and efficiency diagnostic, not rewarded by
itself.

The Hebrew failure is not surprising from an information-theoretic view.  A
byte-level model sees Hebrew as multi-byte UTF-8 sequences; niqqud introduces
additional combining marks; and the corpus mixes pointed and unpointed forms.
For a fixed parameter budget, the effective conditional entropy of Hebrew byte
continuations is much higher unless the model learns stronger morphology-aware
compression.  This supports adding a compact Hebrew byte/morpheme adapter or
more Hebrew-specific curriculum only if it does not hurt FineWeb BPB.

### Plan Update From Geometry

1. Keep the extended geometry suite as a standard evaluation artifact.
2. Fix deterministic validation before using any W&B validation series for
   checkpoint promotion.
3. Add a reward-selective GFlowNet branch objective:
   \[
   R_{\mathrm{branch}}
   =\exp\left(
      -\alpha\operatorname{BPB}
      -\beta\operatorname{BPB}_{\mathrm{answer}}
      +\gamma \Delta_{\mathrm{eff}}
      +\delta H_{\mathrm{toric}}
   \right).
   \]
   This keeps diversity but ties it to lower energy and useful solution spans.
4. Add a toric-memory entropy floor and per-slot histograms.
5. Add an adaptive curvature/viscosity loss:
   \[
   \lambda_\nu\max(0,\bar\kappa-\kappa_\star)^2
   \]
   but gate it off when branch BPB improves, so useful sharp reasoning turns
   are not penalized.
6. Treat Hebrew as a separate diagnostic domain for now.  Do not let poor
   Hebrew BPB dominate the Parameter-Golf objective unless a compact
   morphology-aware representation improves overall validation BPB.
7. Promote branches only when they improve both deterministic BPB and at least
   one reasoning geometry score without worsening toric entropy or trajectory
   dissipation.

## Implemented Rescue Controls

The training code now implements the control bundle above.

- Validation is split into deterministic, GFlowNet-budgeted, and score-first
  streams.  `val/bpb` is now the deterministic no-adaptation value and is the
  only checkpoint-promotion gate.
- The validation loader is non-repeating and non-shuffled, so every evaluation
  begins from the same validation prefix.
- The GFlowNet entropy term is target-based:
  \[
  \mathcal L_{H}=\lambda_H(H(P_F)-H_\star)^2,
  \]
  with \(H_\star=2.05\) nats for the current 16-action policy.
- Toric memory has an entropy floor:
  \[
  \mathcal L_T=\lambda_T\max(0,0.18-H_{\mathrm{toric}})^2.
  \]
- Trajectory flow uses an excess-loss form:
  \[
  \mathcal L_{\mathrm{flow}}
  =\lambda_\nu\max(0,E_{\mathrm{flow}}-0.075)^2.
  \]
- QAT is delayed until step 18,000 and ramps over 4,000 steps, with only the
  eight largest matrix tensors included initially.
- Complex technical/composite rows are mixed with the ordinary stream at a
  deterministic 55% ratio after step 1,000 rather than replacing the whole
  stream.
- Checkpoints remain retained every 250 steps.

The active run was restarted from:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00014750.pt
```

The W&B run is:

```text
https://wandb.ai/amelie-iska-math/toricgt-parameter-golf/runs/oai-rescue-14750-geometry
```

Two tmux sessions are active:

```bash
tmux attach -t toricgt_pg_oai
tmux attach -t toricgt_pg_oai_analysis
```

The analysis watcher waits for the first checkpoint at or after step 16,250.
It then runs W&B metric analysis, budget simplex diagnostics, branch-level 3D
trajectories, Ramachandran-style phase plots, and energy/fitness landscapes on
CPU so the training GPU is not interrupted.  Its synopsis will be written under:

```text
outputs/post_resume_analysis/oai-rescue-14750-geometry/step-00016250/SYNOPSIS.md
```

## Step 20,000 Tuning Update

The step 16,250 diagnostic pass showed that deterministic validation BPB was
improving, toric-memory entropy was recovering, and trajectory flow loss was
moving in the intended direction.  The remaining issue was speed and selectivity:
the GFlowNet action policy remained close to uniform and QAT plus the 55%
complex-row curriculum increased short-horizon training variance.

The next restart therefore keeps the architecture fixed and changes only the
training controls:

- learning rate is reduced from `1.8e-4` to `1.2e-4`;
- complex/composite-row mixing is reduced from `0.55` to `0.35`;
- GFlowNet entropy shaping is strengthened from `0.0015` to `0.003` with the
  same target entropy `2.05`;
- QAT pressure is halved from `1e-6` to `5e-7`;
- QAT coverage is reduced from the eight largest matrices to the four largest
  matrices.

This is a conservative adjustment.  It should preserve the desirable downward
BPB trend and recovered toric entropy while allowing action distributions to
become more task-conditioned and lowering the probability that quantization
noise masks real modeling improvements.

The active restart point is:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00020000.pt
```

The next automatic analysis pass should run after roughly 1,500 additional
steps, targeting the first checkpoint at or after step 21,500.

## Step 34,750 GPU Geometry Analysis And Restart Controls

The GPU analysis suite at step 34,750 inspected W&B metrics, reasoning/K/BPB
triangles, tetrahedra, Ramachandran-style phase plots, energy landscapes, and
per-record embedding-space graph-of-thought trajectories.

The key findings were:

- deterministic validation BPB continued to improve, but slowly;
- budgeted inference did not improve BPB yet: budget 1 had the best simplex BPB
  while budgets 2, 4, and 8 mainly increased complexity and wall time;
- the best branch in the geometry suite was substantially better than the mean
  branch, so branch search contains useful candidates but policy selection is
  not calibrated;
- GFlowNet entropy stayed essentially at \(\log 16\), meaning the action policy
  remains close to uniform;
- trajectory flow/viscous dissipation rose, indicating turbulent embedding-space
  paths without enough likelihood gain;
- QAT and composite-row pressure are likely adding optimization drag.

The restart from

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00034750.pt
```

therefore keeps the architecture fixed and makes only training-control changes:

- `complex_mix_ratio`: `0.35` -> `0.25`;
- `gflownet_action_scale`: `0.06` -> `0.12`;
- `gflownet_entropy_weight`: `0.003` -> `0.01`;
- `gflownet_entropy_target`: `2.05` -> `1.75`;
- `qat_loss_weight`: `5e-7` -> `2e-7`;
- `qat_max_tensors`: `4` -> `2`.

The hypothesis is that the model needs a lower curriculum/QAT load and stronger
policy selectivity before test-time graph-of-thought budget can help BPB.  Until
the GFlowNet policy entropy moves meaningfully below \(\log 16\), checkpoint
promotion should continue to use deterministic BPB and analysis should treat
budget 4/8 as diagnostic rather than default.

## Step 21,250--24,000 Reanalysis And 22,500 Restart Decision

The run restarted from step 21,250 was paused at step 24,111 and reanalyzed on
GPU.  The analysis artifacts are under:

```text
outputs/reanalysis_gpu/oai-resume-21250-selective-gfn/
```

The W&B export for the paused run shows a split signal:

- validation BPB improved slightly from `4.890875` at step 21,500 to `4.881343`
  at step 24,000;
- complexity validation BPB improved from `4.855255` to `4.820594`;
- training BPB/loss worsened over the same window, with the recent train BPB
  slope \(+0.155\) BPB per 1,000 steps and \(t=3.17\);
- GFlowNet entropy moved down only from about `2.772` to `2.754`, still close to
  \(\log 16\), so the policy is not selective enough for inference-time budget
  to help;
- toric-memory entropy declined, approaching the entropy floor.

The same checkpoint-window simplex suite gives:

| checkpoint | simplex mean BPB | simplex budget-1 BPB | simplex budget-8 BPB | geometry mean BPB | best geometry BPB | mean answer BPB |
|---|---:|---:|---:|---:|---:|---:|
| 21,250 | 3.996044 | 3.988463 | 3.997908 | 4.608863 | 3.811952 | 4.567529 |
| 22,500 | 3.993061 | 3.984411 | 3.997107 | 4.607221 | 3.780859 | 4.572008 |
| 24,000 | 4.157946 | 4.120694 | 4.166791 | 4.628992 | 3.975335 | 4.557661 |

The plots support the same conclusion.  At step 22,500 the reasoning/K/BPB
simplex still has a clear low-BPB budget-1 basin, while higher budgets mostly
move outward toward \(K(x)\), MST complexity, and wall time without lowering BPB.
By step 24,000 the simplex and branch clouds move to a higher-BPB region even
though deterministic validation has improved slightly.  The 3D trajectories and
energy landscapes remain useful, but branch selection is not calibrated: the
best terminal branch is much better than the mean branch, and the GFlowNet
policy still explores almost uniformly.

The chosen restart point is therefore:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00022500.pt
```

This is not a model-architecture change.  It is a training-control correction:

- base learning rate `1.2e-4` -> `9e-5`, preserving optimizer moments but
  lowering the resumed cosine schedule;
- complex/composite-row mix `0.25` -> `0.20`, keeping difficult graph-reasoning
  data active while reducing short-horizon BPB drift;
- GFlowNet loss weight `0.01` -> `0.0125`, to make branch-policy learning more
  visible in the total objective;
- GFlowNet entropy shaping `0.01 @ target 1.75` -> `0.02 @ target 2.05`, which
  applies stronger pressure away from uniform \(\log 16\) behavior while
  avoiding premature collapse to only a few actions;
- toric entropy floor `0.18` -> `0.20` and weight `0.002` -> `0.003`, because
  toric entropy was drifting toward the floor;
- trajectory-flow loss weight `0.002` -> `0.003`, to penalize turbulent
  embedding-space paths slightly more while leaving the main BPB objective
  dominant.

The next decision point should be roughly 1,250--2,000 resumed steps after
22,500.  The acceptance criteria are:

1. validation BPB must not regress by more than about `0.01`;
2. simplex budget-1 BPB should remain near or below the 22,500 value;
3. GFlowNet entropy should move below `2.72` without action-diversity collapse;
4. toric-memory entropy should remain above `0.20`;
5. geometry mean BPB should not exceed `4.62`, and best branch BPB should remain
   below `3.85` on the current diagnostic subset.

## Step 23,500--23,750 Reanalysis And 23,600 Restart Consideration

The corrected 22,500 restart was paused at step 23,797 and reanalyzed on GPU
around the requested step-23,600 window.  There is no exact saved checkpoint at
23,600; the adjacent checkpoints are:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00023500.pt
checkpoints/parameter_golf_oai_dense/random_order_step_00023750.pt
```

The analysis artifacts are under:

```text
outputs/reanalysis_gpu/oai-resume-22500-window-corrected/
```

The step-window comparison is:

| checkpoint | simplex mean BPB | simplex budget-1 BPB | simplex budget-8 BPB | geometry mean BPB | best geometry BPB | mean answer BPB | GFlowNet entropy | toric entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 22,500 | 3.993061 | 3.984411 | 3.997107 | 4.607221 | 3.780859 | 4.572008 | 2.766903 | 0.225308 |
| 23,500 | 3.988780 | 3.978290 | 3.990622 | 4.611664 | 3.776056 | 4.567774 | 2.678068 | 0.271882 |
| 23,750 | 3.988625 | 3.983087 | 3.992223 | 4.600572 | 3.786171 | 4.566761 | 2.675859 | 0.251321 |

The corrected controls are doing the intended first-order job.  The GFlowNet
policy is no longer effectively uniform: entropy moved from approximately
`2.77` toward `2.67`, while action diversity stayed high.  This should be
categorized as desired behavior, even though the automatic metric analyzer still
labels entropy decline as undesirable because it assumes "higher is better" for
all entropy metrics.  Here the target is controlled selectivity, not maximum
entropy \(\log 16\).

The simplex plots show that budget-1 remains the strongest BPB point.  Additional
reasoning budget increases \(K(x)\), MST complexity, trajectory length, and
wall-time footprint before it improves compressed likelihood.  The tetrahedra
show the same geometry: low BPB is still located near the low-budget/low-K
corner, while the higher-budget points move toward graph complexity and
diversity.  This means inference-time scaling is producing meaningful
trajectories, but branch ranking is not yet calibrated enough for extra budget
to reduce BPB by default.

The trajectory plots are healthier than the previous 24,000 analysis: paths are
less chaotic, the phase/Ramachandran plots retain separated basins, and energy
landscapes show coherent basins rather than one collapsed attractor.  The mean
path smoothness improves through 23,750, while step 23,500 has the better
best-branch BPB.

The recommended restart point for competition BPB is:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00023500.pt
```

The reason is pragmatic: step 23,500 is the best branch-selection checkpoint in
this window.  It has the best simplex budget-1 BPB (`3.978290`) and best
geometry BPB (`3.776056`).  Step 23,750 has slightly better geometry mean BPB
and mean answer BPB, so it is a reasonable alternative if the next run prioritizes
smooth trajectories over best-branch candidates.  For the Parameter-Golf
objective, the safer restart is 23,500.

### Optional Adaptive Hyperparameter Functions

Adaptive controls should be checkpoint-level, not per-step, because the current
signals are noisy and highly coupled.  The controller should update only after
an evaluation/analysis window, log every knob to W&B, and enforce hard bounds.

1. **GFlowNet entropy target schedule.**  Start broad, then anneal toward
   selective graph-of-thought branching:

   \[
   H^\star(t)=H_{\min}+(H_{\max}-H_{\min})
   \exp\left(-\frac{t-t_0}{\tau_H}\right),
   \qquad
   H_{\max}=\log(E_A),\ H_{\min}\in[1.9,2.2].
   \]

   Viability: high.  This matches the observed need to move away from uniform
   branching without collapsing the action policy.  Use \(\tau_H\approx
   4{,}000\)--\(8{,}000\) steps and clamp the live target to `[1.9, 2.4]`.

2. **Budget-gap driven GFlowNet weight.**  If the best branch is much better
   than the mean branch, increase policy-learning pressure:

   \[
   \lambda_{\mathrm{GFN}}(t+1)=
   \operatorname{clip}\left(
   \lambda_{\mathrm{GFN}}(t)
   \left[1+\eta_g\,\operatorname{EWMA}
   \left(\operatorname{BPB}_{\mathrm{mean}}-\operatorname{BPB}_{\mathrm{best}}\right)
   \right],
   0.008,0.025\right).
   \]

   Viability: high.  The current diagnostic gap says useful candidates exist,
   but the policy/ranker does not yet exploit them.  This is the most directly
   justified adaptive control.

3. **Complex-data mix throttle.**  Reduce composite/long graph rows when short
   horizon BPB drifts upward, then restore them when validation stabilizes:

   \[
   r_{\mathrm{complex}}(t+1)=
   \operatorname{clip}\left(
   r_{\mathrm{complex}}(t)
   -\eta_c\,\max(0,z_{\nabla \mathrm{BPB}})
   +\eta_v\,\mathbf 1[\Delta\mathrm{val\_BPB}<0],
   0.12,0.30\right).
   \]

   Here \(z_{\nabla \mathrm{BPB}}\) is an EWMA-normalized recent train-BPB slope.
   Viability: high if changed slowly at checkpoint boundaries.  It protects the
   Parameter-Golf objective while retaining graph-reasoning curriculum pressure.

4. **Learning-rate damping on positive BPB slope.**

   \[
   \eta_{\mathrm{eff}}(t)=
   \frac{\eta_{\mathrm{cos}}(t)}
   {1+\alpha \max(0,z_{\nabla \mathrm{train\_BPB}})}.
   \]

   Viability: medium.  It is safe when applied as a small multiplicative damping
   factor, but it can overreact to stochastic batch composition.  Prefer using it
   only after two consecutive analysis windows show positive train-BPB slope.

5. **QAT ramp with BPB guard.**

   \[
   \lambda_{\mathrm{QAT}}(t)=
   \lambda_{\max}\,
   \sigma\left(\frac{t-t_q}{\tau_q}\right)
   \exp\left(-\alpha_q\max(0,\operatorname{BPB}_{\mathrm{simplex}}-
   \operatorname{BPB}_{\mathrm{baseline}})\right).
   \]

   Viability: medium-high.  This keeps quantization pressure aligned with the
   artifact goal but backs off if it damages early likelihood.  It should remain
   bounded below the current tiny weight until BPB is clearly improving.

6. **Trajectory-flow viscosity.**  Increase flow regularization when path
   smoothness deteriorates without a likelihood gain:

   \[
   \lambda_{\mathrm{flow}}(t+1)=
   \operatorname{clip}\left(
   \lambda_{\mathrm{flow}}(t)
   \left[1+\eta_f\max(0,s^\star-s_{\mathrm{smooth}})\right],
   0.001,0.006\right).
   \]

   Viability: medium.  It helps prevent turbulent embedding trajectories, but
   too much viscosity can erase useful exploratory branches.

7. **Toric entropy floor controller.**

   \[
   h_{\mathrm{floor}}(t+1)=
   \operatorname{clip}\left(
   q_{0.2}\left(h_{\Theta,t-W:t}\right)-\delta,
   0.18,0.28\right).
   \]

   Viability: medium.  The static floor is currently working, so this is lower
   priority.  It becomes useful if toric entropy repeatedly hits the floor or
   oscillates sharply after GFlowNet changes.

The recommended controller implementation, if added, is a bounded
checkpoint-level callback rather than a new architecture component.  It should
read the latest W&B/eval metrics every `500` steps, update only scalar training
knobs, write a `controller_state.json`, and log `controller/*` metrics.  The
first adaptive run should enable only items 1--3; items 4--7 are useful but more
likely to interact with optimizer state or regularization in hard-to-debug ways.

## GPU Interval Analysis: Step 23,500 to 26,150

Analysis date: 2026-05-29. The active `oai-resume-23500-adaptive`
training process was paused and the GPU was used for all model-side diagnostic
runs. The latest checkpoint written by the active run is
`checkpoints/parameter_golf_oai_dense/random_order_step_00026000.pt`; W&B
history reaches step `26150`, but there is no newer checkpoint from this run.

Primary artifacts:

- Metrics report:
  `outputs/gpu_interval_analysis/oai-resume-23500-adaptive/step-00026000/metrics/automatic_metrics_report.md`
- Metrics contact sheet:
  `outputs/gpu_interval_analysis/oai-resume-23500-adaptive/step-00026000/contact_metrics.png`
- Reasoning simplex outputs:
  `outputs/gpu_interval_analysis/oai-resume-23500-adaptive/step-00026000/simplex/`
- Geometry, trajectory, energy, and phase plots:
  `outputs/gpu_interval_analysis/oai-resume-23500-adaptive/step-00026000/geometry/`
- Simplex/geometry contact sheet:
  `outputs/gpu_interval_analysis/oai-resume-23500-adaptive/step-00026000/contact_simplex_geometry.png`
- 3D trajectory contact sheet:
  `outputs/gpu_interval_analysis/oai-resume-23500-adaptive/step-00026000/contact_trajectories_3d.png`
- Energy/Ramachandran-style phase contact sheet:
  `outputs/gpu_interval_analysis/oai-resume-23500-adaptive/step-00026000/contact_energy_phase.png`

### Metric Categorization

The automatic classifier gives `115` desired metrics, `61` metrics that are
directionally acceptable but weak, and `63` undesirable metrics. The important
manual categorization is:

| Metric family | Behavior | Evidence | Category |
|---|---:|---|---|
| Base train BPB/loss | likelihood drifted upward after the 23.5k resume | `train/bpb` median increased from `4.69603` to `4.93759` (`+5.14%`); `train/loss` increased by the same relative amount; `train/total_loss` increased `+4.82%` | undesirable |
| Recent train BPB slope | drift is weaker in the last few hundred steps but not recovered | recent `train/bpb` slope is `+0.04835` BPB/1k with low t-statistic, while the full-window slope is `+0.19098` BPB/1k | undesirable but possibly stabilizing |
| Validation BPB | improved, but too slowly | `val/bpb` moved from `4.89233` to `4.88581` (`-0.13%`); last value is slightly worse than best `4.88540` | acceptable but not strong enough |
| Complexity validation BPB | improved more than ordinary val BPB | `complexity/val/bpb` moved from `4.84217` to `4.82457` (`-0.36%`) | acceptable but not strong enough |
| Prediction-target NCD | semantic/compressive similarity is not improving | `complexity/val/prediction_target_ncd_lzma_mean` worsened from `0.92175` to `0.92611` | weak/undesirable |
| GFlowNet loss | improved strongly | `train/gflownet_loss` decreased from `2.32838` to `1.79965` (`-22.71%`) | desired |
| GFlowNet entropy/diversity | no collapse, but slowly narrowing | entropy `2.67617 -> 2.65727`; action diversity `0.96388 -> 0.95770` | acceptable, monitor |
| Controller complex mix | moved in the right direction but too late and not far enough | `complex_mix_ratio` decreased from `0.20` to `0.16`, yet train BPB remained elevated | desired direction, insufficient strength |
| Controller GFlowNet weight | moved the wrong way for this interval | `gflownet_loss_weight` increased from `0.01250` to `0.01265` even though base BPB was drifting upward | undesirable |
| Causal audit | clean | future permutation logit error stayed exactly `0` | desired |

### Plot Analysis

The core metric timeseries shows a step-regime change: after the composite
curriculum resume, training BPB/loss jump upward and remain elevated. This is
not an isolated spike; robust spike counts are zero because the issue is a new
plateau, not a one-step outlier. Validation BPB slopes downward, but the
absolute movement is far too small for the competition objective.

The selected metric correlation plot suggests the drift is coupled to the
trajectory-flow terms: train BPB/loss rise with kinetic/viscous trajectory
energy, while GFlowNet loss falls. This is the expected signature of a model
that is learning to produce structured reasoning trajectories, but paying for it
in next-byte likelihood. That tradeoff is acceptable only if validation BPB or
best-branch BPB improves materially; here it does not.

The reasoning simplex evaluation shows budget saturation:

| Budget | BPB | Loss | MST efficiency | Trajectory tokens |
|---:|---:|---:|---:|---:|
| 1 | `4.03596` | `2.79751` | `0.65881` | `2048` |
| 2 | `4.03209` | `2.79483` | `0.68320` | `12288` |
| 4 | `4.03494` | `2.79681` | `0.69107` | `65536` |
| 8 | `4.04786` | `2.80576` | `0.68818` | `65536` |

Budget `2` is the best BPB point, budget `4` only improves MST efficiency, and
budget `8` makes BPB worse. For the current checkpoint, extra inference-time
reasoning beyond budget `4` is over-thinking rather than useful scaling.

The geometry suite reports `mean_bpb=4.74539`, `best_bpb=4.08563`,
`mean_answer_bpb=4.62301`, `best_answer_bpb=3.57531`,
`mean_mst_efficiency=0.50698`, and `mean_path_smoothness=0.090997`. Compared
with the prior step-25k analysis, BPB is slightly better but MST efficiency is
lower and smoothness worsened. Compared with the earlier 21.25k--23.75k GPU
window, the geometry BPB is worse, though those older measurements used a
different sample set and should not be treated as paired statistics.

The 3D embedding-space graph-of-thought trajectories show the qualitative
failure mode. Hebrew text forms a relatively coherent fan-shaped manifold, but
frontier reasoning, GoT math, and competition math show dense attractor regions
with long chaotic excursions. Terminal branches often orbit or skim basins
instead of descending reliably into low-energy/high-answer-likelihood regions.
The energy landscapes contain many local NLL hot spots and irregular ridges; the
Ramachandran-style phase plots show toric phase modes are present and not
collapsed, but the phase modes are not yet aligned strongly enough with lower
BPB terminals.

### Mathematical Interpretation

Let the effective objective be

\[
\mathcal L =
\mathcal L_{\mathrm{BPB}}
+ \lambda_{\mathrm{GFN}}\mathcal L_{\mathrm{GFN}}
+ \lambda_H (H-H^\star)^2
+ \lambda_{\mathrm{flow}}\mathcal L_{\mathrm{flow}}
+ \lambda_{\mathrm{complex}}\mathcal L_{\mathrm{complex}}.
\]

The interval behavior says that the gradient component from
\(\lambda_{\mathrm{GFN}}\mathcal L_{\mathrm{GFN}}\) and the complex-row
curriculum is coherent enough to reduce GFlowNet loss, but not aligned enough
with \(-\nabla \mathcal L_{\mathrm{BPB}}\). In geometric terms, the reasoning
flow is improving its internal trajectory objective while moving along a
likelihood-neutral or likelihood-adverse tangent direction. The model has
entered a shallow BPB floor: validation improves because some regularization and
complex examples help generalization, but base compression is no longer
descending quickly.

The simplex result sharpens this: the first extra reasoning budget is useful,
the second mostly helps graph connectivity, and later budgets add Kolmogorov
program length without lowering BPB. If \(B(k)\) is BPB after reasoning budget
\(k\), then \(B(2)<B(1)\), \(B(4)\approx B(2)\), and \(B(8)>B(2)\); the
discrete second difference has turned positive. The current policy should
therefore penalize long unproductive trajectories until the branch selector
becomes more reliable.

### Recommended Change Set

User decision after reviewing the interval: resume from the earlier step
`23250` checkpoint rather than attempting to recover the later 23.5k-to-26k
run. This is justified because the later interval is worse relative to the
pre-run checkpoint family despite a small validation BPB movement. Enter a short
recovery curriculum from `23250` for `1000`--`1500` steps, then re-analyze.

Recommended config changes:

1. **Complex-row throttle.** Set the live complex mix to `0.08`--`0.10`.
   Use `complex_mix_min=0.06`, `complex_mix_max=0.18`,
   `complex_down_step=0.04`, and `complex_up_step=0.0` until two consecutive
   validation BPB improvements occur. This keeps graph reasoning active but
   restores ordinary byte-likelihood pressure.

2. **GFlowNet weight guard.** Lower `gflownet_loss_weight` to `0.010`.
   Clamp the adaptive controller to `[0.006, 0.014]` and forbid increases when
   `controller/train_bpb_drift > 0.03`. The current policy loss is already
   improving; it does not need more weight while BPB drifts.

3. **Budget cap during training.** Train with reasoning budgets up to `4`;
   keep budget `8` for evaluation only. The simplex shows budget `8` worsens
   BPB while consuming the same maximum trajectory-token allocation as budget
   `4`.

4. **Gentler entropy shaping.** Keep entropy selective but reduce the squared
   entropy pressure by setting `gflownet_entropy_weight=0.01`. Do not lower the
   target further during recovery; the policy is already narrowing and remains
   highly diverse.

5. **Flow damping.** Increase `trajectory_flow_loss_weight` from `0.003` to
   `0.004` only if the next geometry check still shows path smoothness worse
   than `0.085`. The trajectories are turbulent, but over-damping can erase
   useful graph-of-thought exploration.

6. **Controller hysteresis.** Add a hard recovery mode:

   \[
   r_{\mathrm{complex}}\leftarrow \max(0.06,r_{\mathrm{complex}}-0.04),
   \quad
   \lambda_{\mathrm{GFN}}\leftarrow
   \max(0.006,\lambda_{\mathrm{GFN}}-0.001)
   \]

   whenever `train_bpb_drift > 0.08` or validation fails to improve after a
   positive train-BPB window. Recovery mode exits only after two validation
   improvements and nonpositive recent train-BPB slope.

7. **Checkpoint choice.** Resume from
   `checkpoints/parameter_golf_oai_dense/random_order_step_00023250.pt`. Start
   a fresh controller state for this rollback, not the stale controller state
   from the later run. If validation BPB and budget-2 simplex BPB do not improve
   after the recovery window, keep the same checkpoint but reduce complex mix
   again before considering a deeper rollback.

The expected effect is to keep the useful ToricGT components active while
restoring gradient alignment with BPB. The goal for the next analysis window is
not merely lower GFlowNet loss; it is simultaneous improvement in `val/bpb`,
`complexity/val/bpb`, and budget-2 simplex BPB without a further drop in
GFlowNet diversity below roughly `0.95`.
