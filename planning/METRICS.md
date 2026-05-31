# ToricGT OAI Metrics Audit

## 2026-05-31 DEC/Toric/Relative-K Audit

Training remained paused during this audit.  The checkpoint analyzed was:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00035250.pt
```

The updated analysis suite with DEC conservative reasoning diagnostics,
toric-shadow probes, directed persistence-module morphisms, simplicial
trajectory plots, and reasoning/K/BPB simplices was run at:

```text
outputs/reasoning_geometry_suite/oai-dec-toric-relativek-step35250/
```

W&B analysis run:

```text
https://wandb.ai/amelie-iska-math/toricgt-parameter-golf/runs/54ub7jk1
```

Contact sheets reviewed locally:

```text
outputs/reasoning_geometry_suite/oai-dec-toric-relativek-step35250/contact_triangles.png
outputs/reasoning_geometry_suite/oai-dec-toric-relativek-step35250/contact_tetrahedra.png
outputs/reasoning_geometry_suite/oai-dec-toric-relativek-step35250/contact_trajectories.png
outputs/reasoning_geometry_suite/oai-dec-toric-relativek-step35250/contact_topology.png
```

The initial rerender exposed label crowding in the directed-filtration and
simplex figures after adding DEC panels.  The plotting script was patched and
the analysis rerun; the final local figures are readable enough for audit use.

### Numeric Summary

The two-record, six-branch validation geometry audit is small by design; it is
used to verify diagnostic behavior before resuming training, not to estimate
the final leaderboard BPB.

| metric | value | category | interpretation |
|---|---:|---|---|
| mean BPB | `4.9232` | not competitive but expected for this checkpoint slice | The checkpoint is still far from the challenge target; this audit is about geometry and stability. |
| best branch BPB | `4.5328` | as desired relative to branch sampling | GFlowNet/random-order branch selection provides meaningful variance. |
| mean MST efficiency | `0.4099` | as desired but weak | Trajectories have some organization but are not yet tight. |
| mean path smoothness | `0.1026` | as desired | Hidden paths are not exploding; smoothness remains bounded. |
| directed asymmetry | `0.3850` | as desired | Noncommutative/directed topology is nontrivial rather than collapsing to an undirected complex. |
| directed cycle flux | `4.5e-18` | not strong enough | The oriented cycle signal is almost zero; future training should increase useful directed circulation without increasing divergence. |
| triangle density | `0.3033` | as desired | Step complexes contain higher-order simplices. |
| mean Betti-0 | `2.7569` | as desired | Windows preserve multiple local components at lower radii and connect as radius grows. |
| cycle-rank proxy | `0.0069` | not strong enough | Loops are rare; persistent 1D structure needs more signal if analogical topology is to dominate. |
| boundary residual | `0.0` | as desired | The chain-map/boundary surrogate is numerically stable on this audit. |
| Dirichlet energy | `0.5944` | acceptable | Relational map energy is moderate; no immediate instability. |
| DEC conservation loss | `0.1287` | acceptable | Conservative-flow residual is finite and bounded. |
| DEC mass residual | `0.1201` | acceptable but should decrease | Divergence is visible; training should reduce it without erasing directionality. |
| DEC vorticity drift | `0.0180` | as desired | Vorticity is stable across radii. |
| DEC kinetic drift | `0.0041` | as desired | Energy drift is small. |
| DEC Hodge balance | `0.0099` | as desired | Hodge-weighted energy is consistent with the unweighted flow. |
| DEC wedge/interior residual | `0.0619` | acceptable | Convective consistency is nonzero but not dominating. |
| analogical map loss | `0.1433` | as desired but not strong enough | Soft maps exist and are bounded; they need stronger task coupling. |
| directed map loss | `0.1736` | as desired but not strong enough | Directed maps are stable but still loose. |
| transport entropy | `0.9794` | as desired | Transport is not collapsed. |
| exact H0 map rank | `1.0833` | as desired | Persistence-module morphisms are computed and nontrivial. |
| exact H1 map rank | `0.85` | promising but low sample | There is actual loop-level morphism structure, but not enough to judge strength. |
| exact edge validity | `0.9688` | as desired | Most transported edges remain valid. |
| exact triangle validity | `0.9255` | as desired | Most transported 2-simplices remain valid. |
| HDBSCAN stability | `0.9410` | as desired | Radius-parametrized clusters are stable with low noise. |
| HDBSCAN noise fraction | `0.0590` | as desired | Few trajectory points are discarded as outliers. |
| toric fan cells | `10.8333` | as desired | Fan occupancy is nontrivial. |
| toric mean margin | `0.0341` | not strong enough | Active-face decisions are too close to walls. |
| toric min margin | `0.00025` | not as desired | Some decisions sit essentially on fan walls. |
| toric active-face margin | `-1.5690` | not as desired | Probe active-face teacher is still poorly separated. |
| toric binomial residual | `1.4622` | not as desired | Toric ideal relation checks are weak. |
| toric leaf residual | `1.0018` | not as desired | Noncommutative phase leaves are not yet organizing trajectories strongly. |

### Plot Review

The triangle/simplex plots show that branches split primarily along reasoning
time and K/BPB tradeoff axes.  This is useful: the diagnostic is sensitive to
branch choices.  It is not yet ideal: the best BPB branches are not obviously
at the high-reasoning/high-structure boundary, meaning inference-time scaling
is not yet buying enough compression.

The tetrahedra show branch clouds near interior edges rather than clean
vertices.  That indicates multi-objective tradeoffs are real but not sharply
controlled.  No degenerate all-points-one-corner failure was observed.

The 3D embedding trajectories are nontrivial and branch-diverse.  R0 has a
compact tangled branch cloud; R1 shows a clearer sweeping trajectory.  Energy
landscapes have visible local basins, and Ramachandran-style phase plots show
phase organization rather than a single point collapse.

The toric phase/simplicial trajectory plots show dense local Vietoris--Rips
structure over the phase torus, with visible soft analogical maps between
windows.  This supports the intended toric/tropical/simplicial picture, but the
toric leaf residual says the phase projection is still too loose to be used as
a strong training target.

The topology panels show nested edge and triangle densities increasing with
radius, Betti-0 decreasing as expected, zero boundary residual, finite DEC
conservation/mass residuals, and bounded directed asymmetry.  The weak point is
1D cycle strength: cycle-rank and cycle flux are too small, so future updates
should increase directed cycle signal carefully rather than driving all
directed quantities to zero.

### Relative-K Helper Audit

A standalone complexity pass over 128 validation rows was written to:

```text
outputs/complexity/oai-relative-k-val-40250/complexity_summary.json
```

Important values:

| metric | value | interpretation |
|---|---:|---|
| `graph_relative_to_text_helper_k_lzma_mean` | `-4133.66` | Text strongly helps describe graph projections under LZMA. |
| `text_relative_to_graph_helper_k_lzma_mean` | `-4110.22` | Graph projections also strongly help describe text. |
| `text_graph_information_symmetry_gap_k_lzma_mean` | `32.31` | LZMA helper symmetry is reasonably tight relative to payload sizes. |
| `text_graph_information_symmetry_gap_k_zlib_mean` | `3409.13` | Zlib is much less symmetric on these long payloads; use with caution. |

This supports the implementation choice: analogical helper transfer should use
named helper families and report compressor-specific behavior rather than a
single scalar "K".

### Recommended Resume Policy

Resume from step `35250` with the current low-weight DEC, toric, topology, and
relative-K diagnostics enabled.  Do not increase their training weights yet.
The immediate training objective should remain BPB-first while collecting the
new W&B metrics:

```text
train/analogy_step_dec_*
complexity/train/target_helper_cond_k_*
complexity/train/information_symmetry_gap_k_*
complexity/train/analogical_transfer_relative_k_*
complexity/train/prediction_relative_k_reward_*
```

Next adjustment if the next interval stalls: increase toric entropy/fan-margin
regularization slightly and add a mild directed-cycle floor, but only after
checking that BPB and validation rechecks do not regress.

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

## Recovery Reanalysis: Step 24,000 vs Step 25,500

Date: 2026-05-29 UTC.

Run analyzed: `oai-resume-23250-recovery`
(`amelie-iska-math/toricgt-parameter-golf/7qcwgddn`). Training was paused
before analysis. The latest checkpoint from the paused recovery lineage is
`checkpoints/parameter_golf_oai_dense/random_order_step_00025500.pt`; the older
`25750` and `26000` files in the same directory belong to a previous lineage and
are not part of this recovery run. The comparison target requested here is
`checkpoints/parameter_golf_oai_dense/random_order_step_00024000.pt`.

Generated artifacts:

- W&B metrics export and plots:
  `outputs/recovery_analysis/oai-resume-23250-recovery/step-00025500/metrics/`
- Step 24k simplex:
  `outputs/recovery_analysis/oai-resume-23250-recovery/step-00024000/simplex/`
- Step 24k geometry:
  `outputs/recovery_analysis/oai-resume-23250-recovery/step-00024000/geometry/`
- Step 25.5k simplex:
  `outputs/recovery_analysis/oai-resume-23250-recovery/step-00025500/simplex/`
- Step 25.5k geometry:
  `outputs/recovery_analysis/oai-resume-23250-recovery/step-00025500/geometry/`
- Contact sheets:
  `contact_simplex_geometry.png`, `contact_trajectories_3d.png`, and
  `contact_energy_phase.png` in each step directory.

### Metric Categorization

The automatic metric analysis over steps `23260`--`25730` categorized 138
metrics as desired, 54 as desired but too weak/slow, and 50 as not desired.

Desired:

- `train/gflownet_loss` dropped from median `2.2915` to `1.5215`
  (`-33.6%`). The policy objective is learnable and continues to improve.
- `train/contrastive_loss` dropped from median `0.4283` to `0.2916`
  (`-31.9%`). The contrastive regularizer is not the immediate failure point.
- Artifact-size and parameter-count diagnostics remained stable. The model is
  still within the Parameter-Golf byte budget envelope.

Desired but not strong or fast enough:

- `val/bpb` improved from `4.8951` at step `23500` to `4.89299` at `24000`,
  `4.88988` at `24500`, `4.88681` at `25000`, then regressed slightly to
  `4.88727` at `25500`. The total improvement is real but small
  (`-0.12%` over the interval), and the last point no longer improves.
- `complexity/val/bpb` improved from `4.8510` at `23500` to `4.8212` at
  `25500`, but the curve is noisy and partially decoupled from train BPB.
- `complexity/val/argmax_byte_accuracy` stayed flat at about `0.2007`.
  Better BPB is not yet translating into sharper argmax byte accuracy.
- `audit/future_permutation_logit_error` remains zero. Causal/random-order
  leakage checks are behaving correctly but do not explain quality.

Not desired:

- `train/bpb` rose from median `4.5364` to `4.8395`, with recent slope
  `+0.0384 BPB / 1k steps`. This is the central failure signal.
- `train/loss` and `train/total_loss` rose by the same relative amount
  (`+6.7%` and `+6.3%`), confirming that the BPB rise is not a logging artifact.
- `controller/train_bpb_drift` reached `0.1167`, triggering recovery. Recovery
  reduced complex mix and GFlowNet weight, but not enough to restore the
  training descent basin.
- `train/trajectory_flow_loss` increased from median `0.0639` to `0.0824`.
  Graph-of-thought trajectories became less smooth while policy loss improved.
- `train/toric_memory_entropy` fell from median `0.2714` to `0.2482`.
  Toric memory remains active, but phase-channel usage is narrowing.

### Paired Checkpoint Diagnostics

Simplex evaluation on the same four-record validation subset:

| checkpoint | budget | BPB | MST efficiency | trajectory tokens |
|---:|---:|---:|---:|---:|
| 24,000 | 1 | `3.97288` | `0.69005` | `2048` |
| 24,000 | 2 | `3.97998` | `0.72785` | `12288` |
| 24,000 | 4 | `3.98344` | `0.73602` | `65536` |
| 24,000 | 8 | `3.98184` | `0.72773` | `65536` |
| 25,500 | 1 | `3.98351` | `0.65923` | `2048` |
| 25,500 | 2 | `3.98783` | `0.68580` | `12288` |
| 25,500 | 4 | `3.98996` | `0.69456` | `65536` |
| 25,500 | 8 | `3.98746` | `0.69194` | `65536` |

On this simplex subset, step `24,000` dominates step `25,500` on BPB at every
budget and on MST efficiency at every budget. Extra budget still improves MST
structure, but it does not lower BPB; the budget-1 point remains the best BPB
point for both checkpoints.

Geometry evaluation on eight reasoning-heavy records and eighty branches:

| checkpoint | mean BPB | best BPB | mean answer BPB | best answer BPB | MST efficiency | path smoothness |
|---:|---:|---:|---:|---:|---:|---:|
| 24,000 | `4.730398` | `4.011565` | `4.622873` | `3.560267` | `0.524038` | `0.084782` |
| 25,500 | `4.730176` | `4.020794` | `4.623317` | `3.580964` | `0.507104` | `0.104583` |

The global mean BPB is statistically tied, but step `24,000` has better best
branch BPB, better answer BPB, better MST efficiency, and lower path
roughness. This matters because the intended test-time scaling mode depends on
branch selection: the tails and minima are more relevant than a near-tied branch
mean.

### Plot Interpretation

The step-24k simplex plots show budget points moving toward higher MST
efficiency with small BPB penalty. The step-25.5k simplex plots show the same
direction but with uniformly worse BPB colors and weaker MST movement. In both
cases, the BPB color field does not reward longer reasoning budgets yet; this
means the GFlowNet policy is producing richer trajectories before the base LM
can reliably score or select them.

The 3D trajectory contact sheets show persistent attractor clouds with long
excursions. Step `25,500` is visibly more diffuse on several frontier-reasoning
and math records. Hebrew text remains the most coherent manifold. The
Ramachandran-style phase/energy sheets show useful toric modes but no clean
alignment between phase trajectory and low-BPB terminal branches. The energy
landscape has many shallow local basins rather than a small number of reliable
descent channels.

Mathematically, the training objective has entered a gradient-misaligned regime.
Writing

\[
\nabla \mathcal L =
\nabla \mathcal L_{\mathrm{BPB}}
+\lambda_{\mathrm{GFN}}\nabla \mathcal L_{\mathrm{GFN}}
+\lambda_{\mathrm{flow}}\nabla \mathcal L_{\mathrm{flow}}
+\lambda_{\mathrm{cmp}}\nabla \mathcal L_{\mathrm{complex}},
\]

the observed signs imply

\[
\left\langle \nabla \mathcal L_{\mathrm{BPB}},
\lambda_{\mathrm{GFN}}\nabla \mathcal L_{\mathrm{GFN}}
+\lambda_{\mathrm{flow}}\nabla \mathcal L_{\mathrm{flow}}
+\lambda_{\mathrm{cmp}}\nabla \mathcal L_{\mathrm{complex}}
\right\rangle > 0
\]

over the recent interval. GFlowNet and contrastive objectives are improving in
their own coordinates, but their resultant update is partly adverse to byte
likelihood. The correct response is not to remove ToricGT structure; it is to
temporarily lower the auxiliary-gradient norm until BPB re-enters a descending
basin.

### Updated Proposal Before Resuming

Do not resume from step `25,500`. Resume around step `24,000` using
`checkpoints/parameter_golf_oai_dense/random_order_step_00024000.pt`. This
checkpoint is close enough to retain the useful recovery-run validation
improvement, but it precedes most of the train-BPB drift and preserves better
simplex/geometry behavior.

Recommended low-disturbance repair phase for the next `1000`--`1500` steps:

1. **Resume checkpoint**:
   `checkpoints/parameter_golf_oai_dense/random_order_step_00024000.pt`.

2. **Lower the scheduled LR** from `9e-5` to `7e-5` or `6e-5`.
   The current gradients are not exploding, but the auxiliary-gradient mixture
   is changing the basin too quickly. A lower base LR preserves optimizer state
   while reducing step length.

3. **Throttle complex rows harder**:
   start `complex_mix_ratio=0.03`, set `complex_mix_min=0.00`,
   `complex_mix_max=0.08`, `complex_down_step=0.03`, and keep
   `complex_up_step=0.0` until two consecutive validation-BPB improvements and
   nonpositive train-BPB drift. The previous floor `0.06` was not low enough.

4. **Clamp GFlowNet weight lower during repair**:
   start `gflownet_loss_weight=0.006`, set `gflownet_loss_min=0.003`,
   `gflownet_loss_max=0.010`, and forbid increases while
   `controller/train_bpb_drift > 0.00`. GFlowNet loss is already improving; it
   should not compete with the BPB repair gradient.

5. **Reduce entropy and flow shaping temporarily**:
   set `gflownet_entropy_weight=0.005` and
   `trajectory_flow_loss_weight=0.0015` for the repair window. Entropy/diversity
   are already near saturation; flow loss is rising, so further pressure here
   is currently not buying lower BPB.

6. **Keep the ToricGT-specific mechanisms active**:
   retain hybrid/tropical-ring attention, random-order autoregression,
   graph projections, toric memory, Kolmogorov metrics, and checkpoint
   publishing. The plots show these mechanisms are active; the issue is
   weighting and curriculum, not architectural failure.

7. **Analysis trigger**:
   re-run W&B, simplex, and geometry analysis after `1000` steps and again after
   `1500` steps if the run is still descending. Success criterion: train BPB
   median below the 24k--24.25k median, `val/bpb <= 4.889`, simplex budget-1
   BPB no worse than `3.973`, and MST efficiency no worse than `0.70` at budget
   `2` or `4`.

If the repair phase succeeds, gradually reintroduce complex rows by increasing
`complex_mix_max` to `0.12` only after validation and simplex BPB both improve.
If it fails, rollback to `23250` remains safer than advancing to `25500`.

### Repair Phase Implementation

Implemented in `config/train.parameter_golf_random_order_dense.yaml` before
resuming:

- `lr=0.000065`
- `complex_mix_ratio=0.03`
- `complex_mix_min=0.00`, `complex_mix_max=0.08`, `complex_down_step=0.03`
- `gflownet_loss_weight=0.006`
- `gflownet_loss_min=0.003`, `gflownet_loss_max=0.010`
- `gflownet_increase_drift_guard=0.00`
- `gflownet_entropy_weight=0.005`
- `trajectory_flow_loss_weight=0.0015`
- fresh controller state path:
  `checkpoints/parameter_golf_oai_dense/adaptive_controller_state_24000_repair.json`

Resume target:
`checkpoints/parameter_golf_oai_dense/random_order_step_00024000.pt`.

Next automatic analysis target: first new checkpoint at or above step `25000`
from this repair lineage, then a second manual assessment before deciding
whether to keep the repair schedule, loosen the complex curriculum, or roll
back again.

## Bounce Analysis: Step 24,750 vs Step 26,250

Date: 2026-05-29 UTC.

Run analyzed: `oai-resume-24000-repair`
(`amelie-iska-math/toricgt-parameter-golf/vrpn699z`). Training was paused
before this analysis. The latest checkpoint in that run was
`checkpoints/parameter_golf_oai_dense/random_order_step_00026250.pt`.

Generated artifacts:

- W&B metrics export and plots:
  `outputs/bounce_analysis/oai-resume-24000-repair/step-00026250/metrics/`
- Step 24,750 simplex:
  `outputs/bounce_analysis/oai-resume-24000-repair/step-00024750/simplex/`
- Step 24,750 geometry:
  `outputs/bounce_analysis/oai-resume-24000-repair/step-00024750/geometry/`
- Step 26,250 simplex:
  `outputs/bounce_analysis/oai-resume-24000-repair/step-00026250/simplex/`
- Step 26,250 geometry:
  `outputs/bounce_analysis/oai-resume-24000-repair/step-00026250/geometry/`
- Contact sheets:
  `contact_simplex_geometry.png`, `contact_trajectories_3d.png`, and
  `contact_energy_phase.png` in each paired step directory.

### Metric Categorization

The W&B export through step `26270` categorized 138 metrics as desired, 56 as
desired but too weak/slow, and 48 as not desired.

Desired:

- `val/bpb` improved from `4.897071` at step `24500` to `4.888446` at
  step `26000`. The validation curve is shallow but still descending.
- `train/gflownet_loss` fell from median `2.2214` to `1.5741`
  (`-29.1%`). The graph-of-thought policy objective remains learnable.
- `train/contrastive_loss` fell from median `0.4529` to `0.2881`
  (`-36.4%`), so representation alignment itself is not diverging.
- `audit/future_permutation_logit_error` stayed exactly zero. Random-order
  autoregressive decoding remains causal under the future-permutation audit.

Desired but not strong or fast enough:

- `val/bpb` is improving too slowly: recent slope is about
  `-0.0019 BPB / 1k steps`, far below the competition target trajectory.
- `complexity/val/argmax_byte_accuracy` stayed flat around `0.2007`. The
  likelihood distribution improves marginally but is not sharpening.
- `train/gflownet_entropy` and `train/gflownet_action_diversity` are near
  saturation and slightly declining. Exploration is still broad, but broad
  exploration is not being converted into lower BPB.

Not desired:

- `train/bpb` rose from median `4.5454` to `4.9671` (`+9.28%`), with recent
  slope `+0.1145 BPB / 1k steps`.
- `train/loss`, `train/loss_ema`, and `train/total_loss` rose with the same
  sign, so the BPB rise is not a logging artifact.
- `controller/train_bpb_ema` rose from `4.3144` to `4.7018`, while
  `controller/complex_mix_ratio` had already been reduced to zero and
  `controller/gflownet_loss_weight` to `0.003`. The previous controller could
  detect the failure but could not remove it.
- `train/trajectory_flow_loss` rose by about `+33.8%`. The learned embedding
  trajectories became less smooth while policy loss improved.
- `train/toric_memory_entropy` fell by about `12.2%`. Toric memory remains
  active but is narrowing into fewer slots during the bounce.

### Paired Checkpoint Diagnostics

Simplex evaluation on the same four-record subset:

| checkpoint | budget | BPB | MST efficiency |
|---:|---:|---:|---:|
| 24,750 | 1 | `3.990435` | `0.686287` |
| 24,750 | 2 | `3.995005` | `0.727172` |
| 24,750 | 4 | `3.996752` | `0.736277` |
| 24,750 | 8 | `3.998238` | `0.730134` |
| 26,250 | 1 | `3.988555` | `0.653758` |
| 26,250 | 2 | `3.995069` | `0.683082` |
| 26,250 | 4 | `3.998493` | `0.692304` |
| 26,250 | 8 | `3.994977` | `0.688484` |

Step `26,250` has a slightly better budget-1 simplex BPB, but step `24,750`
has much better MST efficiency at budgets `2`, `4`, and `8`. Since the intended
test-time mode selects among branches, this indicates that later training is
not improving the reasoning geometry even when it marginally improves one
short-budget likelihood point.

Geometry evaluation on eight reasoning-heavy records and eighty branches:

| checkpoint | mean BPB | best BPB | mean answer BPB | best answer BPB | MST efficiency | path smoothness |
|---:|---:|---:|---:|---:|---:|---:|
| 24,750 | `4.738420` | `4.025718` | `4.624610` | `3.561768` | `0.511726` | `0.091433` |
| 26,250 | `4.723754` | `4.014822` | `4.615685` | `3.576611` | `0.500373` | `0.099359` |

The later checkpoint is marginally better on mean branch BPB, but worse on best
answer BPB, MST efficiency, and path smoothness. This is exactly the failure
mode we care about: byte likelihood on generic branches is not enough if the
geometry of high-quality reasoning trajectories becomes rougher and less
tree-efficient.

### Cause

The repeated bounce is not explained by gradient explosion: the plotted gradient
norm is bounded and mostly below `0.3`. It is also not explained by the complex
curriculum alone: by the end of the repair run the controller had already
reduced complex microbatches to zero. The remaining signals point to two causes.

First, the train data stream was replaying long file-level stretches after each
rollback. The dataset loader shuffled files, but then consumed whole Parquet
shards. Because the resume stream restarted from the same seed, the run saw an
easy region followed by a harder region roughly every rollback, producing an
apparent lower-bound ricochet after hundreds of steps. This is a nonstationary
minibatch distribution problem:

\[
\widehat{\nabla \mathcal L}_t
  = \nabla \mathcal L_{\mathcal D_{s(t)}}(\theta_t),
\]

where shard state \(s(t)\) changes slowly. The optimizer is then estimating
different local risks over long contiguous windows, rather than a well-mixed
global risk.

Second, auxiliary gradients remain misaligned with BPB in that hard region.
Writing the update as

\[
\nabla \mathcal L
  = \nabla \mathcal L_{\mathrm{BPB}}
    + \lambda_{\mathrm{MTP}}\nabla \mathcal L_{\mathrm{MTP}}
    + \lambda_{\mathrm{GFN}}\nabla \mathcal L_{\mathrm{GFN}}
    + \lambda_{\mathrm{ctr}}\nabla \mathcal L_{\mathrm{ctr}}
    + \lambda_{\mathrm{flow}}\nabla \mathcal L_{\mathrm{flow}},
\]

the observed behavior implies that the projected auxiliary component is not
reliably descent-aligned with the BPB component in the recent shard window:

\[
\left\langle \nabla \mathcal L_{\mathrm{BPB}},
  \nabla \mathcal L - \nabla \mathcal L_{\mathrm{BPB}}\right\rangle > 0
\]

often enough to raise `train/bpb` even while GFlowNet and contrastive losses
improve in their own coordinates.

### Stabilization Changes Before Restarting at 24,750

Implemented before resuming from
`checkpoints/parameter_golf_oai_dense/random_order_step_00024750.pt`:

1. **Row-group interleaving.** The train loader now shuffles `(file, row_group)`
   units rather than consuming whole shards contiguously. This keeps large
   homogeneous shards from dominating several hundred steps.

2. **Resume-aware stream seed.** The train stream seed is offset by the loaded
   checkpoint step. Rollbacks no longer replay the identical easy-then-hard
   sequence. Validation remains fixed.

3. **Lower optimizer step length.** Base LR changed from `6.5e-5` to `5.0e-5`
   while preserving optimizer state. At step `24750`, this lowers the effective
   cosine LR from roughly `3.4e-5` to roughly `2.6e-5`.

4. **Auxiliary-gradient throttling.**
   - `gflownet_loss_weight: 0.006 -> 0.002`
   - `gflownet_entropy_weight: 0.005 -> 0.0015`
   - `trajectory_flow_loss_weight: 0.0015 -> 0.0004`
   - `contrastive_loss_weight: 0.01 -> 0.003`
   - `mtp_loss_weight: 0.05 -> 0.03`
   - `toric_entropy_loss_weight: 0.003 -> 0.001`

5. **More conservative adaptive bounds.**
   - `gflownet_loss_min/max: 0.001/0.006`
   - `complex_mix_min/max: 0.00/0.04`
   - `train_drift_threshold: 0.01`
   - `recovery_train_drift_threshold: 0.04`
   - `recovery_exit_val_improvements: 3`
   - fresh state path:
     `checkpoints/parameter_golf_oai_dense/adaptive_controller_state_24750_stabilize.json`

The architecture is unchanged: dense Parameter-Golf model, random-order
autoregressive decoding, tropical-ring/hybrid attention, toric memory,
embedding-space GFlowNet policy, graph projections, QAT hooks, Kolmogorov
metrics, and checkpoint publishing remain active. The change is to make the
minibatch risk more stationary and reduce non-BPB gradient pressure until the
model exits the bounce region.

## Restart-From-1K Curriculum

The next run restarts from
`checkpoints/parameter_golf_oai_dense/random_order_step_00001000.pt` with
optimizer state reset.  This keeps the useful early representation learned by
the random-order dense ToricGT language model, but discards Adam moments
accumulated under later curricula that repeatedly hit the same BPB floor.

The YAML now exposes the curriculum through `phase_curriculum`:

| absolute steps | phase | intent |
|---:|---|---|
| 1,000-6,000 | `bpb_stabilization` | pure byte-model recovery with GFlowNet and toric entropy pressure disabled |
| 6,000-12,000 | `light_gflownet_alignment` | reintroduce low-weight embedding-space Graph-of-Thought policy learning |
| 12,000-25,000 | `adaptive_reasoning_curriculum` | allow the controller to add larger graph/composite rows only if BPB stays stable |
| 25,000-50,000 | `compression_and_scaling` | activate QAT and stronger reasoning auxiliaries after the likelihood basin is stable |

The mathematical goal is to keep the early optimization vector close to the
byte-likelihood gradient:

\[
\nabla \mathcal L_t
  \approx \nabla \mathcal L_{\mathrm{BPB},t}
  + \epsilon_t \nabla \mathcal L_{\mathrm{aux},t},
\qquad \epsilon_t \ll 1,
\]

until the model has left the high-curvature initialization region.  Once BPB is
monotonically improving on both train and validation windows, the auxiliary
weights increase in phases rather than by per-step oscillation.  This is the
least invasive change consistent with the evidence: architecture, dataset,
random-order graph decoding, tropical-ring attention, toric memory,
Kolmogorov-complexity monitoring, and checkpoint publishing all remain in
place.

Operational controls added for this run:

1. `--reset-optimizer` lets the step-1000 checkpoint initialize weights without
   preserving stale optimizer moments.
2. Row-group interleaving keeps Parquet shards from creating long homogeneous
   easy/hard stretches.
3. Resume-aware stream seeding prevents rollbacks from replaying the identical
   data order.
4. Phase metrics are logged to W&B as `phase/index` and scalar active weights,
   so later analyses can compare behavior across curriculum boundaries.
5. Checkpoints are still retained every 250 steps, making rollback selection
   cheap for this small model.

### Automated Analysis Handoff

The analysis watcher can now trigger an explicit Codex review handoff after it
writes `SYNOPSIS.md`.  The mechanism is intentionally transparent:

```bash
scripts/codex_training_review_resume.sh \
  --analysis-dir outputs/post_resume_analysis/oai-restart-01000-phased/step-00002500 \
  --checkpoint checkpoints/parameter_golf_oai_dense/random_order_step_00002500.pt \
  --step 2500 \
  --run-path amelie-iska-math/toricgt-parameter-golf/1ouz53jk \
  --training-tmux toricgt_pg_oai \
  --tmux-session toricgt_codex_review_00002500
```

`scripts/watch_training_analysis.py` exposes this through
`--codex-review-hook scripts/codex_training_review_resume.sh` and
`--codex-review-tmux-prefix toricgt_codex_review`.  The hook calls
`codex resume` with a structured prompt that points to the exact metric export,
simplex summaries, 3D trajectory diagnostics, Ramachandran-style plots, energy
landscapes, W&B run, checkpoint, and training tmux.  If `--session-id` or
`CODEX_RESUME_SESSION_ID` is provided, it resumes that session; otherwise it
uses `codex exec resume --last`.  The hook deliberately uses the non-interactive
`exec` path because an interactive `codex resume` can block on CLI update prompts
or TUI screens before the analysis prompt is delivered.

The hook does not itself decide to kill or restart training.  It creates a
review/resume turn whose task is to inspect the just-finished analysis, update
`planning/METRICS.md`, implement small high-impact adjustments if justified,
and either leave the current training run intact or pause and resume from an
evidence-selected checkpoint.

For floor-bounce debugging, the preferred operating mode is now interrupting:

```bash
conda run --no-capture-output -n tokengt env PYTHONPATH=src \
  python scripts/watch_training_analysis.py \
  --checkpoint-dir checkpoints/parameter_golf_oai_dense \
  --start-step 1000 \
  --target-step 2250 \
  --run-path amelie-iska-math/toricgt-parameter-golf/1ouz53jk \
  --output-root outputs/post_resume_analysis/oai-restart-01000-phased \
  --min-mtime-unix "$(cat outputs/post_resume_analysis/oai-restart-01000-phased/start_epoch.txt)" \
  --pause-training-before-analysis \
  --device cuda \
  --precision bf16 \
  --codex-review-hook scripts/codex_training_review_resume.sh \
  --codex-review-tmux-prefix toricgt_codex_review \
  --training-tmux toricgt_pg_oai
```

This waits for a fresh checkpoint, sends `Ctrl-C` to the training tmux, runs the
analysis suite with the GPU freed, then opens a Codex review tmux.  The resumed
review is instructed to restart or continue based on evidence and to schedule
the next interrupting analysis roughly 500 steps later.

### Step 2250 Automation Failure And Direct Fix

The step-2250 analysis completed and wrote:

```text
outputs/post_resume_analysis/oai-restart-01000-phased/step-00002250/SYNOPSIS.md
```

The restart did not happen because the first handoff used interactive
`codex resume`, which blocked on the Codex CLI update prompt.  The hook was
changed to use non-interactive `codex exec resume`, but the handoff still left
restart responsibility in another agent process rather than making the training
loop deterministic.  Going forward, interrupting analysis is the authority for
pausing the run; restart decisions can still be reviewed by Codex, but the
operator-visible tmux state must be checked immediately after each analysis.

The metric diagnosis at step 2250 is:

- `train/bpb` rose from a first-window median of `4.14777` to `4.48084`
  (`+8.03%`) with recent slope about `+0.796 BPB / 1k steps`.
- `train/loss` and `train/total_loss` rose by the same percentage, so this is
  not a logging artifact.
- `gflownet_loss_weight`, `trajectory_flow_loss_weight`,
  `gflownet_entropy_weight`, and toric entropy pressure were zero in phase 1.
  The bounce is therefore a base byte-likelihood optimization issue, not a
  GFlowNet auxiliary-gradient issue.
- The drift begins as the absolute LR approaches `4e-5`.  The phase-1 schedule
  was too aggressive for the step-1000 restart.

Direct control change:

- resume from `random_order_step_00001500.pt`, before the bounce;
- reset optimizer moments;
- lower base LR from `4e-5` to `3e-5`;
- extend warmup from `2000` to `3000`;
- set phase-1 `lr_multiplier=0.70`;
- set phase-1 `mtp_loss_weight=0.0` and reduce contrastive weight to `0.0005`.

This makes the effective LR at step 1500 about `1.05e-5`, versus roughly
`3.0e-5` in the failed branch, and keeps the first recovery window as close as
possible to pure BPB descent.

### Curvature-Aware Restart Update: Step 1250

The step-1500 rollback was still too late.  After re-reading the checkpoint
trajectory as a finite-difference curve, the correct rollback point is step
1250: it is the last checkpoint after the large initial BPB drop but before the
loss trajectory enters the floor-bounce basin.

Let \(m_t\) denote checkpoint BPB at step \(t\).  For checkpoints spaced by
\(h=250\), the first and second finite differences are

\[
\Delta_h m_t=m_{t+h}-m_t,\qquad
\Delta_h^2 m_t=m_{t+h}-2m_t+m_{t-h}.
\]

The observed early BPB values were:

| step | train BPB | train loss | best val BPB |
|---:|---:|---:|---:|
| 1000 | 5.21745 | 3.61646 | 4.69611 |
| 1250 | 3.74245 | 2.59407 | 4.69611 |
| 1500 | 3.59144 | 2.48940 | 4.69611 |
| 1750 | 4.05973 | 2.81399 | 4.69611 |
| 2000 | 4.46622 | 3.09575 | 4.69611 |
| 2250 | 4.53774 | 3.14532 | 4.69611 |

The first differences per 1000 steps are:

| interval | BPB delta | slope / 1k steps |
|---|---:|---:|
| 1000 to 1250 | -1.47500 | -5.90001 |
| 1250 to 1500 | -0.15101 | -0.60403 |
| 1500 to 1750 | +0.46829 | +1.87317 |
| 1750 to 2000 | +0.40649 | +1.62595 |
| 2000 to 2250 | +0.07153 | +0.28610 |

The second differences per \(1000^2\) steps are:

| centered window | \(\Delta_h^2\) BPB | curvature / \(1000^2\) steps |
|---|---:|---:|
| 1000, 1250, 1500 | +1.32399 | +21.18393 |
| 1250, 1500, 1750 | +0.61930 | +9.90880 |
| 1500, 1750, 2000 | -0.06181 | -0.98888 |
| 1750, 2000, 2250 | -0.33496 | -5.35939 |

The first derivative is still negative at step 1250, but its magnitude has
already collapsed by roughly an order of magnitude.  The positive second
difference centered at step 1250 is the warning signal: training is decelerating
hard before the first derivative turns positive.  By step 1500 the model is
already too close to the instability, and by 1750 it has crossed into upward
BPB drift.

This changes the restart policy:

1. restart from `random_order_step_00001250.pt`;
2. reset Adam moments;
3. keep the lower LR and longer warmup from the step-1500 repair;
4. use a fresh adaptive-controller state at
   `checkpoints/parameter_golf_oai_dense/adaptive_controller_state_01250_curvature.json`;
5. run the next interrupting analysis at step 1750, not 2000.

### Hessian-Probe Diagnostics

Finite differences diagnose the observed training trajectory, but they do not
directly measure local parameter-space sharpness.  The next run therefore logs
diagnostic-only Hessian-vector probe metrics on a tiny cropped batch:

\[
H_t=\nabla_\theta^2\mathcal L_t(\theta),\qquad
\widehat{\operatorname{tr}}(H_t)=\frac{1}{K}\sum_{k=1}^K v_k^\top H_t v_k,
\]

where \(v_k\) are Rademacher Hutchinson probes on a bounded subset of large
trainable matrices.  A one-step power iteration also reports a Rayleigh
dominant-curvature estimate

\[
\lambda_{\mathrm{probe}}\approx
\frac{v^\top H_t v}{v^\top v}.
\]

These quantities are not used as second-order optimizer updates.  They are
W&B diagnostics only:

- `hessian/train/trace_per_param`: average signed curvature on the probed
  subspace;
- `hessian/train/trace_abs_per_param`: scale of average curvature regardless
  of sign;
- `hessian/train/dominant_curvature`: Rayleigh estimate of local directional
  curvature;
- `hessian/train/dominant_abs_curvature`: sharpness proxy;
- `hessian/train/probe_grad_norm`: gradient norm on the same loss used for the
  HVP.

Interpretation rule: if BPB turns upward while
`hessian/train/dominant_abs_curvature` or `trace_abs_per_param` rises, the
floor is likely a sharpness/step-size problem and the repair should reduce
effective LR or extend warmup.  If BPB turns upward without Hessian growth, the
repair should focus first on data ordering, curriculum mixture, or auxiliary
loss weights.

### Step 1250 Elbow-Momentum Restart

The first curvature-aware step-1250 run was over-damped.  It reset Adam moments,
used base LR \(3\cdot 10^{-5}\), warmup \(3000\), and phase multiplier \(0.70\).
At step 1250 that gives an effective learning rate of roughly

\[
3\cdot 10^{-5}\cdot 0.70\cdot \frac{1250}{3000}
\approx 8.75\cdot 10^{-6}.
\]

The observed BPB stayed mostly in the \(4.5\)--\(4.8\) range through the first
few hundred post-restart steps, and did not reproduce the steep descent seen
from the phased step-1000 restart.  The likely cause is not auxiliary pressure:
the stabilization phase still zeroes GFlowNet, MTP, trajectory-flow, and toric
entropy losses.  The cause is under-powered optimization after discarding Adam
moments from the already-useful steep descent.

The next restart therefore uses step 1250 as an elbow checkpoint but keeps the
optimizer state in the checkpoint.  The scalar control update is:

- base LR \(4\cdot 10^{-5}\);
- warmup \(2500\);
- stabilization `lr_multiplier=0.95`;
- no optimizer reset;
- same zero-auxiliary BPB-stabilization phase;
- fresh controller state at
  `checkpoints/parameter_golf_oai_dense/adaptive_controller_state_01250_elbow_momentum.json`.

The effective LR near step 1250 is then approximately

\[
4\cdot 10^{-5}\cdot 0.95\cdot \frac{1250}{2500}
\approx 1.9\cdot 10^{-5}.
\]

This is about \(2.2\times\) the over-damped run, but still below the
approximately \(3\cdot 10^{-5}\) effective LR associated with the later
1500-to-1750 bounce.  The next interrupting analysis remains at step 1750 and
must compare BPB slope, second differences, Hessian sharpness, and validation
BPB before choosing whether to continue or roll back again.

### Stream-Aligned Step 1250 Restart

The step-1250 elbow-momentum restart still failed to reproduce the fast
phased-1000 descent.  The discrepancy was not a nats/BPB conversion issue.  In
the fast run, the logged byte metrics were already near

| run | step | metric | value |
|---|---:|---|---:|
| `oai-restart-01000-phased` | 1470 | `train/bpb` | \(3.6426\) |
| `oai-restart-01000-phased` | 1500 | `train/bpb` | \(3.5914\) |
| `oai-restart-01000-phased` | 1500 | `controller/train_bpb_ema` | \(\approx 3.6971\) |
| `oai-restart-01250-elbow-momentum` | 1250--1350 | `train/bpb` | mostly \(4.5\)--\(4.8\) |

The causal difference is data stream alignment.  The trainer offsets the
training stream seed by the loaded checkpoint step:

\[
\texttt{stream\_seed}=\texttt{seed}+\texttt{start\_step}.
\]

The fast trajectory resumed at step \(1000\), hence used stream seed
\(17+1000=1017\) and then consumed \(250\) training steps before reaching step
\(1250\).  The later step-1250 restart loaded compatible weights and optimizer
state, but started a new stream seed \(17+1250=1267\).  That changed the local
data distribution and invalidated the comparison.

The trainer now separates the stream origin from the checkpoint step.  If
\(s_{\mathrm{origin}}\) is the desired data-order origin,
\(s_{\mathrm{resume}}\) is the checkpoint step, and \(G\) is gradient
accumulation, it burns in

\[
B=(s_{\mathrm{resume}}-s_{\mathrm{origin}})G
\]

microbatches from the dataloader before resuming optimization.  For the current
restart,

\[
B=(1250-1000)\cdot 16=4000
\]

microbatches.  No model forward or backward pass is performed during burn-in;
only the iterable dataloader is advanced through the same ordinary/complex
microbatch choices that the earlier run would have consumed.  The run logs
`data/stream_origin_step`, `data/stream_burnin_steps`, and
`data/stream_burnin_microbatches` to W&B so future audits can distinguish
weight checkpoint step from data-order state.

Checkpoint saves now archive an existing periodic checkpoint before replacing
the canonical `random_order_step_*.pt` file.  This preserves rollback targets
from earlier attempts while keeping watcher scripts compatible with the
canonical checkpoint names.

### Step 1500 Capture Restart

The stream-aligned run confirmed the diagnosis: the model recovered the fast
trajectory and reached

\[
\operatorname{BPB}_{1500}=3.5919,
\]

but then bounced upward as warmup continued to raise the effective learning
rate.  The local sequence was:

| step | effective LR | train BPB | interpretation |
|---:|---:|---:|---|
| 1500 | \(2.28\cdot10^{-5}\) | \(3.5919\) | useful basin reached |
| 1600 | \(2.43\cdot10^{-5}\) | \(3.7461\) | still acceptable but no longer improving |
| 1700 | \(2.58\cdot10^{-5}\) | \(3.9367\) | upward drift |
| 1750 | \(2.66\cdot10^{-5}\) | \(4.0616\) | basin exit |

The Hessian probe at the evaluation point did not show a large sharpness
explosion:

\[
\widehat{\lambda}_{\mathrm{abs}}\approx 4.83\cdot 10^{-5},
\qquad
\|\nabla\mathcal L_{\mathrm{probe}}\|\approx 0.165.
\]

This means the failure is best treated as a warmup/curriculum overshoot rather
than a need for a second-order optimizer.  The next run restarts from the
stream-aligned step-1500 checkpoint, keeps the step-1000 stream origin, and uses
a capture schedule:

- lower dropout from \(0.10\) to \(0.05\), because the model is underfitting the
  byte stream and still has validation/quantization audits downstream;
- keep GFlowNet, MTP, trajectory-flow, toric-entropy, and QAT losses off during
  the BPB capture interval;
- use `bpb_capture_1500_2000` with `lr_multiplier=1.05`, so the effective LR
  near 1750 sits near \(2.21\cdot10^{-5}\): faster than the over-damped
  restarts but still below the \(2.66\cdot10^{-5}\) bounce point;
- step down to `lr_multiplier=0.82` for 2000--2500 and `0.65` for 2500--6000,
  approximating a \(2.0\cdot10^{-5}\)--\(2.5\cdot10^{-5}\) capture band after
  the warmup ramp.

This is the intended meaning of "drop harder": use the good step-1500 basin as
the restart point, remove avoidable dropout noise, and prevent the LR warmup
from kicking the model back out of the basin before BPB has time to compress.

### Step 1750 Stream-Aligned Audit And Decision

The interrupting watcher analyzed

```text
outputs/post_resume_analysis/oai-restart-01250-stream-origin-1000/step-00001750/
```

against W&B run

```text
amelie-iska-math/toricgt-parameter-golf/36e5gv5d
```

The stream-alignment fix worked.  The first resumed steps returned immediately
to the earlier low-BPB band, and the aligned step-1500 checkpoint has

```text
random_order_step_00001500.pt: train_bpb = 3.5918896520
```

The step-1750 checkpoint, however, regressed:

```text
random_order_step_00001750.pt: train_bpb = 4.0616284461
```

The W&B export through step 1740 reports:

| metric | category | first median | last median | recent slope per 1k |
|---|---|---:|---:|---:|
| `train/bpb` | undesirable | \(3.6904\) | \(3.9031\) | \(+1.5721\) |
| `train/loss_ema` | undesirable | \(2.5391\) | \(2.7373\) | \(+0.9059\) |
| `train/gflownet_loss` | undesirable as diagnostic | \(2.7252\) | \(2.8615\) | \(+0.4667\) |
| `complexity/train/bpb` | desired | \(4.0257\) | \(3.5192\) | \(-7.8735\) |
| `complexity/train/argmax_byte_accuracy` | desired | \(0.3994\) | \(0.4282\) | \(+0.7617\) |
| `train/trajectory_flow_loss` | desired | \(0.1278\) | \(0.1088\) | \(-0.1487\) |
| `train/toric_memory_entropy` | desired but weak | \(0.2786\) | \(0.2779\) | \(+0.0591\) |

The Hessian probe available at step 1500 was small and not sharp:

\[
\lambda_{\mathrm{abs}}\approx 4.83\cdot 10^{-5},
\qquad
|\operatorname{tr}(H)|/P\approx 5.15\cdot 10^{-5}.
\]

This argues against a hard curvature wall at the best checkpoint.  The observed
failure mode is instead a warmup-induced bounce.  With base LR \(4\cdot10^{-5}\),
warmup \(2500\), and phase multiplier \(0.95\), the effective LR rose from

\[
4\cdot10^{-5}\cdot0.95\cdot\frac{1500}{2500}
\approx 2.28\cdot10^{-5}
\]

near the best checkpoint to

\[
4\cdot10^{-5}\cdot0.95\cdot\frac{1750}{2500}
\approx 2.66\cdot10^{-5},
\]

where the EMA trend turned unfavorable.  The next restart therefore uses the
aligned step-1500 checkpoint, preserves Adam state, keeps stream origin at
step \(1000\), and changes the early capture controls:

```yaml
model.dropout: 0.05
adaptive_training.state_path:
  checkpoints/parameter_golf_oai_dense/adaptive_controller_state_01500_capture_schneller.json
phase_curriculum:
  bpb_capture_1500_2000.lr_multiplier: 1.05
  bpb_capture_2000_2500.lr_multiplier: 0.82
  bpb_stabilization_hold.lr_multiplier: 0.65
```

With base LR \(3.6\cdot10^{-5}\) and warmup \(3000\), this gives effective LR

\[
3.6\cdot10^{-5}\cdot1.05\cdot\frac{1750}{3000}
\approx 2.21\cdot10^{-5}
\]

near step \(1750\), deliberately placing the next 500-step window below the
previously observed bounce band while reducing dropout noise.  At step \(2000\)
the phase drops to multiplier \(0.82\), lowering the LR back to
\(1.97\cdot10^{-5}\) before warmup can push the model out of the basin.

#### Plot-Level Interpretation

The core timeseries plot shows a good descent into step 1500 followed by
large-amplitude sawtooth behavior in `train/bpb` and `train/loss`.  The EMA
minimum occurs after the initial recovery but then bends upward; the second
differences alternate sign, which is characteristic of a stochastic floor
bounce rather than a smooth underfitting plateau.

The correlation heatmap shows `train/bpb` is strongly anti-correlated with
trajectory kinetic and viscous terms.  This is acceptable here: smoother,
lower-energy trajectories are being learned, but byte likelihood is not
benefiting monotonically once LR rises.  Therefore trajectory-flow and toric
terms should not be increased yet.

The simplex and tetrahedron plots use only four records, so they are
diagnostics rather than promotion gates.  Their consistent message is that
more reasoning budget and higher MST efficiency do not yet reliably move
samples toward the low-BPB vertex.  Embedding-space GFlowNet branches remain
diverse, but are not sufficiently likelihood-directed.  Since the stabilization
phase deliberately sets GFlowNet loss weight to zero, this is not a reason to
alter the architecture; it is a reason to let byte compression stabilize before
reactivating reasoning pressure.

The 3D reasoning trajectories and Ramachandran-style phase plots show healthy
branch diversity and clear energy basins, but also long outlier chords for math
and GoT examples.  Hebrew examples are currently the easiest: they achieve the
best answer-span BPB in the geometry audit.  The planned restart keeps the
graph-of-thought machinery enabled for measurement but does not amplify it
during the current BPB-stabilization phase.

#### Categorization

Desired:

- stream-origin burn-in restored the low-BPB data-order trajectory;
- `complexity/train/bpb` decreased materially;
- train argmax byte accuracy improved;
- trajectory flow and viscous dissipation improved;
- toric memory entropy stayed far healthier than in the earlier collapsed run;
- random-order causal audit remains clean.

Desired but too weak or slow:

- GFlowNet entropy and action diversity are high, but the policy is still near
  uniform and not yet directed by likelihood;
- LZMA prediction-target NCD improves only weakly;
- toric entropy is stable but not growing;
- validation has only one deterministic point in this window and should not be
  over-interpreted.

Undesirable:

- `train/bpb`, `train/loss`, and `train/loss_ema` rise after the step-1500
  basin;
- logged GFlowNet loss rises as a diagnostic even though it is not active in
  the stabilization objective;
- zlib NCD worsens on the tiny complexity slice;
- more reasoning budget does not yet dominate BPB in the simplex plots.

Decision: restart from `random_order_step_00001500.pt` with lower effective LR
and fresh controller state, then run the next interrupting analysis at step
2000.

### Harder Faster Longer Supervised Sprint

The first step-1500 capture restart recovered the stream but remained too
conservative.  By roughly step \(1650\), train BPB was still hovering around
\(3.7\)--\(3.9\), not falling below the step-1500 low.  The intended next move
is therefore no longer a basin-capture schedule; it is a supervised compression
sprint:

- disable dropout during the BPB sprint: `model.dropout=0.0`;
- keep the stream-aligned `batch_size=2`, `grad_accum_steps=16` structure;
  batch size \(4\) was tested and immediately OOMed on the tropical-ring path
  at roughly \(23.4\) GB used, and changing accumulation also changes iterable
  data state;
- reduce weight decay from \(0.05\) to \(0.03\), because early BPB is dominated
  by underfit compression rather than overfit memorization;
- raise base LR to \(4.2\cdot10^{-5}\) with warmup \(2500\);
- use a short high-pressure phase `bpb_sprint_1500_1800` with multiplier
  \(1.15\), then a longer hold rather than returning immediately to the
  over-damped setting.

The effective LR targets are:

| step | phase | effective LR |
|---:|---|---:|
| 1500 | `bpb_sprint_1500_1800` | \(2.90\cdot10^{-5}\) |
| 1750 | `bpb_sprint_1500_1800` | \(3.38\cdot10^{-5}\) |
| 1800 | `bpb_sprint_1800_2600` | \(2.72\cdot10^{-5}\) |
| 2250 | `bpb_sprint_1800_2600` | \(3.40\cdot10^{-5}\) |
| 2600+ | `bpb_stabilization_hold` | \(\approx3.0\cdot10^{-5}\) |

This intentionally revisits the aggressive LR band, but with three safeguards
that earlier attempts lacked: zero dropout, no auxiliary losses during the
compression sprint, and preserved stream alignment.  The next watcher
should interrupt at step \(1750\), not \(2000\), because this schedule is
deliberately more forceful; if BPB is not below the old \(3.59\) neighborhood
by then, the issue is no longer insufficient LR and the next intervention
should target data curriculum or objective structure rather than pushing LR
higher.

### Text-First BPB Sprint After Strong-LR Failure

The supervised sprint from `random_order_step_00001500.pt` restored the aligned
data stream, disabled dropout, and raised the effective LR into the
\(2.9\cdot10^{-5}\)--\(3.4\cdot10^{-5}\) band.  The first minibatches did not
drop harder: BPB repeatedly revisited the \(3.8\)--\(4.3\) range.  That is a
useful falsification.  It says the immediate bottleneck is not merely step size;
the model is still spending byte capacity on an input representation that is
too noisy for the early Parameter-Golf objective.

The next restart therefore separates the two roles of the corpus:

- the main BPB stream is text-first with `include_graph_projection=false`;
- the delayed complex curriculum retains graph projection through
  `complex_include_graph_projection=true`;
- the restart remains anchored to the step-1000 stream origin and resumes from
  the step-1500 checkpoint, so the comparison is not confounded by a new file
  cycle.

Mathematically, this reduces the entropy of the next-byte target distribution
seen during the compression sprint without removing the ToricGT reasoning
machinery.  Let \(X=(T,G)\) be the text plus graph-projection input and \(B\) be
the byte target.  Early graph JSON/projection text contributes features whose
mutual information with immediate byte prediction is weak relative to its
surface entropy:

\[
  I(B;G\mid T) \ll H(G\mid T)
\]

in the first few thousand steps.  Keeping \(G\) in the main stream therefore
increases optimization noise for BPB.  Delaying \(G\) until the complex
curriculum turns on preserves the graph-of-thought and toric supervision path
after the byte compressor has a stable basin.  This is a curriculum change, not
an architecture retreat.

The next analysis trigger remains step \(1750\).  Desired behavior is immediate
return to the old step-1500 EMA BPB band or better, followed by a negative
short-window slope.  If the text-first run still ricochets, the next change
should adjust the token-order objective or optimizer state; LR increases have
already been tested and should not be repeated blindly.

### Explicit Easy-Medium-Hard Curriculum

A direct shard sample of `data/curated_hf_shards/train/*.parquet` on
2026-05-29 read 858,102 rows across 80 train shards.  The sampled
`estimated_tokens` distribution was:

| percentile | estimated tokens |
|---:|---:|
| 10 | 42 |
| 25 | 123 |
| 40 | 294 |
| 50 | 489 |
| 60 | 1,047 |
| 75 | 8,393 |
| 90 | 44,070 |
| 95 | 70,400 |
| 99 | 108,100 |

The resulting bucket counts were:

| bucket | rule | sampled rows |
|---|---|---:|
| easy | `estimated_tokens <= 128` | 220,323 |
| medium | `129 <= estimated_tokens <= 384` | 169,047 |
| hard | `estimated_tokens >= 385` | 468,732 |

The top sampled families were Jewish Hebrew text (342,780), frontier
GPT-OSS reasoning (326,482), CoT math (171,258), competition math (10,084),
and Opus reasoning (7,498).  This validates a three-tier curriculum rather
than a binary ordinary/complex split: the long tail is real, but exposing it
too early makes BPB chase high-entropy scaffolding before the byte compressor
has settled.

Implementation decision:

- easy stream: unfiltered text-first rows, active at resume.  A live
  1500-step GraphCG restart showed that the nominal `<=128` bucket was short
  but not BPB-easy, producing early BPB around 4.3--4.5 versus the previous
  unfiltered text-first band around 3.5--3.7;
- medium stream: `129..384`, text-first, starts at step 1650 but has zero
  scheduled mass until step 1800;
- hard stream: `>=385`, technical/reasoning keywords, graph projection enabled,
  starts at step 6000 with a 2--4% cap;
- W&B now logs easy, medium, and hard microbatch fractions separately;
- `complex_*` remains as an alias for hard curriculum so old checkpoints,
  adaptive-controller state, and analysis scripts keep working.

The phase curriculum is now:

| steps | easy | medium | hard | purpose |
|---:|---:|---:|---:|---|
| 1500--1800 | 100% | 0% | 0% | recover low-BPB basin without graph noise |
| 1800--2600 | 80% | 20% | 0% | add moderate reasoning rows after capture |
| 2600--6000 | 65% | 35% | 0% | stabilize text likelihood over broader rows |
| 6000--12000 | 53% | 45% | 2% | reintroduce light GFlowNet and hard graph rows |
| 12000+ | 46--51% | 45--50% | 4% | full reasoning curriculum under BPB guardrails |

The effective LR in the 1500--1800 capture window was also reduced from the
failed \(1.15\) multiplier to \(0.90\).  The text-first run showed good early
minibatches around \(3.44\)--\(3.52\) but began bouncing as the warmup pushed
the effective LR above roughly \(3.0\cdot10^{-5}\).  The curriculum restart
therefore tries to make the descent longer by avoiding the high-curvature band,
not by increasing force.

### GraphCG Lattice-Basis Auxiliary Training

The GraphCG codebase uses a learned editing function of the form
\[
    h = z + \alpha d_k
\]
where a direction basis vector is mapped to a normalized edit direction.  Its
core losses make same-direction edits recognizable, penalize cross-direction
similarity, and optionally sparsify the direction vectors.  The analogy
mechanism paper `arXiv:2602.01992` argues that Transformer analogical reasoning
depends on two separable components: geometric alignment of relational
structure in the embedding space, and application of a functor-like map in the
Transformer layers.  For ToricGT, this is exactly the failure mode we want to
control: the byte-level model should not only reduce BPB; its hidden state
should expose stable, reusable directions for graph-of-thought analogies,
toric shifts, Hebrew root transformations, and technical reasoning moves.

Implementation decision:

- add a learned basis `graphcg_direction_basis` with `auto` directions; on the
  current 24 GB 4090 this resolves to 256 directions under the configured
  memory-safety envelope, while older checkpoints with fewer or no basis rows
  are migrated by initializing the new rows;
- edit hidden codes directly as \(z+\alpha d_k\), avoiding a parameter-heavy
  generator and keeping the artifact budget intact;
- use a direction-class contrastive loss so an edit along direction \(k\) at
  \(\alpha\) matches the same direction at \(2\alpha\) more than other axes;
- penalize basis coherence \(\max_{i\ne j}|\langle d_i,d_j\rangle|\) and
  hidden-coordinate correlation in the learned basis;
- keep the base BPB objective dominant: GraphCG starts at only 0.00005 in
  the 1500--1800 capture window, then rises from 0.0001 to 0.0005 after the
  medium/hard reasoning curriculum is active.

W&B metrics added:

| metric | intended behavior |
|---|---|
| `train/graphcg_loss` | should decline or stay bounded without forcing BPB upward |
| `train/graphcg_code_loss` | lower means edit directions are more identifiable |
| `train/graphcg_orthogonal_loss` | lower means the learned basis is closer to orthogonal |
| `train/graphcg_covariance_loss` | lower means hidden coordinates are less entangled in that basis |
| `train/graphcg_basis_coherence` | should remain low; spikes indicate collapsed edit axes |
| `train/graphcg_axis_variance` | should stay nonzero; collapse means unused basis directions |

The next 1750-step analysis should treat GraphCG as successful only if BPB
continues its early descent while coherence and covariance decrease or remain
stable.  If BPB deteriorates but GraphCG geometry improves, reduce
`graphcg_loss_weight` before changing the base optimizer.  If GraphCG loss is
flat and BPB behaves well, keep it as a low-pressure diagnostic until the
medium/hard curriculum begins.

### Analogical Simplex-Tree / Persistent-Homology Loss

The GraphCG basis alone makes relation vectors steerable, but raw vector
analogies are scale-sensitive.  The new analogical lattice loss therefore
promotes maps between *nested simplicial complexes* built on repeated hidden
relation classes.  For a repeated coarse byte relation \(a\to b\), let
\[
    r_i=\frac{h_{t_i+1}-h_{t_i}}{\|h_{t_i+1}-h_{t_i}\|_2+\epsilon}
\]
be normalized hidden arrows.  Within each relation class, compute the normalized
pairwise distance matrix
\[
    \tilde d_{ij}=d(r_i,r_j)/\operatorname{median}_{p<q} d(r_p,r_q).
\]
For radii
\[
    \rho_1<\rho_2<\cdots<\rho_m,
\]
the trainer builds soft Vietoris--Rips complexes
\[
    K_{\rho_1}\subseteq K_{\rho_2}\subseteq\cdots\subseteq K_{\rho_m},
    \qquad
    A_\ell(i,j)=\sigma((\rho_\ell-\tilde d_{ij})/\tau).
\]
The induced inclusion maps \(I_{\ell,\ell+1}:K_{\rho_\ell}\to K_{\rho_{\ell+1}}\)
are represented on 0-chains by row-stochastic prolongation matrices
\[
    P_\ell=\operatorname{rowsum}^{-1}(A_\ell+I).
\]
Because ToricGT's toric memory is noncommutative and the graph-of-thought
process is directed, the implemented complex is not restricted to symmetric
metric topology.  A second directed flag complex uses an antisymmetric
symplectic-style form on normalized relation vectors,
\[
    \Omega(r_i,r_j)=\langle r_i^{(1)},r_j^{(2)}\rangle
                  -\langle r_i^{(2)},r_j^{(1)}\rangle ,
\]
and directed soft edges
\[
    A_\ell^\rightarrow(i,j)=
    \sigma\left((\rho_\ell-\tilde d_{ij}+\gamma\Omega(r_i,r_j))/\tau\right).
\]
Thus \(i\to j\) and \(j\to i\) need not agree.  The directed inclusion maps
\(P_\ell^\rightarrow=\operatorname{rowsum}^{-1}(A_\ell^\rightarrow+I)\)
are monitored through noncommuting chain-map products
\[
    \|P_{\ell+1}^\rightarrow P_\ell^\rightarrow
      -P_\ell^\rightarrow P_{\ell+1}^\rightarrow\|_F^2,
\]
and directed triangle fluxes \(i\to j\to k\to i\).  This is the topological
analogue of the noncommutative torus layer: transport order matters, but the
nested filtration still supplies scale control.
The auxiliary objective combines:

- a relation-vector functor loss that makes equal coarse arrows share
  displacement vectors;
- a lattice-basis reconstruction loss that expresses arrows in the GraphCG
  direction basis;
- a 0D-persistence/MST proxy from nearest-neighbor barcode lengths;
- a soft clique/triangle closure term for 2-simplex consistency;
- an inclusion penalty \(\|\max(0,A_\ell-A_{\ell+1})\|_F^2\);
- a chain-map commutator penalty
  \[
      \|P_{\ell+1}P_\ell-P_\ell P_{\ell+1}\|_F^2,
  \]
  which measures whether nested smoothing maps commute along the filtration.
- directed transitive closure and directed-cycle penalties for the noncommuting
  flag complex.

This gives a cheap persistent-homology surrogate without adding a heavy PH
dependency to the Parameter-Golf path.  It is still faithful to the toric/GoT
interpretation: analogical reasoning is trained as transport between filtered
local complexes, so a reasoning move can persist across scale rather than
depending on one arbitrary embedding norm.

Additional W&B metrics:

| metric | intended behavior |
|---|---|
| `train/analogy_lattice_loss` | small auxiliary pressure; must not dominate BPB |
| `train/analogy_functor_loss` | relation arrows with the same coarse type become more reusable |
| `train/analogy_basis_loss` | relation arrows become expressible in the GraphCG lattice frame |
| `train/analogy_topology_loss` | filtered local complexes become more stable |
| `train/analogy_barcode_loss` | 0D persistence proxy; should not explode |
| `train/analogy_simplex_closure_loss` | soft triangle closure for local clique consistency |
| `train/analogy_filtration_inclusion_loss` | should remain near zero; nonzero means nesting violations |
| `train/analogy_chain_map_loss` | lower means filtration maps commute more cleanly |
| `train/analogy_directed_topology_loss` | directed flag-complex pressure over repeated arrows |
| `train/analogy_directed_transitive_loss` | lower means directed paths close into directed simplices |
| `train/analogy_directed_cycle_loss` | directed triangle holonomy / cycle-flux imbalance |
| `train/analogy_directed_chain_map_loss` | noncommuting directed inclusion-map diagnostic |
| `train/analogy_directed_asymmetry` | verifies the topology is actually directed |
| `train/analogy_directed_skew_norm` | magnitude of the antisymmetric relation form |
| `train/analogy_filtration_edge_density` | guards against empty or saturated complexes |
| `train/analogy_filtration_triangle_density` | tracks higher-order clique growth |

Periodic analysis now renders these objects rather than relying only on scalar
W&B logs. For each selected reasoning record, the watcher writes directed
filtration curves and noncommutative heatmaps to
`outputs/post_resume_analysis/<run>/step-*/geometry/topology/`. The curves show
edge density, soft triangle density, directed asymmetry, and cycle/holonomy flux
as the filtration radius grows. The heatmaps show normalized hidden-arrow
distances, the antisymmetric toric skew matrix, and directed adjacency at
representative filtration radii. Desired behavior is nested inclusion with
nonzero but bounded asymmetry: useful directed structure should appear before
cycle flux or transitive-closure residuals explode.

The loss enters at \(3\cdot10^{-5}\) in the 1500--1800 capture window, then
ramps slowly.  If BPB ricochets again while these topology metrics improve,
the topology term should be delayed rather than removed; the lower LR is the
first guardrail against repeating the 1700-step floor bounce.

## Step 1750 Directed-Topology Restart Review

Analysis directory:
`outputs/post_resume_analysis/oai-restart-01500-directed-topology/step-00001750/`.
The watcher paused the run at checkpoint
`checkpoints/parameter_golf_oai_dense/random_order_step_00001750.pt`.

### Statistical Readout

The relevant BPB window is not a slow plateau. It is a discrete bounce.  In the
W&B export, `train/bpb` improves to a local minimum of `3.4743` at step 1690,
then jumps to `4.5806` at step 1710.  `train/loss` moves in the same direction,
from `2.4082` at step 1690 to `3.1750` at step 1710.  The clipped gradient
norm jumps to `1.1448` at the same time, and the EMA loss continues rising
through step 1740 (`2.9162`), so this is not merely one noisy plotted point.

The finite-difference interpretation is:

- before step 1700, the first difference of BPB is near-flat to negative, with
  a favorable local minimum;
- at step 1710, the first difference becomes sharply positive;
- after step 1710, the instantaneous BPB partially recovers, but the EMA still
  has positive drift because the filter has absorbed a high-loss impulse.

The best saved checkpoint before the bounce is step 1500. There is no saved
1690 or 1700 checkpoint in the current 250-step checkpoint cadence.

### Metric Categories

| behavior | metrics and plots | explanation |
|---|---|---|
| Desired | directed filtration inclusion, topology heatmaps, toric entropy, GraphCG basis diagnostics, `complexity/train/bpb` trend | The directed complexes are nested, noncommutative asymmetry is nonzero but bounded, cycle flux is low, and toric entropy rises instead of collapsing. These are the intended ToricGT structural diagnostics. |
| Desired but too weak or slow | GFlowNet entropy/diversity, branch/test-time-scaling simplex plots, Kolmogorov/NCD proxies | Diversity remains high, but simplex plots show that extra branch budget is not yet selecting lower-BPB terminals. Complexity BPB improves overall but is sparse and noisy. This is acceptable while GFlowNet training weight is zero in the likelihood-capture phase. |
| Undesirable | `train/bpb`, `train/loss`, `train/loss_ema`, `train/total_loss`, recent `train/gflownet_loss`, step-1710 gradient spike | The likelihood objective bounces off a local floor once effective LR enters the observed high-risk band and a sharp batch arrives. Since the topology and toric metrics are sane, removing the ToricGT structure would be the wrong intervention. |

### Mathematical Explanation

Let \(L_t\) be the supervised byte log loss and \(g_t=\nabla_\theta L_t\).  The
step-1710 event is consistent with a sharp minibatch region in which
\(\|g_t\|\) and the local directional curvature are high relative to the
current warmup LR.  The update
\[
    \theta_{t+1}=\theta_t-\eta_t\,\operatorname{AdamW}(g_t)
\]
then overshoots the local basin even if the gradient is clipped at 1.0.  The
EMA loss behaves as the low-pass recursion
\[
    \bar L_t=(1-\alpha)\bar L_{t-1}+\alpha L_t,
\]
so the positive EMA drift after the point spike is evidence that the optimizer
state has moved into a worse neighborhood, not only that the display saw a hard
batch.

The directed topology metrics argue against blaming the new noncommutative
analogy term.  The inclusion residual is controlled, directed asymmetry is
bounded, and cycle flux is small.  In geometric terms, the filtered complexes
are adding edges and triangles monotonically with radius, while the directed
transport form remains nontrivial.  That is the desired scale-stable analogy
regime.  The failure mode is therefore scalar-control instability, not a
structural-model failure.

### Decision

Do not continue from step 1750. Restart from
`random_order_step_00001500.pt` with optimizer state preserved and reduce the
fragile capture-window controls:

- extend the first sprint to steps 1500--2000;
- lower `lr_multiplier` from `0.90` to `0.72`;
- add phase-local `grad_clip_norm=0.75`;
- reduce the tiny topology/analogy pressure from `3e-5` to `1e-5`;
- delay the medium stream and GraphCG ramp into a gentler 2000--3000 phase.

This keeps random-order autoregressive decoding, tropical ring/hybrid
attention, toric memory, dense contest weights, embedding-space GFlowNet
diagnostics, GraphCG, directed topology, and Kolmogorov diagnostics intact.  It
changes only scalar training controls at the empirically identified bounce
point.

## Step 2,000 HDBSCAN-Topology Retry Review

Analysis directory:

```text
outputs/post_resume_analysis/step-00002000/
```

The HDBSCAN-enabled retry from step 1,500 reached a fresh step-2,000 checkpoint,
then the watcher paused training and ran the full GPU analysis suite.  The
result is clear: the new topology machinery is healthy, but the same local BPB
floor-bounce remains.

### Core Statistics

Checkpoint metadata:

| quantity | value |
|---|---:|
| checkpoint train BPB | `4.232872` |
| checkpoint train loss | `2.934004` |
| best validation BPB stored in checkpoint | `4.696106` |
| analyzed W&B last history step | `1990` |

History finite differences:

| event | value |
|---|---:|
| local minimum train BPB | `3.474773` at step `1690` |
| last analyzed train BPB | `4.246687` at step `1990` |
| largest positive first difference | `+1.005582` from step `1700` to `1710` |
| largest positive second difference | centered at step `1700` |

The automatic metric classifier reports:

| category | count |
|---|---:|
| `as_desired` | `181` |
| `as_desired_but_not_strong_or_fast_enough` | `74` |
| `not_as_desired` | `44` |

The important manually interpreted metrics are:

| metric | category | interpretation |
|---|---|---|
| `train/bpb`, `train/loss`, `train/loss_ema` | undesirable | The likelihood objective again leaves the good step-1690 basin after a sharp positive update. |
| `train/grad_norm` | undesirable in the bounce window | The largest destructive BPB jump coincides with high gradient pressure; clipping alone was not enough. |
| `train/toric_memory_entropy` | desired but monitor | Toric entropy is much healthier than earlier collapsed runs, with mean analysis toric entropy around `0.33` on most reasoning records. |
| `train/trajectory_flow_loss` and viscous dissipation | desired but weak | The overall trend improves, but recent slope turns positive after the bounce, so turbulent trajectories return when BPB worsens. |
| `train/gflownet_entropy` and action diversity | desired but not yet active for BPB | Early phase has GFlowNet objective effectively off; entropy movement is diagnostic rather than a failure. |
| `train/analogy_hdbscan_*` and geometry HDBSCAN metrics | desired | Radius-HDBSCAN stability is high and noise is bounded; the robust topology term is not the cause of the BPB failure. |

### Geometry And Plot Readout

The geometry suite analyzed `96` records and `576` branches:

| metric | value |
|---|---:|
| mean BPB | `4.889242` |
| best BPB | `4.051872` |
| mean answer BPB | `4.876394` |
| best answer BPB | `3.610021` |
| mean MST efficiency | `0.714938` |
| mean path smoothness | `0.046129` |
| mean directed topology asymmetry | `0.467627` |
| mean directed cycle flux | `0.006199` |
| mean HDBSCAN cluster count | `1.107205` |
| mean HDBSCAN noise fraction | `0.200338` |
| mean HDBSCAN stability | `0.799662` |
| inclusion violation | `0.0` |

Manual plot inspection:

- The reasoning/K/BPB triangle shows a dense central-to-upper cloud with the
  lowest-BPB points still near the lower reasoning-time side.  Extra reasoning
  structure is visible, but not yet calibrated into lower BPB.
- The reasoning/K/BPB/MST tetrahedron has a compact cloud with no clean
  low-BPB branch separated at high MST efficiency.  MST structure is useful as
  a diagnostic, not yet as a default reward.
- The 3D GoT trajectories branch into readable basins and terminate through
  solution spans, but branches are tightly clustered near the answer basin after
  long straight transports.  This is stable, not yet selective.
- Ramachandran-style phase plots show noncollapsed phase basins.  The model has
  real directed trajectory structure.
- Energy landscapes have coherent low-energy basins but also broad high-energy
  sheets along long transports.  The post-bounce optimizer state raises the
  energy floor rather than destroying geometry.
- Directed nested-simplicial plots show monotone edge/triangle growth, bounded
  directed asymmetry, low cycle flux, and radius-HDBSCAN stability near `0.8`.
  This is the desired noncommutative topology behavior.

### Mathematical Diagnosis

Let \(L_t\) be the per-step byte cross-entropy and let
\(\bar L_t=\alpha\bar L_{t-1}+(1-\alpha)L_t\) be the logged EMA.  The harmful
event is not a slow curriculum drift.  It is a high-curvature impulse:

\[
  \Delta L_{1710}\gg 0,\qquad
  \Delta^2 L_{1700}\gg 0,
\]

followed by a persistently higher \(\bar L_t\).  In optimizer terms, a hard
batch produces an update

\[
  \theta_{t+1}
  =\theta_t-\eta_t\,m_t/\sqrt{v_t+\epsilon},
\]

whose effective norm remains too large even after ordinary gradient clipping.
The topology terms are not implicated because the filtered complexes satisfy
nested inclusion, the HDBSCAN outlier fraction is bounded, and the toric entropy
is healthy.  Therefore the correct intervention is a scalar update damper around
the empirically observed shock window, not removal of GraphCG, persistent
topology, random-order decoding, tropical/ring attention, toric memory, or
embedding-space GoT diagnostics.

### Decision And Implemented Controls

Do not continue from step 2,000.  Restart from:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00001500.pt
```

The update is deliberately minimal and competition-focused:

1. Keep the architecture and objectives fixed.
2. Lower the vulnerable phase-1500--2000 scalar controls:
   - `lr_multiplier`: `0.72 -> 0.62`;
   - `grad_clip_norm`: `0.75 -> 0.62`;
   - `analogy_lattice_loss_weight`: `1e-5 -> 5e-6`.
3. Make phase 2000--3000 gentler:
   - `lr_multiplier`: `0.80 -> 0.68`;
   - `grad_clip_norm`: `0.85 -> 0.72`;
   - `medium_mix_ratio`: `0.10 -> 0.03`;
   - `graphcg_loss_weight`: `5e-5 -> 2e-5`;
   - `analogy_lattice_loss_weight`: `3e-5 -> 1e-5`.
4. Add an early-window shock guard.  If a step in `[1500,2200)` has both
   high loss relative to the EMA and high gradient norm, scale that single
   optimizer update by `0.30`.  This does not skip hard examples and does not
   change the loss.  It only prevents one high-curvature impulse from knocking
   the weights out of the low-BPB basin.

The guard condition is:

\[
  \left[
    \frac{L_t}{\bar L_t}\ge 1.12
    \ \lor\
    L_t-\bar L_t\ge 0.32
  \right]
  \land
  \|g_t\|\ge 0.70.
\]

When active,

\[
  g_t \leftarrow 0.30\,g_t.
\]

New W&B metrics:

| metric | meaning |
|---|---|
| `train/shock_guard_active` | `1` only on damped high-curvature updates |
| `train/shock_guard_update_scale` | update multiplier, normally `1.0` |
| `train/shock_guard_loss_delta` | \(L_t-\bar L_t\) before EMA update |
| `train/shock_guard_loss_ratio` | \(L_t/\bar L_t\) before EMA update |

Acceptance criteria for the next 500-step retry:

1. BPB should remain below `3.8` through the former 1690--1710 shock window.
2. `train/shock_guard_active` should be sparse, ideally only around the hard
   impulse events.
3. Toric entropy should stay above `0.25`.
4. HDBSCAN stability should remain near `0.75--0.85` and noise below `0.25`.
5. Step-2000 checkpoint BPB should be materially below the failed `4.23`.

## Step 2,000 Shock-Guard Retry Review

Analysis directory:

```text
outputs/post_resume_analysis/oai-restart-01500-shockguard/step-00002000/
```

Reviewed at `2026-05-29 23:35 UTC`.

The first shock-guard retry did not solve the floor-bounce.  It reproduced the
same local minimum and rebound:

| quantity | value |
|---|---:|
| checkpoint train BPB | `4.233731` |
| checkpoint train loss | `2.934598` |
| checkpoint best validation BPB | `4.696106` |
| W&B controller validation BPB at step 1,750 | `5.091155` |
| local minimum train BPB | `3.474819` at step `1690` |
| largest positive first difference | `+1.059298` from `1700` to `1710` |
| largest positive second difference | `+0.943592` centered at the shock |
| hessian dominant absolute curvature | `4.05e-05` |
| hessian probe gradient norm | `0.693909` |

Manual metric categorization:

| group | category | reason |
|---|---|---|
| `train/bpb`, `train/loss`, `train/loss_ema` | undesirable | The likelihood objective again exits the step-1690 basin and remains elevated through step 2,000. |
| validation BPB and complexity validation BPB | undesirable | The score-first validation gate worsens from the inherited `~5.03` controller value to `5.091`, and the complexity validation BPB remains `4.93`. |
| shock guard | desired but too weak | It activates at steps `1710` and `1720`, but a `0.30` update scale is still too large for the high-curvature impulse. |
| gradient norm | undesirable in the shock window | The destructive impulse has grad norm `1.30` before clipping and still knocks the trajectory into a higher-loss basin. |
| GFlowNet entropy/diversity/loss | desired but too weak or slow | In this BPB-capture phase the GFlowNet loss is intentionally off, yet diagnostics drift downward after the BPB rebound. |
| toric memory entropy | desired | Entropy improves from roughly `0.28` to `0.31`, showing the toric memory is not collapsing. |
| trajectory flow/viscous loss | desired but too weak | Overall slope is favorable, but recent slope turns positive after the rebound. |
| radius-HDBSCAN topology | desired | Stability is bounded and improving; the geometry suite reports mean stability `0.8005`, noise `0.1995`, and zero inclusion violation. |
| directed topology | desired | Directed asymmetry is nonzero (`0.4678`) with low cycle flux (`0.0062`), which is the intended noncommutative topology signal. |

Plot inspection agrees with the scalar metrics.  The 2-simplex and tetrahedron
plots show a compact central cloud: useful reasoning structure exists, but the
low-BPB branch has not separated.  3D GoT trajectories terminate through answer
regions, but many branches bunch near the terminal basin after long transports.
Ramachandran-style phase plots are noncollapsed.  Energy landscapes are coherent
but broad, with high-energy sheets that match the post-shock elevated loss.  The
directed nested-simplicial plots are healthy: edge/triangle density grows
monotonically, cycle flux is low, HDBSCAN outlier mass decays with radius, and
noncommutative heatmaps show real antisymmetric skew without pathological
inclusion breaks.

The data mix did not explain the rebound.  The logged microbatch fractions were
`medium=0`, `hard=0`, `complex=0` throughout steps `1600--1990`; the failure is
therefore a high-curvature impulse in the easy stream rather than accidental
medium/hard curriculum exposure.  The right local model is

\[
  L_{t+1}-L_t \approx -\eta_t \lVert g_t\rVert^2
  + \frac{1}{2}\eta_t^2 g_t^\top H_t g_t + \xi_t,
\]

where the observed positive \(\Delta L\) and \(\Delta^2L\) imply that either
the stochastic term \(\xi_t\) or the local curvature term overwhelms the first
order descent term around step `1710`.  Because toric entropy and topology are
healthy, the fix should be scalar optimizer control, not removal of ToricGT
structure.

### Decision

Do not continue from step `2000`.  Restart from:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00001500.pt
```

Implemented minimal scalar changes:

| control | old | new | reason |
|---|---:|---:|---|
| phase 1500--2000 `lr_multiplier` | `0.62` | `0.50` | Keep the retry below the empirical high-curvature band. |
| phase 1500--2000 `grad_clip_norm` | `0.62` | `0.45` | Reduce the maximum impulse norm before Adam updates. |
| phase 1500--2000 `analogy_lattice_loss_weight` | `5e-6` | `0` | Remove all non-BPB auxiliary pressure during capture. |
| shock `loss_ratio` | `1.12` | `1.035` | Detect smaller pre-bounce impulses, not only catastrophic spikes. |
| shock `loss_delta` | `0.32` | `0.08` | Trigger from the observed smaller warning bumps. |
| shock `grad_norm` | `0.70` | `0.55` | Catch the 1700--1720 curvature event earlier. |
| shock `update_scale` | `0.30` | `0.05` | Make shock updates nearly no-op instead of merely damped. |
| phase 2000--3000 `lr_multiplier` | `0.68` | `0.54` | Avoid re-entering the basin boundary immediately after recovery. |
| phase 2000--3000 `grad_clip_norm` | `0.72` | `0.55` | Hold update norms near the stabilized capture window. |
| phase 2000--3000 `medium_mix_ratio` | `0.03` | `0` | Delay medium rows until BPB descent is stable again. |
| phase 2000--3000 `graphcg_loss_weight` | `2e-5` | `1e-5` | Preserve light geometry pressure without competing with BPB. |
| phase 2000--3000 `analogy_lattice_loss_weight` | `1e-5` | `0` | Keep lattice objectives diagnostic until the BPB basin is secure. |

Acceptance criteria for the next review:

1. `train/bpb` may spike on the hard batch itself, but it should recover below
   `3.8` by step `1760` and remain below `4.0` at step `2000`.
2. `val/bpb` at the first validation gate should not exceed the inherited
   controller value by more than `0.02`.
3. `train/shock_guard_active` should be sparse but may fire on warning bumps
   before the main impulse.
4. Toric entropy should remain above `0.25`.
5. Radius-HDBSCAN noise should remain below `0.25` and inclusion violation
   should remain `0.0`.

## Step 2,000 Shock-Quarantine Review

Analysis directory:

```text
outputs/post_resume_analysis/oai-restart-01500-shock-quarantine/step-00002000/
```

Reviewed at `2026-05-30 01:06 UTC`.

The stricter shock-quarantine run reproduced the same failure geometry as the
previous retry.  The checkpoint at step `2000` has `train_bpb=4.234739` and
`best_val_bpb=4.696106`; the live W&B validation gate at step `1750` reported
`controller/val_bpb=5.302042`, `complexity/val/bpb=4.989566`, and
`complexity/val/loss=3.458504`.  The local train-BPB minimum remained at
step `1690` with `train/bpb=3.475286`, `train/loss=2.408885`, and
`grad_norm=0.578810`.  The destructive impulse again appeared at step `1710`,
where `train/bpb=4.774089`, the first difference was `+1.183588`, the second
difference was `+1.068373`, `grad_norm=1.665371`, and the shock guard already
had `update_scale=0.05`.

This changes the diagnosis.  The optimizer update was already almost removed,
so the observed jump cannot be explained primarily as an excessive parameter
step.  The correct decomposition is

\[
  L(\theta_{t+1}; z_{t+1}) - L(\theta_t; z_t)
  =
  \underbrace{L(\theta_t; z_{t+1})-L(\theta_t; z_t)}_{\text{stream/data shock}}
  +
  \underbrace{\nabla_\theta L(\theta_t;z_{t+1})^\top \Delta\theta_t
  + O(\|\Delta\theta_t\|^2)}_{\text{optimizer response}}.
\]

Because the same row-window shock survives a near no-op update, the first term
dominates.  The rollback was continuing the step-1000 stream by burning in to
step `1500`, which reintroduced the same high-entropy easy-stream segment after
the checkpoint.  The fix should therefore rotate the resume-aware stream origin,
not weaken the ToricGT architecture.

Manual categorization:

| group | category | reason |
|---|---|---|
| `train/bpb`, `train/loss`, `train/loss_ema` | undesirable | The train likelihood falls into the same local floor near step `1690` and rebounds sharply at `1710`. |
| validation BPB and complexity validation BPB | undesirable | The first validation gate worsened to `5.302`, so the checkpoint is not promotable. |
| shock guard | desired but no longer sufficient | It detects and damps the shock, but the loss observation itself is dominated by the replayed stream segment. |
| gradient norm | undesirable in the impulse window | The high-entropy row segment produces a large gradient norm even under the easy-only curriculum. |
| GFlowNet entropy/action diversity | desired but too weak | Branching remains noncollapsed, but it does not yet produce a separable low-BPB face. |
| toric memory entropy | desired | The toric channel remains live and improves rather than collapsing. |
| radius-HDBSCAN and directed topology | desired | Mean stability is `0.8068`, noise is `0.1932`, inclusion violation is `0.0`, directed asymmetry is `0.4671`, and cycle flux is only `0.0062`. |
| Hessian probes | not decision-grade | The available summary contains NaNs for several Hessian quantities; use finite differences and observed gradients for this decision. |

Plot inspection:

- `geometry/triangles/reasoning_k_bpb.png`: the branch cloud is still compact
  and central; low BPB is not separated by reasoning time or \(K(x)\).
- `geometry/tetrahedra/reasoning_k_bpb_mst.png`: MST efficiency is useful as a
  diagnostic but does not yet identify a clean low-BPB face.
- `geometry/trajectories/..._trajectory_3d.png`: GoT branches are readable but
  tightly bunched near the answer basin, with long transports into the terminal
  region.
- `geometry/trajectories/..._phase_energy.png`: phase clusters are noncollapsed,
  so toric memory should remain active.
- `geometry/trajectories/..._energy_landscape.png`: broad high-energy sheets
  match the elevated post-shock likelihood floor.
- `geometry/topology/..._directed_filtration.png` and
  `..._noncommutative_heatmaps.png`: nested directed topology is healthy, with
  monotone filtration growth, low cycle flux, real antisymmetric skew, and
  stable radius-HDBSCAN behavior.

### Decision

Do not continue from step `2000`.  Restart again from:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00001500.pt
```

Minimal implemented change:

| control | old | new | reason |
|---|---:|---:|---|
| `data.stream_origin_step` | `1000` | `1500` | Change the deterministic resume-aware training stream after loading the step-1500 checkpoint. |
| `data.stream_burnin_steps` | unset | `0` | Avoid replaying the same 1500--2000 microbatch segment. |

All scalar shock-quarantine controls remain intact.  The model remains dense for
the contest artifact, with random-order autoregressive decoding, hybrid
tropical ring attention, toric memory, embedding-space GFlowNet GoT diagnostics,
GraphCG/analogy topology diagnostics, and Kolmogorov-complexity monitors.

Acceptance criteria for the stream-rotated review:

1. `train/bpb` should avoid the old deterministic 1710 impulse; if a new hard
   row appears, recovery below `3.8` within `50--80` steps is acceptable.
2. Step-2000 checkpoint `train_bpb` should be materially below the failed
   `4.2347`.
3. The first validation gate should be below `5.10`; values near or below the
   inherited controller `5.03` are promotable for this phase.
4. Toric entropy should remain above `0.25`.
5. Radius-HDBSCAN noise should remain below `0.25`, stability above `0.75`, and
   inclusion violation at `0.0`.

## Step 2,000 Stream-Rotate Review

Analysis directory:

```text
outputs/post_resume_analysis/oai-restart-01500-stream-rotate/step-00002000/
```

Reviewed at `2026-05-30 02:35 UTC`.

The stream-rotation change improved topology and removed the exact old single
step-1710 impulse, but it did not produce an acceptable likelihood trajectory.
The checkpoint at step `2000` has `train_bpb=4.280771` and `train_loss=2.967204`;
the step-1750 checkpoint is also post-shelf with `train_bpb=4.075140`, so the
best available restart point remains the inherited pre-shelf step-1500
checkpoint with `train_bpb=3.591890`.  W&B validation at the first gate reported
`controller/val_bpb=5.279697`, `complexity/val/bpb=4.923261`, and
`complexity/val/loss=3.412544`: slightly better than the previous
shock-quarantine validation complexity BPB, but still above the acceptance gate.

The local finite-difference picture changed:

| quantity | value |
|---|---:|
| local minimum train BPB | `3.670442` at step `1550` |
| broad shelf maximum | `5.008273` at step `1720` |
| largest positive first difference | `+0.346669` at step `1690` |
| largest positive second difference | `+0.667419` at step `1690` |
| shock-guard active rows | `1590,1600,1610,1630,1660,1670,1690,1700,1710,1720,1730,1860,1870` |
| hessian dominant curvature | `4.81e-06` |
| hessian trace per parameter | `4.70e-06` |

The Hessian probe is now small and well-conditioned, so this is not a sharp
curvature cliff.  It is a broad stochastic-data shelf: high-loss microbatches
arrive repeatedly, and the step-level shock guard only scales the final
accumulated update after the high-loss rows have already dominated the averaged
gradient estimate.  The right local model is a robust stochastic-estimation
problem:

\[
  g_t=\frac1{M}\sum_{m=1}^M \nabla_\theta \ell(\theta;z_{t,m}),\qquad
  \widehat g_t=\frac1{M}\sum_{m=1}^M w_{t,m}\nabla_\theta \ell(\theta;z_{t,m}),
\]

where \(w_{t,m}=\min(1,c_t/\ell(\theta;z_{t,m}))\) for high-loss microbatches.
The raw loss remains logged, but the optimizer receives a Huber-style gradient
estimate whose influence function is bounded.  This is a smaller and better
targeted control than changing architecture or deleting datasets.

Manual categorization:

| group | category | reason |
|---|---|---|
| `train/bpb`, `train/loss`, `train/loss_ema` | undesirable | The broad 1660--1730 shelf pushes BPB above `5.0` and the step-2000 checkpoint is worse than the previous failed checkpoint. |
| validation BPB | undesirable | `controller/val_bpb=5.279697` is still well above the `5.10` gate. |
| complexity validation BPB | desired but too weak | It improved to `4.923261`, but the improvement is not enough to continue from this checkpoint. |
| step-level shock guard | desired but insufficient | It fires repeatedly and correctly, but it acts too late: after microbatch gradients have already accumulated. |
| Hessian probes | desired | Curvature is small and finite, indicating the remedy should be robust stochastic control, not a lower-order model change. |
| GFlowNet entropy/action diversity | desired but too weak | Entropy and diversity remain noncollapsed, but the low-BPB branch has not separated in the simplex/tetrahedron diagnostics. |
| toric memory entropy | desired | `train/toric_memory_entropy` remains around `0.31`, comfortably above the floor. |
| radius-HDBSCAN topology | desired | Stability improved to `0.8366`, noise fell to `0.1634`, and inclusion violation stayed `0.0`. |
| directed topology | desired | Directed asymmetry remained nonzero (`0.4626`) with low cycle flux (`0.00617`). |

Plot inspection:

- `geometry/triangles/reasoning_k_bpb.png`: the cloud shifted and became more
  elongated, but low-BPB points still do not form a clean boundary face.
- `geometry/tetrahedra/reasoning_k_bpb_mst.png`: the point mass is still
  central; MST efficiency is informative but not yet a selection objective.
- `geometry/trajectories/..._trajectory_3d.png`: branches remain readable and
  terminate in a compact answer basin, but branch separation is weak.
- `geometry/trajectories/..._phase_energy.png`: phase occupancy remains
  noncollapsed, supporting continued toric-memory use.
- `geometry/trajectories/..._energy_landscape.png`: energy surfaces are
  smoother than the previous run, matching the smaller Hessian estimate, but
  there is still a high-energy sheet consistent with the broad BPB shelf.
- `geometry/topology/..._directed_filtration.png`: directed nested topology is
  healthier than before; radius-HDBSCAN outlier mass collapses after the first
  radius and cycle flux remains small.

### Decision

Do not continue from step `2000` or `1750`.  Restart again from:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00001500.pt
```

Implemented minimal controls:

| control | old | new | reason |
|---|---:|---:|---|
| `data.stream_origin_step` | `1500` | `2500` | Avoid replaying either analyzed 1500--2000 data segment. |
| `training.robust_micro_loss_guard_enabled` | absent | `true` | Bound high-loss microbatch gradient influence during early BPB capture. |
| `training.robust_micro_loss_guard_start_step` | absent | `1500` | Start exactly at the rollback checkpoint. |
| `training.robust_micro_loss_guard_end_step` | absent | `2600` | Cover the observed shelf and the next validation interval. |
| `training.robust_micro_loss_guard_ratio` | absent | `1.04` | Catch microbatch losses above the short-run EMA band. |
| `training.robust_micro_loss_guard_delta` | absent | `0.10` | Catch absolute loss jumps comparable to the observed shelf onset. |
| `training.robust_micro_loss_guard_min_scale` | absent | `0.10` | Keep high-entropy rows visible but prevent them from controlling the step. |

The guard is deliberately diagnostic-preserving: it logs raw BPB/loss unchanged
and adds W&B metrics
`train/robust_micro_loss_guard_fraction`,
`train/robust_micro_loss_guard_scale`, and
`train/robust_micro_loss_guard_cap`.

Acceptance criteria for the robust-guard review:

1. Raw `train/bpb` may show difficult rows, but the post-shelf EMA must recover
   below `3.9` before step `2000`.
2. `train/robust_micro_loss_guard_fraction` should be sparse to moderate; a
   value near `1.0` for many consecutive steps means the stream itself is too
   hard for this capture phase.
3. Step-2000 checkpoint BPB should beat both failed checkpoints:
   `4.2347` and `4.2808`.
4. `controller/val_bpb` should be below `5.10`, with `5.03` still the near-term
   promotion target.
5. Toric entropy, radius-HDBSCAN stability, directed asymmetry, and inclusion
   violation should remain in their current healthy bands.

## Step 2,000 Robust-Microguard Review

Analysis directory:

```text
outputs/post_resume_analysis/oai-restart-01500-robust-microguard/step-00002000/
```

Reviewed at `2026-05-30 04:20 UTC`.

The robust microbatch guard fixed the catastrophic local train-BPB ricochet but
created a validation/generalization mismatch.  The checkpoint at step `2000`
has `train_bpb=3.920708` and `train_loss=2.717628`, beating the two previous
failed step-2000 checkpoints (`4.2347`, `4.2808`) but still failing promotion.
The live W&B validation gate reported `controller/val_bpb=5.873126`,
`complexity/val/bpb=5.274160`, and `complexity/val/loss=3.655769`.  The local
training window had a real trough at step `1700` with `train/bpb=3.508630`, but
the last-10 mean by step `1990` had rebounded to `3.706021`.  The checkpoint
aggregate (`3.920708`) was written after a harder microbatch than the last live
logged point (`3.692979`).

Manual categorization:

| group | category | reason |
|---|---|---|
| raw `train/bpb`, `train/loss` | desired but too weak | The violent shelf was removed, but the last half-window is mostly flat/oscillatory rather than sharply descending. |
| `train/loss_ema` | desired but too weak | EMA improved early and then flattened around `2.56--2.58`; the negative first derivative lost magnitude after the step-1700 trough. |
| `controller/val_bpb`, `complexity/val/bpb` | undesirable | Validation BPB worsened to `5.873126`/`5.274160`, so the robust guard biased learning away from validation-like hard rows. |
| robust microbatch guard | desired but too strong | It avoided the train ricochet, but mean last-10 guard fraction was `0.275` with a max of `0.625`; persistent selective downweighting changes the stochastic gradient distribution. |
| gradient norm | desired | Mean last-10 grad norm was `0.3997`, with no explosive spike; curvature control is now adequate. |
| Hessian probe | desired but monitor | Dominant curvature was about `1.41e-4` and trace-per-parameter `1.50e-4`: finite, but higher than the stream-rotate probe, consistent with a broader validation mismatch rather than a single sharp cliff. |
| GFlowNet entropy/diversity | desired but too weak | Entropy stayed near `2.74` and action diversity near `0.99`, but simplex budgets did not lower BPB. |
| toric memory entropy | desired but too weak | Entropy stayed live around `0.258`, above the floor but drifting downward from the best `0.274`. |
| GraphCG / analogy topology losses | mixed | Chain/functor/contrastive losses improved, but barcode, HDBSCAN, topology, lattice, and trajectory-flow losses drifted upward, indicating geometric pressure is not yet aligned with likelihood. |
| directed nested topology | desired | Mean inclusion violation was `0.0`, directed asymmetry `0.4381`, cycle flux `0.00668`, HDBSCAN stability `0.9276`, and noise `0.0724`. |

Plot inspection:

- `metrics/core_metric_timeseries.png`: BPB and loss no longer explode, but
  they oscillate around a shallow basin after step `1700`.  GFlowNet
  entropy/diversity are healthy and noncollapsed.  Grad norm decays into a
  stable band.
- `metrics/recent_metric_slopes.png`: the statistically active slopes are
  mostly complexity-compression terms; train BPB lacks a strong recent negative
  slope.
- `geometry/triangles/reasoning_k_bpb.png`: points remain mostly central.
  Lower BPB is not yet a boundary face of reasoning time or \(K(x)\), so extra
  graph-of-thought compute is not translating into likelihood compression.
- `geometry/tetrahedra/reasoning_k_bpb_mst.png`: MST efficiency is useful but
  the low-BPB points are not cleanly separated by MST efficiency.
- `geometry/triangles/toric_memory_control.png` and
  `geometry/tetrahedra/toric_gfn_bpb.png`: toric memory remains active, but
  low-BPB points lean toward toric/order robustness more than GFlowNet
  diversity.
- `geometry/trajectories/*_trajectory_3d.png`: branches are readable and
  terminate in compact answer basins, but many longer excursions carry no BPB
  payoff.
- `geometry/trajectories/*_energy_landscape.png`: the landscape has a broad
  basin plus a high-energy sheet, matching the validation gap.
- `geometry/trajectories/*_phase_energy.png`: phase occupancy is noncollapsed;
  toric channels should stay enabled.
- `geometry/topology/*_directed_filtration.png`: radius filtrations are
  well-ordered, HDBSCAN noise collapses at moderate radius, asymmetry remains
  nonzero, and cycle flux stays low.

Mathematical explanation:

Let \(z_{t,m}\) be microbatch \(m\) at optimizer step \(t\).  The robust guard
changes the estimator from

\[
  g_t=\frac{1}{M}\sum_m \nabla_\theta \ell(\theta_t;z_{t,m})
\]

to

\[
  \widehat g_t=\frac{1}{M}\sum_m
  w_{t,m}\nabla_\theta \ell(\theta_t;z_{t,m}),\qquad
  w_{t,m}=\min\left(1,\frac{c_t}{\ell(\theta_t;z_{t,m})}\right),
\]

with a floor on \(w_{t,m}\).  This bounded-influence estimator is correct for
removing rare destructive shocks, but if \(P(w_{t,m}<1)\) is persistent, then
\(\mathbb E[\widehat g_t]\neq\nabla_\theta\mathbb E[\ell]\).  The observed
guard fraction near `0.275` means the estimator stopped being a shock guard and
became a curriculum reweighting.  Train BPB improved because the high-loss rows
lost influence; validation BPB worsened because those rows encode part of the
held-out distribution.

The geometry diagnostics argue against changing the architecture.  Directed
filtration inclusion is exact, cycle flux is small, and toric phases are
noncollapsed; the problem is scalar control of the early stochastic estimator
and stream mix.  The fix should shorten and soften the robust guard, rotate the
resume stream again, and add a small medium-difficulty validation-like mix so
the model sees hard-enough bytes without letting them dominate the step.

### Decision

Do not continue from step `2000`.  Restart again from:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00001500.pt
```

Implemented minimal controls:

| control | old | new | reason |
|---|---:|---:|---|
| `data.stream_origin_step` | `2500` | `3500` | Avoid replaying either analyzed 1500--2000 stream segment. |
| `data.medium_start_step` | `1650` | `1500` | Make validation-like rows visible immediately after rollback. |
| `data.medium_mix_ratio` | `0.0` | `0.08` | Add a small medium-difficulty component without flooding hard graph rows. |
| `training.lr` | `4.2e-5` | `3.8e-5` | Slightly reduce the warmup-band update size without freezing descent. |
| `training.warmup_steps` | `2500` | `3500` | Keep the step-1500--2000 window below the previous bounce band. |
| `robust_micro_loss_guard_end_step` | `2600` | `1900` | Stop the guard before it becomes a long-lived bias. |
| `robust_micro_loss_guard_ratio` | `1.04` | `1.10` | Trigger only on larger relative outliers. |
| `robust_micro_loss_guard_delta` | `0.10` | `0.18` | Trigger only on larger absolute outliers. |
| `robust_micro_loss_guard_min_scale` | `0.10` | `0.35` | Keep hard rows materially represented in the gradient. |
| `phase bpb_sprint_1500_2000.medium_mix_ratio` | `0.0` | `0.08` | Ensure the phase override does not erase the new medium mix. |
| `phase bpb_sprint_2000_3000.medium_mix_ratio` | `0.0` | `0.12` | Continue validation-like exposure if step 2000 is promotable. |

The architecture remains unchanged: dense Parameter-Golf weights, random-order
autoregressive graph decoding, hybrid tropical ring attention, toric memory,
embedding-space GFlowNet graph-of-thought diagnostics, GraphCG, directed
simplex/persistence diagnostics, and Kolmogorov-complexity monitors all remain
active.

Acceptance criteria for the next step-2000 review:

1. `train/bpb` should remain below `3.85` for the last-10 mean and avoid any
   single-step shelf above `4.4`.
2. `controller/val_bpb` should recover below `5.30`; below `5.10` is
   promotable for continuing beyond step `2000`.
3. `complexity/val/bpb` should improve from `5.274160`; below `5.00` is the
   near-term gate.
4. Robust guard fraction should stay below `0.20` in the last 100 steps; values
   above that mean the guard is still biasing too much data.
5. Toric entropy should remain above `0.25`, directed topology inclusion
   violation should stay `0.0`, HDBSCAN stability should stay above `0.75`, and
   cycle flux should remain below `0.02`.
