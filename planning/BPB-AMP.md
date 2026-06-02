# BPB Amplification Plan

## Purpose

The current BPB training loop is measuring byte compression on the ToricGT curated
reasoning shards:

```text
data/curated_hf_shards/train/*.parquet
data/curated_hf_shards/validation/*.parquet
data/curated_hf_shards/test/*.parquet
```

Those shards are intentionally hard: math, GoT traces, graph reasoning, Hebrew,
biomedicine, topology, and algebraic synthetic tasks.  They are valuable for
ToricGT reasoning, but they are not the OpenAI Parameter Golf benchmark
distribution.  The competition score is FineWeb validation BPB from the local
Parameter Golf scaffold:

```text
amelie-iska/parameter-golf/data/datasets/fineweb10B_byte260/
amelie-iska/parameter-golf/data/datasets/fineweb10B_sp1024/
```

Therefore the plan is not to discard the difficult ToricGT data or demote it to
a late-stage accessory.  The plan is to separate roles:

1. **ToricGT curated data owns the main training substrate**: hard
   graph-of-thought, tree-of-thought, chain-of-thought, topology, Hebrew, math,
   code, biomedicine, and algebraic reasoning.
2. **Embedding-space GFlowNets and test-time scaling stay central**: the model
   should learn to build, retrieve, extend, and score reasoning trajectories,
   not merely compress ordinary text.
3. **FineWeb owns the competition BPB gate and calibration stream**: it is the
   official score distribution and must shape periodic restarts, mixture ratios,
   and final model selection, but it should not erase the ToricGT reasoning
   objective.

The target is a small Parameter-Golf-compliant ToricGT variant trained primarily
as a reasoning model, with official FineWeb BPB monitored and optimized as the
competition constraint.  In other words, BPB is the scoring gate; graph-structured
reasoning is the capability being trained.

## Core Diagnosis

### 1. The Current BPB Number Is Not The Competition BPB

Observed `train/bpb` values around `3.5--5.0` are being produced on curated
ToricGT shards, not on the official FineWeb validation stream.  Those values are
not comparable to leaderboard values around `1.05--1.22`.  The data distributions
are different enough that a plateau on ToricGT curated shards does not imply the
same plateau on FineWeb.

This mismatch is also an opportunity.  If the model is trained primarily on hard
graph-of-thought, tree-of-thought, and chain-of-thought reasoning data and then
performs strongly on the easier FineWeb BPB stream, that is evidence of useful
OOD transfer: the model has learned reusable compression and prediction
structure from reasoning-heavy data rather than memorizing the FineWeb
distribution.

The evidence must be controlled.  A low FineWeb BPB by itself can come from
dataset easiness, byte/tokenizer convention, or a strong n-gram prior.  The
transfer claim becomes credible only when compared against:

- an untrained or randomly initialized model with the same byte pipeline;
- a FineWeb-only model at the same parameter/artifact budget;
- a ToricGT-hard-data-only model evaluated zero-shot on FineWeb;
- a mixed model with measured FineWeb exposure;
- ablations with GFlowNet, GraphCG, toric/tropical, and topology losses removed.

The metric to report is not just `fineweb/val_bpb`, but also the amount of
FineWeb exposure needed to reach that BPB:

```text
transfer_efficiency = BPB_gain_on_FineWeb / FineWeb_training_tokens_seen
```

High transfer efficiency means the difficult reasoning data is producing
generalizable byte-prediction structure.

### 2. The Hard Data Is Still Useful

The curated shards should remain in training because they supply:

- graph-of-thought reasoning traces;
- mathematical and coding proof structure;
- Hebrew root/template graphs;
- biomedicine, biophysics, and structural reasoning records;
- toric, Toric BGG, tropical, Coxeter, braid, persistence, Koszul, and GraphCG
  auxiliary supervision;
- a memory/retrieval library for analogical inference-time scaling.

The failure was not including this data.  The failure was treating its BPB as the
same score as the Parameter Golf FineWeb BPB.

### 3. Explicit-Zero Config Values Must Stay Safe

The recovery branch previously contained patterns of the form:

```python
float(config_get(config, section, key, default) or default)
```

This silently replaces explicit `0.0` config values with the fallback default.
For example, a configured `shock_guard_update_scale: 0.000` can be logged and
used as `0.35`.  Any BPB floor-lock strategy relying on exact zeros is therefore
invalid if this parser behavior returns.  The current implementation uses
typed config accessors for guard scalars so explicit `0.0` values are honored,
and tests cover the shock guard and robust micro-loss guard zeros.  Future
control scalars should use the same accessors instead of `value or default`.

### 4. The Training Controller Needs Two Gates

The controller has been reacting to many metrics at once.  That is too noisy, but
the solution is not a FineWeb-only objective.  The solution is a two-gate
controller:

```text
reasoning gate:  GoT/ToT/CoT quality, GFlowNet reward/diversity,
                 graph validity, analogy transfer, topology, toric consistency
BPB gate:        official-style FineWeb dev/validation BPB,
                 score-first adaptation legality, artifact-size safety
```

Training changes are accepted only when they improve, or at least preserve, both
gates within tolerance.  If the gates disagree, the update should first adjust
mixture ratios and auxiliary weights rather than discarding either objective.

## Dataset Roles

### Competition Stream

Use the local Parameter Golf scaffold to download the official-compatible
FineWeb binary shards.  The byte model should start with `byte260`, since the
current ToricGT Parameter-Golf model uses a byte-level vocabulary with
`vocab_size: 260` and `byte_offset: 4`.

Smoke download:

```bash
cd amelie-iska/parameter-golf
python3 data/cached_challenge_fineweb.py --variant byte260 --train-shards 10
```

Full local download:

```bash
cd amelie-iska/parameter-golf
python3 data/cached_challenge_fineweb.py --variant byte260 --train-shards 80
```

Expected local layout:

```text
amelie-iska/parameter-golf/data/datasets/fineweb10B_byte260/
  fineweb_train_000000.bin
  ...
  fineweb_val_000000.bin
  ...
```

The fixed FineWeb validation split becomes the primary reported competition BPB.
For development, use a train-shard-derived dev split for frequent tuning and run
the full validation split only at promotion gates.  FineWeb also supplies a small
calibration stream during training so the byte distribution does not drift away
from the competition target.

### Hard ToricGT Stream

Keep:

```text
data/curated_hf_shards/train/*.parquet
data/curated_hf_shards/validation/*.parquet
data/curated_hf_shards/test/*.parquet
```

but rename its metrics in logs so they cannot be confused with the official
FineWeb score:

```text
toricgt_aux/train_bpb
toricgt_aux/val_bpb
toricgt_aux/reasoning_exact
toricgt_aux/gflownet_reward
toricgt_aux/graphcg_alignment
toricgt_aux/persistence_stability
toricgt_aux/toric_commutator
```

The hard stream should train structure and should remain the dominant source of
reasoning behavior.  Its BPB is an auxiliary difficulty metric, not the
competition score.

## FineWeb Adapter

Implement a binary shard loader with strict byte accounting:

```text
src/toricgt/fineweb_bin.py
```

Required behavior:

1. stream `.bin` shards without loading all data into RAM;
2. produce fixed-length contexts and next-token targets;
3. preserve document boundaries when requested;
4. support random windows for training;
5. support deterministic sequential scoring for validation;
6. report denominator bytes exactly;
7. round-trip a sample through the local byte vocabulary to verify the
   `byte_offset` convention.

Add a dedicated official-style evaluator:

```text
scripts/evaluate_parameter_golf_fineweb.py
```

The evaluator must report:

```text
fineweb/val_loss
fineweb/val_bpb
fineweb/val_bytes
fineweb/val_tokens
fineweb/score_first_ttt_bpb
fineweb/neural_only_bpb
fineweb/mixer_bpb
```

## Training Mixture

Use a two-stream batch scheduler:

```text
FineWeb batch:    ordinary byte LM next-token prediction
ToricGT batch:    random-order graph projection + reasoning auxiliary losses
```

Let `rho_t` be the probability that a training step uses the hard ToricGT stream.
Unlike a BPB-only curriculum, `rho_t` should normally be high.  Let `phi_t` be
the probability that a step uses the FineWeb calibration stream.  A typical
relationship is:

```text
rho_t + phi_t = 1
rho_t >= 0.60 except during short FineWeb correction windows
```

The schedule should be phase-dependent and analysis-driven.

### Phase 0: Adapter Validation

```text
steps: no model training
rho_t: unused
goal: verify FineWeb loader, BPB denominator, and baseline reproducibility
```

Acceptance:

- byte round-trip passes;
- validation bytes match the scaffold's expected document count;
- official baseline from `amelie-iska/parameter-golf` can be evaluated in the
  same BPB convention or the numerical difference is explained.

### Phase 1: Reasoning-First Warmup With FineWeb Calibration

```text
steps: 0--2k equivalent
rho_t: 0.80--0.90
phi_t: 0.10--0.20
primary loss: ToricGT hard reasoning LM + graph-of-thought/GFlowNet trajectory loss
auxiliary losses: GraphCG basis, light toric/tropical probes, topology diagnostics
```

Goal: make the model a graph-structured reasoner first, while using FineWeb as a
small but steady language-compression calibration stream.  The model should not
learn a generic byte compressor that later receives reasoning as an afterthought.
It should learn random-order graph autoregression and embedding-space trajectory
search from the beginning.

Recommended loss:

```text
L = L_toricgt_lm
  + lambda_gfn L_gflownet_tb
  + lambda_gcg L_graphcg_frame
  + lambda_trop L_tropical_margin
  + lambda_fineweb L_fineweb
```

Initial coefficient scale:

```text
lambda_fineweb: 0.15--0.35
lambda_gfn:     0.01--0.05
lambda_gcg:     1e-4--1e-3
lambda_trop:    1e-4--1e-3
```

The FineWeb coefficient is large enough to prevent BPB drift, but not large
enough to erase graph-of-thought training.

### Phase 2: BPB-Aware Reasoning Stabilization

```text
steps: after hard reasoning losses and FineWeb dev BPB are both descending
rho_t: 0.65--0.85
phi_t: 0.15--0.35
primary loss: ToricGT hard reasoning + FineWeb calibration
auxiliary losses: GraphCG, GFlowNet, toric probes, topology, memory retrieval
```

Recommended loss:

```text
L = L_toricgt_lm
  + lambda_fineweb L_fineweb
  + lambda_gfn L_gflownet_tb
  + lambda_gfn_entropy L_gfn_entropy
  + lambda_gcg L_graphcg
  + lambda_toric L_toric_probe
  + lambda_top L_topology
  + lambda_memory L_trajectory_memory
```

GFlowNet reward/trajectory balance is now a real training term, not merely a
diagnostic.  Its coefficient should be clipped by the BPB gate: if FineWeb BPB
rebounds, reduce `lambda_gfn` and `lambda_memory` temporarily rather than turning
off graph-of-thought training permanently.

### Phase 3: Full ToricGT Reasoning Activation

```text
steps: after both gates are stable
rho_t: 0.60--0.80
phi_t: 0.20--0.40
primary loss: hard reasoning + FineWeb calibration
auxiliary losses: full scheduled ToricGT suite
```

Activate:

- embedding-space graph-of-thought GFlowNet trajectory/subtrajectory balance;
- GraphCG concept-basis disentanglement;
- tropical active-face/margin regularization;
- toric phase, fan, bend, and binomial relation probes;
- Toric BGG finite-certificate probes: `d^2` residual, standard-filtration
  leakage, Koszul-linearity residual, Gale-dual consistency, and trajectory
  signature smoothness;
- persistence/simplex-tree/Koszul diagnostics and light training losses;
- trajectory-memory retrieval distillation;
- Hebrew root-template heads for Hebrew batches;
- code/math/biomed reasoning consistency heads for relevant batches.

The activation rule is gated by both reasoning and BPB:

```text
if FineWeb val BPB worsens beyond tolerance and reasoning metrics improve:
    increase FineWeb calibration ratio temporarily; keep reasoning losses active
elif FineWeb val BPB improves but reasoning metrics collapse:
    increase hard-data ratio and GFlowNet/topology/GraphCG weights
elif both improve:
    continue or slowly increase context/trajectory length
else:
    reduce LR, rollback to pre-curvature checkpoint, or lower high-variance terms
```

### Phase 4: Competition Finalization

```text
primary: two-gate training and validation
competition gate: FineWeb validation BPB
reasoning gate: ToricGT hard validation, GFlowNet/GoT, topology, toric diagnostics
eval: score-first legal adaptation only
```

At this stage, hard ToricGT data should influence:

- learned priors already baked into the model;
- retrieval libraries built only from training data;
- score-first test-time adaptation proposals that remain legal under Parameter
  Golf rules;
- which graph-of-thought trajectories are retrieved, extended, or pruned during
  inference-time scaling.

## Loss Schedule

Use a reasoning-first, BPB-aware composite loss:

```text
L_total(t)
  = L_toricgt_lm
  + lambda_fineweb(t) L_fineweb
  + a_t L_graphcg
  + b_t L_gflownet
  + c_t L_toric
  + d_t L_topology
  + e_t L_memory
  + q_t L_qat
```

where the coefficients are functions of both FineWeb validation behavior and
ToricGT reasoning behavior.

Suggested controller:

```text
fineweb_slope = d EMA(fineweb/dev_bpb) / d step
fineweb_curv  = d^2 EMA(fineweb/dev_bpb) / d step^2
reason_slope  = d EMA(toricgt_aux/reasoning_loss) / d step
gfn_slope     = d EMA(gflownet/tb_loss) / d step

if fineweb_slope < 0 and reason_slope < 0:
    keep or gently increase trajectory length and hard-data ratio
elif fineweb_slope >= 0 and reason_slope < 0:
    temporarily increase lambda_fineweb and phi_t; do not disable GFlowNet
elif fineweb_slope < 0 and reason_slope >= 0:
    increase hard-data ratio and reasoning auxiliaries
else:
    reduce LR, roll back to pre-bounce checkpoint, and lower high-variance terms
```

The FineWeb stream should prevent competition drift; the ToricGT stream should
prevent the model from becoming a shallow compressor with poor reasoning.

## Immediate Repair Protocol

Do not continue blindly from the step-1600 run.  It is less broken than the
step-1700 run, but the logs show that it is still not learning in the desired
way.  The first repair pass should use the existing useful weights near step
1500, while removing optimizer momentum and fixing the config bug.

### Required First Patch

Patch numeric config parsing so explicit zero values are honored.

Bad pattern:

```python
float(config_get(config, section, key, default) or default)
```

Required pattern:

```python
value = config_get(config, section, key, None)
value = default if value is None else value
```

Add a regression test that sets:

```yaml
shock_guard_update_scale: 0.0
shock_guard_grad_norm: 0.0
robust_micro_loss_guard_min_scale: 0.0
```

and verifies that the effective values are exactly zero, not the defaults.

### Restart Point

Use:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00001500.pt
```

The weights are usable.  The optimizer state is suspect.  Restart with one of:

1. full optimizer reset;
2. Adam moment reset only for embeddings, output projection, revealed-prior
   heads, and auxiliary heads;
3. short low-LR burn-in with moments zeroed and weights retained.

The default should be full optimizer reset for the repair pass, because the
observed rebound is consistent with stale momentum pushing through a narrow
low-curvature basin.

### Short BPB Repair Window

For steps `1500--1700` after restart:

- train on hard ToricGT reasoning data plus a FineWeb calibration stream;
- keep GFlowNet branch generation active;
- keep graph-of-thought trajectory logging active;
- keep analysis plots active;
- keep high-variance auxiliary losses out of the gradient unless the BPB gate is
  stable.

This does not mean GFlowNets are disabled.  It means they are used as branch
samplers and replay teachers during the short repair window, while their
trajectory-balance loss is either zero or very small until BPB stops bouncing.

Evaluate at:

```text
1525, 1550, 1575, 1600
```

not only at 1600.  If FineWeb dev BPB and validation-matched calibration BPB do
not improve by 1550, stop and roll back.  A transient train dip alone is not a
promotion signal.

### Validation-Matched Calibration Slice

Build a training-only calibration slice matched to the evaluation distribution by
source family, token length, byte entropy, and document-boundary structure.  It
must not contain validation rows.  Promotion gates should use:

```text
fineweb/dev_bpb
fineweb/full_val_bpb at sparse gates
toricgt_aux/val_bpb
toricgt_aux/reasoning metrics
```

This prevents selection on a noisy train-only valley.

## Online Expert Mixer

Replace the fixed revealed-context prior with a score-first online mixture.
Fixed prior weights are too brittle: current logs show that the revealed context
can slightly hurt the neural BPB when the fixed coefficient is wrong.  Use
Hedge/exponential weights so the model learns when to trust each prior.

For experts `e = 1,\ldots,E`, define:

```text
q_t(x) = sum_e w_e(t) p_e,t(x)
```

After scoring the actual byte `x_t`, update:

```text
w_e(t+1) = w_e(t) p_e,t(x_t)^eta / Z_t
```

where `Z_t` normalizes the weights.  Equivalently:

```text
w_e(t+1) proportional_to w_e(t) exp(-eta * ell_e,t)
ell_e,t = -log p_e,t(x_t)
```

The update is score-before-update, so it is compatible with prequential
Parameter Golf scoring.  It satisfies the standard exponential-weights regret
bound against the best expert in the pool:

```text
sum_t -log q_t(x_t)
  <= min_e sum_t -log p_e,t(x_t) + O(sqrt(T log E)).
```

Initial expert set:

- neural ToricGT byte distribution;
- revealed-prefix unigram model;
- revealed local-neighbor model;
- byte-class/caseops model;
- tiny hashed n-gram or context-tree model;
- optional trajectory-memory retrieval expert trained only on training data;
- optional GFlowNet branch proposal expert.

Log:

```text
mixer/neural_weight
mixer/unigram_weight
mixer/local_neighbor_weight
mixer/caseops_weight
mixer/ngram_weight
mixer/memory_weight
mixer/gfn_weight
mixer/regret_estimate
mixer/bpb
```

The mixer should be active for FineWeb evaluation and for ToricGT random-order
autoregressive scoring, but every expert must obey the same visible-prefix
constraint.

## Oracle Branch Replay Distillation

Use test-time scaling during training without leaking validation targets.

For each training example:

1. sample several random autoregressive orders and GFlowNet graph-of-thought
   branches;
2. score each branch on the training target;
3. keep the branch with lowest BPB or best reward-adjusted BPB;
4. train a small policy/value head to predict that branch from prefix-visible
   state;
5. add the selected trajectory to the training-only trajectory-memory library.

This converts expensive inference-time scaling into supervised route learning.
At evaluation, the model samples or retrieves better branches because it has
learned which trajectories tend to compress and reason well.

Recommended replay objective:

```text
L_replay
  = CE(pi_theta(branch | visible_state), branch_star)
  + beta (V_theta(visible_state) - reward(branch_star))^2
```

where:

```text
reward(branch) =
  - BPB(branch)
  + alpha_reason reasoning_score(branch)
  + alpha_top topology_stability(branch)
  + alpha_toric toric_consistency(branch)
  - alpha_len trajectory_cost(branch)
```

During the short 1500--1700 repair window, compute branch replay targets but
use a very small replay loss.  After the BPB gate stabilizes, increase replay
loss and GFlowNet trajectory-balance loss together.

## Inference-Time Scaling

Keep random-order autoregressive decoding as the ToricGT competition-facing
method, but separate the scoring modes.

### FineWeb Scoring Mode

The official FineWeb evaluator must be left-to-right or otherwise explicitly
normalized in a score-first prequential way.  If random-order decoding is used,
the evaluator must prove that it emits a normalized probability for each scored
byte before seeing that byte.

### GFlowNet Graph-Of-Thought Mode

Use GFlowNet search as a score-first inference-time scaling layer:

1. encode the prefix and candidate reasoning state;
2. retrieve training-only trajectory memories by GraphCG/topology/toric features;
3. sample graph-of-thought continuations in embedding space;
4. map the selected reasoning trajectory to a distributional bias over next
   bytes;
5. score the next byte;
6. only then update legal online state.

This keeps the ToricGT theory active while respecting the Parameter Golf
prequential constraint.

## Expert Mixture For BPB

Replace fixed revealed-prior weights with an online score-first mixture.

For experts `e = 1..E`, define:

```text
q_t(x) = sum_e w_e(t) p_e,t(x)
```

After scoring byte `x_t`, update:

```text
w_e(t+1) ∝ w_e(t) exp(-eta * -log p_e,t(x_t))
```

This gives a standard exponential-weights regret bound against the best expert
in hindsight:

```text
sum_t -log q_t(x_t)
  <= min_e sum_t -log p_e,t(x_t) + O(sqrt(T log E)).
```

Candidate experts:

- neural ToricGT byte model;
- local prefix byte histogram;
- byte-class/caseops expert;
- hashed n-gram/context-tree expert;
- retrieved trajectory-memory expert;
- graph-of-thought GFlowNet proposal expert.

The update is legal because it occurs after scoring.

## Training-Control Fixes

Before the next serious run:

1. replace all `config_get(... ) or default` numeric parsing with helpers that
   preserve explicit zero;
2. add tests proving `0`, `0.0`, and `false` remain explicit values;
3. log both configured and effective shock/robust guard values;
4. stop using a shock guard until this test passes.

Required helper pattern:

```python
def config_float(config, section, key, default):
    value = config_get(config, section, key, None)
    if value is None:
        return float(default)
    return float(value)
```

## Periodic Analysis Protocol

Every `500--600` steps during BPB capture and every `750--1000` steps afterward:

1. pause training before analysis;
2. run FineWeb dev/validation BPB;
3. run ToricGT curated validation metrics;
4. run geometry/topology/GraphCG/GFlowNet analysis plots;
5. write machine-readable summaries:
   - `metrics/category_summary.json`;
   - `simplex/reasoning_simplex_summary.json`;
   - `geometry/reasoning_geometry_summary.json`;
   - `SYNOPSIS.md`;
   - plot contact sheets and interactive trajectory HTML files when available;
6. call the Codex review hook with those paths;
7. have Codex classify metrics into:
   - desired;
   - desired but weak/slow;
   - not desired;
8. have Codex make one explicit controller decision:
   - `CONTINUE`;
   - `ROLLBACK`;
   - `EDIT_AND_RESTART`;
9. resume from the selected checkpoint in a fresh training tmux;
10. schedule the next interrupting watcher immediately after restart.

The active implementation path should use:

```bash
conda run --no-capture-output -n tokengt env PYTHONPATH=src \
  python scripts/watch_training_analysis.py \
  --checkpoint-dir checkpoints/parameter_golf_oai_dense \
  --start-step <resume_step> \
  --target-step <next_gate_step> \
  --run-path <wandb_entity>/<wandb_project>/<run_id> \
  --output-root outputs/post_resume_analysis/<run_name> \
  --min-mtime-unix "$(cat outputs/post_resume_analysis/<run_name>/start_epoch.txt)" \
  --pause-training-before-analysis \
  --training-tmux <training_tmux> \
  --device cuda \
  --precision bf16 \
  --codex-review-hook scripts/codex_training_review_resume.sh \
  --codex-review-tmux-prefix toricgt_codex_review
```

The hook must hand Codex:

- analyzed checkpoint path;
- analyzed step;
- W&B run path;
- W&B metric-category summary;
- FineWeb BPB summary when implemented;
- ToricGT hard-reasoning validation summary;
- simplex/K/BPB plots;
- 3D trajectory plots;
- toric/tropical/chamber plots;
- directed topology and persistence summaries;
- energy/fitness landscape plots;
- `planning/BPB-AMP.md` and `planning/METRICS.md`.

Promotion requires both gates:

```text
FineWeb gate:
  fineweb/dev_bpb or fineweb/val_bpb improves or remains within tolerance
  AND train curvature is not rebounding

Reasoning gate:
  ToricGT hard-reasoning validation does not collapse
  AND GFlowNet trajectory quality/diversity does not collapse
  AND GraphCG/topology/toric diagnostics remain within tolerance
```

If the gates disagree:

```text
FineWeb worse, reasoning better:
  increase FineWeb calibration ratio and lambda_fineweb temporarily;
  keep GFlowNet active as branch replay/search, but damp high-variance GFlowNet
  gradient terms.

FineWeb better, reasoning worse:
  increase hard-data ratio and restore GraphCG/GFlowNet/topology weights;
  do not promote a shallow compressor checkpoint as the main ToricGT candidate.

Both worse:
  rollback to the last pre-curvature checkpoint, reset optimizer moments, lower
  LR, and shrink high-variance auxiliaries.

Both better:
  promote checkpoint, increase trajectory/context length, and schedule next
  watcher.
```

Codex review is not optional.  A paused analysis gate is incomplete until Codex
has consumed the summaries and either resumed training or selected a different
checkpoint/config.  No paused training tmux should be treated as active after an
analysis gate.

## Concrete Implementation Checklist

1. **Fix config zero parsing.**
   - File: `scripts/train_parameter_golf_random_order.py`
   - Test: explicit `shock_guard_update_scale: 0.0` logs and uses `0.0`.

2. **Add FineWeb binary loader.**
   - File: `src/toricgt/fineweb_bin.py`
   - Supports byte260 first.

3. **Add official-style evaluator.**
   - File: `scripts/evaluate_parameter_golf_fineweb.py`
   - Reports `fineweb/val_bpb` separately from `toricgt_aux/val_bpb`.

4. **Add FineWeb-plus-hard-data config.**
   - File: `config/train.parameter_golf_bpb_amp.yaml`
   - Contains two-stream schedule and auxiliary warmup gates.

5. **Add launch script.**
   - File: `scripts/launch_oai_bpb_amp.sh`
   - Downloads or verifies FineWeb shards, starts training, starts periodic
     Codex analysis watcher.

6. **Update W&B namespaces.**
   - `fineweb/*` for competition score.
   - `toricgt_aux/*` for hard reasoning.
   - `geometry/*`, `topology/*`, `graphcg/*`, `gflownet/*` for audits.

7. **Restart training only after adapter validation.**
   - Do not tune BPB plateaus again until the FineWeb BPB gate is live.

## Expected Outcome

The immediate expected improvement is not that ToricGT curated BPB drops to
`1.2`; that was the wrong benchmark.  The expected improvement is:

1. a correct FineWeb BPB number that is comparable to the Parameter Golf
   leaderboard;
2. a faster descent on FineWeb because the early objective is no longer fighting
   hard reasoning auxiliaries;
3. preservation of ToricGT reasoning capacity because the difficult curated data
   remains active after warmup;
4. cleaner promotion decisions because FineWeb BPB and ToricGT reasoning metrics
   are separated instead of collapsed into one ambiguous scalar.

## Non-Negotiables

- Do not train on FineWeb validation targets before scoring them.
- Do not discard the curated ToricGT hard data.
- Do not let auxiliary losses dominate before FineWeb BPB is stable.
- Do not compare curated-shard BPB to Parameter Golf leaderboard BPB.
- Do not resume serious BPB training until explicit-zero config parsing is
  fixed and tested.
