# ToricGT OAI Metrics Audit

## 2026-06-04 Seq4096 R75 Step-3550 BPB Gate Review

Run:

```text
analyzed run: amelie-iska-math/toricgt-parameter-golf/toricgt_seq4096_4k_recovery_r75_20260604T182013Z
analyzed checkpoint: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_4k_recovery_r75_20260604T182013Z/toricgt_seq4096_4k_recovery_r75_20260604T182013Z_step_003550.pt
analysis: outputs/live_periodic_reviews/toricgt_seq4096_4k_recovery_r75_20260604T182013Z_nonblocking/step-00003550
requested analysis also inspected: outputs/live_periodic_reviews/toricgt_seq4096_4k_recovery_r75_20260604T182013Z/step-00003550
loop: parameter_golf_bpb_target, iteration 4 / 100, target BPB <= 1.2
```

Status:

```text
primary BPB signal: openai_parameter_golf/bpb = 1.2184563243984596
step-3550 FineWeb-style validation BPB/loss: 1.2184563243984596 / 2.057312464685708
step-3550 train BPB/loss: 1.1952148714795368 / 2.0485405921936035
generalization gap: 0.0233 BPB
target gap: 0.018456324398459678
validation slope from 3500 to 3550: -0.0026 BPB per 100 steps
required slope to hit 1.2 by step 4000: -0.004111111111111102 BPB per 100 steps
projected target step from validation slope: 4261.538461538417
sampled seq4096 OAI competition BPB/loss: 1.3283079544692817 / 2.297511398792267
sampled OAI eval scope: 16 local FineWeb sp1024 validation sequences, seq_len=256
requested-analysis sampled OAI BPB/loss: 1.2141407373159094 / 2.1170952320098877
requested-analysis sampled OAI eval scope: 1 local FineWeb sp1024 validation sequence, seq_len=256
metric categories: desired=42, weak_or_slow=29, undesirable=3
artifact probe: int8+zlib total 15,885,422 bytes, 114,578 bytes under 16 MB
live continuation while review was running: step 3600 val BPB/loss = 1.2169 / 2.0547
```

Metric and plot categorization:

- Desired: primary FineWeb-style validation BPB improved monotonically across
  the active loop (`1.221661` at step 3450, `1.219828` at step 3500,
  `1.218456` at step 3550), and the live continuation improved again to
  `1.2169` at step 3600.  Train BPB is below target at the analyzed checkpoint.
  GraphCG basis behavior is strong for token/norm trajectories
  (`spectral_entropy` about `0.948` and `0.928`, effective ranks about `73.4`
  and `65.6`, off-diagonal coherence around `1e-8`).  Koszul `d2` residual is
  zero, BGG resolution and Gale-dual consistency are approximately one, toric
  fan entropy is high for R0/R1/R4, and generated BPB, simplex, 3D trajectory,
  Ramachandran-style phase-energy, energy-landscape, GraphCG, tropical,
  topology, and BGG/Koszul plots were present and inspectable.
- Desired but too weak or slow: validation BPB remains above the `1.2` target,
  and the measured descent velocity is too small for a step-4000 crossing.
  Transfer has only one paired validation interval in this analysis.  The
  sampled OAI competition probe is a short sp1024/seq_len-256 check and is
  worse (`1.3283`) than the run's primary W&B FineWeb-style BPB, so it is a
  calibration warning rather than a promotion metric.  GraphCG/MST structure is
  weak for layer and attention trajectories: R2/R3 effective ranks are only
  about `2.68` and `3.03`, MST efficiencies are `0.0568` and `0.0653`, and
  Slepian leakage is high (`0.729` and `0.641`).  GFlowNet/branch replay and
  test-time-scaling evidence remain sidecar/proxy level.
- Undesirable: the step-3550 run is still off-track for the 4K gate, train
  samples remain noisy (`1.1209` train BPB at step 3537 but `1.2255` train BPB
  at live step 3600), complexity/NCD remains high (`recent_full_log_ncd_lzma`
  about `0.8875`), and tropical chamber motion is active enough to be a
  pressure signal rather than a settled basin (`R0` crossing rate about
  `0.963`, plateau pressure about `1.17`).  No Hessian trace/sharpness probe was
  available, and validation second differences are underdetermined from the
  two-point r75 validation slice.

Mathematical/statistical interpretation:

- FineWeb BPB is the competition gate.  The validation first difference from
  step 3500 to 3550 is negative, so the rollback trigger of a positive first
  derivative is not met.  The velocity shortfall is real:
  `0.004111 - 0.002600 = 0.001511` BPB per 100 steps, which projects target
  crossing near step `4262`, but this is a slow-descent problem rather than
  evidence of a floor-bounce basin at the analyzed boundary.
- The train/validation gap is about `0.0233` BPB, so train dips are not
  sufficient authority.  The proposal's `best_observed_bpb=1.1209` is a noisy
  train sample, not the competition validation gate.  The live step-3600
  validation improvement supplies an immediate control against overreacting to
  that train-sample minimum.
- The two-gate rule remains intact.  The BPB gate is improving but too slow;
  the reasoning gate is alive but uneven.  GraphCG and BGG/Koszul consistency
  support keeping structural sidecars, while weak layer/attention MST,
  Slepian leakage, analogical transport weakness, and tropical churn argue
  against increasing pre-threshold auxiliary weight.
- A strong FineWeb-style result after limited FineWeb exposure may still be
  OOD transfer from hard reasoning data, but this run must not overclaim it.
  Controls remain required against tokenizer effects, n-gram/context baselines,
  dataset easiness, FineWeb-only exposure, hard-only exposure, mixed-budget
  exposure, and auxiliary ablations.

The adjustment proposal was rerun on the analysis directory:

```text
/home/iska/miniconda3/envs/tokengt/bin/python scripts/propose_training_adjustments.py \
  --analysis-dir outputs/live_periodic_reviews/toricgt_seq4096_4k_recovery_r75_20260604T182013Z_nonblocking/step-00003550 \
  --target-bpb 1.2 --gate-step 4000 --checkpoint-step 3550 \
  --wandb-run-path amelie-iska-math/toricgt-parameter-golf/toricgt_seq4096_4k_recovery_r75_20260604T182013Z
```

It recommended `restart_from_best_checkpoint_with_damped_structural_sidecars`.
I used that as evidence, not authority, and rejected it for this handoff because
the primary validation derivative is negative, Hessian/second-difference
evidence is absent, the recommendation is driven partly by a train-sample
minimum and by missing the step-4000 velocity, and the live continuation already
improved to `val_bpb=1.2169` at step 3600.

Decision: `CONTINUE`.

Reason: the analyzed checkpoint is the best primary validation BPB in the loop,
the first derivative is still negative, and no curvature evidence justifies a
rollback.  The run is still off-track for a 4K crossing, so the BPB-first review
loop remains active, but the smallest high-impact action is to keep the same
config and let the active supervisor/watchers collect the next nonblocking
checkpoint review.  No JEPA, architecture, tokenizer, data, model-training
code, config scalar, or checkpoint artifact change was made.  A watcher-only checkpoint
filename compatibility patch was made so the generic non-pausing watcher can
see Seq4096 `<run>_step_*.pt` checkpoints.

No better-strategy sentinel was written.  The BPB-first loop remains active
until `BPB <= 1.2`, 100 review iterations complete, or a better documented
strategy replaces ordinary recovery replays.

Operational handoff at the decision boundary:

```text
action: CONTINUE
active supervisor: scripts/watch_seq4096_4k_recovery.py
active supervisor tmux: toricgt_seq4096_4k_gate_r75_20260604T182013Z
active supervisor log: logs/toricgt_seq4096_4k_recovery_r75_20260604T182013Z.4k_gate.txt
active training tmux: toricgt_seq4096_4k_recovery_r75_20260604T182013Z
active training process: train_gpt.py verified running
active W&B run: amelie-iska-math/toricgt-parameter-golf/toricgt_seq4096_4k_recovery_r75_20260604T182013Z
active training log: amelie-iska/parameter-golf/logs/toricgt_seq4096_4k_recovery_r75_20260604T182013Z.txt
active checkpoint dir: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_4k_recovery_r75_20260604T182013Z
analyzed checkpoint: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_4k_recovery_r75_20260604T182013Z/toricgt_seq4096_4k_recovery_r75_20260604T182013Z_step_003550.pt
latest live checkpoint at decision time: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_4k_recovery_r75_20260604T182013Z/toricgt_seq4096_4k_recovery_r75_20260604T182013Z_step_003600.pt
latest live validation BPB/loss: 1.2169 / 2.0547 at step 3600
non-pausing live watcher: toricgt_seq4096_4k_recovery_r75_nonblocking_watch_3550
non-pausing watcher implementation: scripts/watch_seq4096_analysis.py
non-pausing watcher output root: outputs/live_periodic_reviews/toricgt_seq4096_4k_recovery_r75_20260604T182013Z_nonblocking
next watcher target observed: step 3600 analysis started without pausing training
requested handoff loop status: outputs/live_periodic_reviews/toricgt_seq4096_4k_recovery_r75_20260604T182013Z/step-00003550/logs/bpb_codex_loop_status.json
nonblocking loop status: outputs/live_periodic_reviews/toricgt_seq4096_4k_recovery_r75_20260604T182013Z_nonblocking/step-00003550/logs/bpb_codex_loop_status.json
additional requested generic watcher: toricgt_watch_training_analysis_r75_3650
additional generic watcher implementation: scripts/watch_training_analysis.py
additional generic watcher log: logs/toricgt_seq4096_4k_recovery_r75_20260604T182013Z.watch_training_analysis_3650.txt
additional generic watcher target: checkpoint >= 3650 on CPU, no pause-training flag
loop env: BPB_TARGET=1.2, BPB_MAX_REVIEW_ITERATIONS=100,
          BPB_LOOP_STATE=/home/iska/Documents/amelie/bio/ToricGT/outputs/toricgt_seq4096_4k_recovery_r75_20260604T182013Z_live_bpb_codex_loop_state.json,
          BPB_LOOP_STOP_FILE=/home/iska/Documents/amelie/bio/ToricGT/outputs/toricgt_seq4096_4k_recovery_r75_20260604T182013Z_live_bpb_codex_loop_stop,
          BPB_LOOP_NAME=parameter_golf_bpb_target
```

The next analysis is non-interrupting.  The existing compact watcher
`scripts/watch_seq4096_analysis.py` is still running, and the requested generic
`scripts/watch_training_analysis.py` sidecar is now also running on CPU for
step 3650 without `--pause-training-before-analysis`.  The generic watcher was
patched to match both `random_order_step_*.pt` and Seq4096 `<run>_step_*.pt`
checkpoint names.

Post-decision live-state update:

After the `CONTINUE` decision, the r75 supervisor analyzed the live step-3600
checkpoint and observed another improvement (`val_bpb=1.2169`, `val_loss=2.0547`)
but still projected a miss of the step-4000 BPB gate.  A concurrent supervisor
review escalated the active path to r77.  This is not a reversal of the
step-3550 decision: the finite difference stayed negative, but the off-track
velocity triggered the supervisor's projected-miss recovery policy after the
next checkpoint.  The stale r76 recovery attempt was stopped; r77 is the active
training and supervisor path.

```text
active action after supervisor escalation: r77 recovery continuation from r75 step 3600
active training tmux: toricgt_seq4096_4k_recovery_r77_20260604T183710Z
active supervisor: scripts/watch_seq4096_4k_recovery.py
active supervisor tmux: toricgt_seq4096_4k_gate_r77_20260604T183710Z
active supervisor log: logs/toricgt_seq4096_4k_recovery_r77_20260604T183710Z.4k_gate.txt
active W&B run: amelie-iska-math/toricgt-parameter-golf/toricgt_seq4096_4k_recovery_r77_20260604T183710Z
active training log: amelie-iska/parameter-golf/logs/toricgt_seq4096_4k_recovery_r77_20260604T183710Z.txt
active checkpoint dir: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_4k_recovery_r77_20260604T183710Z
resume checkpoint: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_4k_recovery_r75_20260604T182013Z/toricgt_seq4096_4k_recovery_r75_20260604T182013Z_step_003600.pt
resume controls: reset_optimizer=0, reset_rng=0, reset_loader=0
r77 scalar controls: tied_embed_lr=0.0345, matrix_lr=0.018, scalar_lr=0.018,
                     bigram_bias_lr=0.014, advanced_loss_scale=0.018,
                     graphcg=0.032, toric_tropical=0.006, slepian=0.02,
                     koszul_bgg=0.0006, analogy=0.0006
generic non-pausing watcher tmux: toricgt_watch_training_analysis_r77_3650
generic non-pausing watcher log: logs/toricgt_seq4096_4k_recovery_r77_20260604T183710Z.watch_training_analysis_3650.txt
generic non-pausing watcher output root: outputs/live_periodic_reviews/toricgt_seq4096_4k_recovery_r77_20260604T183710Z_watch_training_analysis
generic non-pausing watcher target: checkpoint >= 3650 on CPU, no pause-training flag
generic watcher loop env: BPB_TARGET=1.2, BPB_MAX_REVIEW_ITERATIONS=100,
                          BPB_LOOP_STATE=/home/iska/Documents/amelie/bio/ToricGT/outputs/toricgt_seq4096_4k_recovery_r75_20260604T182013Z_live_bpb_codex_loop_state.json,
                          BPB_LOOP_STOP_FILE=/home/iska/Documents/amelie/bio/ToricGT/outputs/toricgt_seq4096_4k_recovery_r75_20260604T182013Z_live_bpb_codex_loop_stop,
                          BPB_LOOP_NAME=parameter_golf_bpb_target
```

The active r77 gate supervisor and its compact `watch_seq4096_analysis.py`
sidecar were launched by the supervisor/concurrent review using r77-local loop
state files.  The required generic `scripts/watch_training_analysis.py` sidecar
for the next review preserves the supplied r75 loop environment and is
non-blocking; training remains active while it waits for the step-3650
checkpoint.

Implementation note: the generic watcher uses its invoking Python interpreter
for child analysis subprocesses, avoiding tmux environments where a plain
`python` executable is not present on `PATH`.  It also treats simplex and
geometry suites as optional so a RandomOrderLM-only diagnostic can fail cleanly
on compact Seq4096 GPT checkpoints while BPB/W&B analysis and the proposal
still complete.

Latest live-state update:

The r77 step-3650 generic watcher completed the BPB/W&B sidecar path without
pausing training.  It recorded `val_bpb=1.2171306166080218`, a small positive
validation bump from the r77 step-3600 best of `1.2169196232244919`, and
launched the next Codex review as a non-blocking sidecar.  The active supervisor
then escalated to the damped r78 path, which resumes the r75 step-3600
checkpoint with optimizer/RNG/loader state preserved and lower auxiliary
pressure.

```text
current active training tmux: toricgt_seq4096_4k_recovery_r78_20260604T184450Z
current active W&B run: amelie-iska-math/toricgt-parameter-golf/toricgt_seq4096_4k_recovery_r78_20260604T184450Z
current active training log: amelie-iska/parameter-golf/logs/toricgt_seq4096_4k_recovery_r78_20260604T184450Z.txt
current active checkpoint dir: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_4k_recovery_r78_20260604T184450Z
current active supervisor tmux: toricgt_seq4096_4k_gate_r78_20260604T184450Z
current active supervisor log: logs/toricgt_seq4096_4k_recovery_r78_20260604T184450Z.4k_gate.txt
current active resume checkpoint: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_4k_recovery_r75_20260604T182013Z/toricgt_seq4096_4k_recovery_r75_20260604T182013Z_step_003600.pt
current active latest checkpoint: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_4k_recovery_r78_20260604T184450Z/toricgt_seq4096_4k_recovery_r78_20260604T184450Z_step_003600.pt
current active latest BPB/loss: 1.2169 / 2.0547 at step 3600
current generic non-pausing watcher: toricgt_watch_training_analysis_r78_3650
current generic watcher log: logs/toricgt_seq4096_4k_recovery_r78_20260604T184450Z.watch_training_analysis_3650.txt
current generic watcher target: fresh checkpoint >= 3650 on CPU, no pause-training flag
current generic watcher loop env: BPB_TARGET=1.2, BPB_MAX_REVIEW_ITERATIONS=100,
                                  BPB_LOOP_STATE=/home/iska/Documents/amelie/bio/ToricGT/outputs/toricgt_seq4096_4k_recovery_r75_20260604T182013Z_live_bpb_codex_loop_state.json,
                                  BPB_LOOP_STOP_FILE=/home/iska/Documents/amelie/bio/ToricGT/outputs/toricgt_seq4096_4k_recovery_r75_20260604T182013Z_live_bpb_codex_loop_stop,
                                  BPB_LOOP_NAME=parameter_golf_bpb_target
```

Final live-state correction for the step-3600 handoff:

The active supervisor subsequently escalated again to the low-train-BPB capture
branch, r79, after the r77/r78 sidecar reviews observed the validation bump at
step 3650.  This remains the same `EDIT_AND_RESTART` decision class: preserve
the r75 step-3600 checkpoint, keep optimizer/RNG/loader state, and apply only
scalar BPB-capture controls.  The step-3650 r77 sidecar was non-pausing and
completed its Seq4096-compatible geometry/simplex analysis; its Codex hook found
the `toricgt_codex_review_seq4096_live_00003650` session already present, so no
training pause was introduced.

```text
current active training tmux: toricgt_seq4096_4k_recovery_r79_low_bpb_capture_20260604T185003Z
current active W&B run: amelie-iska-math/toricgt-parameter-golf/toricgt_seq4096_4k_recovery_r79_low_bpb_capture_20260604T185003Z
current active training log: amelie-iska/parameter-golf/logs/toricgt_seq4096_4k_recovery_r79_low_bpb_capture_20260604T185003Z.txt
current active checkpoint dir: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_4k_recovery_r79_low_bpb_capture_20260604T185003Z
current active supervisor: scripts/watch_seq4096_4k_recovery.py
current active supervisor tmux: toricgt_seq4096_4k_gate_r79_low_bpb_capture_20260604T185003Z
current active supervisor log: logs/toricgt_seq4096_4k_recovery_r79_low_bpb_capture_20260604T185003Z.4k_gate.txt
current active resume checkpoint: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_4k_recovery_r75_20260604T182013Z/toricgt_seq4096_4k_recovery_r75_20260604T182013Z_step_003600.pt
current active fresh checkpoints: step 3600 scheduled, step 3608 low_train_bpb
current active latest checked BPB/loss: 1.2180 / 2.0565 at step 3608
current generic non-pausing watcher: toricgt_watch_training_analysis_r79_3650
current generic watcher log: logs/toricgt_seq4096_4k_recovery_r79_low_bpb_capture_20260604T185003Z.watch_training_analysis_3650.txt
current generic watcher target: fresh checkpoint >= 3650 on CPU, no pause-training flag
current compact sidecar watcher: toricgt_seq4096_4k_analysis_r79_low_bpb_capture_20260604T185003Z
corrected loop env for r79 gate and analysis sidecars: BPB_TARGET=1.2,
    BPB_MAX_REVIEW_ITERATIONS=100,
    BPB_LOOP_STATE=/home/iska/Documents/amelie/bio/ToricGT/outputs/toricgt_seq4096_4k_recovery_r74_20260604T180722Z_live_bpb_codex_loop_state.json,
    BPB_LOOP_STOP_FILE=/home/iska/Documents/amelie/bio/ToricGT/outputs/toricgt_seq4096_4k_recovery_r74_20260604T180722Z_live_bpb_codex_loop_stop,
    BPB_LOOP_NAME=parameter_golf_bpb_target
```

Implementation note: `scripts/watch_training_analysis.py` now detects compact
Seq4096 checkpoints (`model.tok_emb.weight`) and dispatches their structural
analysis to `scripts/evaluate_seq4096_reasoning_geometry_suite.py`, which writes
the expected `geometry/` and `simplex/` outputs without loading the checkpoint
through the incompatible RandomOrderLM diagnostic path.  The watcher still runs
without `--pause-training-before-analysis`.

## 2026-06-04 Seq4096 R75 Step-3600 BPB Gate Review

Run:

```text
W&B run: amelie-iska-math/toricgt-parameter-golf/toricgt_seq4096_4k_recovery_r75_20260604T182013Z
analysis dir: outputs/live_periodic_reviews/toricgt_seq4096_4k_recovery_r75_20260604T182013Z/step-00003600
checkpoint: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_4k_recovery_r75_20260604T182013Z/toricgt_seq4096_4k_recovery_r75_20260604T182013Z_step_003600.pt
checkpoint step: 3600
target BPB: 1.2
loop iteration: 3 / 100
```

Status:

```text
primary BPB signal: openai_parameter_golf/bpb = 1.2169196232244919
FineWeb validation BPB/loss: 1.2169196232244919 / 2.0547178091232596
FineWeb train BPB/loss: 1.2255411781645588 / 2.0819263458251953
sampled official-style OAI BPB/loss: 1.2183651952387373 / 2.1244614124298096
sampled OAI eval scope: 1 local FineWeb sp1024 validation sequence, seq_len=256
target gap: 0.0169196232244919
recent validation velocity: about 0.0029-0.0032 BPB drop per 100 steps
required velocity to hit 1.2 by step 4000: about 0.004225 BPB per 100 steps
projected target step: 4182.76
metric categories: desired=47, weak_or_slow=22, undesirable=5
artifact probe: 15,886,755 bytes, 113,245 bytes under the 16 MB controller limit
```

Metric and plot review:

- Desired: official-style FineWeb validation BPB improved from `1.219800000`
  at step 3500 to `1.2184563243984596` at step 3550 and
  `1.2169196232244919` at step 3600.  Validation loss moved with it
  (`2.0596 -> 2.057312464685708 -> 2.0547178091232596`).  The sampled local
  FineWeb SP1024 OAI probe was consistent with the primary signal at
  `1.2183651952387373`, but its one-sequence scope keeps it high variance.
  GraphCG off-diagonal coherence stayed near zero in the geometry records,
  Koszul `d2` residual was zero, BGG resolution and Gale-dual consistency were
  essentially one, and the simplex, 3D trajectory, phase-energy,
  energy-landscape, GraphCG, tropical, topology, and BGG/Koszul plots were
  generated and inspected.
- Desired but too weak or slow: BPB remains above target by
  `0.0169196232244919`.  The recent validation velocity is below the required
  gate velocity by roughly `0.0010-0.0015` BPB per 100 steps, so the line still
  crosses after the 4K gate.  Transfer evidence is weak with only two paired
  validation intervals.  The reasoning gate is alive but rough: GraphCG basis
  condition numbers are very large on some records, MST efficiency and path
  smoothness are weak, analogical transport is poor, Slepian leakage is uneven,
  and GFlowNet/branch replay and test-time-scaling diagnostics remain
  proxy-level rather than promotion evidence.
- Undesirable: train BPB is noisy and bounced at the validation-aligned step
  (`train_bpb=1.2255411781645588` at step 3600 after lower train samples
  around step 3550).  Train loss and total loss were categorized undesirable in
  the metric summary.  Tropical chamber crossings and plateau pressure remain
  high, trajectory paths are long and kinked, relative-K/NCD proxies are still
  high enough to withhold compression claims, and no Hessian trace/sharpness
  probe was available.

Mathematical/statistical interpretation:

- FineWeb BPB is the competition gate.  The validation first differences are
  still negative: approximately `-0.001343676` from step 3500 to 3550 and
  `-0.001536701` from step 3550 to 3600.  The second difference is slightly
  favorable rather than a validation floor bounce, so a rollback is not
  justified by sign or curvature.  The failure mode is insufficient descent
  speed.
- The train/validation phase plane shows poor transfer: train BPB crossed below
  the target band around step 3550 and then moved back above validation by step
  3600, while validation descended slowly.  This is consistent with an
  optimizer-transfer mismatch or noisy train floor bounce, not with a clean
  deployable basin.
- The proposal script was re-run on the analysis directory.  It reported
  `primary_action=restart_from_best_checkpoint_with_damped_structural_sidecars`,
  current/best gate BPB `1.2169196232244919`, recent drop about `0.0029` BPB
  per 100 steps, and projected target step `4182.76`.  I treated this as
  evidence for scalar intervention, not as authority to add architecture.  Its
  OAI competition field missed the sampled OAI probe, so that blind spot was
  not used against the primary FineWeb signal.
- The reasoning gate stays as sidecar evidence.  The simplex/geometry plots
  were non-collapsed but not promotion-grade.  GraphCG/topology/toric/BGG
  diagnostics should shape branch and auxiliary control, not replace the
  FineWeb BPB objective before the target checkpoint.  The two-gate rule
  therefore calls for scalar mixture/auxiliary control rather than discarding
  either objective.
- No OOD-transfer claim is justified here.  The run is already using the local
  FineWeb SP1024 stream, and the sampled OAI eval is too small.  Tokenizer,
  n-gram/context, dataset-easiness, FineWeb-only, hard-data-only, mixed-budget,
  and auxiliary-ablation controls are still required before claiming
  hard-reasoning transfer.
- Hessian/sharpness probes were not available in this analysis directory; the
  decision used finite differences, BPB velocity, transfer diagnostics, and
  geometry/topology sidecars.

Decision: `EDIT_AND_RESTART`.

The final active handoff is scalar recovery from the analyzed step-3600
checkpoint.  The first recovery was r77; after r77 emitted
`step:3650/4000 val_loss:2.0551 val_bpb:1.2171`, the gate supervisor advanced
to r78 with more damping, then to r79 low-BPB capture.  I accepted this as the
live `EDIT_AND_RESTART` handoff because it keeps training active, preserves the
Parameter-Golf/ToricGT architecture, and changes only runtime scalar controls
plus a low-train-BPB checkpoint/validation trigger.  No checkpoint was deleted
and no better-strategy stop sentinel was written.

The current r79 restart is active.  It loaded the r75 step-3600 checkpoint with
`reset_optimizer=0`, `reset_rng=0`, and `reset_loader=0`; the tmux process is
running and the generic CPU sidecar is waiting for the fresh step-3650
checkpoint without pausing training.

Operational handoff:

```text
active training tmux: toricgt_seq4096_4k_recovery_r79_low_bpb_capture_20260604T185003Z
active supervisor: scripts/watch_seq4096_4k_recovery.py
active supervisor tmux: toricgt_seq4096_4k_gate_r79_low_bpb_capture_20260604T185003Z
active supervisor log: logs/toricgt_seq4096_4k_recovery_r79_low_bpb_capture_20260604T185003Z.4k_gate.txt
active training log: amelie-iska/parameter-golf/logs/toricgt_seq4096_4k_recovery_r79_low_bpb_capture_20260604T185003Z.txt
active checkpoint dir: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_4k_recovery_r79_low_bpb_capture_20260604T185003Z
W&B run: amelie-iska-math/toricgt-parameter-golf/toricgt_seq4096_4k_recovery_r79_low_bpb_capture_20260604T185003Z
resume checkpoint: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_4k_recovery_r75_20260604T182013Z/toricgt_seq4096_4k_recovery_r75_20260604T182013Z_step_003600.pt
resume controls: reset_optimizer=0, reset_rng=0, reset_loader=0
runtime scalar delta: tied_embed_lr=0.0345, matrix_lr=0.018, scalar_lr=0.018,
                      bigram_bias_lr=0.014, advanced_loss_scale=0.018,
                      graphcg=0.032, toric_tropical=0.006, slepian=0.02,
                      koszul_bgg=0.0006, analogy=0.0006,
                      advanced_loss_sample_tokens=320, fan_bins=12,
                      advanced_loss_every=4, warmup_steps=80,
                      advanced_loss_max_ce_ratio=0.00075
low-BPB capture controls: checkpoint_on_train_bpb_below=1.13,
                          cooldown_steps=10, max_captures=4,
                          validate_on_train_bpb_checkpoint=1
nonblocking watcher tmux: toricgt_watch_training_analysis_r79_3650
nonblocking watcher script: scripts/watch_training_analysis.py
nonblocking watcher target: fresh checkpoint >= step 3650
nonblocking watcher log: logs/toricgt_seq4096_4k_recovery_r79_low_bpb_capture_20260604T185003Z.watch_training_analysis_3650.txt
nonblocking watcher output root: outputs/live_periodic_reviews/toricgt_seq4096_4k_recovery_r79_low_bpb_capture_20260604T185003Z_watch_training_analysis
nonblocking watcher device/precision: cpu / fp32
nonblocking watcher pause behavior: no --pause-training-before-analysis
required loop env: BPB_TARGET=1.2, BPB_MAX_REVIEW_ITERATIONS=100,
                   BPB_LOOP_STATE=/home/iska/Documents/amelie/bio/ToricGT/outputs/toricgt_seq4096_4k_recovery_r75_20260604T182013Z_live_bpb_codex_loop_state.json,
                   BPB_LOOP_STOP_FILE=/home/iska/Documents/amelie/bio/ToricGT/outputs/toricgt_seq4096_4k_recovery_r75_20260604T182013Z_live_bpb_codex_loop_stop,
                   BPB_LOOP_NAME=parameter_golf_bpb_target
```

## 2026-06-04 Seq4096 R74 Step-3400 BPB Gate Review

Run:

```text
analyzed run: amelie-iska-math/toricgt-parameter-golf/toricgt_seq4096_4k_recovery_r74_20260604T180722Z
analyzed checkpoint: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_4k_recovery_r74_20260604T180722Z/toricgt_seq4096_4k_recovery_r74_20260604T180722Z_step_003400.pt
analysis: outputs/live_periodic_reviews/toricgt_seq4096_4k_recovery_r74_20260604T180722Z/step-00003400
loop: parameter_golf_bpb_target, iteration 1 / 100 at handoff, target BPB <= 1.2
```

Status:

```text
step-3400 primary FineWeb validation BPB/loss: 1.2230 / 2.0650
step-3400 target gap: 0.0230
sampled official-style OAI BPB/loss: 1.2072933580979746 / 2.1051554679870605
sampled OAI eval scope: 1 local FineWeb validation sequence, seq_len=256
step-3400 metric categories: desired=28, weak_or_slow=4, undesirable=7
live superseding check while review was running: step 3450 val BPB/loss = 1.2217 / 2.0627
step-3450 loop best: openai_parameter_golf/bpb = 1.221661249861833
step-3450 validation slope from 3400: -0.0026 BPB per 100 steps
step-3450 projected target step: 4284.6
step-3450 metric categories: desired=46, weak_or_slow=26, undesirable=7
artifact probe: int8+zlib total 15,876,187 bytes, 123,813 bytes under 16 MB
```

Metric and plot categorization:

- Desired: validation BPB improved from `1.2230` at step 3400 to `1.2217`
  at step 3450; the sampled official-style FineWeb check was close to target
  at `1.2073`; GraphCG basis plots were nearly diagonal with off-diagonal
  coherence around `1e-8`; Koszul `d2` residual stayed `0`; BGG resolution and
  Gale-dual consistency were about `0.986--1.0`; toric fan entropy was high for
  token, norm, and memory trajectories; the BPB, simplex, 3D trajectory,
  phase-energy, energy-landscape, GraphCG, tropical, topology, and BGG plots
  were generated and inspectable.
- Desired but too weak or slow: FineWeb validation BPB still misses the `1.2`
  gate; the step-3450 descent velocity projects to step `4284.6`, beyond the
  4K gate; the one-sequence official-style sample is high-variance calibration
  evidence rather than a promotion metric; transfer has only a single paired
  validation interval; GFlowNet branch replay and test-time-scaling remain
  proxy-level in this compact analysis; GraphCG/topology/toric/BGG signals
  support sidecar control but not heavier pre-threshold objective weight.
- Undesirable: train BPB bounced after the local dip (`1.1827` best train BPB
  then `1.2403` latest in the exported W&B slice) with recent slope `+28.76`
  BPB per 1K W&B steps and `t=4.29`; tropical chamber crossing rates remained
  high (`0.75--0.963`) with plateau pressure on several trajectories; layer and
  attention MST efficiency were low (`0.0567` and `0.0653`) and path smoothness
  was weak; standard leakage/fitting residuals were nonzero (`~0.078--0.132`);
  no Hessian trace/sharpness or second-difference probe was available to justify
  a more aggressive curvature call.

Mathematical/statistical interpretation:

- FineWeb BPB is the competition gate.  At step 3400 the validation gap was
  `0.0230`, requiring `0.003833` BPB drop per 100 steps to hit the 4K target
  with no validation velocity estimate yet.  The live step-3450 point improved
  by `0.0013` over 50 steps, but the resulting `0.0026` BPB per 100 steps is
  still below the velocity required for the remaining runway.
- The sampled OAI BPB (`1.207293`) is consistent with possible OOD transfer
  from hard reasoning exposure, but it is one short FineWeb validation sequence.
  It must be controlled against tokenizer effects, n-gram/context baselines,
  dataset easiness, FineWeb-only exposure, hard-only exposure, mixed-budget
  exposure, and auxiliary-ablation explanations before treating it as proof of
  transfer.
- The reasoning gate is alive but not the promotion bottleneck.  Relative-K
  proxies were moderate (`0.639--0.812`) and recent log NCD was high
  (`0.8869`), so compression structure is present but not decisive.  GraphCG
  and BGG/Koszul consistency are desired sidecar evidence; analogical transport
  accuracy is mostly `0`; toric/tropical phase behavior is active but noisy.
  The two-gate rule therefore keeps BPB primary and keeps structural families
  as sidecar/controller signals until the FineWeb gate clears.

The adjustment proposal was rerun on the analysis directory:

```text
/home/iska/miniconda3/envs/tokengt/bin/python scripts/propose_training_adjustments.py \
  --analysis-dir outputs/live_periodic_reviews/toricgt_seq4096_4k_recovery_r74_20260604T180722Z/step-00003400 \
  --target-bpb 1.2 --gate-step 4000 --checkpoint-step 3400 \
  --wandb-run-path amelie-iska-math/toricgt-parameter-golf/toricgt_seq4096_4k_recovery_r74_20260604T180722Z
```

It recommended `restart_policy=no_restart_until_next_gate`, kept the
pre-threshold objective BPB-clean, and recommended preparing branch controls if
the next validation gate confirms the miss.  I used it as supporting evidence,
not as authority.

Decision: `CONTINUE`.

Reason: there is no positive validation first difference at the reviewed
boundary.  The live step-3450 validation point improved from step 3400, so the
rollback trigger is not met.  The run is still off-track for the 4K gate, but
the smallest high-impact action is to let the active supervisor carry the same
config to the next dense checkpoint, then let the existing gate watcher restart
only if the projected-miss/patience rules fire.  No JEPA, architecture, data,
tokenizer, config scalar, or checkpoint artifact change was made.

No better-strategy sentinel was written.  The BPB-first loop remains active
until `BPB <= 1.2`, 100 reviews complete, or a better documented strategy
replaces ordinary recovery replays.

Operational handoff:

```text
action: CONTINUE
active supervisor: scripts/watch_seq4096_4k_recovery.py
active supervisor tmux: toricgt_seq4096_4k_gate_r74_20260604T180722Z
active supervisor log: logs/toricgt_seq4096_4k_recovery_r74_20260604T180722Z.4k_gate.txt
active training tmux: toricgt_seq4096_4k_recovery_r74_20260604T180722Z
active training process: train_gpt.py verified running
active W&B run: amelie-iska-math/toricgt-parameter-golf/toricgt_seq4096_4k_recovery_r74_20260604T180722Z
active training log: amelie-iska/parameter-golf/logs/toricgt_seq4096_4k_recovery_r74_20260604T180722Z.txt
active checkpoint dir: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_4k_recovery_r74_20260604T180722Z
analyzed checkpoint: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_4k_recovery_r74_20260604T180722Z/toricgt_seq4096_4k_recovery_r74_20260604T180722Z_step_003400.pt
latest live checkpoint at decision time: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_4k_recovery_r74_20260604T180722Z/toricgt_seq4096_4k_recovery_r74_20260604T180722Z_step_003450.pt
latest live validation BPB/loss: 1.2217 / 2.0627 at step 3450
non-pausing live watcher: toricgt_seq4096_live_50step_toricgt_seq4096_4k_recovery_r74_20260604T180722Z
non-pausing watcher output root: outputs/live_periodic_reviews/toricgt_seq4096_4k_recovery_r74_20260604T180722Z
next watcher target: step 3500
loop status: outputs/live_periodic_reviews/toricgt_seq4096_4k_recovery_r74_20260604T180722Z/step-00003450/logs/bpb_codex_loop_status.json
loop env: BPB_TARGET=1.2, BPB_MAX_REVIEW_ITERATIONS=100,
          BPB_LOOP_STATE=/home/iska/Documents/amelie/bio/ToricGT/outputs/toricgt_seq4096_4k_recovery_r74_20260604T180722Z_live_bpb_codex_loop_state.json,
          BPB_LOOP_STOP_FILE=/home/iska/Documents/amelie/bio/ToricGT/outputs/toricgt_seq4096_4k_recovery_r74_20260604T180722Z_live_bpb_codex_loop_stop,
          BPB_LOOP_NAME=parameter_golf_bpb_target
```

The next analysis remains non-interrupting.  For this compact Seq4096 run the
compatible watcher is `scripts/watch_seq4096_analysis.py`, which defaults the
compact OAI eval to CPU and does not have a pause-training flag.  The generic
`scripts/watch_training_analysis.py` watcher was not started because it only
matches `random_order_step_*.pt` checkpoints and would not see these
`<run>_step_*.pt` Seq4096 checkpoints.

## 2026-06-04 Seq4096 R72 Step-3550 BPB Gate Review

Run:

```text
analyzed run: amelie-iska-math/toricgt-parameter-golf/toricgt_seq4096_4k_recovery_r72_20260604T173538Z
analyzed checkpoint: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_4k_recovery_r72_20260604T173538Z/toricgt_seq4096_4k_recovery_r72_20260604T173538Z_step_003550.pt
analysis: outputs/live_periodic_reviews/toricgt_seq4096_4k_recovery_r72_20260604T173538Z/step-00003550
loop: parameter_golf_bpb_target, iteration 3 / 100 at handoff, target BPB <= 1.2
```

Status:

```text
primary BPB signal: openai_parameter_golf/bpb = 1.228238804953698
best observed loop BPB: 1.226657554487925 at step 3550
r72 best validation BPB/loss: 1.2198 / 2.0596 at step 3500
r72 step-3550 validation BPB/loss: 1.2282 / 2.0738
r72 step-3550 train BPB/loss: 1.2204 / 2.0587
sampled official-style OAI BPB/loss: 1.2122616356447082 / 2.1138
sampled OAI eval scope: 1 local FineWeb validation sequence, seq_len=256
metric categories: desired=39, weak_or_slow=31, undesirable=4
recent validation slope: +0.0168 BPB per 100 steps
gate velocity state: off_track_unreachable
```

Metric and plot categorization:

- Desired: GraphCG basis plots stayed mostly diagonal with low off-diagonal
  coherence; BGG/Koszul diagnostics had zero `d2_residual` and full
  resolution/Gale-dual consistency where present; topology filtrations were
  noncollapsed; train BPB was still near the target; the BPB, simplex,
  tetrahedron, 3D trajectory, phase-energy, topology, GraphCG, tropical, and
  energy-landscape plots were generated and inspectable.
- Desired but too weak or slow: FineWeb-style BPB is close but still above the
  `1.2` gate; the sampled official-style OAI result is encouraging but too
  small to promote; GraphCG/topology/BGG/tropical sidecars show structure but
  have too little paired validation history to justify heavier objective
  weight; GFlowNet/branch replay and test-time-scaling evidence is proxy-level;
  MST efficiency, path smoothness, and trajectory-length diagnostics show
  coherent paths but not a short smooth descent into a low-BPB basin.
- Undesirable: validation BPB worsened from `1.2198` at step 3500 to `1.2282`
  at step 3550, giving a positive first difference and missing the gate
  velocity requirement; the train/validation split suggests transfer failure
  despite short train-BPB dips after step 3550; tropical chamber churn and
  plateau pressure remain too active; Kolmogorov/NCD complexity proxies are
  still high; no Hessian or second-difference probe was available to overturn
  the finite-difference rollback evidence.

Mathematical/statistical interpretation:

- FineWeb BPB remains the competition gate.  The full validation BPB miss and
  positive validation derivative dominate the decision.  The one-sequence
  official-style OAI sample (`1.2122616`) is consistent with possible transfer
  from hard reasoning exposure, but this remains a hypothesis until controlled
  against tokenizer effects, n-gram/context-tree baselines, dataset easiness,
  FineWeb-only exposure, hard-only exposure, mixed-budget exposure, and
  auxiliary-ablation explanations.
- The first derivative is enough to reject `CONTINUE`: `d BPB / d step` turned
  positive in the validation signal, while the gate needs a negative BPB
  velocity to cross `1.2` by step 4000.  Hessian sharpness/trace and
  second-difference probes were absent, so they were not used as authority.
- The reasoning gate is alive but not promotion-grade.  GraphCG and
  BGG/Koszul consistency are desired; topology and simplex structure are
  present; toric/tropical phase behavior is active; but the BPB-transfer map
  still recommends keeping structural families as sidecar/controller evidence
  until the FineWeb BPB gate clears.

The adjustment proposal was rerun on the analysis directory:

```text
/home/iska/miniconda3/envs/tokengt/bin/python scripts/propose_training_adjustments.py \
  --analysis-dir outputs/live_periodic_reviews/toricgt_seq4096_4k_recovery_r72_20260604T173538Z/step-00003550 \
  --target-bpb 1.2 --gate-step 4000 --checkpoint-step 3550
```

It recommended preparing a mid-run recovery branch, keeping the pre-threshold
competition objective BPB-clean, and treating GraphCG/topology/toric/tropical
metrics as evidence rather than automatic authority.  I used it as supporting
evidence, not as a command.

Decision: `EDIT_AND_RESTART`.

Reason: r72 repeated the validation rebound after the r71/r72 step-3500 low, so
plain continuation is not justified.  The active restart handoff is r73, a
deeper selected rollback/edit branch from the warmdown r52 step-3250 checkpoint
with more runway, `tied_embed_lr=0.034`, matrix/scalar LR held at `0.018`, dense
50-step validation/checkpointing, and small guarded structural losses
(`advanced_loss_scale=0.03`, GraphCG `0.03`, toric/tropical `0.008`, Slepian
`0.015`, Koszul/BGG `0.0015`, analogy `0.001`).  This preserves the Seq4096
Parameter-Golf architecture and uses mixture/auxiliary weights rather than
discarding either the competition BPB gate or the reasoning gate.  No JEPA,
architecture, data, tokenizer, or checkpoint artifact change was made.

No better-strategy sentinel was written.  The current BPB-first loop remains
active until `BPB <= 1.2`, 100 review iterations, or a better documented
strategy replaces ordinary cliff/recovery replays.

Operational handoff:

```text
current action: CONTINUE
active training tmux: toricgt_seq4096_4k_recovery_r73_20260604T174607Z
active W&B run: amelie-iska-math/toricgt-parameter-golf/toricgt_seq4096_4k_recovery_r73_20260604T174607Z
active training log: amelie-iska/parameter-golf/logs/toricgt_seq4096_4k_recovery_r73_20260604T174607Z.txt
active checkpoint dir: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_4k_recovery_r73_20260604T174607Z
selected resume checkpoint: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_warmdown_r52_20260604T125517Z/toricgt_seq4096_warmdown_r52_20260604T125517Z_step_003250.pt
latest saved active checkpoint: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_4k_recovery_r73_20260604T174607Z/toricgt_seq4096_4k_recovery_r73_20260604T174607Z_step_003350.pt
latest active validation BPB/loss: 1.2246 / 2.0676 at step 3350
active gate tmux: toricgt_seq4096_4k_gate_r73_20260604T174607Z
active gate log: logs/toricgt_seq4096_4k_recovery_r73_20260604T174607Z.4k_gate.txt
active mirror tmux: toricgt_seq4096_4k_mirror_r73_20260604T174607Z
active full-diagnostics tmux: toricgt_seq4096_4k_full_diag_r73_20260604T174607Z
live sidecar supervisor: toricgt_seq4096_live_periodic_full_analysis_supervisor
live sidecar supervisor log: logs/seq4096_live_periodic_full_analysis_supervisor.log
non-pausing live watcher: toricgt_seq4096_live_full_analysis_toricgt_seq4096_4k_recovery_r73_20260604T174607Z
non-pausing watcher output root: outputs/live_periodic_reviews/toricgt_seq4096_4k_recovery_r73_20260604T174607Z
live loop status: outputs/live_periodic_reviews/toricgt_seq4096_4k_recovery_r73_20260604T174607Z/step-00003350/logs/bpb_codex_loop_status.json
loop env: BPB_TARGET=1.2, BPB_MAX_REVIEW_ITERATIONS=100,
          BPB_LOOP_STATE=/home/iska/Documents/amelie/bio/ToricGT/outputs/toricgt_seq4096_4k_recovery_r71_20260604T172807Z_live_bpb_codex_loop_state.json,
          BPB_LOOP_STOP_FILE=/home/iska/Documents/amelie/bio/ToricGT/outputs/toricgt_seq4096_4k_recovery_r71_20260604T172807Z_live_bpb_codex_loop_stop,
          BPB_LOOP_NAME=parameter_golf_bpb_target
```

The next review remains non-interrupting.  For compact Seq4096 checkpoints the
compatible sidecar is `scripts/watch_seq4096_analysis.py`, not the historical
`scripts/watch_training_analysis.py` RandomOrderLM loader; the active watcher
has no pause-training flag and the supervisor owns crash/stall recovery.
`scripts/watch_seq4096_4k_recovery.py` now also propagates the stable BPB loop
environment into recursive gate watchers so future r74+ restarts do not fork
the loop state.

## 2026-06-04 Seq4096 R71 Step-3550 BPB Gate Review

Run:

```text
analyzed run: amelie-iska-math/toricgt-parameter-golf/toricgt_seq4096_4k_recovery_r71_20260604T172807Z
analyzed checkpoint: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_4k_recovery_r71_20260604T172807Z/toricgt_seq4096_4k_recovery_r71_20260604T172807Z_step_003550.pt
analysis: outputs/live_periodic_reviews/toricgt_seq4096_4k_recovery_r71_20260604T172807Z/step-00003550
loop: parameter_golf_bpb_target, iteration 2 / 100 at handoff, target BPB <= 1.2
```

Status:

```text
primary BPB signal: openai_parameter_golf/bpb = 1.226657554487925
sampled official-style OAI BPB/loss: 1.230380069869588 / 2.145411729812622
sampled OAI eval scope: 1 local FineWeb validation sequence, seq_len=256
best validation BPB in r71 gate window: 1.2198 at step 3500
step-3550 validation BPB/loss: 1.2267 / 2.0712
step-3550 train BPB/loss: 1.2031 / 2.0620
metric categories: desired=39, weak_or_slow=27, undesirable=6
checkpoint params: 18,108,488
artifact size margin: 111,328 bytes under the 16 MB limit
```

Metric and plot categorization:

- Desired: GraphCG off-diagonal coherence stayed bounded; tropical active-face
  margin stayed finite; GraphCG loss improved over the full window; GraphCG
  spectral entropy stayed near `0.997`; toric fan entropy stayed high; Koszul
  `d2_residual` stayed zero; BGG resolution and Gale-dual consistency stayed at
  or near `1.0`; the BPB, simplex, 3D trajectory, phase-energy, energy
  landscape, GraphCG, tropical, topology, and BGG plots were nonblank and
  inspectable.
- Desired but too weak or slow: the best validation BPB remained `0.0198` above
  target; current validation BPB was `0.0267` above target; the OAI-style sample
  was worse than the W&B primary BPB and too small to promote; transfer
  efficiency still had only one paired validation interval; relative-K and
  Kolmogorov proxies showed structure but not decisive BPB transfer; MST
  efficiency was uneven (`R0/R1` about `0.34--0.35`, layer/attention near
  `0.06`); path smoothness was low for layer, attention, and memory paths;
  tropical chamber crossings and plateau pressure were high enough to signal
  exploration but not stable descent.
- Undesirable: validation BPB worsened from `1.2198` at step 3500 to `1.2267`
  at step 3550, a positive first difference of `+0.0069` over 50 steps
  (`+0.0138` BPB per 100 steps); train loss and total loss drifted upward in
  the aggregate metric stats despite recent train BPB drops; advanced runtime
  stability flags moved more than expected; attention and memory trajectories
  showed jagged long phase/energy hops instead of a smooth terminal approach.

Mathematical/statistical interpretation:

- FineWeb BPB remains the competition gate.  The local official-style sample
  (`1.230380`) and the W&B primary BPB (`1.226658`) both miss `1.2`; the sample
  is one short FineWeb validation sequence, so it is calibration evidence with
  high variance, not a promotion metric.  No OOD-transfer claim is justified
  without tokenizer, n-gram, dataset-easiness, FineWeb-only, hard-only,
  mixed-exposure, and auxiliary-ablation controls.
- The finite-difference evidence is enough for intervention even without
  Hessian probes: the validation first derivative changed sign in the wrong
  direction after the step-3500 low, and the gate velocity report requires a
  positive drop of `0.005933` BPB per 100 steps to hit `1.2` by step 4000.
  Hessian trace/sharpness and second-difference probes were not available, so
  they were not used as authority.
- The reasoning gate is alive but subordinate to the BPB miss.  GraphCG basis
  behavior is mostly stable, BGG/Koszul consistency is healthy, and topology is
  noncollapsed, but branch/test-time-scaling and GFlowNet-style evidence is
  proxy-level only in this compact Seq4096 analysis.  The two-gate rule says to
  keep advanced objectives as sidecar/controller signals until competition BPB
  clears the gate.

The adjustment proposal was rerun on the analysis directory:

```text
/home/iska/miniconda3/envs/tokengt/bin/python scripts/propose_training_adjustments.py \
  --analysis-dir outputs/live_periodic_reviews/toricgt_seq4096_4k_recovery_r71_20260604T172807Z/step-00003550 \
  --target-bpb 1.2 --gate-step 4000 --checkpoint-step 3550 \
  --wandb-run-path amelie-iska-math/toricgt-parameter-golf/toricgt_seq4096_4k_recovery_r71_20260604T172807Z
```

It again recommended preparing a mid-run recovery branch, keeping the primary
competition objective BPB-clean, and using GraphCG/Koszul/BGG/topology and
toric/tropical diagnostics as sidecar controller evidence.  I used that as
evidence, not as an automatic authority.

Decision: `EDIT_AND_RESTART`.

Reason: the step-3550 validation derivative is positive, so continuing from the
analyzed checkpoint is not justified.  The active recovery already implements
the smallest scalar restart consistent with the data: resume from the last
dense checkpoint before the rebound, reset optimizer/RNG/loader, keep the
Seq4096 architecture and all Parameter-Golf constraints, keep advanced losses
log-only and sidecar-scaled, hold matrix/scalar LR, and use the recovery tied
embedding LR (`0.0357` instead of r71 `0.034`).

No better-strategy sentinel was written.  The current loop remains the best
available strategy: BPB-first recovery with dense review, sidecar structural
diagnostics, and rollback/edit restarts until `BPB <= 1.2`, the 100-review cap,
or a genuinely better documented strategy.

Operational handoff and live supersession:

```text
r71 decision checkpoint: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_4k_recovery_r71_20260604T172807Z/toricgt_seq4096_4k_recovery_r71_20260604T172807Z_step_003500.pt
immediate recovery run: toricgt_seq4096_4k_recovery_r72_20260604T173538Z
r72 W&B run: amelie-iska-math/toricgt-parameter-golf/toricgt_seq4096_4k_recovery_r72_20260604T173538Z
r72 confirmation: step 3550 val BPB/loss = 1.2282 / 2.0738
r72 sidecar loop status: iteration 3 / 100, same r71 live BPB loop state
current live training tmux: toricgt_seq4096_4k_recovery_r73_20260604T174607Z
current live training log: amelie-iska/parameter-golf/logs/toricgt_seq4096_4k_recovery_r73_20260604T174607Z.txt
current live checkpoint dir: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_4k_recovery_r73_20260604T174607Z
current live W&B run: amelie-iska-math/toricgt-parameter-golf/toricgt_seq4096_4k_recovery_r73_20260604T174607Z
current live resume checkpoint: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_warmdown_r52_20260604T125517Z/toricgt_seq4096_warmdown_r52_20260604T125517Z_step_003250.pt
current live saved checkpoint: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_4k_recovery_r73_20260604T174607Z/toricgt_seq4096_4k_recovery_r73_20260604T174607Z_step_003250.pt
current live step-3250 validation BPB/loss: 1.2290 / 2.0751
current live controls: tied_embed_lr=0.034, matrix_lr=0.018, scalar_lr=0.018, advanced_loss_scale=0.03, graphcg=0.03, toric_tropical=0.008, slepian=0.015, koszul_bgg=0.0015, analogy=0.001
current live status: training process active on GPU, non-pausing Seq4096 sidecar analysis attached
```

Implementation note: `scripts/codex_training_review_resume.sh` now detects
compact Seq4096 `train_gpt.py` as active training before launching fallback
continues.  `scripts/supervise_seq4096_live_analysis.py` and
`scripts/watch_seq4096_4k_recovery.py` now preserve BPB loop environment
variables when present, falling back to per-run loop files only when no parent
loop is defined.  These are sidecar/supervisor safety edits only; no model
architecture, JEPA, data, or weight artifact change was made.

## 2026-06-04 Seq4096 R71 Step-3500 BPB Gate Review

Run:

```text
analyzed run: amelie-iska-math/toricgt-parameter-golf/toricgt_seq4096_4k_recovery_r71_20260604T172807Z
analyzed checkpoint: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_4k_recovery_r71_20260604T172807Z/toricgt_seq4096_4k_recovery_r71_20260604T172807Z_step_003500.pt
analysis: outputs/live_periodic_reviews/toricgt_seq4096_4k_recovery_r71_20260604T172807Z/step-00003500
loop: parameter_golf_bpb_target, iteration 1 / 100, target BPB <= 1.2
```

Status:

```text
full FineWeb validation BPB/loss: 1.2198 / 2.0596
sampled official-style OAI BPB/loss: 1.2018589627023633 / 2.0956795
sampled OAI eval scope: 1 local FineWeb validation sequence, seq_len=256
train BPB near review: 1.1989 -> 1.1714 -> 1.1780 -> 1.1803
metric categories: desired=37, weak_or_slow=13, undesirable=1
checkpoint params: 18,108,488
artifact size margin: 116,138 bytes under the 16 MB limit
```

Metric and plot categorization:

- Desired: train loss and total loss continued to fall in the short W&B export;
  the checkpoint stayed under the artifact limit; GraphCG disentanglement was
  high on average (`0.730`); Koszul `d2_residual` was zero; BGG
  `resolution_consistency` was almost exact (`0.999846`); Gale-dual
  consistency was `1.0`; topology chain commutator was near numerical zero; the
  simplex, tetrahedron, trajectory, phase, chamber, energy, GraphCG, topology,
  and BGG plots were nonblank and structurally coherent.
- Desired but too weak or slow: full FineWeb validation BPB missed the gate by
  `0.0198`; the sampled OAI BPB missed by `0.001859` and is too small a sample
  to promote; BPB transfer efficiency had no paired train/validation history;
  relative-K gain was small (`0.0158`); analogical transport accuracy was weak
  (`0.033`); Slepian concentration/leakage were mixed (`0.726` / `0.274`);
  MST efficiency and path smoothness were low (`0.206` / `0.054`); trajectory
  length was large (`586.6` mean, one R4 path above `1363`); toric fan entropy
  was present (`0.791`) but margin was thin (`0.016`); tropical chamber
  crossings were frequent (`17.4` mean, chamber-crossing rate `0.873`).
- Undesirable: `advanced/graphcg_loss` was the only metric classified
  undesirable in the review export and drifted upward; train BPB rebounded after
  the step-3502 low; the later r71 step-3550 validation check worsened to
  `1.2267`, giving a positive validation first derivative; GraphCG basis
  conditioning was very high (`2.399e9` mean, `1.187e10` max); trajectory plots
  showed jagged long hops and phase/chamber churn instead of smooth terminal
  approach.

Mathematical/statistical interpretation:

- FineWeb BPB is the competition gate.  The full validation value `1.2198` is
  above target, and the sampled official-style OAI value `1.20185896` is close
  but still above `1.2`.  Because the OAI-style sample is one short local
  FineWeb sequence, it is encouraging calibration evidence only.  Any future
  claim of OOD transfer from hard reasoning data still needs controls against
  tokenizer convention, n-gram/context-tree baselines, dataset easiness,
  FineWeb-only exposure, hard-only transfer, mixed-exposure budget, and
  auxiliary ablations.
- The r71 step-3500 analysis alone did not contain enough validation history
  for a slope, second difference, or Hessian-backed rollback decision.  The
  BPB gate velocity report required a validation drop of about `0.00396` BPB per
  100 steps to reach `1.2` by the 4K gate.  The subsequent r71 step-3550
  validation BPB rose to `1.2267`, so the observed finite difference changed
  sign in the wrong direction before there was enough runway to recover.
- The reasoning gate is coherent but not strong enough to trade away BPB.
  Graph-of-thought and branch/test-time-scaling evidence is mostly proxy-level
  here: simplex and geometry views show structured branches, but low-BPB regions
  are not reliably selected.  The Kolmogorov/relative-K proxies indicate modest
  compression gain rather than a decisive reasoning breakthrough.  Toric BGG,
  Koszul, and Gale-dual consistency are healthy diagnostics, while standard
  leakage, fitting-minor residuals, GraphCG conditioning, trajectory roughness,
  and tropical chamber churn argue for keeping advanced objectives as sidecars
  until BPB clears the gate.

The adjustment proposal was run on the analysis directory:

```text
/home/iska/miniconda3/envs/tokengt/bin/python scripts/propose_training_adjustments.py \
  --analysis-dir outputs/live_periodic_reviews/toricgt_seq4096_4k_recovery_r71_20260604T172807Z/step-00003500
```

It recommended preparing a mid-run recovery branch, keeping the pre-threshold
competition objective BPB-clean, and using advanced metrics as sidecar evidence
rather than heavier loss terms.  I used this as evidence, not authority.  After
the r71 step-3550 validation derivative turned positive, the active gate
supervisor executed the smallest matching recovery: resume from the analyzed
step-3500 checkpoint, reset optimizer/RNG/loader, keep advanced losses
log-only, hold matrix/scalar learning rates, and raise only the tied embedding
LR from `0.034` to `0.0357`.

Decision: `EDIT_AND_RESTART`.

No tracked architecture or JEPA-style change was made.  The ToricGT
Parameter-Golf architecture, random-order autoregressive graph decoding,
tropical/hybrid attention, toric memory, dense contest weights, embedding-space
GFlowNet graph-of-thought data path, hard reasoning data path, GraphCG,
topology, toric, BGG, Kolmogorov diagnostics, and BPB-first two-gate rule were
preserved.
The better-strategy stop sentinel was not written because there is not yet a
replacement strategy better than the active BPB recovery loop.

Operational handoff:

```text
selected resume checkpoint: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_4k_recovery_r71_20260604T172807Z/toricgt_seq4096_4k_recovery_r71_20260604T172807Z_step_003500.pt
active training tmux: toricgt_seq4096_4k_recovery_r72_20260604T173538Z
active gate supervisor tmux: toricgt_seq4096_4k_gate_r72_20260604T173538Z
active training log: amelie-iska/parameter-golf/logs/toricgt_seq4096_4k_recovery_r72_20260604T173538Z.txt
active checkpoint dir: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_4k_recovery_r72_20260604T173538Z
active saved checkpoint: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_4k_recovery_r72_20260604T173538Z/toricgt_seq4096_4k_recovery_r72_20260604T173538Z_step_003500.pt
active W&B run: amelie-iska-math/toricgt-parameter-golf/toricgt_seq4096_4k_recovery_r72_20260604T173538Z
restart scalar: TIED_EMBED_LR=0.0357, MATRIX_LR=0.018, SCALAR_LR=0.018
resume controls: RESET_OPTIMIZER_ON_RESUME=1, RESET_RNG_ON_RESUME=1, RESET_LOADER_ON_RESUME=1
```

Next analysis handoff:

```text
seq4096 live 50-step watcher tmux: toricgt_seq4096_live_50step_toricgt_seq4096_4k_recovery_r72_20260604T173538Z
seq4096 live 250-step watcher tmux: toricgt_seq4096_live_full_analysis_toricgt_seq4096_4k_recovery_r72_20260604T173538Z
next target steps: 3550 and 3750
watcher script: scripts/watch_seq4096_analysis.py, the seq4096-compatible non-pausing watcher
pause behavior: non-interrupting; no pause-training-before-analysis flag is used
loop env: BPB_TARGET=1.2, BPB_MAX_REVIEW_ITERATIONS=100,
          BPB_LOOP_STATE=/home/iska/Documents/amelie/bio/ToricGT/outputs/toricgt_seq4096_4k_recovery_r71_20260604T172807Z_live_bpb_codex_loop_state.json,
          BPB_LOOP_STOP_FILE=/home/iska/Documents/amelie/bio/ToricGT/outputs/toricgt_seq4096_4k_recovery_r71_20260604T172807Z_live_bpb_codex_loop_stop,
          BPB_LOOP_NAME=parameter_golf_bpb_target
```

Operational addendum after the r72 handoff:

```text
r72 step-3550 validation BPB/loss: 1.2282 / 2.0738
r72 result: worse than the r71/r72 step-3500 gate value; target still missed
final active recovery run: toricgt_seq4096_4k_recovery_r73_20260604T174607Z
final active W&B run: amelie-iska-math/toricgt-parameter-golf/toricgt_seq4096_4k_recovery_r73_20260604T174607Z
final active training tmux: toricgt_seq4096_4k_recovery_r73_20260604T174607Z
final active gate tmux: toricgt_seq4096_4k_gate_r73_20260604T174607Z
final active training log: amelie-iska/parameter-golf/logs/toricgt_seq4096_4k_recovery_r73_20260604T174607Z.txt
final active checkpoint dir: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_4k_recovery_r73_20260604T174607Z
final active resume checkpoint: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_warmdown_r52_20260604T125517Z/toricgt_seq4096_warmdown_r52_20260604T125517Z_step_003250.pt
final active live watcher tmux: toricgt_seq4096_live_50step_toricgt_seq4096_4k_recovery_r73_20260604T174607Z
final active live watcher loop env: original r71 live BPB loop state/stop file, BPB_TARGET=1.2, BPB_MAX_REVIEW_ITERATIONS=100, BPB_LOOP_NAME=parameter_golf_bpb_target
```

The final active r73 process was already using the GPU when r72 was found
inactive.  It keeps the same compact Seq4096 model family, tokenizer, dense
FineWeb path, Parameter-Golf BPB target, and diagnostic families, but it is a
stronger supervisor-owned recovery from the earlier warmdown checkpoint with
structural sidecar losses active.  I preserved that active process instead of
stopping it.  A later manual r73 attempt from the r71 step-3500 checkpoint
exited with CUDA OOM because this active run already occupied the GPU; its
failed sidecar watchers were stopped and no checkpoint was deleted.

## 2026-06-03 Automated Review: Step 5,000 FineWeb-Revealed BPB Recovery Gate

The watcher analyzed:

```text
outputs/post_resume_analysis/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/step-00005000/
```

Analyzed checkpoint:

```text
checkpoints/parameter_golf_all_phases_fineweb_from0/random_order_step_00005000.pt
```

The BPB loop target remains `<= 1.2`.  This was review iteration `5 / 100`.
The loop primary BPB is still `best_val_bpb=5.349750123909216`, first reached at
step `4750`; the target is not reached.  The post-analysis official-style
FineWeb/OAI eval on decoded local `fineweb10B_sp1024` validation bytes reported
`oai_competition/bpb=4.520416072849183`, loss `3.1333136558532715`, over 8 eval
batches.  This is the competition calibration gate.  The hard ToricGT validation
gate remains the reasoning gate.

### Metric Categorization

| category | read |
|---|---|
| desired | official-style FineWeb/OAI BPB improved again (`4.52796404743025` W&B step-4750 summary to `4.520416072849183` post-analysis step-5000 eval); hard validation best BPB did not regress (`5.349750123909216`); exported hard validation BPB/loss slopes remain negative (`val/bpb -0.046803/1k`, `val/loss -0.032442/1k`); checkpoint train BPB is lower than the late W&B training samples (`4.49188378560791`); future-permutation logit audit remains zero; GFlowNet entropy/diversity are high (`2.771388`, `0.998646`); GraphCG basis coherence and covariance improved; Koszul exactness and syzygy residuals remain small; Toric BGG resolution consistency is high (`0.934541`); topology inclusion/boundary residuals and directed cycle flux are effectively zero; exact edge validity is high (`0.972594`), Slepian concentration/leakage are clean (`1.0` / `0.0`) |
| desired but too weak or slow | both gates remain far above `1.2`; recent train BPB/loss slopes are weakly positive (`+0.131640/1k`, `+0.091246/1k`) despite lower medians over the full window; hard validation improvement is slow and the best metric has plateaued for one checkpoint; score-first hard validation gives only a small gain (`5.342478` vs deterministic `5.349750`); simplex BPB is nearly flat across budgets (`6.5473--6.5571`) and argmax byte accuracy is only `0.10034`; geometry branch replay has a large mean-best gap (`mean BPB 7.388789`, best `5.036372`, gap `2.352417`); mean answer BPB is weak (`7.530481`, best answer `4.912411`); MST efficiency is moderate (`0.477016`) and path smoothness is low (`0.124134`); hard-val argmax byte accuracy is stuck (`0.173828`), and `complexity/val/prediction_target_ncd_lzma_mean` worsened in the W&B window |
| undesirable | BPB target missed by a large margin; this phase is still FineWeb-only calibration (`fineweb_mix_ratio=1.0`, hard/medium/complex ratios `0.0`), so the current FineWeb result is not evidence of OOD transfer from hard reasoning data; OAI eval uses 8 batches and must not be overclaimed without tokenizer, n-gram, dataset-easiness, FineWeb-only, hard-only zero-shot, mixed-exposure, and GFlowNet/GraphCG/toric/topology ablation controls; toric active-face margin remains strongly negative (`geometry mean -2.827209`, train around `-3.109375`); toric shadow margins are thin (`mean 0.023280`, min `0.000166`); toric memory entropy is collapsed-low (`0.029454`); BGG standard leakage remains high (`0.478637`); exact triangle validity is only `0.497139`; HDBSCAN noise is material (`0.226020`); Hessian trace, dominant curvature, HVP norm, and probe grad norm are `NaN`, so Hessian evidence cannot justify a rollback |

Core values:

| metric | value | read |
|---|---:|---|
| checkpoint train BPB / loss | `4.49188378560791` / `3.1135365813970566` | lower than most late W&B samples, but train stream is noisy |
| post-analysis official-style FineWeb BPB / loss | `4.520416072849183` / `3.1333136558532715` | competition calibration gate available, improved, target failed |
| W&B OAI/FineWeb BPB at export | `4.52796404743025` | last exported W&B summary before the post-analysis eval |
| hard validation best BPB | `5.349750123909216` | loop primary BPB, nonworse since step 4750 |
| hard score-first / GFlowNet BPB | `5.342478009500132` / `5.35063996214478` | score-first helps slightly; GFlowNet does not beat deterministic |
| geometry mean / best BPB | `7.388788839181264` / `5.03637170791626` | large branch-selection gap |
| geometry mean / best answer BPB | `7.530481064764796` / `4.912411313446559` | answer-span quality remains weak |
| GFlowNet loss / entropy / diversity | `1.630382` / `2.771388` / `0.998646` | alive, diverse, not selecting low-BPB branches reliably |
| GraphCG basis / coherence / covariance | `0.140295` / `0.068848` / `0.146411` | frame is stable enough; utility is weak |
| Toric BGG resolution/leakage/Gale/signature | `0.934541` / `0.478637` / `0.149744` / `0.007681` | consistency good, leakage high |
| Koszul exactness / syzygy / affine coverage | `0.013567` / `0.006784` / `1.0` | stable diagnostic |
| topology exact edge / triangle validity | `0.972594` / `0.497139` | edge complex good, triangles weak |
| topology directed asymmetry / cycle flux | `0.456544` / `4.52e-18` | noncommutative structure without cycle explosion |
| HDBSCAN clusters/noise/stability | `0.914931` / `0.226020` / `0.773980` | structured but noisy |
| MST efficiency / path smoothness | `0.477016` / `0.124134` | useful graph geometry, rough trajectories |
| toric active-face entropy / margin | `0.336880` / `-2.827209` geometry mean | entropy present, margins inverted |
| toric shadow occupied cells / mean margin / min margin | `15.25` / `0.023280` / `0.000166` | broad fan coverage, thin stability margin |
| Slepian concentration / leakage / effective modes | `1.0` / `0.0` / `3.419880` | desired spectral audit |
| Hessian probe | probe loss finite, curvature/trace/HVP `NaN` | unusable rollback evidence |

### Plot Review

Reviewed generated metric plots, simplex plots, geometry triangles/tetrahedra,
3D reasoning trajectories, Ramachandran-style phase-energy plots, toric
phase/winding plots, energy landscapes, directed nested-simplicial topology
plots, exact persistence/Koszul audits, and noncommutative heatmaps.

Plot categorization:

| diagnostic | category | evidence |
|---|---|---|
| core metric timeseries | desired but weak | validation BPB/loss decline cleanly; train BPB/loss are noisy and recently tilt up |
| recent slopes/correlations | mixed | validation and OAI slopes are favorable; train BPB correlates with GFlowNet loss and recent train slope is weakly adverse |
| simplex plots | desired but too weak/slow | budget movement changes reasoning/K proxies more than BPB; low-BPB contour is not strongly exploited |
| geometry tetrahedra | desired but weak | lower-BPB branches exist, but lower BPB is not consistently aligned with MST, toric entropy, or GFlowNet axes |
| 3D reasoning trajectories | desired but weak | branches are structured and nonblank, with visible off-manifold excursions and jagged fan-like spans |
| Ramachandran-style phase plots | desired but weak | chamber exploration is broad and non-collapsed, but phase paths are tangled rather than terminal-directed |
| energy landscapes | desired but weak | basins exist, but many typical branches remain far worse than the best branch |
| exact persistence/topology | mixed | edge validity, H0 behavior, inclusion, and zero cycle flux are healthy; H1/triangle validity and clustering noise are weak |
| toric phase/shadow/Slepian | mixed | fan occupancy and Slepian concentration are good; active-face and shadow margins are too thin |

### Mathematical Reading

Let `B_fw` be official-style FineWeb/OAI BPB and `B_hard` be hard ToricGT
validation BPB.  The loop history is:

```text
B_hard best: 5.382641, 5.373152, 5.359213, 5.349750, 5.349750
B_fw W&B:    4.546705, 4.540764, 4.532134, 4.527964
B_fw local step-5000 post-analysis: 4.520416
```

The hard validation first differences through step 4750 are negative
(`-0.009490`, `-0.013939`, `-0.009463`), then the best metric is flat at step
5000.  The FineWeb/OAI first differences are also negative, including the local
step-5000 post-analysis improvement from `4.527964` to `4.520416`.  The last
second differences are mildly positive, so descent is decelerating, but there is
no positive validation first derivative and no valid Hessian trace/sharpness
probe.  This is slow descent with noisy train samples, not a demonstrated
floor-bounce basin.

The branch/test-time-scaling diagnostics show a selection problem rather than a
representation collapse.  GFlowNet entropy and action diversity are high, and
branch replay finds terminals more than 2 BPB better than the mean branch, but
the learned policy does not yet put enough mass on those low-BPB terminals.
Because the BPB gate is still improving, the small existing GFlowNet/GraphCG
anchors should be preserved and monitored instead of promoted into heavier
auxiliary losses during byte warmup.

The two-gate rule is respected: FineWeb/OAI is the competition calibration gate,
and hard ToricGT reasoning plus GFlowNet/GraphCG/topology/toric/Koszul/BGG
diagnostics are the reasoning gate.  Both gates are weak but non-regressing
enough to continue.  Strong FineWeb performance after limited FineWeb exposure
would be possible OOD transfer only after controls against tokenizer convention,
n-gram/context-tree baselines, dataset easiness, FineWeb-only training,
hard-only zero-shot transfer, mixed exposure, and auxiliary ablations.

The adjustment proposal was run and saved:

```text
python scripts/propose_training_adjustments.py \
  --analysis-dir outputs/post_resume_analysis/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/step-00005000
```

It recommended lower LR or rollback, tiny GraphCG/GFlowNet/entropy/analogy
activation, and keeping toric/Koszul losses zero while margins are weak.  I used
that as evidence, not authority.  I rejected `EDIT_AND_RESTART` because the
active byte-warmup phase already has FineWeb-only calibration, tiny GFlowNet and
GraphCG anchors, zero toric/Koszul/topology losses, and improving FineWeb/OAI and
hard validation gates.  I rejected `ROLLBACK` because the validation first
derivative is not positive, the primary hard BPB is nonworse, the FineWeb
post-analysis BPB improved, and Hessian evidence is invalid.

### Decision

Action: `CONTINUE`.

No code or config scalar was changed.  The better-strategy stop sentinel was
not written.  The live training tmux was inspected and observed advancing past
step `5150` under the active supervisor, same config, and same W&B run.  Since
the supervisor had already continued the checkpoint lineage and the step-5250
watcher is non-interrupting CPU/fp32, I preserved the live supervisor-owned
process instead of killing and replaying the same run.

Operational handoff:

```text
supervisor tmux: toricgt_supervisor_fineweb_bpb_recovery
training tmux: toricgt_fineweb_bpb_recovery_live
watcher tmux: toricgt_fineweb_bpb_recovery_watcher
training config: config/train.parameter_golf_all_phases.yaml
analyzed checkpoint: checkpoints/parameter_golf_all_phases_fineweb_from0/random_order_step_00005000.pt
active training log: logs/parameter_golf_all_phases/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/supervisor/restart_001_20260603T124401Z/train.log
W&B run path: amelie-iska-math/toricgt-parameter-golf/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z
W&B URL: https://wandb.ai/amelie-iska-math/toricgt-parameter-golf/runs/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z
next watcher target: 5250
next watcher log: logs/parameter_golf_all_phases/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/supervisor/analysis_target_00005250_20260603T154409Z/watcher.log
watcher command: scripts/watch_training_analysis.py without --pause-training-before-analysis
watcher device/precision: cpu / fp32
loop env: BPB_TARGET=1.2, BPB_MAX_REVIEW_ITERATIONS=100,
          BPB_LOOP_STATE=/home/iska/Documents/amelie/bio/ToricGT/logs/parameter_golf_all_phases/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/supervisor/bpb_codex_loop_state.json,
          BPB_LOOP_STOP_FILE=logs/parameter_golf_all_phases/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/supervisor/bpb_codex_loop_stop,
          BPB_LOOP_NAME=all_phases_supervised_watchdog
```

## 2026-06-04 r77 Step 3650 BPB Review

Analyzed checkpoint:

```text
amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_4k_recovery_r77_20260604T183710Z/toricgt_seq4096_4k_recovery_r77_20260604T183710Z_step_003650.pt
```

Analysis directory:

```text
outputs/live_periodic_reviews/toricgt_seq4096_4k_recovery_r77_20260604T183710Z_watch_training_analysis/step-00003650
```

The BPB gate is still the binding constraint.  The official-style FineWeb/OAI
signal is `openai_parameter_golf/bpb = 1.2171306166080218` at step `3650`,
above the active target `1.2`.  The best loop BPB remains
`1.2169196232244919` at step `3600`, so the finite difference over the
available validation points is positive:

```text
3600 -> 3650: delta BPB = +0.000230617, about +0.000461 per 100 steps
required to hit 1.2 by step 4000: about -0.00489 per 100 steps
```

There are only two validation points in this export, so there is no reliable
second-difference or Hessian evidence.  The first derivative is nevertheless
positive at the gate boundary, and the train-side series is worse: `train/bpb`,
`fineweb/train_bpb`, and `openai_parameter_golf/train_bpb` are all categorized
`not_as_desired`, with median train BPB moving from `1.19101` to `1.21579`
and recent slope about `+0.44999` per 1k steps.  `train/loss` and
`train/total_loss` also moved up.  The core metric plot shows oscillatory train
loss/BPB with an upward moving average, while validation BPB is nearly flat but
slightly rising from the prior best.

Metric behavior categories:

```text
desired: 44
desired but too weak or slow: 21
undesirable: 14
```

Desired: auxiliary clamps are active and non-explosive; BGG standard leakage,
GraphCG off-diagonal coherence, Slepian leakage, and tropical active-face
margin are stable enough as probes; target metadata and checkpoint/runtime
signals are healthy.

Desired but too weak or slow: FineWeb/OAI validation BPB, validation loss,
best-val BPB, BPB gap-to-target, GraphCG spectral entropy, toric fan entropy,
Slepian/Pollak loss, Koszul/BGG loss, and weighted unscaled auxiliary loss are
not collapsing, but the improvement is too slow to satisfy the target gate.

Undesirable: train BPB/loss/total loss are rising; GraphCG loss and auxiliary
loss are moving against the intended lower-is-better objective.  This is
consistent with a local floor bounce rather than a clean descent basin.

Reasoning gate coverage is incomplete for this checkpoint.  Simplex and
geometry diagnostics both failed while loading the configless checkpoint
(`KeyError: 'config'`), so there are no valid generated simplex plots, 3D GoT
trajectories, Ramachandran-style phase plots, MST/smoothness/trajectory-length
summaries, energy/fitness landscapes, persistence topology plots, or branch
replay diagnostics to promote from.  The available W&B structural diagnostics
show low absolute structural pressure (`structural_recapture_score ~= 0.0495`),
with tropical/complexity pressure the largest family in the W&B summary and
the proposal compressing the missing geometry/topology evidence into a
trajectory-memory pressure.  That is too incomplete to override the BPB gate.

FineWeb exposure is direct here (`train_loader: fineweb10B_sp1024`,
`fineweb/val_bpb` present), so this checkpoint is not evidence for OOD transfer
from hard ToricGT reasoning data.  Any later transfer claim still needs
tokenizer controls, n-gram/context controls, dataset-easiness controls,
FineWeb-only controls, hard-data-only controls, and equal-budget mixed controls.

`scripts/propose_training_adjustments.py` was rerun on the analysis directory
with target `1.2`, gate step `4000`, and checkpoint step `3650`.  It reported
`status=off_track`, current primary BPB `1.2171306166080218`,
required drop `0.004894` per 100 steps, recent drop `0.0`, and primary action
`prepare_midrun_recovery_branch`.  I used this as evidence, not authority.  Its
`restart_policy=no_restart_until_next_gate` is weaker than the explicit handoff
rule to roll back before the first positive BPB derivative.

Decision: `ROLLBACK`.

No code or config scalar was changed, and the better-strategy stop sentinel was
not written.  The selected rollback point is the dense step-3600 checkpoint
before the positive first derivative.  The active successor run is already
training from that point under the same Parameter-Golf architecture and scalar
family:

```text
gate/supervisor tmux: toricgt_seq4096_4k_gate_r79_low_bpb_capture_20260604T185003Z
training tmux: toricgt_seq4096_4k_recovery_r79_low_bpb_capture_20260604T185003Z
training config: config/train.parameter_golf_random_order_dense.yaml
resume checkpoint: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_4k_recovery_r75_20260604T182013Z/toricgt_seq4096_4k_recovery_r75_20260604T182013Z_step_003600.pt
training log: amelie-iska/parameter-golf/logs/toricgt_seq4096_4k_recovery_r79_low_bpb_capture_20260604T185003Z.txt
gate log: logs/toricgt_seq4096_4k_recovery_r79_low_bpb_capture_20260604T185003Z.4k_gate.txt
W&B run path: amelie-iska-math/toricgt-parameter-golf/toricgt_seq4096_4k_recovery_r79_low_bpb_capture_20260604T185003Z
```

The non-interrupting next analysis sidecar is active with
`scripts/watch_training_analysis.py` on CPU/fp32, target step `3650`, and no
`--pause-training-before-analysis` flag:

```text
watcher tmux: toricgt_watch_training_analysis_r79_3650
watcher log: logs/toricgt_seq4096_4k_recovery_r79_low_bpb_capture_20260604T185003Z.watch_training_analysis_3650.txt
watcher output root: outputs/live_periodic_reviews/toricgt_seq4096_4k_recovery_r79_low_bpb_capture_20260604T185003Z_watch_training_analysis
loop env: BPB_TARGET=1.2, BPB_MAX_REVIEW_ITERATIONS=100,
          BPB_LOOP_STATE=/home/iska/Documents/amelie/bio/ToricGT/outputs/toricgt_seq4096_4k_recovery_r75_20260604T182013Z_live_bpb_codex_loop_state.json,
          BPB_LOOP_STOP_FILE=/home/iska/Documents/amelie/bio/ToricGT/outputs/toricgt_seq4096_4k_recovery_r75_20260604T182013Z_live_bpb_codex_loop_stop,
          BPB_LOOP_NAME=parameter_golf_bpb_target
```

## Seq4096 R74 Step-3500 BPB Review

Analysis directory:

```text
outputs/live_periodic_reviews/toricgt_seq4096_4k_recovery_r74_20260604T180722Z/step-00003500
```

Analyzed checkpoint:

```text
amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_4k_recovery_r74_20260604T180722Z/toricgt_seq4096_4k_recovery_r74_20260604T180722Z_step_003500.pt
```

Primary BPB gate status: not reached.  The loop primary metric is
`openai_parameter_golf/bpb=1.2198282837822922` at step 3500, so the active
target gap is `0.019828283782292267` against `BPB_TARGET=1.2`.  W&B
`fineweb/val_bpb` and `val/bpb` agree with the primary signal.  Validation loss
is `2.0596289606362665`; train BPB at the validation checkpoint is `1.2156`.

Official-style FineWeb/OAI calibration is high variance at this checkpoint.  The
generated compact OAI eval used one 256-token validation sequence and reported
`oai_competition/bpb=1.2017869051762973`, which is near the threshold but not a
promotion-grade estimate.  A CPU rerun over 16 sampled 256-token sequences
reported `oai_competition/bpb=1.328491831232304`.  Treat the one-sequence result
as evidence that some validation slices are nearly solved, not as proof that the
competition BPB gate has crossed 1.2.  Controls are still needed against
tokenizer, n-gram/context-tree, and validation-slice easiness explanations.

Metric categories from the W&B/statistical export:

- Desired: 51 metrics.  BPB and validation loss are descending, artifact export
  remains legal (`15,883,737` bytes, `116,263` bytes under the 16 MB limit), the
  compact geometry suite is non-collapsed, BGG/Koszul `d2` residuals are zero in
  the proxy records, and GraphCG off-diagonal coherence is near numerical zero.
- Desired but too weak or slow: 25 metrics.  Validation BPB improvement is real
  but marginal: `3400 -> 3450` drops `0.0013`, and `3450 -> 3500` drops
  `0.0019`.  The sparse second difference is favorable (`-0.0006`), but the
  projected target crossing is still about step `4119`, after the 4K gate.
  Train-to-validation transfer is weak: train BPB drops `0.0155` while
  validation BPB drops `0.0019`, for transfer efficiency `0.1226`.
- Undesirable: 3 metrics.  Two are stable advanced-control toggles logged under
  an undesirable stable category; the third is `bpb/improvement_from_initial`,
  whose category appears sign-inverted for an improvement metric.  The stronger
  undesirable signal is qualitative/statistical: broader sampled OAI BPB is much
  worse than the one-sequence result, so the near-threshold OAI point is not
  robust.

Plot review covered the BPB descent/target-zone plots, velocity requirement,
transfer efficiency, advanced metric evidence/control maps, simplex
triangles/tetrahedra, 3D reasoning trajectories, Ramachandran-style phase-energy
plots, topology/persistence audits, GraphCG basis plots, tropical chamber plots,
and energy/tetrahedron landscapes.  The plots are structured and nonblank.
Behavior is mixed: BPB geometry is moving in the right direction, but transfer
efficiency, analogical transport, layer/attention MST efficiency, and trajectory
smoothness are not yet strong.

Reasoning-gate read:

- GFlowNet/GoT branch quality is not directly available as a full training
  metric in this Seq4096 export; the geometry proxy is the available sidecar.
  It does not justify increasing branch/reasoning loss weight before the BPB
  gate clears.
- Kolmogorov proxies are useful but not yet decisive.  Recent full-log
  `NCD_lzma=0.8892497009058281`; geometry relative-K values range from about
  `0.642` to `0.811`, with relative-K gain mostly zero except layer/attention
  controls.  Compression structure exists, but it is not yet transferring
  cleanly into validation BPB.
- GraphCG basis behavior is good for embeddings and memory
  (`disentanglement_score` about `0.93-0.95` for embedding trajectories and
  `0.819` for bigram memory) but weak for layer/attention control records
  (`0.448-0.505`) with very large basis condition numbers.  Keep it as a tiny
  sidecar/controller signal, not a larger objective.
- Persistence/simplex/Koszul/BGG topology is coherent as a proxy:
  `bgg_resolution_consistency` is `~1.0`, `bgg_gale_dual_consistency=1.0`,
  `koszul_d2_residual=0`, and standard leakage is moderate (`0.0886-0.1319`).
  This supports preserving diagnostics, not changing scalar weights.
- MST and trajectory geometry are uneven.  Embedding MST efficiency is usable
  (`0.341-0.351`) and smoother for the norm-ordered trajectory
  (`path_smoothness=0.0968`), while layer and attention controls have low MST
  efficiency (`0.0568-0.0653`) and rough paths.  Bigram memory has the longest
  trajectory (`1310.64`) and strong terminal confidence, but low smoothness.
- Toric/tropical behavior is active rather than collapsed.  Toric fan entropy
  ranges from `0.626` to `0.925`; tropical chamber crossing rates are high
  (`0.75-0.963`), and the FineWeb curve diagnostic has active-face entropy
  `0.3271` with zero plateau pressure.  This is desired exploration, but too
  noisy to use as a heavier pre-threshold loss.
- Hessian/sharpness probes were not present in this analysis directory.  The
  checkpoint finite differences do not show a floor-bounce basin: validation
  first differences are negative and the sparse second difference is negative.

`scripts/propose_training_adjustments.py` was rerun on the analysis directory.
It recommended `restart_from_best_checkpoint_with_damped_structural_sidecars`.
I treat that as evidence of velocity risk, not as authority: the proposal is
driven by projected miss of the 4K gate, while the actual sparse validation
finite differences are still descending and there is no positive-curvature
rollback signal.  The BPB transfer controller also recommends a pre-threshold
BPB-clean objective with advanced families in sidecar mode.

Decision: `CONTINUE`.

No code or config scalar was changed, and the better-strategy stop sentinel was
not written.  The active continuation is R75:

```text
training tmux: toricgt_seq4096_4k_recovery_r75_20260604T182013Z
training log: amelie-iska/parameter-golf/logs/toricgt_seq4096_4k_recovery_r75_20260604T182013Z.txt
resume checkpoint: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_4k_recovery_r74_20260604T180722Z/toricgt_seq4096_4k_recovery_r74_20260604T180722Z_step_003500.pt
new checkpoint dir: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_4k_recovery_r75_20260604T182013Z
W&B run: amelie-iska-math/toricgt-parameter-golf/toricgt_seq4096_4k_recovery_r75_20260604T182013Z
resume controls: reset_optimizer=0, reset_rng=0, reset_loader=0
```

No active `scripts/supervise_parameter_golf_training.py` process was observed;
the Seq4096 tmux bundle is the active training owner.  A non-interrupting
Seq4096 sidecar watcher was launched for the next checkpoint because the generic
`watch_training_analysis.py` only matches native `random_order_step_*.pt`
checkpoints and would not see compact Seq4096 `*_step_*.pt` files:

```text
watcher tmux: toricgt_seq4096_4k_recovery_r75_nonblocking_watch_3550
target step: 3550
watcher log: logs/toricgt_seq4096_4k_recovery_r75_20260604T182013Z.nonblocking_3550_analysis.txt
output root: outputs/live_periodic_reviews/toricgt_seq4096_4k_recovery_r75_20260604T182013Z_nonblocking
pause behavior: non-interrupting; no training pause signal
compact OAI eval: cpu, seq_len=256, val_max_sequences=16
loop env: BPB_TARGET=1.2, BPB_MAX_REVIEW_ITERATIONS=100,
          BPB_LOOP_STATE=outputs/toricgt_seq4096_4k_recovery_r74_20260604T180722Z_live_bpb_codex_loop_state.json,
          BPB_LOOP_STOP_FILE=outputs/toricgt_seq4096_4k_recovery_r74_20260604T180722Z_live_bpb_codex_loop_stop,
          BPB_LOOP_NAME=parameter_golf_bpb_target
```

## 2026-06-03 Revealed FineWeb BPB Recovery Step-5250 Review

Run:

```text
wandb: amelie-iska-math/toricgt-parameter-golf/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z
checkpoint: checkpoints/parameter_golf_all_phases_fineweb_from0/random_order_step_00005250.pt
analysis: outputs/post_resume_analysis/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/step-00005250
loop: all_phases_supervised_watchdog, iteration 6 / 100, target BPB <= 1.2
```

Status:

```text
checkpoint train_bpb: 4.594683061499462
checkpoint train_loss: 3.18479160964489
checkpoint best_val_bpb: 5.349750123909216 at step 4750
checkpoint best_oai_competition_bpb: 4.523199440070747 at step 5000
official-style OAI/FineWeb BPB: 4.5242843066323895 from local_fineweb10B_sp1024_validation_decoded_to_utf8_bytes
official-style OAI/FineWeb loss: 3.1359949111938477
metric categories: desired=427, weak/slow=188, undesirable=160
geometry records: 4 selected records, 24 branches
geometry mean_bpb: 7.542314887046814
geometry best_bpb: 5.062435150146484
geometry best_answer_bpb: 4.934705768443206
mean MST efficiency: 0.47992876045229343
mean path smoothness: 0.1354596856298935
mean directed asymmetry: 0.4589943156735468
mean directed cycle flux: 5.537630829179353e-18
```

Metric and plot categorization:

- Desired: official-style FineWeb/OAI BPB improved from `4.546705` at step
  4000 to the best observed `4.52319944` at step 5000.  Future-permutation
  audit stayed zero, artifact size and parameter count stayed stable, GFlowNet
  entropy and action diversity stayed high (`~2.771` and `~0.9985`), GraphCG
  basis coherence decreased from `0.07910` to `0.06641`, Toric BGG
  `resolution_consistency` stayed high (`0.94082`), `d2_residual` stayed low
  (`0.06304`), and `koszul_linearity_residual=0`.
- Desired but too weak or slow: FineWeb/OAI BPB is still far from the target
  `1.2`, OAI eval covers only 8 batches, hard curated validation is still high
  (`best_val_bpb=5.34975`), train BPB/loss have only shallow robust recent
  improvement, simplex color ranges remain narrow around poor BPB, branch mean
  BPB is much worse than best branch BPB, MST efficiency is only `0.48`, and
  path smoothness is only `0.135`.
- Undesirable: hard validation BPB has positive first differences after step
  4750, OAI/FineWeb BPB is flat to slightly positive from 5000 to 5250, branch
  search gap is large (`mean_bpb - best_bpb ~= 2.48`), `graphcg/basis_loss`
  drifted upward, analogical map residuals drifted, chart-transition
  resolution shift drifted, `revealed_neighbor_context_norm` moved too much
  for a stable diagnostic, toric shadow margin is only `0.02305`, active-face
  margin is strongly negative (`-2.886` in geometry), toric memory entropy is
  low, and Toric BGG standard leakage remains high (`0.45835`).

Mathematical/statistical interpretation:

- FineWeb is the competition BPB gate.  Its finite differences were
  `-0.00594`, `-0.00863`, `-0.00417`, `-0.00476`, then `+0.000013` over the
  4000--5250 checkpoint sequence.  The positive value at 5250 is tiny, but the
  second difference is positive at the last interval, so the low-BPB direction
  has stalled at the step-5000 checkpoint.
- Hard ToricGT validation is the reasoning gate, not the competition score.
  Its `val/bpb` first differences were `-0.00949`, `-0.01394`, `-0.00946`,
  then `+0.00446` and `+0.01343`.  That is a clearer floor-bounce signature,
  but the two-gate rule selects the rollback point from the FineWeb gate, so
  the selected checkpoint is step 5000 rather than the earlier hard-validation
  best at 4750.
- Train BPB and train loss are noisy.  The robust W&B metric report estimates
  recent slopes of `-0.1638` BPB/1k steps and `-0.1135` loss/1k steps, but
  checkpoint-level train BPB jumps by `+0.1028` from 5000 to 5250.  This is a
  stochastic microbatch signal and does not override deterministic validation.
- Hessian data is not current enough to drive the action.  The only trace and
  dominant-curvature probe in the export is at step 4250 (`trace=-13.4776`,
  dominant curvature `8.6e-6`), so rollback is based on BPB finite differences,
  not on a new sharpness measurement.
- GFlowNet branch replay is alive but undertrained as a selector.  High
  entropy/diversity means there is no collapse, but geometry shows average
  branches are much worse than the best terminal.  That supports keeping the
  tiny active GFlowNet weights, not promoting a larger auxiliary loss while BPB
  is stalled.
- Kolmogorov and relative-K proxies are mixed.  `complexity/val/bpb` improved
  slightly to `5.15833`, validation zlib/lzma NCD proxies remain high, argmax
  byte accuracy is flat at `0.173828`, and analogical transfer gains exist but
  are not enough to certify a compression regime change.
- GraphCG basis behavior is mostly acceptable, but not strong.  Basis
  coherence improved and covariance was categorized desired, while basis loss
  drifted upward.  This is a monitor signal, not a reason to change the
  Parameter-Golf architecture.
- Persistence/simplex/Koszul topology is coherent.  Exact/inclusion residuals
  are near zero, directed cycle flux is numerical zero, HDBSCAN stability is
  usable (`0.77284` in geometry), and Slepian leakage is zero.  Topology remains
  a diagnostic until lower BPB accompanies the structure.
- Toric/tropical behavior is structured but weak as a BPB control.  Active-face
  entropy is nonzero, but margins are poor; toric shadow occupied fan cells are
  populated, but the low margin means heavy toric/Koszul losses would likely
  fight the BPB gate.
- Plot review covered the metric plots, recent-slope bars, simplex
  triangles/tetrahedra, geometry triangles/tetrahedra, 3D GoT trajectories,
  Ramachandran-style phase-energy plots, toric phase winding collections,
  energy landscapes, and topology audits.  The plots are nonblank and
  structured, but branch manifolds are long/diffuse and do not concentrate into
  a low-BPB attractor.
- No OOD-transfer claim is justified for this phase.  `fineweb_mix_ratio=1.0`
  and hard/complex mix ratios are zero, so the FineWeb result is direct
  FineWeb calibration exposure.  Future transfer claims still need tokenizer,
  n-gram/context-tree, dataset-easiness, FineWeb-only, hard-data-only, and
  equal-budget mixed controls.

The adjustment proposal from `scripts/propose_training_adjustments.py` was run
on the analysis directory and saved as:

```text
outputs/post_resume_analysis/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/step-00005250/metrics/training_adjustment_proposal.md
outputs/post_resume_analysis/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/step-00005250/metrics/training_adjustment_proposal.json
```

It recommended `decrease_or_rollback`, plus tiny GraphCG/GFlowNet/contrastive
activation and continued zero heavy toric/Koszul losses.  I treated the
proposal as evidence, not authority.  The active config already has a BPB-first
FineWeb warmup with tiny GraphCG/GFlowNet weights, no heavy toric/Koszul loss,
and high GFlowNet entropy, so I accepted the rollback part and rejected a
config edit for this iteration.

Decision: `ROLLBACK`.

Selected checkpoint:

```text
checkpoints/parameter_golf_all_phases_fineweb_from0/random_order_step_00005000.pt
```

No code or config scalar was changed.  The better-strategy stop sentinel was
not written.  Training was explicitly relaunched from the selected checkpoint
with the same config and the required BPB loop environment.

Operational handoff:

```text
supervisor tmux: toricgt_supervisor_fineweb_bpb_recovery
training tmux: toricgt_fineweb_bpb_recovery_live
training config: config/train.parameter_golf_all_phases.yaml
rollback checkpoint: checkpoints/parameter_golf_all_phases_fineweb_from0/random_order_step_00005000.pt
rollback log: logs/parameter_golf_all_phases/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/supervisor/rollback_00005250_to_00005000_20260603T162235Z/train.log
W&B run path: amelie-iska-math/toricgt-parameter-golf/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z
watcher tmux: toricgt_fineweb_bpb_recovery_watcher
next watcher target: 5500
next watcher log: logs/parameter_golf_all_phases/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/supervisor/analysis_target_00005500_20260603T161710Z/watcher.log
watcher device/precision: cpu / fp32
watcher pause behavior: non-interrupting; no --pause-training-before-analysis
loop env: BPB_TARGET=1.2, BPB_MAX_REVIEW_ITERATIONS=100,
          BPB_LOOP_STATE=/home/iska/Documents/amelie/bio/ToricGT/logs/parameter_golf_all_phases/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/supervisor/bpb_codex_loop_state.json,
          BPB_LOOP_STOP_FILE=logs/parameter_golf_all_phases/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/supervisor/bpb_codex_loop_stop,
          BPB_LOOP_NAME=all_phases_supervised_watchdog
```

## 2026-06-04T17:56Z Seq4096 R73 Step 3250 BPB Gate Handoff

Context:

```text
run: toricgt_seq4096_4k_recovery_r73_20260604T174607Z
analysis directory: outputs/live_periodic_reviews/toricgt_seq4096_4k_recovery_r73_20260604T174607Z/step-00003250
analyzed checkpoint: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_4k_recovery_r73_20260604T174607Z/toricgt_seq4096_4k_recovery_r73_20260604T174607Z_step_003250.pt
W&B run: amelie-iska-math/toricgt-parameter-golf/toricgt_seq4096_4k_recovery_r73_20260604T174607Z
target BPB: 1.2
loop iteration at handoff: 4 / 100
```

Decision: `CONTINUE`.

No code or config scalars were changed, and no better-strategy stop sentinel was
written.  I explicitly resumed/confirmed the training process with `SIGCONT` and
tmux flow resume, then verified that the run advanced from the analyzed step
3250 to step 3300.

Metric categories:

- Desired: Seq4096 sampled OAI/FineWeb evaluator is available and near target
  (`oai_competition/bpb=1.2124380193422888`, sampled, one sequence, seq len
  256); the official-style validation BPB improved after resume from `1.2290`
  at step 3250 to `1.2270` at step 3300; GraphCG basis diagonalization is strong
  in embedding/tropical/memory records (`disentanglement_score` about
  `0.827--0.948`, off-diagonal coherence near `1e-8`); BGG resolution and
  Gale-dual consistency are `1.0`; Koszul `d2` residual is `0.0`; Slepian
  concentration is effectively `1.0` for embedding/tropical/memory records.
- Desired but too weak or slow: the analyzed full validation BPB is still
  `0.029` above target at step 3250, with only one validation point in the r73
  export and no paired train/validation transfer slope; the required 3250->4000
  validation velocity is `0.0038667` BPB per 100 steps; train BPB at the first
  resumed checkpoint is higher (`train_bpb=1.2451` at step 3300); analogical
  transport is weak (`0.0` on most records, `0.1667` on directed layer flow);
  MST efficiency is good only for embedding/tropical records and weak for
  layer/attention flow; hard-reasoning/GFlowNet branch quality is mostly proxy
  evidence rather than direct held-out GoT/ToT/CoT BPB in this review.
- Undesirable: prior branches showed a floor-bounce from r71 step 3500 to 3550
  (`1.2198 -> 1.226657554487925`, a positive first difference of about
  `+0.0137` BPB per 100 steps) and r72 step 3550 worsened further; current r73
  tropical phase plots have frequent chamber crossings and high plateau pressure
  in some records; GraphCG basis condition numbers are very large despite low
  off-diagonal coherence; BGG standard leakage is nonzero (`0.0835--0.1319`);
  Hessian/sharpness probes and second differences are unavailable for r73.

Mathematical/statistical read:

- The step-3250 analysis alone was `off_track_unreachable` because the velocity
  estimate was undefined and the gap was `0.029`.  After active resume, the
  first finite difference is favorable: `1.2290 -> 1.2270` over 50 steps, or
  about `-0.004` BPB per 100 steps.  From step 3300 the remaining target gap is
  `0.027`, requiring about `0.00386` BPB per 100 steps through step 4000.  This
  justifies continuing to the next non-interrupting gate rather than rolling
  back immediately.
- The sampled OAI result is encouraging but not authoritative: it used a single
  sampled decoded FineWeb sequence.  Any OOD-transfer claim from limited FineWeb
  exposure still needs tokenizer, n-gram, dataset-easiness, FineWeb-only,
  hard-data-only, and equal-budget mixed controls.
- The prior positive derivative at r71 3500->3550 is real floor-bounce evidence,
  but r73 restarted earlier at step 3250 and has now produced a negative first
  derivative.  A rollback to the last pre-bounce dense checkpoint is therefore
  premature until r73 shows a renewed positive derivative or curvature signal.
- Topology/Koszul/BGG/GraphCG/tropical metrics are useful sidecar controls under
  the BPB-AMP two-gate rule.  They do not justify discarding the FineWeb BPB gate
  or promoting auxiliary losses beyond the current scalar controls while BPB is
  still above target.

`scripts/propose_training_adjustments.py` was rerun on the analysis directory
with the r73 run path and target.  Its proposal reported `primary_action:
prepare_midrun_recovery_branch`, `restart_policy: no_restart_until_next_gate`,
and `target_status: off_track`.  I treated that as evidence rather than
authority: because the post-resume finite difference is now negative and the
checkpoint lacks second-difference/Hessian evidence of a new basin, the selected
action remains `CONTINUE`.

Operational handoff:

```text
training tmux: toricgt_seq4096_4k_recovery_r73_20260604T174607Z
training log: amelie-iska/parameter-golf/logs/toricgt_seq4096_4k_recovery_r73_20260604T174607Z.txt
latest verified checkpoint: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_4k_recovery_r73_20260604T174607Z/toricgt_seq4096_4k_recovery_r73_20260604T174607Z_step_003350.pt
latest verified validation: step 3350, val_loss 2.0676, val_bpb 1.2246
supervisor tmux: toricgt_seq4096_live_periodic_full_analysis_supervisor
supervisor log: logs/seq4096_live_periodic_full_analysis_supervisor.log
sidecar watcher tmux: toricgt_seq4096_live_full_analysis_toricgt_seq4096_4k_recovery_r73_20260604T174607Z
sidecar watcher: scripts/watch_seq4096_analysis.py, non-pausing, Codex review hook disabled
reason for watcher substitution: scripts/watch_training_analysis.py expects RandomOrderLM random_order_step checkpoints; this run uses compact Seq4096 GPT-style checkpoints, so the Seq4096-compatible watcher is the safe non-interrupting sidecar.
loop env preserved: BPB_TARGET=1.2, BPB_MAX_REVIEW_ITERATIONS=100, BPB_LOOP_NAME=parameter_golf_bpb_target,
                    BPB_LOOP_STATE=/home/iska/Documents/amelie/bio/ToricGT/outputs/toricgt_seq4096_4k_recovery_r71_20260604T172807Z_live_bpb_codex_loop_state.json,
                    BPB_LOOP_STOP_FILE=/home/iska/Documents/amelie/bio/ToricGT/outputs/toricgt_seq4096_4k_recovery_r71_20260604T172807Z_live_bpb_codex_loop_stop
```

Resume note: because this is a rollback within the same W&B run, W&B warns that
rows below its previously seen step are ignored until the replay passes the old
run step.  Local training, checkpointing, and the non-blocking CPU watcher are
active, so training is not left waiting on Codex review.

## 2026-06-03 Automated Review: Step 4,750 FineWeb-Revealed BPB Recovery Gate

The watcher analyzed:

```text
outputs/post_resume_analysis/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/step-00004750/
```

Analyzed checkpoint:

```text
checkpoints/parameter_golf_all_phases_fineweb_from0/random_order_step_00004750.pt
```

The BPB loop target remains `<= 1.2`.  This was review iteration `4 / 100`.
The loop primary BPB is `best_val_bpb=5.349750123909216` at step `4750`;
the target is not reached.

### Metric Categorization

| category | read |
|---|---|
| desired | both BPB gates improved monotonically across the reviewed checkpoints: hard validation `5.382641461921426 -> 5.373151803212118 -> 5.359212953324989 -> 5.349750123909216`, and W&B official-style FineWeb/OAI BPB `4.54670499689477 -> 4.540763684238064 -> 4.532134282748372 -> 4.52796404743025`; the dedicated local OAI eval on decoded `fineweb10B_sp1024` validation bytes is available at `4.5264447526925276` BPB, loss `3.1374924182891846`, 8 eval batches; train BPB/loss medians drift down slightly in the exported window; GFlowNet entropy and action diversity remain high (`2.771354`, `0.998676`) without collapse; GraphCG axis entropy remains high (`0.997391`) and chart energy decreases; topology boundary/inclusion/cycle-flux diagnostics stay effectively zero; exact topology edge validity is healthy; Koszul exactness (`0.013288`) and syzygy residuals are small; Toric BGG resolution consistency remains high (`0.931139`); Slepian concentration/leakage are clean (`1.0` / `0.0`) |
| desired but too weak or slow | both gates are still far above `1.2`; recent train BPB slope is negative but weak (`-0.0621/1k`, `t=-0.579`), and validation improvement is only about `-0.0450/1k` by the loop-point fit; score-first hard validation gives only a small legal adapter gain (`5.342478` vs deterministic `5.349750`); simplex BPB is nearly flat (`6.55255--6.56342`) despite larger reasoning budgets and better MST efficiency; geometry branch replay finds useful terminals but not reliably (`mean BPB 7.3254`, best `5.0279`, mean-best gap `2.2976`; mean answer BPB `7.4614`, best answer `4.9076`); MST efficiency is moderate (`0.4756`) and path smoothness is low (`0.1411`); hard reasoning records are uneven, with English reasoning near `5.6` BPB, GoT/math better at best branch, and Hebrew much worse around `13` BPB; Kolmogorov proxies are mixed, with hard-val argmax byte accuracy stuck at `0.173828` and prediction target NCD worsening to `0.935378` |
| undesirable | the BPB target is missed by a large margin; the current phase is still FineWeb-only calibration (`fineweb_mix_ratio=1.0`, hard/medium/complex fractions `0.0`), so this is not evidence for OOD transfer from hard reasoning data; OAI validation uses only 8 batches and must not be overclaimed without tokenizer, n-gram, dataset-easiness, FineWeb-only, hard-only, mixed-exposure, and ablation controls; toric active-face margins remain strongly negative (`train -3.165039`, geometry mean `-2.787145`) and toric shadow margins are thin (`mean 0.023200`, min `0.000135`); toric memory entropy is low (`0.029247`); BGG standard leakage remains high (`0.487591`); exact triangle validity remains weak (`0.486968`); HDBSCAN noise is still material (`0.2269` geometry mean); Hessian trace, dominant curvature, HVP norm, and probe grad norm are `NaN` in the W&B export, so Hessian evidence cannot justify a rollback |

Core values:

| metric | value | read |
|---|---:|---|
| checkpoint train BPB | `4.564344289991357` | improving at the checkpoint but noisy |
| checkpoint train loss | `3.1637623757123947` | same direction as train BPB |
| dedicated local official-style FineWeb BPB | `4.5264447526925276` | competition calibration gate available, failed target |
| W&B OAI/FineWeb BPB | `4.52796404743025` | monotone improvement from prior gates |
| hard validation BPB | `5.349750123909216` | loop primary BPB, improved from steps 4000/4250/4500 |
| hard score-first BPB | `5.342478009500132` | small legal adapter gain only |
| hard GFlowNet BPB | `5.35063996214478` | does not beat deterministic validation |
| geometry mean / best BPB | `7.325421591599782` / `5.027864456176758` | branch replay gap remains large |
| geometry mean / best answer BPB | `7.461353426504306` / `4.907620220758822` | answer-span quality remains weak |
| GFlowNet entropy / diversity / loss | `2.771354` / `0.998676` / `1.528211` | alive, not selecting low-BPB branches reliably |
| GraphCG axis entropy / basis loss / chart energy | `0.997391` / `0.138160` / `4.851279` | stable frame, weak utility |
| Toric BGG resolution/leakage/Gale/signature | `0.931139` / `0.487591` / `0.190301` / `0.003901` | stable diagnostics, leakage high |
| Koszul exactness / syzygy | `0.013288` / `0.006645` | stable diagnostic |
| topology exact edge / triangle validity | `0.966353` / `0.486968` | edge complex good, triangles weak |
| topology directed asymmetry / cycle flux | `0.462828` / `5.48e-18` | noncommutative structure without cycle explosion |
| HDBSCAN clusters/noise/stability | `0.9132` / `0.2269` / `0.7731` | structured but noisy |
| MST efficiency / path smoothness | `0.475562` / `0.141096` | useful graph geometry, rough trajectories |
| toric active-face entropy / margin | `0.379849` / `-2.787145` geometry mean | entropy present, margins inverted |
| toric shadow cells / min margin | `15.75` / `0.000135` | broad fan coverage, thin stability margin |
| Slepian concentration / leakage / effective modes | `1.0` / `0.0` / `3.419585` | desired spectral audit |

### Plot Review

Reviewed generated metric plots, simplex plots, geometry triangles/tetrahedra,
3D reasoning trajectories, Ramachandran-style phase-energy plots, toric
phase/winding plots, energy landscapes, topology contact plots, exact
persistence/Koszul audits, noncommutative heatmaps, toric shadow audits, and
Slepian audits.

Plot categorization:

| diagnostic | category | evidence |
|---|---|---|
| core metric timeseries | desired but weak | validation and OAI BPB drift down; train BPB is noisy but not reversing the gate |
| recent slopes | mixed | train BPB/loss slopes are weakly negative; complexity NCD and several reasoning proxies move adversely |
| simplex plots | desired but too weak/slow | budget and MST placement improve more than BPB; low-BPB contours are not strongly exploited |
| geometry tetrahedra | desired but weak | low-BPB branches exist, but lower BPB is not consistently aligned with higher MST, smoother flow, or toric/GFlowNet axes |
| 3D reasoning trajectories | desired but weak | nonblank structured GoT branches exist; many branches remain fan-like or jagged |
| Ramachandran-style phase plots | desired but weak | chamber exploration is broad and non-collapsed, but phase paths are tangled rather than cleanly terminal-directed |
| energy landscapes | desired but weak | basins exist but the mean branch remains far worse than the best branch |
| exact persistence/Koszul/topology | mixed | edge validity, inclusion, H0 behavior, and zero cycle flux are healthy; H1/triangle validity and clustering noise remain weak |
| toric shadow/Slepian audits | mixed | fan occupancy and Slepian concentration are good; active-face and shadow margins are too thin |

### Mathematical Reading

Let `B_fw` be the official-style FineWeb/OAI BPB and `B_hard` be hard ToricGT
validation BPB.  Through the loop checkpoints:

```text
B_hard: 5.382641, 5.373152, 5.359213, 5.349750
B_fw:   4.546705, 4.540764, 4.532134, 4.527964
```

The first differences are still negative:

```text
Delta B_hard: -0.009490, -0.013939, -0.009463
Delta B_fw:   -0.005941, -0.008630, -0.004170
```

The last second difference is mildly positive after the stronger step-4500
improvement, but there is no positive validation first derivative and the
Hessian export is unusable (`NaN`).  This is slow descent, not a demonstrated
floor-bounce basin.

The branch/test-time-scaling diagnostics show a policy-selection problem:
geometry has a mean-best BPB gap of about `2.30`, while GFlowNet entropy and
diversity remain high.  The policy can sample diverse branches but does not yet
concentrate probability on the lower-BPB terminals.  Because the current phase
is BPB-first FineWeb warmup, the correct response is to monitor and preserve the
tiny GFlowNet/GraphCG anchors, not to promote heavy toric/Koszul/topology losses
while margins are weak.

The two-gate rule is respected as follows: the FineWeb gate is the competition
calibration/evaluation gate and is improving too slowly; the reasoning gate is
alive but also weak/noisy.  The gates do not disagree enough to justify changing
mixture ratios or discarding either objective.  Future strong FineWeb results
after limited FineWeb exposure should still be treated as possible OOD transfer
only after controls against tokenizer convention, n-gram/context-tree baselines,
dataset easiness, FineWeb-only training, hard-only zero-shot transfer, mixed
exposure, and GFlowNet/GraphCG/toric/topology ablations.

The adjustment proposal was run and saved:

```text
python scripts/propose_training_adjustments.py \
  --analysis-dir outputs/post_resume_analysis/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/step-00004750 \
  --output-json outputs/post_resume_analysis/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/step-00004750/training_adjustment_proposal.json \
  --output-md outputs/post_resume_analysis/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/step-00004750/training_adjustment_proposal.md
```

It recommended lower LR or rollback, tiny GraphCG/GFlowNet/entropy/analogy
activation, and keeping toric/Koszul losses zero while margins are weak.  I used
that as evidence, not authority.  I rejected `EDIT_AND_RESTART` because the
active byte-warmup phase already has FineWeb-only training, tiny GraphCG and
GFlowNet anchors, zero toric/Koszul losses, and monotone validation/OAI BPB
improvement.  I rejected `ROLLBACK` because the validation first derivative is
negative and there is no reliable Hessian or curvature evidence of a bounce
basin.

### Decision

Action: `CONTINUE`.

No code or config scalar was changed.  The better-strategy stop sentinel was
not written.  Training was not left inactive: the live training tmux was
explicitly inspected and observed advancing past step `4890` from the same
supervisor-owned run and config.  Since the active supervisor had already
continued this checkpoint lineage and the next watcher is non-interrupting, I
preserved the live process instead of killing and replaying the same W&B run.

Operational handoff:

```text
supervisor tmux: toricgt_supervisor_fineweb_bpb_recovery
training tmux: toricgt_fineweb_bpb_recovery_live
training config: config/train.parameter_golf_all_phases.yaml
analyzed checkpoint: checkpoints/parameter_golf_all_phases_fineweb_from0/random_order_step_00004750.pt
active training log: logs/parameter_golf_all_phases/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/supervisor/restart_001_20260603T124401Z/train.log
W&B run path: amelie-iska-math/toricgt-parameter-golf/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z
W&B URL: https://wandb.ai/amelie-iska-math/toricgt-parameter-golf/runs/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z
watcher tmux: toricgt_fineweb_bpb_recovery_watcher
next watcher target: 5000
next watcher log: logs/parameter_golf_all_phases/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/supervisor/analysis_target_00005000_20260603T151207Z/watcher.log
watcher device/precision: cpu / fp32
watcher pause behavior: non-interrupting; no --pause-training-before-analysis
loop env: BPB_TARGET=1.2, BPB_MAX_REVIEW_ITERATIONS=100,
          BPB_LOOP_STATE=/home/iska/Documents/amelie/bio/ToricGT/logs/parameter_golf_all_phases/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/supervisor/bpb_codex_loop_state.json,
          BPB_LOOP_STOP_FILE=logs/parameter_golf_all_phases/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/supervisor/bpb_codex_loop_stop,
          BPB_LOOP_NAME=all_phases_supervised_watchdog
```

## 2026-06-03 Automated Review: Step 4,250 FineWeb-Revealed BPB Recovery Gate

The watcher analyzed:

```text
outputs/post_resume_analysis/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/step-00004250/
```

Analyzed checkpoint:

```text
checkpoints/parameter_golf_all_phases_fineweb_from0/random_order_step_00004250.pt
```

The BPB loop target remains `<= 1.2`.  The loop primary BPB is
`best_val_bpb=5.373151803212118` at step `4250`; the target is not reached.

### Metric Categorization

| category | read |
|---|---|
| desired | official-style FineWeb BPB remains available on the local decoded `fineweb10B_sp1024` validation shard (`oai_competition/bpb=4.53821558928288`, loss `3.145651340484619`, 8 eval batches); the hard validation gate improved from step 4000 (`best_val_bpb 5.382641461921426 -> 5.373151803212118`); the W&B window still has a weak downward median drift for train BPB/loss (`train/bpb` median `4.60605 -> 4.55052`, `train/loss` median `3.19267 -> 3.15418`); GFlowNet entropy and diversity remain high (`2.7708`, `0.9985`) without collapse; GraphCG basis coherence is stable and decreasing (`0.07910 -> 0.07471` median); topology boundary, inclusion, and directed cycle flux stay effectively zero; exact topology edge validity is high (`0.9735` mean); Toric BGG resolution consistency remains high (`0.9319`) and standard leakage improved in the recent window (`0.4637` last); Koszul exactness and syzygy residuals stay small; Slepian concentration/leakage are clean (`1.0` / `0.0`) |
| desired but too weak/slow | both BPB gates are far above `1.2` (`FineWeb 4.5382`, hard best `5.3732`); checkpoint train BPB worsened from step 4000 to 4250 (`4.60533 -> 4.631999`) even though hard validation improved; the proposal script measured a weak positive recent train BPB slope (`+0.0331/1k`, `t=0.16`); score-first validation improves hard BPB only slightly (`5.37177` vs deterministic `5.38264` from the exported run summary); branch search has a useful best but poor mean (`mean BPB 7.1902`, best `5.0204`, mean-best gap `2.1698`); simplex budget scaling is too slow for the gain (`budget 8` BPB `6.5347` vs `budget 1` BPB `6.5360` with much more wall time); MST efficiency is coherent but modest (`0.4582` mean); trajectory smoothness is low (`0.1501` mean) and path length is long (`834.2` mean); toric shadow fan coverage is broad (`15.79` occupied cells), but margins are very thin (`mean_margin 0.0239`, `min_margin 1.23e-4`) |
| undesirable | the primary BPB target is missed by a large margin; the last exported train finite differences are locally positive (`train/bpb` last step slope about `+13.18/1k`, positive second-difference estimate), so train BPB is noisy and cannot be used as a promotion signal; GraphCG axis variance is categorized not-as-desired (`+17.2%` relative median change, high CV); toric memory entropy is deteriorating in the W&B window (`0.03537 -> 0.02904` median, recent slope `-0.0202/1k`, `t=-3.45`); tropical active-face margins remain strongly negative (`train -3.1445`, geometry mean `-2.5289`); toric binomial residual and leaf residual are high (`1.3196`, `0.9875` geometry means); Toric BGG standard leakage remains high in absolute terms (`0.4637`); exact triangle validity is weak (`0.4815` mean); the W&B analysis export contains `NaN` Hessian trace/dominant-curvature/HVP values at step 4000, so there is no reliable Hessian rollback signal in the exported metrics |

Core values:

| metric | value | read |
|---|---:|---|
| checkpoint train BPB | `4.631999` | worse than step 4000, but not the promotion gate |
| checkpoint train loss | `3.210657` | same checkpoint rebound as train BPB |
| official-style FineWeb BPB | `4.538216` | competition calibration gate available, failed target |
| W&B best FineWeb BPB | `4.533575` | best competition signal so far in this loop |
| hard validation BPB | `5.373152` | loop primary BPB, improved from step 4000 |
| hard score-first BPB | `5.371769` | small legal adapter gain in exported run summary |
| complexity hard-val BPB | `5.171556` in live log near checkpoint | hard validation still weak |
| geometry mean / best BPB | `7.190219` / `5.020374` | branch replay gap remains large |
| geometry mean / best answer BPB | `7.318581` / `4.895271` | answer-span quality remains weak |
| GFlowNet entropy / diversity / loss | `2.770810` / `0.998465` / `1.659062` | alive; policy not yet exploiting branch gap |
| GraphCG basis/coherence | `basis_loss 0.137829`, `coherence 0.074707` | stable frame, weak utility |
| Toric BGG resolution/leakage/Gale/signature | `0.931879` / `0.463675` / `0.108838` / `0.006989` | mostly stable diagnostics; leakage still high |
| Koszul exactness/syzygy | `0.013268` / `0.006635` | stable diagnostic |
| topology exact edge / triangle validity | `0.973500` / `0.481498` | edge complex good, triangles weak |
| topology directed asymmetry / cycle flux | `0.465348` / `5.80e-18` | noncommutative structure without cycle explosion |
| topology HDBSCAN clusters/noise/stability | `0.9097` / `0.2263` / `0.7737` | structured but not cleanly clustered |
| MST efficiency / path smoothness / path length | `0.458176` / `0.150068` / `834.193` | useful graph geometry, rough trajectories |
| toric active-face entropy / margin | `0.454431` / `-2.528925` geometry mean | entropy present, margins inverted |
| toric shadow cells / min margin | `15.791667` / `0.000123` | broad coverage, thin stability margin |
| Slepian concentration / leakage / effective modes | `1.0` / `0.0` / `3.423214` | desired spectral audit |
| W&B Hessian probe | `probe_loss 3.194189`, trace/curvature/HVP `NaN` | unusable rollback evidence in export |
| live-log Hessian near checkpoint | trace `-13.4776`, dominant curvature `8.60e-6`, HVP norm `0.01256`, grad norm `0.3446` | small-curvature probe after export; not a floor-bounce signal |

### Plot Review

Reviewed representative generated artifacts from the metrics, simplex, geometry,
trajectory, Ramachandran-style phase, energy-landscape, and topology plot sets:

```text
metrics/core_metric_timeseries.png
metrics/recent_metric_slopes.png
simplex/reasoning_k_bpb_triangle.png
simplex/reasoning_k_bpb_mst_tetrahedron.png
geometry/triangles/energy_landscape.png
geometry/tetrahedra/reasoning_k_bpb_mst.png
geometry/tetrahedra/toric_gfn_bpb.png
geometry/trajectories/*_trajectory_3d.png
geometry/trajectories/*_toric_phase_simplicial_trajectory.png
geometry/topology/*_exact_persistence_morphisms.png
geometry/topology/*_toric_shadow_audit.png
```

Plot categorization:

| diagnostic | category | evidence |
|---|---|---|
| core metric timeseries | desired but weak | smoothed train BPB/loss drift down over the window, but the late samples turn up |
| recent slopes | mixed | strongest recent movement is in GraphCG/topology/complexity audits; BPB slope is weak and noisy |
| simplex plots | desired but too weak/slow | BPB is nearly flat around `6.535--6.543`; larger budgets improve MST/reasoning placement more than BPB |
| geometry tetrahedra | desired but weak | low-BPB branches are visible, but higher-MST or higher-toric branches are not consistently the lower-BPB branches |
| 3D reasoning trajectories | desired but weak | nonblank structured GoT branches exist, but long jagged excursions and dense answer-span columns match low path smoothness |
| Ramachandran-style phase plots | desired but weak | broad chamber crossings show exploration rather than collapse, but phase paths are too tangled for clean terminal selection |
| energy landscapes | desired but weak | lower-BPB branches cluster away from high MST-efficiency points; useful basins exist but are not typical |
| exact persistence/Koszul audits | mixed | edge validity and H0 map behavior are healthy; H1/triangle validity is weak |
| toric shadow/Slepian audits | mixed | Newton fan coverage and Slepian concentration are good; tropical margins remain too thin |

### Mathematical Reading

Let `B_fw` be official-style FineWeb BPB, `B_hard` be hard ToricGT validation
BPB, and `B_train` be the current train BPB.  At the step-4250 gate:

```text
B_fw ~= 4.538, B_hard ~= 5.373, B_train ~= 4.632.
```

Both gates fail the `1.2` target.  The two-gate rule does not support discarding
either objective: the FineWeb gate is lower and competition-calibrated, while the
hard reasoning gate still detects weak branch, topology, and toric behavior.
The gates do not strongly disagree in the promotion sense because `B_hard`
improved, but the train checkpoint finite difference is unfavorable:

```text
Delta B_train(3750->4000) = -0.038785
Delta B_train(4000->4250) = +0.026669
Delta^2 B_train = +0.065454

Delta B_hard(4000->4250) = -0.009490
```

That is a noisy train rebound with a better hard-validation checkpoint, not a
rollback case.  The available W&B Hessian data are `NaN`, and the live-log
curvature probe near the checkpoint has very small dominant curvature, so
second-difference/Hessian evidence is insufficient to label a floor-bounce
basin.

The branch/test-time-scaling math is also clear: the simplex budget increase
has sub-basis-point BPB gain at much higher trajectory cost, while the full
geometry branch search has a large mean-best gap.  The right interpretation is
weak branch selection, not a need for larger search budgets.  Kolmogorov proxies
show analogical transfer gains, but prediction-target NCD remains high
(`~0.926--0.930` in simplex records, `~0.927` hard-val live log), so compression
quality is still weak.

The proposal script was run and used as evidence, not authority:

```text
python scripts/propose_training_adjustments.py \
  --analysis-dir outputs/post_resume_analysis/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/step-00004250 \
  --output-json outputs/post_resume_analysis/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/step-00004250/proposed_training_adjustments.json \
  --output-md outputs/post_resume_analysis/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/step-00004250/proposed_training_adjustments.md
```

It recommended rollback/lower LR plus tiny GraphCG, GFlowNet, entropy, analogy,
and contrastive pressure.  I rejected an edit at this gate because the active
byte-warmup phase already has tiny GraphCG/GFlowNet/contrastive anchors, heavy
toric/Koszul losses are explicitly contraindicated by weak margins, and the
primary hard-validation gate improved.  I also rejected rollback because it
would discard the only improved hard-validation checkpoint without a reliable
Hessian or validation-floor-bounce signal.

FineWeb performance after limited or mixed exposure should still be treated
conservatively.  This run is currently in FineWeb-only calibration
(`fineweb_mix_ratio=1.0`), so any strong FineWeb result would need the controls
from `BPB-AMP.md` before being called OOD transfer: tokenizer/byte denominator
checks, n-gram and dataset-easiness baselines, FineWeb-only controls, hard-only
zero-shot evaluation, and GFlowNet/GraphCG/toric/topology ablations.

### Decision

Action: `CONTINUE`.

No code or config edit was made.  No stop sentinel was written because this
analysis did not find a better BPB optimization strategy to replace the active
loop.  No rollback was selected because the hard validation gate improved and
the evidence for a floor-bounce basin is insufficient.

The active supervisor was verified keeping training alive past the analyzed
checkpoint, so I preserved the supervisor-owned continuation rather than killing
and replaying the same W&B run.  The training tmux was observed advancing past
step `4400` from the same checkpoint lineage and config.

| item | value |
|---|---|
| supervisor tmux | `toricgt_supervisor_fineweb_bpb_recovery` |
| training tmux | `toricgt_fineweb_bpb_recovery_live` |
| active training resume | `checkpoints/parameter_golf_all_phases_fineweb_from0/random_order_step_00003750.pt` |
| analyzed checkpoint | `checkpoints/parameter_golf_all_phases_fineweb_from0/random_order_step_00004250.pt` |
| current config | `config/train.parameter_golf_all_phases.yaml` |
| W&B run path | `amelie-iska-math/toricgt-parameter-golf/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z` |
| W&B URL | `https://wandb.ai/amelie-iska-math/toricgt-parameter-golf/runs/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z` |
| training log | `logs/parameter_golf_all_phases/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/supervisor/restart_001_20260603T124401Z/train.log` |
| supervisor log | `logs/parameter_golf_all_phases/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/supervisor/supervisor.log` |
| BPB loop state | `logs/parameter_golf_all_phases/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/supervisor/bpb_codex_loop_state.json` |
| next watcher tmux | `toricgt_fineweb_bpb_recovery_watcher` |
| next watcher target | `4500` |
| next watcher log | `logs/parameter_golf_all_phases/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/supervisor/analysis_target_00004500_20260603T140604Z/watcher.log` |

The next watcher is already non-interrupting sidecar analysis on CPU.  Its
process command omits `--pause-training-before-analysis` and preserves the loop
environment:

```text
BPB_TARGET=1.2
BPB_MAX_REVIEW_ITERATIONS=100
BPB_LOOP_STATE=logs/parameter_golf_all_phases/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/supervisor/bpb_codex_loop_state.json
BPB_LOOP_STOP_FILE=logs/parameter_golf_all_phases/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/supervisor/bpb_codex_loop_stop
BPB_LOOP_NAME=all_phases_supervised_watchdog
```

## 2026-06-03 Automated Review: Step 4,000 FineWeb-Revealed BPB Recovery Gate

The watcher analyzed:

```text
outputs/post_resume_analysis/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/step-00004000/
```

Analyzed checkpoint:

```text
checkpoints/parameter_golf_all_phases_fineweb_from0/random_order_step_00004000.pt
```

The BPB loop target remains `<= 1.2`.  The loop primary BPB is
`best_val_bpb=5.382641461921426` at step `4000`, so the target is not reached.

### Metric Categorization

| category | read |
|---|---|
| desired | official-style FineWeb BPB is now available and much lower than the earlier all-phases probes (`oai_competition/bpb=4.5432656878428155`, W&B last `4.54670499689477`, best `4.5335748094319985`); checkpoint finite differences are favorable at the decision point (`train_bpb 4.644115 -> 4.605330`, first difference `-0.038785`, second difference `-0.025250` from steps `3750 -> 4000`); the hard validation gate finally improved (`best_val_bpb 5.852095 -> 5.382641`); GFlowNet entropy/action diversity are not collapsed (`2.7705`, `0.9982`); topology boundary, inclusion, and directed cycle losses remain effectively zero; exact edge validity is high; Toric BGG standard leakage and Gale consistency trend down in the W&B window; Slepian concentration/leakage stay `1.0` / `0.0`; artifact accounting stays under the target |
| desired but too weak/slow | FineWeb BPB and hard validation BPB are both still far above `1.2`; train BPB slope is negative but small over the exported window (`about -0.57/1k` by direct OLS, proposal `-0.31/1k`); score-first hard validation helps only slightly (`5.371769` vs deterministic `5.382641`); simplex BPB is nearly flat (`6.5356--6.5439`) despite improved MST efficiency; geometry branch search has useful isolated branches but weak mean quality (`mean BPB 7.0731`, best `4.9969`, mean-best gap `2.0762`); MST efficiency is coherent but modest (`0.4453`); path smoothness is low (`0.1632` mean); GraphCG is stable but not strongly useful (`basis_coherence 0.0767`, tiny axis variance); Koszul exactness is stable but weakly drifting (`0.0135`, syzygy `0.00675`) |
| undesirable | active training is FineWeb-only in the current phase (`fineweb_mix_ratio=1.0`, hard/medium/complex fractions `0.0`), so the hard-reasoning gate is mostly transfer/audit rather than active supervision until later phases; hard reasoning records remain poor, especially Hebrew (`mean BPB 12.1882`) and frontier reasoning (`5.6433`); branch/test-time scaling does not recover a competitive terminal; toric memory entropy is low (`0.0348` train, geometry task means mostly `0.03--0.09`); tropical active-face margins remain strongly negative (`train -3.1499`, geometry mean `-2.5021`) with very thin toric shadow margins (`min margin mean about 1.41e-4`); exact triangle validity is weak (`0.4680` mean); Hessian trace, dominant curvature, HVP norm, and probe grad norm are `NaN`, so there is no usable Hessian rollback signal |

Core values:

| metric | value | read |
|---|---:|---|
| checkpoint train BPB | `4.605330` | improving at the decision point, but far above target |
| checkpoint train loss | `3.192171` | same finite-difference behavior as train BPB |
| official-style FineWeb BPB | `4.543266` | competition calibration gate available, failed target |
| W&B best FineWeb BPB | `4.533575` | best competition signal in this run |
| hard validation BPB | `5.382641` | loop primary BPB, improved from prior best but failed target |
| hard score-first BPB | `5.371769` | small legal adapter gain only |
| complexity hard-val BPB | `5.183304` | confirms hard validation is still weak |
| geometry mean / best BPB | `7.073144` / `4.996897` | branch replay gap remains large |
| geometry mean / best answer BPB | `7.193858` / `4.875293` | answer-span quality remains weak |
| GFlowNet entropy / diversity / loss | `2.770546` / `0.998181` / `1.521934` | alive, but branch policy is not yet exploiting the branch gap |
| GraphCG basis/coherence | `basis_loss 0.132802`, `coherence 0.076660` | stable but too weak |
| Toric BGG resolution/leakage/Gale/signature | `0.934628` / `0.474522` / `0.113989` / `0.004059` | mostly diagnostic; leakage still high |
| Koszul exactness/syzygy | `0.013506` / `0.006754` | stable diagnostic |
| topology exact edge / triangle validity | `0.973275` / `0.468043` | edge complex good, triangles weak |
| topology directed asymmetry / cycle flux | `0.465994` / `6.99e-18` | noncommutative structure without cycle explosion |
| toric active-face entropy / margin | `0.458797` / `-2.502140` geometry mean | entropy useful, margins inverted |
| toric shadow cells / min margin | `15.833333` / `0.000141` | broad coverage, thin stability margin |
| Slepian concentration / leakage | `1.0` / `0.0` | desired |

### Plot Review

Reviewed representative generated artifacts from the metrics, simplex, geometry,
trajectory, Ramachandran-style phase, energy-landscape, and topology plot sets:

```text
metrics/core_metric_timeseries.png
metrics/recent_metric_slopes.png
simplex/reasoning_k_bpb_triangle.png
simplex/reasoning_k_bpb_mst_tetrahedron.png
geometry/triangles/reasoning_k_bpb.png
geometry/triangles/trajectory_flow.png
geometry/trajectories/*_trajectory_3d.png
geometry/trajectories/*_phase_energy.png
geometry/trajectories/*_energy_landscape.png
geometry/topology/*_directed_filtration.png
geometry/topology/*_exact_persistence_morphisms.png
geometry/topology/*_toric_shadow_audit.png
```

Plot categorization:

| diagnostic | category | evidence |
|---|---|---|
| core metric timeseries | desired but weak | train BPB/loss are noisy with spikes, but the smoothed line drifts down into step 4000 |
| recent slopes | mixed | useful negative slopes are dominated by complexity variance metrics; BPB slopes are favorable but not statistically strong |
| simplex plots | desired but weak | MST efficiency improves with budget, but BPB stays around `6.54`; search geometry is organized but not decisive |
| geometry reasoning simplex | desired but weak | branches cluster by record and budget, but low-BPB regions are not reached by most branches |
| 3D reasoning trajectories | desired but weak | trajectories are nonblank and structured, with lengths around `779--1056`; outlier jumps and dense crossing remain visible |
| Ramachandran-style phase plots | desired but weak | phase occupancy is broad rather than collapsed, but high local NLL points remain scattered through chambers |
| energy landscapes | desired but weak | basins exist, but high-energy islands and long branch excursions indicate weak terminal selection |
| directed filtration/topology | mixed | edge density, triangle density, boundary residuals, and cycle flux are healthy; analogical map and DEC residuals remain nontrivial |
| exact persistence/Koszul audits | mixed | exact edge validity is high, but exact triangle validity is too weak for a strong topology gate |
| toric shadow audits | mixed | occupied Newton fan cells and Slepian concentration are good; tropical margins are too thin for toric losses to own gradients |

### Mathematical Reading

Let `B_fw` be official-style FineWeb BPB, `B_hard` be hard ToricGT validation
BPB, and `B_train` be the current training BPB.  At step 4000:

```text
B_fw ~= 4.54, B_hard ~= 5.38, B_train ~= 4.61.
```

Both gates fail the `1.2` target, but they do not disagree in direction:
checkpoint metadata shows `B_hard` improved from the previous best, and the
last two checkpoint finite differences for `B_train` are negative.  This is not
a floor-bounce rollback case:

```text
Delta B_train(3500->3750) = -0.013536
Delta B_train(3750->4000) = -0.038785
Delta^2 B_train = -0.025250
```

Hessian probes cannot override that because the curvature estimates are `NaN`.
The statistical read is therefore slow descent with high variance, not a
positive-curvature basin.

The strong FineWeb improvement relative to earlier probes should not be called
OOD transfer yet.  The current phase is FineWeb calibration only, so a lower
FineWeb BPB is expected direct exposure.  A later transfer claim still needs the
controls from `BPB-AMP.md`: tokenizer/byte accounting checks, n-gram and
dataset-easiness baselines, FineWeb-only controls, hard-data-only zero-shot
FineWeb evaluation, and GFlowNet/GraphCG/toric/topology ablations.

The proposal script was run and used as evidence, not authority:

```text
scripts/propose_training_adjustments.py --analysis-dir .../step-00004000
```

It recommended holding or slightly increasing LR and adding tiny GraphCG,
GFlowNet, analogy, entropy, and contrastive weights because branch replay has a
large mean-best gap.  I rejected an edit for this gate: the BPB gate is still
the first objective, recent BPB/checkpoint differences are improving, toric and
Koszul margins are explicitly not ready for heavier losses, and interrupting a
live supervisor-owned run for tiny auxiliary changes would add restart risk
without a clear two-gate gain.

### Decision

Action: `CONTINUE`.

No code or config edit was made.  No stop sentinel was written because this is
not a better-strategy replacement for the BPB loop.  No rollback was selected
because step 4000 has negative first and second BPB checkpoint differences and a
new hard-validation best.

The active supervisor had already resumed/kept the same run alive and the
training tmux was verified advancing past step 4000, so I preserved that
supervisor-owned continuation rather than killing and replaying from the same
checkpoint.

| item | value |
|---|---|
| supervisor tmux | `toricgt_supervisor_fineweb_bpb_recovery` |
| training tmux | `toricgt_fineweb_bpb_recovery_live` |
| active training resume | `checkpoints/parameter_golf_all_phases_fineweb_from0/random_order_step_00003750.pt` |
| analyzed checkpoint | `checkpoints/parameter_golf_all_phases_fineweb_from0/random_order_step_00004000.pt` |
| current config | `config/train.parameter_golf_all_phases.yaml` |
| W&B run path | `amelie-iska-math/toricgt-parameter-golf/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z` |
| W&B URL | `https://wandb.ai/amelie-iska-math/toricgt-parameter-golf/runs/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z` |
| training log | `logs/parameter_golf_all_phases/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/supervisor/restart_001_20260603T124401Z/train.log` |
| supervisor log | `logs/parameter_golf_all_phases/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/supervisor/supervisor.log` |
| BPB loop state | `logs/parameter_golf_all_phases/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/supervisor/bpb_codex_loop_state.json` |
| next watcher tmux | `toricgt_fineweb_bpb_recovery_watcher` |
| next watcher target | `4250` |
| next watcher log | `logs/parameter_golf_all_phases/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/supervisor/analysis_target_00004250_20260603T133403Z/watcher.log` |

The next watcher is already non-interrupting sidecar analysis on CPU.  Its
process command omits `--pause-training-before-analysis` and preserves the loop
environment:

```text
BPB_TARGET=1.2
BPB_MAX_REVIEW_ITERATIONS=100
BPB_LOOP_STATE=logs/parameter_golf_all_phases/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/supervisor/bpb_codex_loop_state.json
BPB_LOOP_STOP_FILE=logs/parameter_golf_all_phases/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/supervisor/bpb_codex_loop_stop
BPB_LOOP_NAME=all_phases_supervised_watchdog
```

## 2026-06-03 Automated Review: Step 1,250 All-Phases All-Metrics Gate

The watcher analyzed:

```text
outputs/post_resume_analysis/toricgt-all-phases-all-metrics-20260603T012949Z/step-00001250/
```

Analyzed checkpoint:

```text
checkpoints/parameter_golf_all_phases/random_order_step_00001250.pt
```

The previous training process had been paused/interrupted by the watcher; a new
training tmux was required for any continuation.

### Metric Categorization

| category | read |
|---|---|
| desired | train BPB/loss improved over the exported W&B window (`train/bpb` median `3.9517 -> 3.5387`, recent slope `-5.834/1k`, `t=-5.53`); GFlowNet loss decreased (`10.8079 -> 8.5740`); toric active-face entropy improved (`0.4332 -> 0.6129`); Toric BGG standard leakage improved (`0.4314 -> 0.4029`); Gale-dual consistency drifted downward; topology boundary, inclusion, and directed cycle losses remained zero; Koszul exactness and syzygy residuals stayed small; GraphCG basis coherence stayed stable |
| desired but too weak/slow | official-style FineWeb BPB was available but far from the target and worsened from the earlier W&B probe (`oai_competition/bpb 8.9451` near step `1002` to `9.4288` at step `1250`); hard validation BPB was `9.7109`, score-first validation BPB was only slightly better at `9.4679`; branch/test-time scaling found useful isolated branches but not competitive means (`geometry mean BPB 8.2307`, best BPB `4.1894`, best answer BPB `3.6176`); simplex budget scaling improved BPB only from `4.9034` to `4.8804`; MST efficiency was coherent but modest (`0.3355` geometry mean, `0.4743` best simplex budget); GraphCG basis loss edged down only `0.33%`; topology persistence declined slowly; toric active-face margins remained negative |
| undesirable | the two BPB gates disagreed with training loss: train BPB improved while FineWeb and hard validation stayed high or worsened; GFlowNet entropy and action diversity collapsed (`entropy 0.0824 -> 0.0529`, diversity `0.0298 -> 0.0191`); toric memory entropy fell (`0.0697 -> 0.0389`); toric binomial and geometry losses rose; GraphCG covariance rose; validation argmax byte accuracy was `0.0`; Hessian trace, dominant curvature, and HVP norm were `NaN`; the checkpoint had only coarse saved states (`500`, `1000`, `1250`), so no dense floor-capture point existed between 1000 and 1250 |

Core values:

| metric | value | read |
|---|---:|---|
| checkpoint train BPB | `3.539561` | lower than step 1000, but not promotion-ready |
| checkpoint best hard-val BPB | `8.238147` | primary loop BPB, far above target `1.2` |
| hard validation BPB at gate | `9.710925` | weak generalization |
| score-first hard validation BPB | `9.467908` | legal adapter helps slightly but not enough |
| official-style FineWeb BPB | `9.428791` | competition gate failed |
| prior FineWeb probe | `8.945133` near step `1002` | selected rollback evidence |
| train BPB minimum in W&B window | `3.497505` at step `1220` | local train descent exists |
| last train BPB first diff | `+0.009757` over steps `1245 -> 1250` | small local rebound |
| last train BPB second diff | `+0.021323` | weak floor-bounce warning, not dense enough for a cliff rollback |
| GFlowNet entropy/diversity | `0.057767` / `0.020857` | collapsed relative to target `2.0` |
| GraphCG basis/coherence | `basis_loss 0.2732`, `coherence 0.000484` | stable but too weak |
| Toric BGG resolution/leakage/Gale | `0.88998` / `0.40073` / `0.29057` | diagnostic-only; leakage still high |
| Koszul exactness/syzygy | `0.012716` / `0.006359` | stable diagnostic |
| topology directed asymmetry/cycle flux | `0.48539` / `5.6e-18` geometry mean | noncommutative structure without cycle explosion |
| toric active-face entropy/margin | `0.61317` / `-1.82064` geometry mean | entropy useful; margins inverted |
| toric shadow cells/min margin | `15.96` / `0.000148` | coverage visible; margin too thin |
| Slepian concentration/leakage | `1.0` / `0.0` | desired |

### Plot Review

Reviewed representative generated artifacts from the 93 PNG/HTML plot files:

```text
metrics/core_metric_timeseries.png
metrics/recent_metric_slopes.png
metrics/selected_metric_correlations.png
simplex/reasoning_k_bpb_triangle.png
simplex/reasoning_k_bpb_mst_tetrahedron.png
geometry/triangles/reasoning_k_bpb.png
geometry/triangles/trajectory_flow.png
geometry/tetrahedra/toric_gfn_bpb.png
geometry/trajectories/*_trajectory_3d.png
geometry/trajectories/*_phase_energy.png
geometry/trajectories/*_energy_landscape.png
geometry/topology/*_directed_filtration.png
geometry/topology/*_exact_persistence_morphisms.png
geometry/topology/*_commutative_algebra_audit.png
geometry/topology/*_toric_shadow_audit.png
geometry/topology/*_toric_slepian_audit.png
```

Plot categorization:

| diagnostic | category | evidence |
|---|---|---|
| core metric timeseries | mixed | train BPB descends overall, but the competition probe worsens and the last train finite differences turn positive |
| recent slopes | mixed | BPB/loss slopes are favorable on train; entropy/diversity and memory slopes are unfavorable |
| simplex triangles/tetrahedra | desired but weak | larger search budgets improve MST/reasoning placement and BPB slightly, but the best simplex BPB remains about `4.88` |
| 3D reasoning trajectories | desired but weak | trajectories are nonblank and organized, but path lengths remain long (`~1270--1825`) and not aligned with low BPB on GoT/CoT records |
| Ramachandran-style phase-energy plots | desired but weak | phase structure is visible without a single catastrophic phase, but toric active-face margins remain negative |
| energy/fitness landscapes | desired but weak | basins exist, yet high-BPB branches dominate frontier, GoT, and CoT records |
| topology audits | mixed | exact edge validity is high and boundary/inclusion residuals are zero; exact triangle validity and HDBSCAN clustering remain too weak |
| toric shadow/Slepian audits | mixed | Slepian concentration/leakage are clean; toric shadow margins are too thin for toric loss activation |

### Mathematical Reading

Let \(B_{\rm train}\), \(B_{\rm hard}\), and \(B_{\rm fw}\) denote the train,
hard-reasoning validation, and FineWeb BPB estimates.  The observed update
reduces \(B_{\rm train}\), but \(B_{\rm fw}\) rises from about `8.945` to
`9.429`, and \(B_{\rm hard}\) stays near `9.71`.  That means the descent
direction is not aligned with the competition gate:

```text
Delta B_train < 0, but Delta B_fw > 0 and B_hard >> target.
```

This is a two-gate failure, not a promotion case.  The hard-data-only warmup is
learning local byte likelihood on the curated stream while failing to calibrate
the official FineWeb byte distribution.  The BPB-AMP two-gate rule therefore
favors a mixture edit, not deleting the reasoning objective and not a plain
continue.

The adjustment proposal was used as evidence rather than authority.  It
recommended tiny GFlowNet/GraphCG/contrastive activations because branch search
has a large mean-best gap (`4.0413` mean-minus-best geometry BPB).  I did not
activate those losses yet because the BPB gate is failing first and GFlowNet
entropy/diversity are diagnostic collapses, not an active-gradient emergency.
The smaller high-impact move is to add FineWeb calibration exposure while
keeping ToricGT hard-reasoning data dominant.

No strong FineWeb transfer claim is available here.  If a later gate shows a
sharp FineWeb improvement after limited FineWeb exposure, it must be controlled
against tokenizer convention, n-gram/Dirichlet baselines, FineWeb-only training,
hard-data-only zero-shot evaluation, and dataset easiness.

### Decision

Action: `EDIT_AND_RESTART`.

Minimal edits:

```text
scripts/train_parameter_golf_random_order.py
config/train.parameter_golf_all_phases.yaml
tests/test_toric_bgg.py
```

The training loop now has an optional `fineweb_calibration` stream that reuses
the existing decoded SentencePiece FineWeb byte loader.  It is gated by
`fineweb_mix_ratio`, defaults off unless configured, logs actual FineWeb
microbatch fractions, and is included in phase controls.  The all-phases config
uses `fineweb_mix_ratio: 0.35` during the current warmup, then reduces to
`0.30`, `0.25`, and `0.20` in later phases so ToricGT reasoning remains the
dominant stream.  The checkpoint directory was changed to:

```text
checkpoints/parameter_golf_all_phases_fineweb35_from1000
```

so replayed step numbers do not overwrite the analyzed checkpoint.

Selected resume checkpoint:

```text
checkpoints/parameter_golf_all_phases/random_order_step_00001000.pt
```

Reason: step 1000 is the last saved checkpoint before the official-style
FineWeb probe worsened; step 1250 improves train BPB but fails both validation
gates.  No stop sentinel was written because this is an in-loop two-gate
calibration edit, not a replacement for the review loop.

Validation before launch:

```text
python -m py_compile scripts/train_parameter_golf_random_order.py
PYTHONPATH=src pytest -q tests/test_toric_bgg.py
FineWeb loader smoke: [2, 1024] byte-token batches, token IDs 36..244
```

Fresh training and watcher sessions:

| item | value |
|---|---|
| training tmux | `toricgt_all_phases_fineweb35_from1000_20260603T021204Z` |
| watcher tmux | `toricgt_all_phases_fineweb35_from1000_20260603T021204Z_watcher` |
| W&B run path | `amelie-iska-math/toricgt-parameter-golf/toricgt-all-phases-fineweb35-from1000-20260603T021204Z` |
| training log | `logs/training/toricgt-all-phases-fineweb35-from1000-20260603T021204Z.train.log` |
| watcher log | `logs/training/toricgt-all-phases-fineweb35-from1000-20260603T021204Z.watcher.log` |
| config | `config/train.parameter_golf_all_phases.yaml` |
| analysis root | `outputs/post_resume_analysis/toricgt-all-phases-fineweb35-from1000-20260603T021204Z` |
| fresh min mtime | `1780452724` |
| next target step | `1500` |

The watcher was launched with:

```text
scripts/watch_training_analysis.py
  --target-step 1500
  --pause-training-before-analysis
  --device cuda
  --precision bf16
  --codex-review-hook scripts/codex_training_review_resume.sh
  --codex-review-tmux-prefix toricgt_codex_review
```

Live verification:

```text
optimizer_state_loaded: true
trainer status: stepping on RTX 4090
watcher status: waiting for a fresh checkpoint >= 1500
```

## 2026-06-02 Automated Review: Step 2,250 Revealed-Context BPB-Cliff Gate

The watcher analyzed:

```text
outputs/post_resume_analysis/oai-bpb-revealed-context-02025-20260602T190948Z/step-00002250/
```

Analyzed checkpoint:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00002250.pt
```

The watcher paused/interrupted the previous training process after producing the
step-2250 analysis.  A fresh resume was required; the previous tmux is not an
active run.

### Metric Categorization

| category | read |
|---|---|
| desired | best validation BPB in the current run occurred at step 2050 (`val/bpb 5.015896`, `score_first_bpb 5.011459`), and the whole 1500--2200 validation curve still improved from the original `5.27` range; geometry mean BPB improved slightly relative to step 2025 (`4.8340 -> 4.8209`), best branch BPB improved (`4.0772 -> 4.0551`), and best answer BPB improved (`3.6093 -> 3.6027`); MST efficiency remained coherent (`0.7096`); topology inclusion, boundary, and variety-complex residuals stayed zero; exact edge validity remained high (`0.9749`, directed `0.9301`); Toric BGG resolution consistency improved to `0.9431`; standard leakage improved to `0.5110`; toric active-face entropy stayed useful (`0.8775` geometry); toric Slepian concentration/leakage stayed `1.0` / `0.0`; artifact accounting remained below the cap with training-only probes excluded |
| desired but too weak/slow | validation improved over the long window but no longer improved at the gate (`controller/val_improved=0`, `controller/recovery_active=1`); hard-reasoning geometry has useful isolated branches but weak mean quality (`mean BPB 4.8209`, `frontier_gptoss_reasoning` mean `5.3377`); GraphCG basis coherence is stable (`0.0173`) but axis variance is tiny; GFlowNet entropy/diversity are alive but low (`2.0569`, `0.7454`) while GFlowNet loss is high (`7.7387`) and weight is zero; exact triangle validity (`0.6771`) and HDBSCAN stability (`0.8471`) weakened relative to the prior analysis; toric shadow has good cell coverage (`13.08`) but a very thin minimum margin (`0.000165`); active-face margins remain negative (`geometry -0.8292`, train about `-0.8860`); Kolmogorov/NCD proxies are mostly flat, with validation argmax byte accuracy only about `0.199--0.200` |
| undesirable | checkpoint train BPB entered a floor-bounce basin: checkpoint sequence `2050: 4.0779`, `2075: 3.6885`, `2100: 3.8878`, `2125: 4.2687`, `2200: 4.4822`, `2225: 4.6892`, `2250: 4.6478`; validation also degraded after the 2050 best (`2100: 5.0165`, `2150: 5.0175`, `2200: 5.0179`); recent train BPB slope was strongly positive (`+3.321/1k`, `t=4.44`); shock metrics detected high-loss impulses (`loss_ratio` around `1.04`, `loss_delta` around `0.12` at the gate), but `shock_guard_active=0` and robust microbatch capping was zero because the global guard window still ended at step 1800; GFlowNet loss rose by about `130%` in the report, entropy/diversity dropped by about `26%`; Toric BGG Gale consistency worsened to `0.2296`; official FineWeb BPB and Hessian/sharpness probes were not available |

Core values:

| metric | value | read |
|---|---:|---|
| checkpoint train BPB | `4.647771` | too high for promotion |
| W&B last train BPB | `4.648819` | same floor-bounce state |
| W&B train BPB minimum in replay | `3.770102` at step `2060` | confirms a low basin existed before the rebound |
| best checkpoint before positive checkpoint derivative | `random_order_step_00002075.pt` | selected rollback point |
| validation BPB at step 2050 | `5.015896` | best validation gate in this replay |
| validation BPB at step 2200 | `5.017904` | degraded from best |
| score-first validation BPB at step 2200 | `5.013939` | legal adapter remains slightly better than deterministic |
| controller recovery active | `1` | controller also marked the run as in recovery |
| controller train BPB drift | `0.138340` | not acceptable for continue |
| train loss / loss EMA at gate | `3.222316` / `3.104914` | loss shock and EMA drift |
| GFlowNet entropy / diversity / loss | `2.056935` / `0.745366` / `7.738666` | degraded diagnostic, not an active gradient source |
| GFlowNet loss weight | `0.0` | correct for BPB cliff replay |
| GraphCG coherence / axis variance | `0.017334` / `1.46e-4` | stable but weak |
| Koszul exactness residual | `0.016170` | stable diagnostic |
| Toric BGG resolution consistency | `0.943121` | desired diagnostic |
| Toric BGG standard leakage | `0.510998` | improved but too high for loss activation |
| Toric BGG Gale consistency | `0.229594` | not ready |
| Toric BGG loss weight | `0.0` | correct for cliff recovery |
| geometry mean / best BPB | `4.820879` / `4.055124` | hard-reasoning gate weak but not collapsed |
| geometry mean / best answer BPB | `4.694436` / `3.602702` | isolated strong branches remain |
| geometry MST efficiency / smoothness | `0.709618` / `0.060457` | coherent but not a BPB driver |
| exact edge / directed edge validity | `0.974863` / `0.930075` | desired |
| exact triangle validity | `0.677108` | too weak for topology loss activation |
| HDBSCAN stability / noise | `0.847059` / `0.152941` | usable but weaker than prior gate |
| toric active-face entropy / margin | `0.877461` / `-0.829183` | entropy desired; margin not stable |
| toric shadow occupied cells / min margin | `13.083333` / `0.000165` | coverage visible; margin too thin |
| toric Slepian concentration / leakage | `1.0` / `0.0` | desired phase concentration |

Finite differences over fresh checkpoint files:

```text
step 2050: train_bpb 4.077936
step 2075: train_bpb 3.688517  first diff -0.389419
step 2100: train_bpb 3.887838  first diff +0.199321, second diff +0.588740
step 2125: train_bpb 4.268689  first diff +0.380850
step 2150: train_bpb 4.021380  first diff -0.247308
step 2175: train_bpb 3.975372  first diff -0.046008
step 2200: train_bpb 4.482183  first diff +0.506811
step 2225: train_bpb 4.689205  first diff +0.207022
step 2250: train_bpb 4.647771  first diff -0.041434
```

The checkpoint-level first positive derivative appears immediately after the
step-2075 low, with a large positive second difference.  That is the statistical
rollback criterion.  The absence of an official FineWeb probe means this gate
cannot claim competition-transfer improvement; the fixed validation stream is
still the BPB gate available here.

### Plot Review

Reviewed representative generated plots:

```text
metrics/core_metric_timeseries.png
metrics/recent_metric_slopes.png
simplex/reasoning_k_bpb_triangle.png
simplex/reasoning_k_bpb_mst_tetrahedron.png
geometry/tetrahedra/reasoning_k_bpb_mst.png
geometry/tetrahedra/energy_control.png
geometry/trajectories/*_trajectory_3d.png
geometry/trajectories/*_phase_energy.png
geometry/trajectories/*_energy_landscape.png
geometry/topology/*_directed_filtration.png
geometry/topology/*_exact_persistence_morphisms.png
geometry/topology/*_commutative_algebra_audit.png
geometry/topology/*_toric_shadow_audit.png
geometry/topology/*_toric_slepian_audit.png
```

Plot categorization:

| diagnostic | category | evidence |
|---|---|---|
| core metric timeseries | undesirable for continuation | train BPB/loss show two valley-and-rebound arcs; the second rebound accelerates after step 2200 while validation flattens and then worsens |
| simplex BPB/reasoning plots | desired but weak | budget scaling improves MST/reasoning placement but does not create a clear low-BPB branch family |
| reasoning/BPB/MST tetrahedron | desired but weak | Hebrew branches still occupy the low-BPB region; GoT/CoT/frontier branches remain far from the low-BPB vertex |
| 3D trajectories and energy landscapes | desired but weak | coherent basins exist, but long low-density jumps and high-energy branches remain visible |
| Ramachandran-style phase plots | desired | phase structure is organized and not dominated by one pathological phase defect |
| directed filtration/topology | mixed | edge validity and zero boundary residuals are good; exact triangle validity and HDBSCAN stability slipped |
| toric shadow audit | desired but weak | active Newton-face coverage and bend structure are visible, but margins are near zero and bends spike, so toric geometry should remain audit-only |
| Toric BGG/Koszul audits | mixed | resolution consistency and `d^2` style residuals are stable; standard leakage and Gale consistency are not ready for training loss |

### Mathematical Reading

The run walked through a narrow byte-likelihood basin.  In the available
checkpoint series, the low at step 2075 is followed by a positive first
difference and a large positive second difference at 2100.  Validation confirms
the same direction: after the best validation point at step 2050, subsequent
validation probes worsen.  This is not a case where hard reasoning improves
while the competition BPB gate is merely noisy; both gates fail to promote the
step-2250 checkpoint.

The mechanism is consistent with the BPB cliff plan.  The loss shock detector
observed high-loss impulses, but the global shock and robust microbatch guard
windows had expired at 1800, so the gradients were not damped in the documented
2000--2500 floor-capture window.  The fix is therefore scalar training control,
not an architecture change.

Reasoning diagnostics still matter.  The geometry suite shows that toric,
topological, Koszul, BGG, GFlowNet, GraphCG, and Kolmogorov diagnostics are
informative, but none is strong enough to override the BPB rollback.  The BGG
and Koszul probes remain training-only diagnostics with zero loss, exactly as
the late-phase toggle policy requires.

### Decision

Action: `EDIT_AND_RESTART`.

Minimal config edits:

```text
config/train.parameter_golf_random_order_dense_valmix35_from1000.yaml
```

The global guards now match the documented 2000--2500 cliff policy:

```text
shock_guard_start_step: 2000
shock_guard_end_step: 2500
shock_guard_loss_ratio: 1.018
shock_guard_loss_delta: 0.045
shock_guard_grad_norm: 0.18
shock_guard_update_scale: 0.006
robust_micro_loss_guard_start_step: 2000
robust_micro_loss_guard_end_step: 2500
robust_micro_loss_guard_ratio: 1.018
robust_micro_loss_guard_delta: 0.020
robust_micro_loss_guard_min_scale: 0.08
```

Resume from:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00002075.pt
```

Optimizer state is preserved; `--reset-optimizer` was not used.  Existing
step-2100 through step-2250 checkpoint files were copied to:

```text
checkpoints/parameter_golf_oai_dense/pre_step2250_rollback_20260602T2250_review/
```

so the step-numbered replay does not silently discard the analyzed states.
These checkpoint artifacts are not committed or pushed.

Fresh training and watcher sessions:

| item | value |
|---|---|
| training tmux | `toricgt_oai_bpb_revealed_02075_20260602T194909Z` |
| watcher tmux | `toricgt_watch_bpb_revealed_02075_20260602T194909Z` |
| W&B run path | `amelie-iska-math/toricgt-parameter-golf/oai01500revealed20260602T174238Z` |
| run name | `oai-bpb-revealed-context-02075-20260602T194909Z` |
| training log | `logs/training/oai-bpb-revealed-context-02075-20260602T194909Z.log` |
| watcher log | `logs/training/oai-bpb-revealed-context-02075-20260602T194909Z.watcher.log` |
| analysis root | `outputs/post_resume_analysis/oai-bpb-revealed-context-02075-20260602T194909Z` |
| fresh min mtime | `1780429749` |
| next target step | `2325` |

The watcher was launched with:

```text
scripts/watch_training_analysis.py
  --target-step 2325
  --pause-training-before-analysis
  --device cuda
  --precision bf16
  --codex-review-hook scripts/codex_training_review_resume.sh
  --codex-review-tmux-prefix toricgt_codex_review
```

Live verification:

```text
optimizer_state_loaded: true
trainer status: stepping on RTX 4090
watcher status: waiting for a fresh checkpoint >= 2325
```

## 2026-06-02 Automated Review: Step 2,025 Revealed-Context BPB-Cliff Gate

The watcher analyzed:

```text
outputs/post_resume_analysis/oai-bpb-revealed-context-01525-20260602T175354Z/step-00002025/
```

Analyzed checkpoint:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00002025.pt
```

The watcher paused/interrupted the previous training process after producing the
step-2025 checkpoint.  A fresh resume was required; the previous tmux is not an
active run.

### Metric Categorization

| category | read |
|---|---|
| desired | deterministic validation BPB improved monotonically in the watcher window (`5.2723 -> 5.0162`) with five consecutive controller improvements; score-first BPB was slightly better than deterministic (`5.0118` vs `5.0162`); complexity validation BPB improved to `4.8313`; topology inclusion and boundary residuals were zero; exact edge validity stayed high (`0.9745`, directed `0.9316`); HDBSCAN stability was high (`0.8574`); Toric BGG resolution consistency was useful as a diagnostic (`0.9316`); standard leakage improved (`0.6100 -> 0.4750`); toric active-face entropy increased (`0.8924`); toric memory entropy improved (`0.3201`); Slepian concentration/leakage remained `1.0` / `0.0`; artifact accounting stayed within the cap with training-only probes excluded |
| desired but too weak/slow | train BPB was noisy and locally higher (`4.2693` checkpoint, `4.2868` W&B last row), but the controller EMA drift was still negative (`-0.0505`) and the LR had already dropped to `2.38e-6`; hard-reasoning geometry had usable best branches (`best BPB 4.0772`, best answer BPB `3.6093`) but weak mean branch quality (`4.8340` mean BPB); MST efficiency (`0.7082`) and path smoothness (`0.0590`) were coherent but not yet likelihood-aligned; GraphCG basis coherence rose to `0.0172`; Koszul exactness stayed stable (`0.01675`) but higher-order exact triangle validity remained only `0.6843`; simplex budget scaling improved reasoning/K-helper axes but did not monotonically improve BPB |
| undesirable | no official FineWeb BPB or Hessian/sharpness probe was available at this gate; checkpoint first differences near the analyzed point were positive (`1975->2000: +0.0129`, `2000->2025: +0.0926`) with positive second difference (`+0.0796`); GFlowNet entropy and diversity declined after the controller zeroed the GFlowNet loss (`entropy 2.0046`, diversity `0.7238`, loss `6.4904`, weight `0`); Toric BGG Gale consistency worsened to `0.2327`; standard leakage remains too high for supervision; tropical active-face margins remain negative (`train -0.7280`, geometry mean `-0.8353`); toric phase-leaf/binomial residuals are not ready to drive optimization |

Core values:

| metric | value | read |
|---|---:|---|
| checkpoint train BPB | `4.269286` | noisy local train value |
| W&B last train BPB | `4.286762` | categorized as not desired because the train median rose |
| W&B train BPB minimum in window | `3.767812` | confirms high batch noise in the cliff window |
| deterministic validation BPB | `5.016239` | primary BPB gate; best in the current watcher window |
| score-first validation BPB | `5.011840` | legal score-first adapter is not hurting the gate |
| complexity validation BPB | `4.831316` | auxiliary validation stream improved |
| controller consecutive validation improvements | `5` | supports continuation rather than rollback |
| controller train BPB EMA / drift | `4.294570` / `-0.050506` | weak descent despite noisy instantaneous train BPB |
| train loss / loss EMA | `2.971357` / `2.974124` | same train-noise pattern as BPB |
| GFlowNet entropy / diversity / loss | `2.004571` / `0.723836` / `6.490376` | diagnostic degraded after weight zeroing |
| GFlowNet loss weight | `0.0` | degradation is not an active gradient source |
| GraphCG coherence / axis variance | `0.017212` / `5.03e-05` | frame is alive but drifting |
| Koszul exactness residual | `0.016751` | stable diagnostic |
| Toric BGG resolution consistency | `0.931583` | desired diagnostic |
| Toric BGG standard leakage | `0.475046` | improving but too high for loss activation |
| Toric BGG Gale consistency | `0.232722` | not ready |
| Toric BGG loss weight | `0.0` | correct for cliff recovery |
| geometry mean / best BPB | `4.834022` / `4.077204` | hard-reasoning gate weak but not collapsed |
| geometry mean / best answer BPB | `4.700738` / `3.609255` | good isolated branches, weak mean |
| geometry MST efficiency / smoothness | `0.708169` / `0.059001` | coherent but not yet a BPB driver |
| topology directed asymmetry / cycle flux | `0.393213` / `4.81e-18` | noncommutative direction without cycle explosion |
| topology HDBSCAN stability / noise | `0.857368` / `0.142632` | desired |
| exact edge / directed edge validity | `0.974521` / `0.931560` | desired |
| exact triangle validity | `0.684285` | too weak for topology loss activation |
| toric active-face entropy / margin | `0.892363` / `-0.728027` | entropy desired; margin still inverted |
| toric shadow occupied cells / min margin | `12.875` / `0.000251` | fan coverage visible; margins thin |
| toric Slepian concentration / leakage | `1.0` / `0.0` | desired phase concentration |

Finite differences:

```text
step 1975: train_bpb 4.163801
step 2000: train_bpb 4.176723  first diff +0.012922
step 2025: train_bpb 4.269286  first diff +0.092563, second diff +0.079640
```

Those local train differences are a warning, but they are not sufficient
rollback evidence because validation has a stronger, statistically clearer
negative slope:

```text
val/bpb recent slope: -0.220650 per 1k steps, t = -3.2709
controller/val_bpb recent slope: -0.144146 per 1k steps, t = -5.2827
```

No Hessian trace, Hessian sharpness, or official FineWeb BPB probe was present
in this export.  Under the two-gate rule, I treat the strong validation trend as
the checkpoint-selection gate and the hard-reasoning metrics as diagnostics to
preserve for later recovery.

### Plot Review

Reviewed representative generated plots:

```text
metrics/core_metric_timeseries.png
metrics/recent_metric_slopes.png
simplex/reasoning_k_bpb_triangle.png
simplex/reasoning_k_bpb_mst_tetrahedron.png
geometry/tetrahedra/reasoning_k_bpb_mst.png
geometry/tetrahedra/energy_control.png
geometry/trajectories/*_trajectory_3d.png
geometry/trajectories/*_phase_energy.png
geometry/trajectories/*_energy_landscape.png
geometry/topology/*_directed_filtration.png
geometry/topology/*_exact_persistence_morphisms.png
geometry/topology/*_commutative_algebra_audit.png
geometry/topology/*_toric_shadow_audit.png
geometry/topology/*_toric_slepian_audit.png
```

Plot categorization:

| diagnostic | category | evidence |
|---|---|---|
| core metric timeseries | mixed | validation/score-first curves descend cleanly; train BPB is noisy; GFlowNet entropy/diversity collapsed after its loss was quarantined |
| simplex BPB/reasoning triangle | desired but weak | budget 1 gives the lowest sampled BPB (`4.3846`); higher budgets improve reasoning/K-helper placement but move away from the low-BPB vertex |
| reasoning/BPB/MST tetrahedron | desired but weak | Hebrew branches form the low-BPB cluster; GoT/CoT/frontier reasoning branches remain farther from the low-BPB vertex |
| 3D trajectories | desired but weak | terminal basins are coherent, with several long exploratory jumps from branch starts; search is structured but not yet consistently likelihood-improving |
| Ramachandran-style phase plots | desired | phase mass is organized near wrap boundaries and axes without one dominant high-energy phase defect |
| energy landscapes | desired but weak | arcs/basins are visible, but long low-density excursions remain and branch quality is not yet uniformly low BPB |
| directed filtration/topology | desired | nested inclusion and oriented boundary residuals are effectively zero; edge validity is high; cycle flux is essentially zero |
| toric shadow audit | desired but weak | occupied active cells and bends are visible, but active-face margins are very small and branch fan entropy is not yet a robust certificate |
| Toric BGG/Koszul audits | mixed | \(d^2\)-style consistency and exactness diagnostics are stable; standard leakage, Gale consistency, and higher simplex validity are not ready for optimization |

### Mathematical Reading

This gate is not a floor-bounce rollback.  The train checkpoint sequence has a
local positive first and second difference, but the validation sequence is
descending with five consecutive improvements after the controller lowered the
learning rate.  Statistically, the train trace is high-variance because the
curated hard-reasoning stream mixes Hebrew, GoT/ToT/CoT, and frontier-reasoning
records; the validation slope is the cleaner gate signal in this window.

The reasoning gate says to preserve the current architecture and weights but not
to reactivate auxiliary gradients.  GFlowNet entropy/diversity decayed because
the BPB-cliff schedule intentionally set the GFlowNet loss weight to zero.  That
is undesirable for later graph-of-thought search, but it is not the source of
the current BPB gradient.  Toric BGG resolution consistency is already useful,
yet the high standard leakage and worsened Gale-dual consistency mean BGG must
remain diagnostic-only.  The same applies to topology/Koszul and toric geometry:
finite complexes are internally consistent, but active-face margins and
triangle validity are too weak for late-phase losses.

The two-gate interpretation is therefore:

```text
BPB gate:      continue, because deterministic validation and score-first BPB improve.
Reasoning gate: preserve diagnostics, but keep GFlowNet/GraphCG/topology/toric/BGG
                losses quarantined through the cliff window.
```

If a strong FineWeb result appears later after limited FineWeb exposure, it
should be treated as possible OOD transfer from hard reasoning data only after
controlling against tokenizer convention, n-gram/Dirichlet priors, FineWeb-only
baselines, and dataset easiness.  No official FineWeb probe was available here,
so this gate cannot make that claim.

### Decision

Action: `CONTINUE`.

Resume from:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00002025.pt
```

No code/config edits were made.  Optimizer state is preserved; `--reset-optimizer`
was not used.

Fresh training and watcher sessions:

| item | value |
|---|---|
| training tmux | `toricgt_oai_bpb_revealed_02025_20260602T190948Z` |
| watcher tmux | `toricgt_watch_bpb_revealed_02025_20260602T190948Z` |
| W&B run path | `amelie-iska-math/toricgt-parameter-golf/oai01500revealed20260602T174238Z` |
| run name | `oai-bpb-revealed-context-02025-20260602T190948Z` |
| training log | `logs/training/oai-bpb-revealed-context-02025-20260602T190948Z.log` |
| watcher log | `logs/training/oai-bpb-revealed-context-02025-20260602T190948Z.watcher.log` |
| analysis root | `outputs/post_resume_analysis/oai-bpb-revealed-context-02025-20260602T190948Z` |
| fresh min mtime | `1780427388` |
| next target step | `2250` |

The watcher was launched with:

```text
scripts/watch_training_analysis.py
  --target-step 2250
  --pause-training-before-analysis
  --device cuda
  --precision bf16
  --codex-review-hook scripts/codex_training_review_resume.sh
  --codex-review-tmux-prefix toricgt_codex_review
```

Live verification:

```text
optimizer_state_loaded: true
trainer status: stepping on RTX 4090
watcher status: waiting for a fresh checkpoint >= 2250
```

## 2026-06-02 Automated Review: Step 1,525 Revealed-Context Recovery Gate

The watcher analyzed:

```text
outputs/post_resume_analysis/oai-bpb-revealed-context-01500-20260602T174238Z/step-00001525/
```

Analyzed checkpoint:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00001525.pt
```

The watcher paused the training process before analysis.  A Codex review hook
was spawned in:

```text
toricgt_codex_review_bpb_cliff_00001525
```

### Metric Categorization

| category | read |
|---|---|
| as desired | training restarted cleanly from the step-1500 checkpoint with Adam reset, then produced a fresh step-1525 checkpoint; GFlowNet entropy stayed high (`2.76`) with action diversity near one; GraphCG basis coherence stayed very low; exact topology edge validity was high (`0.973` mean, `0.935` directed); inclusion and variety-of-complexes residuals were zero; HDBSCAN stability was high (`0.884`); Toric BGG resolution consistency was useful as a diagnostic (`0.901`) and signature smoothness was low (`0.0021`); artifact accounting stayed below the cap with training-only probes excluded |
| desired but too weak/slow | train BPB at the checkpoint was worse than the resume point (`3.5454 -> 4.0688`), but the short post-checkpoint log had already moved back toward lower values (`3.896`, `3.880` at steps 1526--1527); simplex reasoning budgets improved MST/reasoning scores but not BPB; mean geometry BPB was high (`4.930`) while best branch BPB was useful (`4.044`); MST efficiency (`0.707`) and path smoothness (`0.050`) were coherent but not yet compression-aligned; toric shadow margins were positive but tiny (`min 0.00022`) |
| undesirable | no official-style FineWeb BPB was available at this gate; controller validation BPB in W&B summary remained high/stale (`5.2686`) and cannot promote the checkpoint; active-face margins were still inverted (`train -1.3745`, geometry mean `-1.696`); Toric BGG standard leakage was high (`0.610`), so BGG supervision must remain off; simplex budget scaling made BPB slightly worse (`4.340` at budget 1 vs `4.362` at budget 4); exact triangle validity was only `0.685`, so higher-order topology is not ready to drive optimization |

Core values:

| metric | value | read |
|---|---:|---|
| step-1500 train BPB | `3.545400` | resume point before Adam reset |
| step-1525 checkpoint train BPB | `4.068817` | worse than source checkpoint, but only 25 reset steps |
| step-1525 checkpoint train loss | `2.820289` | same short-window rebound |
| best validation BPB in checkpoint metadata | `4.696106` | unchanged; no fresh official-style FineWeb gate |
| W&B controller validation BPB | `5.268565` | stale/sparse calibration, not a promotion signal |
| simplex BPB by budget | `4.340`, `4.344`, `4.362`, `4.357` | branch scaling does not yet reduce BPB |
| simplex MST efficiency by budget | `0.726`, `0.779`, `0.808`, `0.815` | reasoning structure improves with budget |
| geometry mean / best BPB | `4.930245` / `4.043945` | mean branch quality weak; best Hebrew branch useful |
| geometry mean / best answer BPB | `4.814122` / `3.610226` | answer-span scoring has isolated strong branches |
| geometry MST efficiency | `0.706840` | desired but weak |
| geometry path smoothness | `0.050277` | desired |
| topology exact edge / directed edge validity | `0.973121` / `0.935311` | desired |
| topology exact triangle validity | `0.684593` | too weak for topology loss activation |
| topology HDBSCAN stability/noise | `0.883843` / `0.116157` | desired |
| Toric BGG resolution consistency | `0.900758` | good diagnostic |
| Toric BGG standard leakage | `0.610030` | not ready for supervision |
| Toric BGG Gale consistency | `0.106010` | usable diagnostic, not a loss yet |
| toric active-face margin | `-1.374512` train, `-1.695964` geometry | not as desired |
| toric Slepian concentration/leakage | `1.0` / `0.0` | desired phase concentration |

Finite differences are underdetermined for this restarted run.  The W&B export
contained only one post-restart row, so Hessian/second-difference claims are
not statistically supported.  Direct checkpoint metadata gives:

\[
  B_{1500}=3.545400,\qquad B_{1525}=4.068817,
\]

but the training log immediately after the checkpoint showed lower noisy
microbatch values at steps `1526` and `1527`.  I treat this as an early Adam
reset transient rather than evidence for rollback.

### Plot Review

Reviewed representative generated plots:

```text
simplex/reasoning_k_bpb_triangle.png
geometry/tetrahedra/toric_gfn_bpb.png
geometry/trajectories/*_trajectory_3d.png
geometry/trajectories/*_phase_energy.png
geometry/trajectories/*_energy_landscape.png
geometry/topology/*_commutative_algebra_audit.png
geometry/topology/*_directed_filtration.png
geometry/topology/*_toric_shadow_audit.png
geometry/topology/*_toric_slepian_audit.png
```

Plot categorization:

| diagnostic | category | evidence |
|---|---|---|
| simplex BPB/reasoning triangle | mixed | budget 1 is best for BPB; budgets 4/8 improve reasoning/MST placement but move away from the low-BPB vertex |
| Toric/GFlowNet/BPB tetrahedron | desired but weak | diversity and toric entropy are visible, but most points cluster away from the low-BPB vertex |
| 3D trajectories | desired but weak | dense terminal basins exist with a few long exploratory jumps; search is structured but not yet consistently likelihood-improving |
| Ramachandran-style phase plots | desired | phase mass is structured near axes and wrap boundaries without a single pathological high-NLL phase region |
| energy landscapes | desired but weak | coherent basin/arc appears, but long low-density excursions remain |
| Koszul/commutative algebra audit | mixed | \(d_1d_2\) and variety residuals are clean, while Fitting/BE and \(H_1\) obstruction panels still show localized violations |

### Mathematical Explanation

This gate is a two-objective disagreement in miniature.  The reasoning gate is
alive: entropy, branch diversity, edge-level persistence morphisms, toric phase
concentration, and BGG \(d^2\)-style resolution consistency are all visible.
The competition gate is not yet favorable: no fresh FineWeb BPB exists, sparse
validation is stale/high, and branch/test-time scaling does not lower BPB.

Mathematically, the current auxiliary geometry should remain a diagnostic.  The
standard-filtration mass is still mostly outside the allowed order ideal, so
turning on Toric BGG would push hidden states toward a finite complex before
the byte likelihood has recovered.  Likewise, negative active-face margins mean
the tropical/toric fan decisions are not stable wall-crossing certificates yet.
The correct intervention is therefore not scalar editing; the config already
sets `toric_bgg_loss_weight`, `koszul_persistence_loss_weight`, toric geometry,
GFlowNet, trajectory-flow, and trajectory-memory loss weights to zero in this
recovery band.

### Decision

`CONTINUE` from:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00001525.pt
```

No code or config scalar changes were made for this gate.  The continuation is
a fresh tmux process, not the paused training process:

```text
training tmux: toricgt_oai_bpb_revealed_01525_20260602T175354Z
watcher tmux:  toricgt_watch_bpb_revealed_01525_20260602T175354Z
W&B run:       amelie-iska-math/toricgt-parameter-golf/oai01500revealed20260602T174238Z
train log:     logs/training/oai-bpb-revealed-context-01525-20260602T175354Z.log
watcher log:   logs/training/oai-bpb-revealed-context-01525-20260602T175354Z.watcher.log
analysis root: outputs/post_resume_analysis/oai-bpb-revealed-context-01525-20260602T175354Z
```

The trainer loaded optimizer state from the analyzed checkpoint
(`optimizer_state_loaded=true`).  The next interrupting analysis target is:

```text
target step: 2025
min mtime:   1780422834
```

Acceptance criteria for the next gate:

1. fresh train BPB should recover below the step-1525 transient and preferably
   approach or beat the step-1500 source BPB;
2. sparse validation/FineWeb-style BPB, if available, must not deteriorate;
3. branch budget scaling should stop increasing BPB, or stay disabled as an
   inference-time promotion path;
4. GFlowNet entropy should remain near `log(16)` without increasing loss
   pressure;
5. Toric BGG standard leakage must fall materially before any nonzero
   `toric_bgg_loss_weight`;
6. active-face margins should become less negative before toric geometry losses
   are promoted from diagnostics;
7. exact triangle validity should move above `0.69` before topology losses are
   made active.

## 2026-06-02 Automated Review: Step 4,500 Curve-Lock Replay

The watcher analyzed:

```text
outputs/post_resume_analysis/oai-bpb-graphcg-gfn-04000-curvelock-20260602T120225Z/step-00004500/
```

Analyzed checkpoint:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00004500.pt
```

The run was stopped for review by the watcher.  The target training tmux
session was no longer active when the review began, so there was no additional
process to kill before selecting the replay point.

### Metric Categorization

| category | count | read |
|---|---:|---|
| as desired | `182` | full-window train BPB and loss improved; GFlowNet entropy remains near `log(16)`; action diversity remains near one; directed edge validity, toric winding, Slepian concentration, and random-order audits remain structurally healthy |
| as desired but too weak/slow | `119` | GFlowNet loss, branch BPB, answer BPB, MST efficiency, path smoothness, and toric fan occupancy remain coherent but are not yet converting enough search structure into lower byte likelihood |
| not as desired | `100` | the checkpoint sequence exits a substantially better step-4200 likelihood basin; recent train BPB/loss slopes are positive; active-face margins are still low/inverted; exact triangle validity remains below the desired topological-consistency band |

Core values:

| metric | value | read |
|---|---:|---|
| analyzed train BPB at step 4500 | `3.593540` | better than the full-window start, but worse than the fresh local low |
| analyzed train loss at step 4500 | `2.490852` | tracks the BPB rebound |
| best fresh checkpoint BPB | `3.394715` at step `4200` | materially better than the analyzed endpoint |
| W&B first-window train BPB median | `3.88868` | descent over the whole run is real |
| W&B last-window train BPB median | `3.55915` | full-window relative change `-8.47%` |
| recent train BPB slope per 1k steps | `+0.385096` | not as desired; local floor exit |
| recent train loss slope per 1k steps | `+0.266928` | not as desired; same floor exit |
| GFlowNet loss median change | `3.01946 -> 2.78558` | desired over the full window |
| recent GFlowNet loss slope per 1k steps | `+0.971214` | not as desired; auxiliary loss rebounds with BPB |
| GFlowNet entropy | about `2.77249` | desired; policy is noncollapsed |
| GFlowNet action diversity | about `0.99871` | desired |
| mean branch BPB | `4.99202` | too weak |
| best branch BPB | `4.03176` | too weak relative to train BPB |
| mean answer BPB | `4.87519` | too weak |
| best answer BPB | `3.61069` | desired but not yet competitive |
| MST efficiency | `0.69660` | desired but weak |
| path smoothness | `0.07175` | desired |
| directed asymmetry | `0.38929` | desired; noncommutative directed topology remains active |
| exact edge validity | `0.97795` | desired |
| exact directed edge validity | `0.92643` | desired |
| exact triangle validity | `0.66978` | not as desired |
| HDBSCAN stability | `0.87218` | desired |
| transport entropy | `0.99715` | desired |
| active-face margin | `-1.81413` | not as desired |
| toric shadow min margin | `0.000162` | too weak |
| toric shadow occupied fan cells | `12.79167` | desired |
| Slepian concentration/leakage | `1.0 / 0.0` | desired |

Fresh checkpoint finite differences after the step-4000 restart:

| window | train BPB delta |
|---|---:|
| `4050 -> 4100` | `-0.403438` |
| `4100 -> 4150` | `-0.059985` |
| `4150 -> 4200` | `-0.171485` |
| `4200 -> 4250` | `+0.202553` |
| `4250 -> 4300` | `+0.116091` |
| `4300 -> 4350` | `-0.141940` |
| `4350 -> 4400` | `+0.030166` |
| `4400 -> 4450` | `+0.035739` |
| `4450 -> 4500` | `-0.043784` |

Second differences:

| window | second difference |
|---|---:|
| `4050 -> 4100 -> 4150` | `+0.343453` |
| `4100 -> 4150 -> 4200` | `-0.111500` |
| `4150 -> 4200 -> 4250` | `+0.374038` |
| `4200 -> 4250 -> 4300` | `-0.086462` |
| `4250 -> 4300 -> 4350` | `-0.258031` |
| `4300 -> 4350 -> 4400` | `+0.172106` |
| `4350 -> 4400 -> 4450` | `+0.005573` |
| `4400 -> 4450 -> 4500` | `-0.079523` |

The local low is:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00004200.pt
```

with train BPB `3.394715`.  The exit signal is strong:

\[
  B_{4250}-B_{4200}=0.202553,\qquad
  B_{4300}-B_{4250}=0.116091.
\]

The immediate curvature across the low is

\[
  (B_{4250}-B_{4200})-(B_{4200}-B_{4150})
  =0.202553-(-0.171485)=0.374038>0.
\]

The later partial recovery into step `4500` does not justify continuing,
because the endpoint remains `0.198825` BPB worse than step `4200`.

### Plot Review

Reviewed:

```text
metrics/core_metric_timeseries.png
metrics/selected_metric_correlations.png
metrics/recent_metric_slopes.png
simplex/reasoning_k_bpb_triangle.png
simplex/reasoning_k_bpb_mst_tetrahedron.png
simplex/reasoning_k_bpb_diversity_tetrahedron.png
geometry/triangles/*.png
geometry/tetrahedra/*.png
geometry/trajectories/*_trajectory_3d.png
geometry/trajectories/*_energy_landscape.png
geometry/trajectories/*_phase_energy.png
geometry/trajectories/*_toric_phase_simplicial_trajectory.png
geometry/trajectories/*_projected_simplicial_toric_geometry.html
geometry/topology/*_exact_persistence_morphisms.png
geometry/topology/*_directed_filtration.png
geometry/topology/*_toric_shadow_audit.png
geometry/topology/*_toric_slepian_audit.png
```

Plot categorization:

| diagnostic | category | evidence |
|---|---|---|
| core BPB/loss timeseries | mixed | full-window slope is negative, but the recent slope is positive and the checkpoint replay shows a clear exit from the step-4200 low |
| GFlowNet panels | desired but weak | entropy and action diversity are healthy, but GFlowNet loss rebounds with BPB rather than reducing branch BPB |
| simplex and tetrahedra | too weak | high-reasoning and high-complexity branches remain too far from the low-BPB vertex; the best branch improves but does not become a reliable terminal basin |
| 3D trajectories and energy landscapes | desired but weak | paths are coherent and smooth, but long chords and high endpoint energy remain; search is structured but not yet sufficiently compression-aligned |
| exact persistence morphisms | mixed | edge and directed validity are strong, but triangle validity around `0.67` leaves higher-order simplex consistency below target |
| toric/simplicial phase plots | desired but weak | projected leaves, fan cells, and local complexes are visible, but active-face and shadow margins are too small for stable wall-crossing control |

### Mathematical Explanation

The likelihood objective found a better basin at step `4200`, then exited with
large positive first differences.  If \(B_t\) is checkpoint BPB, the discrete
gradient \(\nabla B_t=B_{t+50}-B_t\) changes from `-0.171485` over
`4150 -> 4200` to `+0.202553` over `4200 -> 4250`.  The corresponding
positive second difference is a classical floor-bounce signature: the optimizer
steps across a narrow local basin faster than the auxiliary geometry can align
with the byte-likelihood curvature.

The auxiliary systems are not collapsed.  Entropy, action diversity, directed
edge validity, toric fan occupancy, HDBSCAN stability, and Slepian leakage are
all acceptable.  The failure is not a lack of structure; it is excess transverse
curvature from still-active GraphCG/GFlowNet/topology directions relative to the
small BPB basin.  The correct intervention is therefore scalar damping, not an
architectural rollback.

### Decision

Restart from step `4200`, not from `4500`.

Implemented scalar-only changes:

| control | old | new |
|---|---:|---:|
| robust micro guard ratio | `1.035` | `1.030` |
| robust micro guard delta | `0.040` | `0.032` |
| robust micro guard min scale | `0.15` | `0.12` |
| controller `gflownet_loss_max` | `2e-6` | `1e-6` |
| controller recovery down step | `1e-6` | `5e-7` |
| `3250-6000` LR multiplier | `0.020` | `0.016` |
| `3250-6000` grad clip | `0.20` | `0.18` |
| `3250-6000` medium mix ratio | `0.08` | `0.06` |
| `3250-6000` GFlowNet loss weight | `1e-6` | `5e-7` |
| `3250-6000` GFlowNet entropy weight | `2e-7` | `1e-7` |
| `3250-6000` GraphCG loss weight | `1e-5` | `5e-6` |
| `3250-6000` contrastive weight | `1e-6` | `5e-7` |

Next target: fresh step `4700`.

Acceptance criteria:

1. step `4200` remains a stable low rather than immediately rebounding;
2. at least one fresh checkpoint beats `3.394715`;
3. sparse validation BPB does not worsen;
4. GFlowNet entropy remains near `log(16)` and action diversity remains above
   `0.998`;
5. branch and answer BPB start moving toward the train BPB basin;
6. exact triangle validity returns above `0.69` or stops degrading;
7. toric active-face and shadow margins increase from the current near-zero or
   inverted state.

## 2026-06-02 Automated Review: Step 4,300 Basin-Lock Replay

The watcher analyzed:

```text
outputs/post_resume_analysis/oai-bpb-graphcg-gfn-03800-basinlock-20260602T105427Z/step-00004300/
```

Analyzed checkpoint:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00004300.pt
```

The replay was stopped for review after step `4300`.  The target training tmux
session was already paused by the watcher, so no additional process kill was
required before choosing a restart checkpoint.

### Metric Categorization

| category | count | read |
|---|---:|---|
| as desired | `336` | full-window train BPB/loss decreased from the high-noise post-resume samples; sparse deterministic validation BPB and validation loss moved slightly downward; GFlowNet entropy remains near `log(16)` and action diversity remains near `0.9986`; exact edge validity, directed validity, inclusion checks, toric phase structure, and Slepian leakage stay healthy |
| as desired but too weak/slow | `141` | step `4000` recovers almost all of the step-3800 low but does not improve on it; validation gates are sparse; branch BPB remains around `5`; MST efficiency and path smoothness are useful but do not yet lower answer BPB |
| not as desired | `148` | the run exits the step-4000 low with a positive second difference; GFlowNet loss rebounds around the same window; active-face margins remain inverted/low; exact triangle validity remains below target |

Core values:

| metric | value | read |
|---|---:|---|
| analyzed train BPB at step 4300 | `3.676559` | too high for promotion |
| best fresh checkpoint BPB | `3.491981` at step `4000` | close to the step-3800 source low, but not lower |
| sparse validation BPB | `5.55029` | slight downward movement in the replay, but weaker than the prior ultralow run |
| complexity validation BPB | rising from about `5.238` to `5.244` | not as desired |
| complexity validation prediction-target NCD | falling from about `0.929` to `0.921` | desired |
| GFlowNet entropy/action diversity | `2.77248` / `0.99864` | desired |
| mean branch BPB | `4.99074` | too weak |
| best branch BPB | `4.03230` | too weak |
| mean answer BPB | `4.87349` | too weak |
| best answer BPB | `3.61037` | desired but marginal |
| MST efficiency | `0.69670` | desired but weak |
| path smoothness | `0.07120` | desired, improved |
| exact edge validity | `0.97813` | desired |
| exact directed edge validity | `0.92597` | desired |
| exact triangle validity | `0.67368` | not as desired |
| toric shadow fan cells | `12.58` | desired |
| toric shadow min margin | `0.000140` | too weak |
| Slepian concentration/leakage | `1.0 / 0.0` | desired |

Fresh checkpoint finite differences after the step-3800 restart:

| window | train BPB delta |
|---|---:|
| `3850 -> 3900` | `-0.375422` |
| `3900 -> 3950` | `-0.021743` |
| `3950 -> 4000` | `-0.221114` |
| `4000 -> 4050` | `+0.001916` |
| `4050 -> 4100` | `+0.177784` |
| `4100 -> 4150` | `-0.171611` |
| `4150 -> 4200` | `+0.245268` |
| `4200 -> 4250` | `-0.029085` |
| `4250 -> 4300` | `-0.039694` |

The local low is:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00004000.pt
```

with train BPB `3.491981`.  It is the last checkpoint before the first
positive derivative.  The curvature evidence is:

\[
  B_{4050}-B_{4000}=0.001916,\qquad
  B_{4100}-B_{4050}=0.177784,
\]

so

\[
  (B_{4100}-B_{4050})-(B_{4050}-B_{4000})=0.175869>0.
\]

This is again a floor-bounce basin exit.  The later negative deltas into step
`4300` are not enough to justify continuing from `4300`, because the checkpoint
is still `0.1846` BPB worse than step `4000`.

### Plot Review

Reviewed:

```text
metrics/core_metric_timeseries.png
metrics/selected_metric_correlations.png
metrics/recent_metric_slopes.png
simplex/reasoning_k_bpb_triangle.png
simplex/reasoning_k_bpb_mst_tetrahedron.png
geometry/trajectories/*_trajectory_3d.png
geometry/trajectories/*_energy_landscape.png
geometry/trajectories/*_phase_energy.png
geometry/trajectories/*_toric_phase_simplicial_trajectory.png
geometry/trajectories/*_projected_simplicial_toric_geometry.html
geometry/topology/*_exact_persistence_morphisms.png
geometry/topology/*_directed_filtration.png
geometry/topology/*_toric_shadow_audit.png
geometry/topology/*_toric_slepian_audit.png
```

Plot categorization:

| diagnostic | category | evidence |
|---|---|---|
| core BPB/loss timeseries | mixed | full-window slope is down, but the checkpoint sequence exits the step-4000 low |
| sparse validation panels | desired but weak | deterministic validation BPB moves down slightly, but complexity BPB rises |
| selected correlations | not as desired | GFlowNet loss remains positively correlated with BPB; trajectory kinetic/dissipation anti-correlates with BPB but not enough to reduce branch BPB |
| simplex/tetrahedra | too weak | the best branch remains near the low-BPB vertex, while higher-budget branches still occupy high-reasoning/high-BPB regions |
| 3D trajectories and energy landscapes | too weak | the flow is coherent and smoother, but still has long chords and endpoint energy that do not form stable low-BPB basins |
| exact persistence morphisms | mixed | edge validity is high, but triangle validity remains around `0.674` |
| toric/simplicial phase plots | desired but cluttered | toric winding and local VR structure are real, but active-face margins are too small and the plot remains dense |

### Mathematical Explanation

The basin-lock run lowered the learning rate enough to revisit the good
likelihood basin, but not enough to stay there.  In finite-difference terms,
the discrete derivative

\[
  \nabla B_t \approx B_{t+50}-B_t
\]

turns positive at step `4000`, and the second difference is positive over the
next span.  Because GFlowNet entropy and GraphCG axes are noncollapsed, the
auxiliary systems are not failing by collapse.  They are still adding
transverse curvature to the byte-likelihood objective before the local basin is
flat enough.  The geometry plots support that read: trajectory smoothness is
improving, but branch BPB and answer BPB remain high.  In other words, search
has coherent structure, but the structure is not yet aligned with compression.

### Decision

Restart from step `4000`, not from `4300`.

Implemented scalar-only changes:

| control | old | new |
|---|---:|---:|
| robust micro guard ratio | `1.04` | `1.035` |
| robust micro guard delta | `0.055` | `0.040` |
| robust micro guard min scale | `0.20` | `0.15` |
| controller `gflownet_loss_max` | `5e-6` | `2e-6` |
| controller recovery down step | `2e-6` | `1e-6` |
| `3250-6000` LR multiplier | `0.026` | `0.020` |
| `3250-6000` grad clip | `0.22` | `0.20` |
| `3250-6000` medium mix ratio | `0.10` | `0.08` |
| `3250-6000` GFlowNet loss weight | `2.5e-6` | `1e-6` |
| `3250-6000` GFlowNet entropy weight | `5e-7` | `2e-7` |
| `3250-6000` GraphCG loss weight | `1.5e-5` | `1e-5` |
| `3250-6000` contrastive weight | `3e-6` | `1e-6` |

Next target: fresh step `4500`.

Acceptance criteria:

1. step `4000` is preserved as a stable low rather than immediately exited;
2. at least one fresh checkpoint beats `3.491981`;
3. deterministic validation BPB remains nonincreasing;
4. complexity validation BPB stops rising while prediction-target NCD continues
   falling;
5. GFlowNet entropy remains near `log(16)` and action diversity remains above
   `0.998`;
6. exact triangle validity recovers above `0.69` or at least stops degrading.

## 2026-06-02 Automated Review: Step 4,100 Ultralow-Curvature Replay

The watcher analyzed:

```text
outputs/post_resume_analysis/oai-bpb-graphcg-gfn-03600-ultralowcurv-20260602T094341Z/step-00004100/
```

Analyzed checkpoint:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00004100.pt
```

The replay was stopped for review after step `4100`.  The requested training
tmux session was no longer active after the watcher pause, so no additional
process interruption was required before restarting.

### Metric Categorization

| category | count | read |
|---|---:|---|
| as desired | `344` | validation BPB, validation loss, and complexity validation BPB improved slightly across the sparse gates; GFlowNet entropy remains near `log(16)`; action diversity remains near `0.9986`; exact edge validity, directed edge validity, inclusion checks, toric phase winding, Slepian concentration, and random-order causality audits remain healthy |
| as desired but too weak/slow | `154` | train BPB found a better local low at step `3800`, but did not stay there; GraphCG and topology metrics remain structurally meaningful, but their benefits are not yet converting into branch BPB; simplex and trajectory plots show useful organization but too much high-reasoning/high-BPB mass |
| not as desired | `127` | train BPB and train loss leave the step-3800 basin with positive curvature; GFlowNet loss rebounds after the same window; active-face margins remain inverted/low; exact triangle validity and several Koszul/topology residuals are still below the target range |

Core values at the analyzed checkpoint:

| metric | value | read |
|---|---:|---|
| analyzed train BPB | `3.624333` | worse than the fresh step-3800 low |
| analyzed train loss | `2.512196` | same rebound as BPB |
| sparse validation BPB | `5.547263` | improved versus the previous low-curvature review |
| complexity validation BPB | `5.233380` | improved versus the previous low-curvature review |
| GFlowNet entropy | `2.77247` | desired; exploration remains noncollapsed |
| GFlowNet action diversity | `0.99864` | desired |
| mean branch BPB | `4.98947` | too weak; search is not yet compressing enough |
| best branch BPB | `4.02904` | improved but not close enough to train BPB |
| mean answer BPB | `4.87336` | improved, still too high |
| MST efficiency | `0.69730` | desired but weak |
| path smoothness | `0.07259` | desired, improved |
| directed topology asymmetry | `0.38935` | desired; noncommutative directed structure is present |
| exact edge validity | `0.97776` | desired |
| exact directed edge validity | `0.92735` | desired |
| exact triangle validity | `0.67122` | not as desired |
| active-face margin | `-1.81055` | not as desired |
| toric shadow occupied fan cells | `12.75` | desired; fan occupancy is nontrivial |
| toric shadow min margin | `0.000175` | too weak; walls are too close |
| Slepian concentration / leakage | `1.0 / 0.0` | desired |

Fresh checkpoint finite differences after the step-3600 restart:

| window | train BPB delta |
|---|---:|
| `3650 -> 3700` | `+0.016759` |
| `3700 -> 3750` | `-0.061667` |
| `3750 -> 3800` | `-0.179698` |
| `3800 -> 3850` | `+0.065059` |
| `3850 -> 3900` | `+0.197493` |
| `3900 -> 3950` | `-0.024604` |
| `3950 -> 4000` | `-0.184700` |
| `4000 -> 4050` | `+0.110049` |
| `4050 -> 4100` | `-0.024158` |

The best fresh checkpoint in this replay is:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00003800.pt
```

with train BPB `3.485194`, which beats the earlier step-3600 low.  The bad
news is the immediate curvature exit:

\[
  B_{3850}-B_{3800}=0.065059,\qquad
  B_{3900}-B_{3850}=0.197493,
\]

and therefore

\[
  (B_{3900}-B_{3850})-(B_{3850}-B_{3800})=0.132434>0.
\]

This is the signature of a basin exit rather than ordinary stochastic noise:
the first derivative becomes positive immediately after the new low, and the
second derivative is positive over the first post-low pair.

### Plot Review

Reviewed:

```text
metrics/core_metric_timeseries.png
metrics/selected_metric_correlations.png
metrics/recent_metric_slopes.png
simplex/reasoning_k_bpb_triangle.png
simplex/reasoning_k_bpb_mst_tetrahedron.png
geometry/triangles/*.png
geometry/tetrahedra/*.png
geometry/trajectories/*_trajectory_3d.png
geometry/trajectories/*_energy_landscape.png
geometry/trajectories/*_phase_energy.png
geometry/trajectories/*_toric_phase_simplicial_trajectory.png
geometry/trajectories/*_toric_phase_winding_collection.png
geometry/trajectories/*_projected_simplicial_toric_geometry.html
geometry/topology/*_toric_shadow_audit.png
geometry/topology/*_exact_persistence_morphisms.png
geometry/topology/*_directed_filtration.png
geometry/topology/*_toric_slepian_audit.png
```

Plot categorization:

| diagnostic | category | evidence |
|---|---|---|
| core BPB/loss timeseries | too weak | validation gates improve, but train BPB exits the step-3800 low immediately |
| GFlowNet entropy/diversity | as desired | entropy stays near `log(16)` and diversity remains above `0.998` |
| branch simplex/tetrahedra | too weak | the best branch is near the low-BPB vertex, but most higher-budget branches still live in high-reasoning/high-BPB regions |
| 3D trajectories | too weak | trajectory geometry is coherent, but dense branch clouds and long chords remain |
| energy landscapes | too weak | local NLL bands improved, but branch endpoints still do not settle into low-BPB basins |
| phase/Ramachandran plots | as desired | phase bands and toric winding are structured rather than random |
| toric shadow audit | too weak | fan cells and bends are visible; active-face margins remain too small |
| exact persistence morphisms | mixed | edge validity is high, while triangle validity remains around `0.671` |
| Slepian/PSWF toric audit | as desired | concentration is `1.0` and leakage is `0.0` |

### Mathematical Explanation

Let \(B_t\) be checkpoint train BPB and let

\[
  L_{\rm eff}
  =L_{\rm BPB}
  +\lambda_{\rm GFN}L_{\rm GFN}
  +\lambda_{\rm GraphCG}L_{\rm GraphCG}
  +\lambda_{\rm topo}L_{\rm topo}
  +\lambda_{\rm toric}L_{\rm toric}.
\]

The auxiliary geometry terms are behaving as useful coordinate-system
regularizers: GraphCG axes are noncollapsed, GFlowNet entropy is healthy,
directed topology is nontrivial, and toric phase shadows are coherent.  The
failure mode is not mechanism collapse.  It is early-window curvature: once
\(B_t\) reaches the local low at step `3800`, the auxiliary-gradient and
medium-stream components add enough transverse curvature that the byte
likelihood descent direction exits the basin.  The empirical evidence is the
positive first derivative after the local low and the positive second
difference over `3800 -> 3850 -> 3900`.

The correct response is scalar, not architectural: reduce step size, medium
stream pressure, and auxiliary gradient amplitude inside the likelihood-capture
phase while preserving random-order autoregressive graph decoding, tropical
ring/hybrid attention, toric memory, dense contest weights, GraphCG,
embedding-space GFlowNet graph-of-thought, relative Kolmogorov diagnostics,
directed persistence, Koszul audits, and toric probes.

### Decision

Restart from step `3800`, not from `4100`, because step `3800` is the only
fresh checkpoint that beats the previous step-3600 low and the subsequent
finite differences show a curvature exit.

Implemented scalar-only changes:

| control | old | new |
|---|---:|---:|
| robust micro guard ratio | `1.05` | `1.04` |
| robust micro guard delta | `0.075` | `0.055` |
| robust micro guard min scale | `0.25` | `0.20` |
| controller `gflownet_loss_max` | `1e-5` | `5e-6` |
| controller `gflownet_relax_step` | `1e-6` | `0.0` |
| controller recovery down step | `5e-6` | `2e-6` |
| `3250-6000` LR multiplier | `0.032` | `0.026` |
| `3250-6000` grad clip | `0.24` | `0.22` |
| `3250-6000` medium mix ratio | `0.15` | `0.10` |
| `3250-6000` GFlowNet loss weight | `5e-6` | `2.5e-6` |
| `3250-6000` GFlowNet entropy weight | `1e-6` | `5e-7` |
| `3250-6000` GraphCG loss weight | `2e-5` | `1.5e-5` |
| `3250-6000` contrastive weight | `5e-6` | `3e-6` |

Next target: fresh step `4300`.

Acceptance criteria:

1. no immediate positive second-difference exit after the step-3800 source;
2. train BPB remains below `3.485194` or establishes a lower local minimum;
3. validation BPB and complexity validation BPB keep their slight downward
   movement;
4. GFlowNet entropy remains near `log(16)` and action diversity stays above
   `0.998`;
5. exact edge validity remains above `0.97`, directed edge validity above
   `0.92`, inclusion violation `0.0`, Slepian leakage `0.0`;
6. exact triangle validity recovers above `0.69` or at least stops degrading.

## 2026-06-02 Automated Review: Step 4,100 Low-Curvature Replay

The watcher analyzed:

```text
outputs/post_resume_analysis/oai-bpb-graphcg-gfn-03600-lowcurv-20260602T083044Z/step-00004100/
```

Analyzed checkpoint:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00004100.pt
```

The replay was stopped for review after step `4100`.  The current training
tmux session was no longer active after the watcher pause, so no additional
process interruption was required for the restart decision.

### Metric Categorization

| category | count | read |
|---|---:|---|
| as desired | `364` | byte likelihood and GFlowNet loss decrease over the whole replay; exact edge validity, directed validity, inclusion/boundary checks, Slepian leakage, phase winding, and random-order causality audits remain healthy |
| as desired but too weak/slow | `128` | validation gates are sparse and flat-to-rising; GFlowNet entropy/action diversity remain noncollapsed but do not convert to BPB; GraphCG axes are stable but not accelerating byte compression |
| not as desired | `133` | no checkpoint beats the previous step-3600 BPB low; validation/complexity validation are not improving; trajectory viscous dissipation rises; triangle validity and several topology residuals worsen |

Core W&B behavior:

| metric | first median | last median | relative change | recent slope / 1k |
|---|---:|---:|---:|---:|
| train BPB | `3.86899` | `3.67417` | `-5.04%` | `-0.48863` |
| train loss | `2.68178` | `2.54674` | `-5.04%` | `-0.33869` |
| total loss | `2.68198` | `2.54694` | `-5.04%` | `-0.33871` |
| validation BPB | `5.55248` sparse median | `5.55248` sparse median; controller gate rose to about `5.5560` | weak/slow | not promotion-ready |
| complexity validation BPB | `5.24985` sparse median | `5.24985` sparse median | weak/slow | not promotion-ready |
| prediction-target NCD, validation | `0.92667` | `0.92667` | weak/slow | not promotion-ready |
| GFlowNet loss | `3.09379` | `2.85383` | `-7.76%` | `-1.39735` |
| GFlowNet entropy | `2.77237` | `2.77242` | nearly flat | `-0.00024` |
| GFlowNet action diversity | `0.99860` | `0.99854` | nearly flat | `-0.00113` |
| trajectory viscous dissipation | `0.02247` | `0.02745` | `+22.17%` | `+0.01185` |

Fresh checkpoint finite differences:

| window | train BPB delta |
|---|---:|
| `3650 -> 3700` | `-0.229978` |
| `3700 -> 3750` | `+0.087649` |
| `3750 -> 3800` | `+0.011434` |
| `3800 -> 3850` | `-0.073616` |
| `3850 -> 3900` | `+0.151312` |
| `3900 -> 3950` | `-0.224203` |
| `3950 -> 4000` | `+0.104350` |
| `4000 -> 4050` | `-0.134646` |
| `4050 -> 4100` | `+0.020280` |

The best fresh checkpoint in this replay was step `4050` with train BPB
`3.561287`, but this remains worse than the replay source checkpoint at
step `3600` with train BPB `3.520679`.  The second difference over
`4000 -> 4050 -> 4100` is `+0.154926`, indicating another positive-curvature
exit from the local descent direction.

### Geometry And Plot Review

Reviewed:

```text
metrics/core_metric_timeseries.png
metrics/selected_metric_correlations.png
metrics/recent_metric_slopes.png
simplex/reasoning_k_bpb_triangle.png
simplex/reasoning_k_bpb_mst_tetrahedron.png
geometry/trajectories/R2_gss1147-got_math_500k_got_math_trajectory_3d.png
geometry/trajectories/R2_gss1147-got_math_500k_got_math_energy_landscape.png
geometry/trajectories/R2_gss1147-got_math_500k_got_math_phase_energy.png
geometry/trajectories/R2_gss1147-got_math_500k_got_math_toric_phase_simplicial_trajectory.png
geometry/trajectories/R2_gss1147-got_math_500k_got_math_toric_phase_winding_collection.png
geometry/topology/R2_gss1147-got_math_500k_got_math_toric_shadow_audit.png
geometry/topology/R2_gss1147-got_math_500k_got_math_exact_persistence_morphisms.png
geometry/topology/R2_gss1147-got_math_500k_got_math_directed_filtration.png
geometry/topology/R2_gss1147-got_math_500k_got_math_toric_slepian_audit.png
```

Plot categorization:

| diagnostic | category | evidence |
|---|---|---|
| core BPB/loss timeseries | as desired but too weak | full-window decrease is real, but the curve rebounds after the previous low and validation gates rise |
| GFlowNet entropy/diversity | as desired | entropy stays near `log(16)` and diversity near `0.9985`, so exploration is alive |
| branch simplex/tetrahedra | too weak | the best branch is near the low-BPB vertex, but most branches cluster in high-reasoning/high-BPB regions |
| 3D trajectories | too weak | trajectories are coherent, but still contain long chords and dense branch clouds that do not compress answers |
| phase/Ramachandran plots | as desired | phase bands and toric winding are structured rather than random |
| toric shadow audit | too weak | occupied fan cells and bends are visible, but active-face margins are near zero and branch BPB remains high |
| directed persistence morphisms | mixed | edge validity is high, but exact triangle validity dropped to `0.6698` and cycle/rank residuals worsened |
| Slepian/PSWF toric audit | as desired | concentration `1.0`, leakage `0.0`, finite phase shadow remains coherent |

Key geometry values:

| metric | value | read |
|---|---:|---|
| mean branch BPB | `4.99534` | still far from train BPB |
| best branch BPB | `4.02664` | slightly improved, not enough |
| mean answer BPB | `4.88001` | worsened slightly from the previous review |
| best answer BPB | `3.61179` | slight improvement |
| MST efficiency | `0.69651` | improved, but not enough to lower BPB |
| path smoothness | `0.07516` | improved |
| directed asymmetry | `0.38856` | slightly improved |
| exact edge validity | `0.97743` | desired |
| exact directed edge validity | `0.92933` | desired |
| exact triangle validity | `0.66976` | not as desired |
| Buchsbaum-Eisenbud multiplier residual | `0.00415` | worsened |
| active-face margin | `-1.81283` | not as desired; still inverted/low-margin |
| mean toric bend | `1.84837` | improved |
| leaf residual | `1.00008` | slightly worse |
| Slepian concentration / leakage | `1.0 / 0.0` | desired |

### Mathematical Explanation

Let \(B_t\) denote checkpoint train BPB and let \(G_t\) denote the auxiliary
geometric gradient contribution from GFlowNet, GraphCG, directed persistence,
and toric probes.  The replay satisfies

\[
  B_{4050} - B_{4000} < 0,\qquad
  B_{4100} - B_{4050} > 0,
\]

with positive second difference

\[
  (B_{4100}-B_{4050})-(B_{4050}-B_{4000})=0.154926.
\]

This is a curvature reversal rather than a monotone descent channel.  The
geometric diagnostics explain the source: the toric/persistence systems are
alive, but branch BPB remains around \(5\), active-face margins are nearly
zero or inverted, and triangle/rank residuals worsened.  In the effective loss

\[
L_{\rm eff}=L_{\rm BPB}
  +\lambda_{\rm GFN}L_{\rm GFN}
  +\lambda_{\rm GraphCG}L_{\rm GraphCG}
  +\lambda_{\rm topo}L_{\rm topo}
  +\lambda_{\rm toric}L_{\rm toric},
\]

the auxiliary terms are useful as regularized coordinate systems, but in this
early window their gradients add curvature before the byte model has captured
the local likelihood basin.  Since edge validity, phase concentration, and
GFlowNet entropy are healthy, removing mechanisms would be the wrong response.
The correct response is scalar: reduce learning-rate curvature, reduce
medium-stream pressure, and cap auxiliary gradient amplitude while replaying
from the last known lower-BPB basin checkpoint.

### Decision

Restart from step `3600` again, not from `4050` or `4100`.

Implemented scalar-only changes:

| control | old | new |
|---|---:|---:|
| shock guard grad norm | `0.30` | `0.28` |
| shock guard update scale | `0.015` | `0.010` |
| robust micro guard ratio | `1.06` | `1.05` |
| robust micro guard delta | `0.10` | `0.075` |
| robust micro guard min scale | `0.32` | `0.25` |
| controller `gflownet_loss_max` | `0.00003` | `0.00001` |
| controller `gflownet_relax_step` | `0.000005` | `0.000001` |
| controller recovery down step | `0.00001` | `0.000005` |
| `3250-6000` LR multiplier | `0.045` | `0.032` |
| `3250-6000` grad clip | `0.28` | `0.24` |
| `3250-6000` medium mix ratio | `0.25` | `0.15` |
| `3250-6000` GFlowNet loss weight | `1e-5` | `5e-6` |
| `3250-6000` GFlowNet entropy weight | `2e-6` | `1e-6` |
| `3250-6000` GraphCG loss weight | `3e-5` | `2e-5` |
| `3250-6000` contrastive weight | `1e-5` | `5e-6` |

No architecture was changed.  Random-order autoregressive graph decoding,
dense contest weights, tropical ring/hybrid attention, toric memory, GraphCG,
embedding-space GFlowNet graph-of-thought, relative Kolmogorov diagnostics,
directed persistence, Koszul audits, and toric probes remain active.

Next target: fresh step `4100`.

Acceptance criteria:

1. at least one fresh checkpoint beats step-3600 train BPB `3.520679`;
2. no immediate positive second-difference exit after the new local low;
3. validation BPB and complexity validation BPB stop rising at the next gate;
4. GFlowNet entropy remains near `log(16)` and action diversity stays above
   `0.998`;
5. exact edge validity remains above `0.97`, directed edge validity above
   `0.92`, inclusion violation `0.0`, Slepian leakage `0.0`;
6. exact triangle validity recovers above `0.69` or stops degrading.

## 2026-06-02 Automated Review: Step 3,800 Low-Curvature Hold Replay

The watcher analyzed:

```text
outputs/post_resume_analysis/oai-bpb-graphcg-gfn-03300-curvcap-20260602T070622Z/step-00003800/
```

Analyzed checkpoint:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00003800.pt
```

The fresh replay window after the step-3300 restart produced a clear local
minimum at step `3600`:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00003600.pt
```

### Metric Categorization

| category | count | read |
|---|---:|---|
| as desired | `342` | feature systems remain alive: GFlowNet entropy/action diversity are noncollapsed, directed persistence morphisms are computed, toric phase audits are structured, and exact boundary/inclusion errors remain zero |
| as desired but too weak/slow | `143` | train BPB decreases over the full replay but only by `1.79%`; GraphCG and analogy terms are stable but not materially accelerating validation BPB |
| not as desired | `140` | validation BPB, complexity validation BPB, and branch BPB move the wrong way after the step-3600 low; finite differences show a floor-bounce basin |

Core W&B behavior:

| metric | first median | last median | relative change | recent slope / 1k |
|---|---:|---:|---:|---:|
| train BPB | `3.73412` | `3.66742` | `-1.79%` | `-0.0688` |
| train loss | `2.58830` | `2.54206` | `-1.79%` | `-0.0477` |
| total loss | `2.58859` | `2.54235` | `-1.79%` | `-0.0477` |
| validation BPB | `5.54494` median, controller gate rose from about `5.5394` to `5.5505` | flat in the sparse report, positive in gates | undesirable |
| complexity validation BPB | about `5.234` to `5.257` in the plotted gates | positive | undesirable |
| GFlowNet loss | `2.92006` | `2.84043` | `-2.73%` | `-2.5129` |
| GFlowNet entropy | `2.77242` | `2.77246` | near `0%` | noncollapsed |
| action diversity | `0.998616` | `0.998608` | near `0%` | noncollapsed |

Fresh checkpoint finite differences:

| window | train BPB delta |
|---|---:|
| `3350 -> 3400` | `-0.066283` |
| `3400 -> 3450` | `+0.113945` |
| `3450 -> 3500` | `-0.029881` |
| `3500 -> 3550` | `-0.069380` |
| `3550 -> 3600` | `-0.161846` |
| `3600 -> 3650` | `+0.046484` |
| `3650 -> 3700` | `+0.077951` |
| `3700 -> 3750` | `+0.022308` |
| `3750 -> 3800` | `+0.000889` |

Second differences show the same curvature event: the discrete second
difference over `3550 -> 3600 -> 3650` is `+0.208329`, so the optimizer crossed
out of a low-BPB basin immediately after the `3600` checkpoint.

### Plot Review

Reviewed:

```text
metrics/core_metric_timeseries.png
metrics/selected_metric_correlations.png
simplex/reasoning_k_bpb_triangle.png
simplex/reasoning_k_bpb_mst_tetrahedron.png
geometry/trajectories/R2_gss1147-got_math_500k_got_math_trajectory_3d.png
geometry/trajectories/R2_gss1147-got_math_500k_got_math_energy_landscape.png
geometry/trajectories/R2_gss1147-got_math_500k_got_math_phase_energy.png
geometry/trajectories/R2_gss1147-got_math_500k_got_math_toric_phase_simplicial_trajectory.png
geometry/trajectories/R2_gss1147-got_math_500k_got_math_toric_phase_winding_collection.png
geometry/topology/R2_gss1147-got_math_500k_got_math_toric_shadow_audit.png
geometry/topology/R2_gss1147-got_math_500k_got_math_exact_persistence_morphisms.png
geometry/topology/R2_gss1147-got_math_500k_got_math_directed_filtration.png
```

Visual conclusions:

1. Core BPB/loss plots form a low near `3580-3600`, then rebound.  The sparse
   validation gates rise across the same interval, so this is not a promotion
   point.
2. Correlations show GFlowNet loss and gradient norm positively aligned with
   BPB/loss, while trajectory kinetic/dissipation terms are negatively aligned
   with BPB.  This favors keeping GFlowNet active but extremely light during
   likelihood capture.
3. The simplex and tetrahedron plots show branches remain in the high-BPB
   interior rather than moving toward the low-BPB vertex.  Test-time scaling is
   alive but not yet improving the contest objective.
4. The 3D trajectory and energy landscape are coherent but still have long
   excursions and high-energy outliers; the best branch is not enough to rescue
   validation.
5. Ramachandran-style phase and toric winding plots show structured projected
   phase bands, not random phase noise.  Noncommutative toric memory remains
   useful as an audit signal.
6. Exact persistence-module morphism and directed-filtration plots remain
   mathematically healthy: boundary residual and inclusion violation are zero,
   exact edge validity is high, and cycle flux is effectively zero.  The weak
   point is conversion to BPB, not topology collapse.

### Desired

| diagnostic | value |
|---|---:|
| fresh local low | step `3600`, train BPB `3.520679` |
| exact morphisms computed | `20.0` |
| topology boundary residual | `0.0` |
| topology inclusion violation | `0.0` |
| exact edge validity | `0.977131` |
| exact directed edge validity | `0.930216` |
| directed cycle flux | `~6.05e-18` |
| HDBSCAN stability | `0.876573` |
| Slepian concentration / leakage | `1.0 / 0.0` |
| GFlowNet entropy / diversity | `2.77246 / 0.99861` |

### Desired But Too Weak Or Slow

| diagnostic | value | issue |
|---|---:|---|
| full-window train BPB change | `-1.79%` | real improvement, but slower than the desired cliff-like descent |
| mean branch BPB | `4.9960` | worse than base train BPB and not validation-promoting |
| best branch BPB | `4.0299` | a useful branch exists, but not a competitive endpoint |
| mean answer BPB | `4.8781` | answer-span compression remains weak |
| MST efficiency | `0.6927` | coherent, not improving enough |
| path smoothness | `0.0761` | acceptable, but long excursions persist |
| GraphCG loss | flat near `5.459` | steerable frame is stable but not yet accelerating BPB |

### Not As Desired

| diagnostic | value / trend | read |
|---|---|---|
| train BPB finite difference | positive after `3600` | first derivative changed sign after the local low |
| second difference | `+0.208329` over `3550 -> 3600 -> 3650` | sharp curvature/floor-bounce signal |
| validation BPB gate | rises to about `5.5505` | validation direction worsens |
| complexity validation BPB | rises to about `5.257` | byte-compression generalization worsens |
| branch simplex BPB | still around `5.0` mean | inference-time search not converting to BPB |
| active-face margin | `-1.8288` | tropical faces remain low-margin/inverted |
| toric leaf residual | `0.9985` | phase leaves are audit-consistent but not yet a loss to strengthen |
| exact triangle validity | `0.6941` | usable but weaker than the previous handoff |

### Mathematical Explanation

Let \(B_t\) be checkpoint train BPB.  The discrete optimizer trace satisfies

\[
  B_{3600} < B_{3550},\qquad
  B_{3650}-B_{3600}>0,\qquad
  B_{3700}-B_{3650}>0.
\]

Thus the local directional derivative along the optimizer path switches from
negative to positive at step `3600`, and the positive second difference
indicates a high-curvature exit from the basin rather than a noisy one-step
fluctuation.  The validation gates rising concurrently imply the effective
objective

\[
  L_{\rm eff}
  = L_{\rm BPB}
    + \lambda_{\rm GFN}L_{\rm GFN}
    + \lambda_{\rm GraphCG}L_{\rm GraphCG}
    + \lambda_{\rm topo}L_{\rm topo}
\]

is still over-curved for the post-low phase.  The geometric systems are not
collapsing, so the right fix is scalar: lower the step size and auxiliary
gradient contribution during the basin capture window.  Strengthening toric,
Koszul, or persistence losses now would add curvature when active-face margins
and branch BPB say the byte model is not ready for that pressure.

### Decision

Restart from step `3600`, not `3800`.

Implemented scalar-only changes:

| control | old | new |
|---|---:|---:|
| robust micro guard ratio | `1.08` | `1.06` |
| robust micro guard delta | `0.14` | `0.10` |
| robust micro guard min scale | `0.40` | `0.32` |
| controller `gflownet_loss_max` | `0.00008` | `0.00003` |
| controller `gflownet_relax_step` | `0.00002` | `0.000005` |
| controller recovery down step | `0.00004` | `0.00001` |
| `3250-6000` LR multiplier | `0.06` | `0.045` |
| `3250-6000` grad clip | `0.30` | `0.28` |
| `3250-6000` medium mix ratio | `0.35` | `0.25` |
| `3250-6000` GFlowNet loss weight | `2e-5` | `1e-5` |
| `3250-6000` GFlowNet entropy weight | `5e-6` | `2e-6` |
| `3250-6000` GraphCG loss weight | `4e-5` | `3e-5` |
| `3250-6000` analogy lattice weight | `1e-6` | `0` |
| `3250-6000` contrastive weight | `2e-5` | `1e-5` |

No architecture or data feature family was removed.  Random-order
autoregressive graph decoding, dense contest weights, tropical ring/hybrid
attention, toric memory, GraphCG axes, embedding-space GFlowNet graph-of-thought,
relative Kolmogorov diagnostics, directed persistence, Koszul audits, and toric
probes remain active.

Next target: fresh step `4100`.

Acceptance criteria:

1. at least one checkpoint remains below step-3600 train BPB `3.520679`;
2. no two consecutive positive 50-step BPB deltas after a new low;
3. validation BPB and complexity validation BPB stop rising at the next gate;
4. controller-reported GFlowNet weight stays below `3e-5`;
5. GFlowNet entropy/action diversity remain noncollapsed;
6. exact edge validity remains above `0.97`, directed edge validity above
   `0.92`, inclusion violation `0.0`, Slepian leakage `0.0`.

## 2026-06-02 Automated Review: Step 3,500 Curvature-Capture Replay

The watcher analyzed:

```text
outputs/post_resume_analysis/oai-bpb-graphcg-gfn-03000-curvcap-20260602T054043Z/step-00003500/
```

Training was still alive as an orphaned tmux-launched process even though the
tmux session listing only showed the Codex review session.  I stopped only the
active `03000_curvcap` process group after the review, leaving checkpoints
intact.

Analyzed checkpoint:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00003500.pt
```

Best checkpoint found in the fresh replay window:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00003300.pt
```

### Metric Categorization

| category | count | read |
|---|---:|---|
| as desired | `227` | core likelihood improved over the full 3000-3500 replay; GraphCG, toric phase, persistence, and branch diversity remain active |
| as desired but too weak/slow | `87` | GFlowNet entropy/action diversity are noncollapsed but mostly flat; topology and toric audits are coherent but not yet converting to validation BPB |
| not as desired | `87` | checkpoint finite differences show a post-3300 rebound; validation remains far above the inherited best gate |

Core W&B behavior:

| metric | first median | last median | relative change | recent slope / 1k |
|---|---:|---:|---:|---:|
| train BPB | `3.88766` | `3.68032` | `-5.33%` | `-0.0244` in the automated report; local 3370-3490 slope is negative but EMA had already turned positive after the low |
| train loss | `2.69472` | `2.55101` | `-5.33%` | `-0.0169` |
| total loss | `2.69521` | `2.55137` | `-5.34%` | `-0.0170` |
| GFlowNet loss | `3.07216` | `2.85438` | `-7.09%` | decreases overall, but strongly co-varies with BPB |
| GFlowNet entropy | `2.77242` | `2.77248` | near `0%` | healthy but saturated |
| action diversity | `0.998611` | `0.998708` | near `0%` | healthy but not improving |

Checkpoint finite differences:

| window | train BPB delta |
|---|---:|
| `3050 -> 3100` | `-0.182124` |
| `3100 -> 3150` | `+0.145250` |
| `3150 -> 3200` | `-0.065660` |
| `3200 -> 3250` | `-0.057311` |
| `3250 -> 3300` | `-0.213448` |
| `3300 -> 3350` | `+0.108755` |
| `3350 -> 3400` | `+0.094319` |
| `3400 -> 3450` | `+0.068880` |
| `3450 -> 3500` | `-0.062681` |
| `3500 -> 3550` | `+0.036912` |

The saved step-`3300` checkpoint has train BPB `3.4573897600`, better than
step `3000` (`3.5026167257`), step `3500` (`3.6666615173`), and step `3550`
(`3.7035734657`).  This is the last clean checkpoint before the sustained
positive finite-difference sequence.

### Plot Review

Reviewed:

```text
metrics/core_metric_timeseries.png
metrics/selected_metric_correlations.png
metrics/recent_metric_slopes.png
simplex/reasoning_k_bpb_triangle.png
simplex/reasoning_k_bpb_mst_tetrahedron.png
geometry/trajectories/R2_gss1147-got_math_500k_got_math_trajectory_3d.png
geometry/trajectories/R2_gss1147-got_math_500k_got_math_energy_landscape.png
geometry/trajectories/R2_gss1147-got_math_500k_got_math_phase_energy.png
geometry/trajectories/R2_gss1147-got_math_500k_got_math_toric_phase_simplicial_trajectory.png
geometry/topology/R2_gss1147-got_math_500k_got_math_exact_persistence_morphisms.png
```

Visual conclusions:

1. BPB/loss form a U-shaped local basin: descent through the `3280-3300`
   neighborhood followed by rebound.
2. Spearman correlations show GFlowNet loss positively correlated with BPB and
   gradient norm; entropy and action diversity remain noncollapsed.
3. The energy landscape is thinner and less chaotic than earlier runs, but
   branch excursions remain long.
4. Ramachandran-style phase plots show structured bands rather than uniform
   phase noise, so toric phase memory is active.
5. Persistence-module morphism plots are coherent: boundary residual and
   inclusion violations are zero, edge validity is high, and 2-simplex validity
   is weaker but usable as an audit.
6. The toric/simplicial overlay is visually dense, but the numerical topology
   summaries do not indicate collapse.

### Desired

| diagnostic | value |
|---|---:|
| best fresh checkpoint | step `3300`, train BPB `3.4573897600` |
| exact persistence morphisms | `20.0` |
| topology boundary residual | `0.0` |
| topology inclusion violation | `0.0` |
| directed cycle flux | `~5.55e-18` |
| exact edge validity | `0.979499` |
| directed edge validity | `0.927638` |
| HDBSCAN stability | `0.876248` |
| Slepian concentration / leakage | `1.0 / 0.0` |
| action diversity | `~0.9987` |

### Desired But Too Weak Or Slow

| diagnostic | value | issue |
|---|---:|---|
| mean branch BPB | `4.9914` | not yet competitive with base likelihood |
| best branch BPB | `4.0302` | useful branch exists, but not enough to promote |
| mean answer BPB | `4.8738` | answer-span branch compression remains weak |
| MST efficiency | `0.6961` | coherent but not improving |
| path smoothness | `0.0753` | acceptable but excursions remain long |
| GFlowNet entropy | `2.7725` | saturated, not producing stronger low-BPB branches |
| GraphCG loss | flat around `5.46` | stable steering frame, little recent improvement |

### Not As Desired

| diagnostic | value / trend | read |
|---|---|---|
| validation BPB | controller gate around `5.5358`; inherited best checkpoint still `4.6961` | current basin not validation-promoted |
| checkpoint BPB deltas | positive from `3300 -> 3450` | first derivative turns positive after the local low |
| train BPB EMA | minimum near `3310`, then increases | floor-bounce basin |
| GFlowNet controller weight | relaxed to `0.001` despite tiny phase schedule | auxiliary direction overrode BPB hold |
| active-face margin | `-1.8128` | tropical faces remain low-margin/inverted |
| toric binomial residual | `1.2643` | still audit-scale, not ready for stronger weight |
| toric leaf residual | `1.0020` | phase leaves remain diagnostics rather than loss pressure |
| exact triangle validity | `0.7114` | topology is usable but too weak for promotion |

### Mathematical Explanation

The checkpoint sequence is a one-dimensional trace of the stochastic objective
along the optimizer path.  Around step `3300`, the first finite difference
\(\Delta L_t=L_{t+50}-L_t\) changes sign:

\[
  \Delta L_{3250}<0,\qquad
  \Delta L_{3300}>0,\qquad
  \Delta L_{3350}>0.
\]

This is the signature of crossing a local low in the effective objective

\[
  L_{\rm eff}
  = L_{\rm BPB}
    + \lambda_{\rm GFN}L_{\rm GFN}
    + \lambda_{\rm GraphCG}L_{\rm GraphCG}
    + \lambda_{\rm topo}L_{\rm topo}.
\]

The run did not fail structurally: phase plots, directed filtrations, exact
persistence morphisms, and GraphCG axes remain coherent.  The failure is scalar:
the adaptive controller relaxed \(\lambda_{\rm GFN}\) to `1e-3`, while the
phase schedule intended \(\lambda_{\rm GFN}\le 4e-5\).  Since GFlowNet loss is
positively correlated with BPB in this window, that relaxation increases the
component of the update pointing away from the low-BPB basin.

Geometrically, the trajectory machinery is producing noncollapsed search, but
the active-face margins are small and toric leaf/binomial residuals remain
audit-scale.  Therefore increasing geometric losses would add curvature before
the byte model has stabilized.  The appropriate intervention is not a new
architecture; it is a smaller post-low scalar step and a controller cap.

### Decision

Restart from step `3300`, not step `3500` or `3550`.

Implemented scalar-only changes:

| control | old | new |
|---|---:|---:|
| global shock guard grad norm | `0.32` | `0.30` |
| global shock guard update scale | `0.025` | `0.015` |
| controller `gflownet_loss_max` | `0.001` | `0.00008` |
| controller `gflownet_relax_step` | `0.0005` | `0.00002` |
| controller recovery down step | `0.00075` | `0.00004` |
| `3250-6000` LR multiplier | `0.08` | `0.06` |
| `3250-6000` grad clip | `0.32` | `0.30` |
| `3250-6000` GFlowNet loss weight | `4e-5` | `2e-5` |
| `3250-6000` GFlowNet entropy weight | `1e-5` | `5e-6` |
| `3250-6000` analogy lattice weight | `2e-6` | `1e-6` |
| `3250-6000` contrastive weight | `3e-5` | `2e-5` |

No code path, model architecture, dataset stream, or feature family was
removed.  Random-order autoregressive graph decoding, dense contest weights,
tropical ring/hybrid attention, toric memory, GraphCG axes, embedding-space
GFlowNet graph-of-thought, relative Kolmogorov diagnostics, directed
persistence, Koszul audits, and toric probes remain active.

Next target: fresh step `3800`.

Acceptance criteria:

1. checkpoint train BPB stays below the step-3300 value for at least one saved
   checkpoint, preferably below `3.40`;
2. no two consecutive positive 50-step BPB deltas after the new low;
3. controller-reported GFlowNet loss weight stays below `8e-5`;
4. validation/complexity BPB at the next gate improves over the current
   controller gate (`5.5358`) and preferably over `5.2601` complexity val BPB;
5. GFlowNet entropy/action diversity remain noncollapsed;
6. exact edge validity remains above `0.97`, directed edge validity above
   `0.92`, inclusion violation `0.0`, and Slepian leakage `0.0`.

## 2026-06-01 Automated Review: Step 2,000 Valmix35-r2 Replay

The watcher paused training at a fresh step-`2000` checkpoint from:

```text
toricgt_oai_bpb_valmix35r2_01500_20260601T220127Z
```

Analysis outputs:

```text
outputs/post_resume_analysis/oai-bpb-valmix35r2-01500-20260601T220127Z/step-00002000/
```

### Metric Categorization

| category | count | read |
|---|---:|---|
| as desired | `199` | topology, toric audits, GFlowNet diversity, and most train likelihood terms remain coherent |
| as desired but too weak/slow | `87` | r2 improved a few gates by tiny amounts, but not enough to promote |
| not as desired | `106` | the same 1700-1800 positive finite-difference bounce remains |

Checkpoint and validation comparison:

| run / gate | train BPB | val BPB at 1750 | complexity val BPB | mean branch BPB | mean answer BPB |
|---|---:|---:|---:|---:|---:|
| previous step-2000 retry | `3.5362` | `5.4281` | `5.1358` | `4.9338` | `4.8148` |
| valmix35-r2 step-2000 | `3.5388` | `5.4245` | `5.1340` | `4.9313` | `4.8125` |

This is a real but tiny improvement in validation and branch diagnostics.  It
does not justify the old post-2000 sprint, but it also does not justify another
rollback to 1500: the repeated r2 replay reproduced the same impulse almost
exactly, which suggests the bad mode is tied to data-order curvature around
that interval rather than a random optimizer accident.

Finite-difference summary:

| window | train BPB slope / 1k | second derivative / 1k^2 | interpretation |
|---|---:|---:|---|
| `1501-1700` | `-2.562` | `+13.564` | strong descent but convex upward |
| `1700-1800` | `+3.696` | `-130.653` | same sharp bounce |
| `1800-1990` | `-0.034` | `+2.094` | flat post-bounce shelf |
| `1501-1990` | `-0.374` | `+4.544` | net descent too shallow |

### Plot Review

Reviewed the generated sheets:

```text
plots_review_contact_sheet.png
plots_review_contact_sheet_all_geometry.png
```

Visual conclusions:

1. Core BPB/loss curves still show the same bounce around `1740-1790`.
2. Reasoning/K/BPB simplices and tetrahedra keep branch points near the
   interior rather than pulling them to the low-BPB boundary.
3. 3D graph-of-thought trajectories remain coherent and branching.
4. Ramachandran-style phase plots remain diverse; no phase collapse is visible.
5. Toric phase-winding and embedded torus/simplicial plots render correctly.
6. Exact persistence morphisms, directed filtrations, and noncommutative
   heatmaps remain coherent.

### Desired

| diagnostic | value |
|---|---:|
| exact persistence morphisms | `20.0` |
| topology boundary residual | `0.0` |
| topology inclusion violation | `0.0` |
| directed cycle flux | `~6.63e-18` |
| exact edge validity | `0.974342` |
| directed edge validity | `0.938115` |
| Slepian concentration / leakage | `1.0 / 0.0` |
| GFlowNet action diversity | `0.9943` at end |

### Too Weak Or Slow

| diagnostic | value | issue |
|---|---:|---|
| best branch BPB | `4.0415` | no meaningful improvement |
| mean branch BPB | `4.9313` | still worse than the step-1500 branch audit |
| mean answer BPB | `4.8125` | slight improvement but weak |
| MST efficiency | `0.7024` | stable but not improving |
| HDBSCAN stability | `0.8835` | acceptable but not stronger |
| toric fan entropy | `0.6803` | active but below the previous retry |

### Undesirable

| diagnostic | value / trend | read |
|---|---|---|
| validation BPB | `5.4245` | still far worse than the inherited `4.6961` gate |
| train BPB finite differences | positive bounce `1700-1800` | curvature/impulse persists |
| active-face margin | `-1.7222` | still inverted/low-confidence |
| toric binomial residual | `1.6292` | worse than the previous retry |
| toric leaf residual | `0.9987` | audit-only; not ready for loss pressure |
| exact triangle validity | `0.6878` | worse than desired; keep topology losses off |

### Decision

Continue from the fresh step-`2000` checkpoint, but do not use the historical
`bpb_sprint_2000_3000` phase.  The r2 replay reproduced the same bounce, so a
third rollback to 1500 is unlikely to change the local data-order curvature.
The safer experiment is to carry the slightly improved r2 checkpoint forward
with a low-update hold phase and measure the next validation gate.

Implemented scalar-only changes:

| phase | old | new |
|---|---|---|
| `2000-2500` | `bpb_sprint_2000_3000` | `bpb_postbounce_valmix_hold_2000_2500` |
| `lr_multiplier` | `0.60` | `0.10` |
| `grad_clip_norm` | `0.55` | `0.36` |
| `medium_mix_ratio` | `0.05` | `0.35` |
| `contrastive_loss_weight` | `2.5e-4` | `2e-5` |
| auxiliary geometry losses | off | off |

No architecture changes.  No JEPA.  Random-order autoregressive graph decoding,
tropical ring/hybrid attention, toric memory, dense contest weights,
embedding-space GFlowNet graph-of-thought, relative Kolmogorov diagnostics,
GraphCG axes, directed persistence, Koszul audits, and toric probes remain
unchanged.

Next target: fresh step `2500`.

Acceptance criteria:

1. `val/bpb < 5.4245`, preferably below `5.3285`;
2. `complexity/val/bpb < 5.1340`, preferably below `5.0305`;
3. train BPB slope over `2000-2500` materially negative;
4. mean branch BPB below `4.9275`;
5. no new shock cascade;
6. inclusion violation `0.0`, HDBSCAN stability above `0.88`, Slepian leakage
   `0.0`.

## 2026-06-01 Automated Review: Step 2,000 Damped Early Replay

Training was already paused by the watcher.  No active trainer remained in the
requested tmux session; only the Codex review tmux was present, and the 4090 was
idle.  I reviewed the W&B export, checkpoint metadata, metric summaries, and
generated plot contact sheets for:

```text
outputs/post_resume_analysis/oai-bpb-damped-01500-20260601T203131Z/step-00002000/
```

Analyzed checkpoint:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00002000.pt
```

### Metric Categorization

| category | count | read |
|---|---:|---|
| as desired | `198` | core train likelihood descended over the full 1500-2000 window; topology and toric audit machinery remained coherent |
| as desired but too weak/slow | `83` | descent became shallow after the bounce; geometry exists but does not yet lower validation/test-time branch BPB |
| not as desired | `111` | validation at the only gate worsened, branch BPB worsened, and several complexity/topology variances rose |

Core W&B behavior:

| window | train BPB slope / 1k | BPB second derivative / 1k^2 | interpretation |
|---|---:|---:|---|
| `1501-1700` | `-2.564` | `+13.589` | strong descent but already convex upward |
| `1700-1800` | `+3.687` | `-131.406` | sharp bounce impulse after the local minimum |
| `1800-1990` | `-0.037` | `+1.956` | essentially flat recovery |
| `1501-1990` | `-0.375` | `+4.539` | net improvement, but too slow and too curved |

Validation and checkpoint gates:

| checkpoint / gate | train BPB | validation BPB | branch mean BPB | branch answer BPB |
|---|---:|---:|---:|---:|
| step `1500` analysis | `3.5454` | previous gate only | `4.9275` | `4.8044` |
| W&B eval at `1750` | `3.7296` | `5.4281` | n/a | n/a |
| step `2000` analysis | `3.5362` | n/a | `4.9338` | `4.8148` |

The step-2000 raw train BPB is only `0.0092` lower than step 1500, while the
known validation gate at step 1750 worsened and branch/answer BPB weakened.
This is not a promotion-quality improvement.

### Desired Behavior

The following should be preserved:

| diagnostic | value |
|---|---:|
| checkpoint train BPB | `3.536199` |
| checkpoint train loss | `2.451106` |
| exact persistence morphisms | `20.0` |
| topology boundary residual | `0.0` |
| topology inclusion violation | `0.0` |
| exact edge validity | `0.974807` |
| directed edge validity | `0.933642` |
| HDBSCAN stability | `0.883708` |
| directed cycle flux | `~6.55e-18` |
| Slepian concentration / leakage | `1.0 / 0.0` |
| toric fan-cell entropy | `0.691788` |
| toric binomial residual | `1.603516`, improved but still high |

The plot review confirmed that the 3D graph-of-thought trajectories branch
coherently, Ramachandran-style phase plots remain diverse, toric phase-winding
collections render both the flat irrational winding and torus projection, exact
persistence-module morphism plots are coherent, and the noncommutative heatmaps
show structured directed adjacency rather than numerical collapse.

### Desired But Too Weak Or Slow

| diagnostic | step-2000 value | issue |
|---|---:|---|
| train BPB recent slope | `-0.037 / 1k` | effectively flat after the rebound |
| GFlowNet entropy | `2.7549` | healthy but drifting down |
| GFlowNet action diversity | `0.9941` | high but slightly decreasing |
| best branch BPB | `4.0414` | not better than the step-1500 branch best |
| MST efficiency | `0.7023` | slightly worse than step 1500 |
| path smoothness | `0.0584` | worse than step 1500 |
| HDBSCAN stability | `0.8837` | still usable, but just below the previous gate |

The simplex and tetrahedron plots place most branches near the interior rather
than near the low-BPB boundary.  This means the reasoning geometry is alive but
not yet converted into likelihood gains.  Increasing auxiliary weights would be
premature; the failure is a scalar optimization/curriculum mismatch.

### Not As Desired

| diagnostic | behavior | read |
|---|---|---|
| `val/bpb` | `5.4281` at step `1750` | worse than the previous deterministic gate |
| `complexity/val/bpb` | `5.1358` at step `1750` | complexity probe agrees validation is weak |
| train BPB finite differences | hard negative slope, then hard positive slope, then flat | floor-bounce basin |
| `mean_bpb` branch audit | `4.9338`, worse than step 1500 | test-time branches are not improving |
| `mean_answer_bpb` | `4.8148`, worse than step 1500 | answer spans are not improving |
| toric active-face margin | `-1.7319`, worse than step 1500 | tropical faces are still inverted/low-margin |
| toric leaf residual | `0.9987` | phase leaves remain audit-only |
| DEC conservation/mass residuals | rose relative to step 1500 | topology regularization should stay off |

### Mathematical Explanation

The observed BPB curve is consistent with a stochastic objective
\[
  L_\alpha(\theta)=(1-\alpha)L_{\rm easy}(\theta)+\alpha L_{\rm med}(\theta)
\]
whose local Hessian along the optimizer direction becomes sharply positive
near the 1700-step region.  The 1500-1700 window has the right sign
\(\langle \nabla L_\alpha, \Delta\theta\rangle<0\), but the positive quadratic
term
\[
  \tfrac12\Delta\theta^\top H_\alpha\Delta\theta
\]
grows enough to reverse the finite difference by 1700-1800.  After that, the
guarded updates reduce the damage but leave the run on a nearly flat shelf.

The geometry diagnostics support this interpretation: filtrations, persistence
morphisms, toric winding, and Slepian concentration remain coherent, so the
architecture did not fail.  The bad behavior is the validation-aligned gradient
not being strong enough relative to curvature and high-BPB microbatch impulses.

### Decision

Do not continue from step `2000`.  Restart from the last checkpoint before the
first strong positive derivative:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00001500.pt
```

Implemented scalar-only changes:

| control | previous 1500-2000 retry | new 1500-2000 retry |
|---|---:|---:|
| phase name | `bpb_curvature_damped_1500_2000` | `bpb_valmix_curvature_damped_1500_2000` |
| `lr_multiplier` | `0.18` | `0.12` |
| `grad_clip_norm` | `0.45` | `0.40` |
| `medium_mix_ratio` | `0.18` | `0.35` |
| `contrastive_loss_weight` | `5e-5` | `3e-5` |
| `shock_guard_grad_norm` | `0.42` | `0.36` |
| `shock_guard_update_scale` | `0.06` | `0.04` |
| robust guard ratio/delta | `1.10 / 0.16` | `1.08 / 0.14` |
| robust guard min scale | `0.45` | `0.40` |

No architecture changes.  No JEPA.  Random-order autoregressive graph decoding,
tropical ring/hybrid attention, toric memory, dense contest weights,
embedding-space GFlowNet graph-of-thought, relative Kolmogorov diagnostics,
GraphCG axes, directed persistence, Koszul audits, and toric probes remain
unchanged.  Auxiliary GFlowNet/topology/toric/QAT losses remain diagnostic-only
for this gate.

Next target: fresh step `2000`.

Acceptance criteria:

1. `val/bpb < 5.4281`, preferably below the prior `5.3285` gate;
2. `complexity/val/bpb < 5.1358`, preferably below `5.0305`;
3. train BPB finite-difference slope after 1700 non-positive or materially
   smaller than `+3.687 / 1k`;
4. no repeated shock-guard cascade;
5. mean branch BPB below `4.9275` and best branch BPB below `4.0414`;
6. HDBSCAN stability above `0.88`, inclusion violation `0.0`, Slepian leakage
   `0.0`;
7. toric active-face margin no worse than `-1.7017`.

## 2026-06-01 Automated Review: Step 40,000 Medium-Mix 8% Retry

Training was paused by the watcher at a fresh step-`40000` checkpoint from
`toricgt_oai_bpb_valmix8_39500_20260601T143436Z`.  The training tmux was no
longer active and the GPU was idle at review time, so this review makes a new
restart decision from the checkpoint and analysis evidence.

Analyzed checkpoint:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00040000.pt
```

Analysis outputs:

```text
outputs/post_resume_analysis/oai-bpb-valmix8-39500-20260601T143436Z/step-00040000/
```

### Metric Categorization

Automatic categories were unchanged at the aggregate level:

| category | count |
|---|---:|
| as desired | `304` |
| as desired but too weak or slow | `114` |
| not as desired | `201` |

Fresh local train BPB stayed in the acceptable recovery band:

| step band | median BPB | mean BPB | max BPB |
|---:|---:|---:|---:|
| `39500` | `3.6330` | `3.6699` | `4.301` |
| `39550` | `3.6570` | `3.7092` | `4.313` |
| `39600` | `3.6195` | `3.6590` | `4.320` |
| `39650` | `3.6285` | `3.6671` | `4.294` |
| `39700` | `3.6465` | `3.7205` | `4.340` |
| `39750` | `3.6345` | `3.6587` | `4.292` |
| `39800` | `3.5590` | `3.6193` | `4.354` |
| `39850` | `3.6855` | `3.7288` | `4.382` |
| `39900` | `3.6205` | `3.6582` | `4.393` |
| `39950` | `3.5830` | `3.6747` | `4.478` |

Validation:

| step | primary val BPB | complexity val BPB | controller train BPB EMA |
|---:|---:|---:|---:|
| `39500` | `4.933094` | previous gate | `3.663048` before restart |
| `39750` | `4.937515` | `4.824397` | n/a |
| `40000` | `4.939186` | `4.825921` | `3.644690` |

The medium-mix sweep is monotone in the useful direction but too weak:

| retry | controls | primary val BPB | complexity val BPB |
|---|---|---:|---:|
| easy-only | `lr=0.88`, medium `0.00` | `4.946108` | `4.832355` |
| 3% medium | `lr=0.52`, medium `0.03` | `4.941543` | `4.827858` |
| 8% medium | `lr=0.36`, medium `0.08` | `4.939186` | `4.825921` |

The derivative is still positive relative to the 39.5k gate, but the sequence
shows the direction correction is real.

### Desired Behavior

The train-stream BPB shelf is under control, and topology/toric diagnostics
remain valid:

| diagnostic | value |
|---|---:|
| checkpoint train BPB | `3.626682` |
| checkpoint train loss | `2.513824` |
| exact persistence morphisms | `20.0` |
| topology boundary residual | `0.0` |
| topology inclusion violation | `0.0` |
| exact edge validity | `0.979229` |
| directed edge validity | `0.949034` |
| HDBSCAN stability | `0.938070` |
| directed cycle flux | `~4.65e-18` |
| Slepian concentration | `1.0` |
| Slepian leakage | `0.0` |
| occupied fan cells | `12.25` |

Contact sheets were generated and reviewed under:

```text
outputs/post_resume_analysis/oai-bpb-valmix8-39500-20260601T143436Z/step-00040000/plot_contact_sheets/
```

The plot review shows coherent noncommutative heatmaps, radius filtrations,
exact persistence morphisms, toric winding, Slepian/PSWF phase plots,
Ramachandran-style phase clouds, 3D GoT branches, and energy landscapes.
Nothing points to a failed geometry module.

### Desired But Too Weak Or Slow

| diagnostic | value | trend |
|---|---:|---|
| mean branch BPB | `4.750429` | slightly better than 3% retry |
| best branch BPB | `4.029984` | still weak |
| mean answer BPB | `4.630563` | slight improvement |
| best answer BPB | `3.615508` | not a new best |
| MST efficiency | `0.498475` | flat |
| path smoothness | `0.086943` | better than 3% retry |
| active-face entropy | `0.838499` | active |
| active-face margin | `-0.972900` | still negative |
| toric binomial residual | `0.506851` | undesirable |
| toric leaf residual | `0.998823` | undesirable |

The simplex plots still place most branch points away from low-BPB vertices.
The model is preserving exploration geometry but has not turned it into lower
deterministic validation BPB.

### Mathematical Explanation

Let
\[
  g_\alpha=(1-\alpha)g_e+\alpha g_m
\]
be the mixed recovery gradient.  The observed sequence is consistent with
\[
  \langle \nabla L_v,g_{0.08}\rangle
  <
  \langle \nabla L_v,g_{0.03}\rangle
  <
  \langle \nabla L_v,g_{0}\rangle
\]
in the bad direction, but still not crossing zero.  Increasing the medium
component is therefore justified, while lowering the learning-rate multiplier
slightly keeps the high-BPB row impulses from dominating the optimizer state.

The second key point is that geometry metrics are approximately invariant under
these scalar changes: HDBSCAN stability, exact morphisms, Slepian leakage, and
cycle flux remain stable.  This means the scalar curriculum can be adjusted
without disrupting the ToricGT-specific structure.

### Decision

Do not continue from `40000`.  Restart from:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00039500.pt
```

Implemented scalar-only changes:

| control | old | new | reason |
|---|---:|---:|---|
| `phase/lr_multiplier` | `0.36` | `0.30` | keep update bounded while increasing medium exposure |
| `phase/medium_mix_ratio` | `0.08` | `0.18` | test whether validation derivative can cross zero |
| `phase/hard_mix_ratio` | `0.00` | `0.00` | unchanged |
| `phase/complex_mix_ratio` | `0.00` | `0.00` | unchanged |

Architecture remains unchanged.  No JEPA.  Keep random-order autoregressive
decoding, tropical ring/hybrid attention, toric memory, dense contest weights,
GFlowNet graph-of-thought, Kolmogorov diagnostics, directed persistence,
GraphCG axes, Koszul audits, and toric probes.

Next target: fresh `40000`.

Acceptance criteria:

1. `val/bpb <= 4.933094`;
2. if primary val only approaches the gate, require `complexity/val/bpb <
   4.825921` and lower branch BPB;
3. train median BPB below `3.85` and no repeated impulse above `4.5`;
4. topology inclusion violation `0.0`, HDBSCAN stability above `0.90`, Slepian
   leakage `0.0`.

## 2026-06-01 Automated Review: Step 40,000 Low-LR Medium-Mix Retry

Training was paused by the watcher at a fresh step-`40000` checkpoint from the
run `toricgt_oai_bpb_valmix_39500_20260601T130711Z`.  The GPU was idle at
review time and no training session remained active, so the decision below is a
restart decision, not an interruption of live optimization.

Analyzed checkpoint:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00040000.pt
```

Analysis outputs:

```text
outputs/post_resume_analysis/oai-bpb-valmix-39500-20260601T130711Z/step-00040000/
```

The W&B export is still partially stale because the same run id was resumed
from a lower local step.  Fresh values for the retry are taken from
`logs/training/oai-bpb-valmix-39500-20260601T130711Z.log`.

### Metric Categorization

The automatic audit reported the same broad category counts as the prior
seed-rotate review:

| category | count |
|---|---:|
| as desired | `304` |
| as desired but too weak or slow | `114` |
| not as desired | `201` |

Local train BPB stayed controlled throughout the 39.5k--40k window:

| step band | median train BPB | mean train BPB | max train BPB |
|---:|---:|---:|---:|
| `39500` | `3.6320` | `3.6451` | `4.293` |
| `39550` | `3.6340` | `3.6643` | `4.302` |
| `39600` | `3.6240` | `3.6368` | `4.296` |
| `39650` | `3.6180` | `3.6345` | `4.296` |
| `39700` | `3.6400` | `3.6688` | `4.311` |
| `39750` | `3.6245` | `3.6278` | `4.322` |
| `39800` | `3.5605` | `3.5971` | `4.323` |
| `39850` | `3.6800` | `3.6795` | `4.250` |
| `39900` | `3.5630` | `3.5935` | `4.287` |
| `39950` | `3.6160` | `3.6382` | `4.296` |

The validation gate did not recover:

| step | primary val BPB | complexity val BPB | controller train BPB EMA |
|---:|---:|---:|---:|
| `39500` | `4.933094` | previous gate | `3.682681` before restart |
| `39750` | `4.939244` | `4.827275` | n/a |
| `40000` | `4.941543` | `4.827858` | `3.663048` |

Compared with the easy-only retry, the low-LR 3% medium mix helped:
`val/bpb` improved from `4.946108` to `4.941543` and
`complexity/val/bpb` improved from `4.832355` to `4.827858`.  It still failed
the promotion gate because the primary validation derivative remains positive
relative to step `39500`.

### Desired Behavior

The following should be preserved:

| diagnostic | value | read |
|---|---:|---|
| train checkpoint BPB | `3.630631` | stable likelihood recovery |
| train checkpoint loss | `2.516562` | stable |
| exact persistence morphisms | `20.0` | desired |
| topology boundary residual | `0.0` | desired |
| topology inclusion violation | `0.0` | desired |
| exact edge validity | `0.979004` | desired |
| directed edge validity | `0.951329` | desired |
| HDBSCAN stability | `0.938314` | desired |
| directed cycle flux | `~5.13e-18` | desired |
| Slepian leakage | `0.0` | desired |
| Slepian concentration | `1.0` | desired |
| toric occupied fan cells | `12.0` | active |

The generated plots remained visually valid.  Contact sheets were generated
locally for review under:

```text
outputs/post_resume_analysis/oai-bpb-valmix-39500-20260601T130711Z/step-00040000/plot_contact_sheets/
```

The 3D graph-of-thought trajectories, Ramachandran-style phase plots, energy
landscapes, irrational toric winding collections, noncommutative heatmaps,
radius filtrations, exact persistence-module morphisms, and toric/Slepian
audits are all nonblank and structured.  The topology/geometry stack should
not be disabled.

### Desired But Too Weak Or Slow

| diagnostic | value | read |
|---|---:|---|
| mean branch BPB | `4.751865` | slightly better than easy-only but still weak |
| best branch BPB | `4.030099` | worse than prior best |
| mean answer BPB | `4.632189` | weak |
| best answer BPB | `3.619655` | weaker than earlier gates |
| mean MST efficiency | `0.498363` | below useful threshold |
| mean path smoothness | `0.089855` | not improving enough |
| active-face entropy | `0.840971` | active, not decisive |
| active-face margin | `-0.963135` | still negative |
| toric phase recurrence | `0.028038` | present, low |
| transport entropy | `0.993750` | healthy but not yet linked to BPB |

The simplex/tetrahedron sheets show branches clustered away from the low-BPB
vertices.  This means the inference-time geometry is expressive but not yet
aligned strongly enough with the byte-likelihood objective.

### Undesirable

| metric | behavior |
|---|---|
| primary validation BPB | `4.933094 -> 4.939244 -> 4.941543` |
| primary validation derivative | still positive over both 250-step intervals |
| complexity validation | better than easy-only, still not better than earlier gates |
| toric binomial residual | `0.505173`, worse than desired |
| toric phase leaf residual | `0.998815`, still effectively uncorrected |
| geometry best BPB | `4.030099`, not a new low |

The mathematical read is that the 3% medium mixture changed the validation
directional derivative in the right direction but not enough.  If the recovery
gradient is
\[
  g_\alpha=(1-\alpha)g_e+\alpha g_m,
\]
where \(g_e\) is the easy-stream gradient and \(g_m\) is the medium-row
gradient, the observed validation drift gives
\[
  -\eta \langle \nabla L_v,g_{0.03}\rangle > 0,
\]
but the magnitude is smaller than for \(g_0\).  The next intervention should
therefore increase \(\alpha\) modestly and reduce \(\eta\) enough that the
hard/complex shelf remains absent.  This is a scalar control change, not an
architecture change.

### Decision

Do not continue from step `40000`.  Restart again from the last checkpoint
before the validation derivative turned positive:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00039500.pt
```

Implemented the smallest scalar correction:

| control | old | new | reason |
|---|---:|---:|---|
| `phase/lr_multiplier` | `0.52` | `0.36` | reduce validation drift and floor-bounce step size |
| `phase/medium_mix_ratio` | `0.03` | `0.08` | make the recovery gradient more validation-aligned |
| `phase/hard_mix_ratio` | `0.00` | `0.00` | avoid hard-row shelf |
| `phase/complex_mix_ratio` | `0.00` | `0.00` | avoid graph-projection shelf |

No JEPA and no architecture changes.  Keep random-order autoregressive graph
decoding, tropical ring/hybrid attention, toric memory, dense contest weights,
embedding-space GFlowNet graph-of-thought, Kolmogorov diagnostics,
GraphCG-style directions, directed persistence, Koszul audits, toric probes,
and PSWF/Slepian torus audits.

Next review target: `40000`.

Acceptance criteria:

1. primary `val/bpb <= 4.933094`;
2. if primary val only ties, require `complexity/val/bpb < 4.827858`;
3. train BPB median remains below `3.8` without repeated `>4.3` impulses;
4. mean branch BPB below `4.751865` or best branch BPB below `4.030099`;
5. topology inclusion violation `0.0`, HDBSCAN stability above `0.90`, and
   Slepian leakage `0.0`.

## 2026-06-01 Automated Review: Step 40,000 Seed-Rotate Retry

Training was paused by the watcher before this review.  The requested training
session `toricgt_oai_bpb_seedrotate_39500_20260601T114442Z` was no longer
active and the GPU was idle, so the review made an explicit restart decision
from the checkpoint evidence rather than letting a stale tmux state stand.

Analyzed checkpoint:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00040000.pt
```

W&B run:

```text
https://wandb.ai/amelie-iska-math/toricgt-parameter-golf/runs/oai37500r2-20260601T024939Z
```

Analysis outputs:

```text
outputs/post_resume_analysis/oai-bpb-seedrotate-39500-20260601T114442Z/step-00040000/
```

The W&B export in the analysis folder still contains stale pre-resume history
around step `39990`; the fresh segment is therefore read from
`logs/training/oai-bpb-seedrotate-39500-20260601T114442Z.log` and from the
checkpoint-local analysis summaries.

### Metric Categorization

The automatic audit reported:

| category | count |
|---|---:|
| as desired | `304` |
| as desired but too weak or slow | `114` |
| not as desired | `201` |

Fresh local train BPB medians were stable and much healthier than the previous
high-BPB shelf:

| step band | median train BPB |
|---:|---:|
| `39500` | `3.6305` |
| `39550` | `3.6295` |
| `39600` | `3.6240` |
| `39650` | `3.6170` |
| `39700` | `3.6370` |
| `39750` | `3.6120` |
| `39800` | `3.5695` |
| `39850` | `3.6585` |
| `39900` | `3.5590` |
| `39950` | `3.6225` |

Checkpoint-local finite differences show fast train-stream recovery followed
by deceleration:

| checkpoint | train BPB |
|---:|---:|
| `39500` | `5.000765` |
| `39750` | `3.606788` |
| `40000` | `3.560902` |

For equal 250-step spacing, the first difference changes from about `-1.394`
to `-0.0459`.  The second difference is strongly positive, so the model is
already approaching a local train-stream floor by the end of the retry.

The decisive problem is validation:

| step | primary val BPB | complexity val BPB | controller train BPB EMA |
|---:|---:|---:|---:|
| `39500` | `4.933094` | checkpoint gate | n/a |
| `39750` | `4.942227` | `4.831072` | n/a |
| `40000` | `4.946108` | `4.832355` | `3.682681` |

Thus the seed rotation fixed the raw training shelf, but easy-only recovery
moved the validation distribution in the wrong direction.

### Desired Behavior

The branch/topology machinery remains coherent and should be preserved:

| diagnostic | value | read |
|---|---:|---|
| train-stream BPB band | `~3.56--3.66` | good recovery |
| GFlowNet loss trend | decreasing in W&B export | desired |
| exact persistence morphisms | `20.0` | desired |
| topology boundary residual | `0.0` | desired |
| topology inclusion violation | `0.0` | desired |
| exact edge validity | `0.980684` | desired |
| directed edge validity | `0.951687` | desired |
| HDBSCAN stability | `0.938531` | desired |
| Slepian leakage | `0.0` | desired |
| directed cycle flux | `~6.5e-18` | desired |
| occupied fan cells | `11.75` | active |

The contact sheets confirm that the 3D graph-of-thought trajectories,
Ramachandran-style phase plots, energy landscapes, irrational toric winding
collections, noncommutative simplex heatmaps, radius filtrations, exact
persistence-module morphism plots, and toric/Slepian audits render and carry
nontrivial structure.  There is no evidence that JEPA-like or architecture
level changes are needed here.

### Desired But Too Weak Or Slow

| diagnostic | value | read |
|---|---:|---|
| mean branch BPB | `4.754877` | weak; worse than prior 40k audit |
| best branch BPB | `4.029133` | weak; worse than prior 40k audit |
| mean answer BPB | `4.632791` | weak |
| best answer BPB | `3.612397` | usable but not improving |
| MST efficiency | `0.500401` | modest |
| path smoothness | `0.092219` | worse than prior 40k audit |
| toric active-face entropy | `0.840383` | active but not decisive |
| toric active-face margin | `-0.959147` | improved but still negative |

The simplex/tetrahedron plots place most branches away from the low-BPB vertex.
The model is exploring richly, but inference-time geometry is not yet pulling
branches into reliably better terminal basins.

### Undesirable

| metric | behavior |
|---|---|
| primary `val/bpb` | worsened `4.933094 -> 4.942227 -> 4.946108` |
| `complexity/val/bpb` | worsened to `4.832355` |
| toric memory entropy | still decaying in W&B export |
| toric binomial residual | `0.493601`, not improving |
| toric phase leaf residual | `1.000005`, still effectively uncorrected |
| train second difference | positive; train-stream descent already decelerating |

The mathematical explanation is a distribution-gradient mismatch.  Let
\(P_e\) be the easy stream and \(P_v\) the validation stream.  The retry took
steps approximately in direction
\[
  g_e=\nabla_\theta \mathbb{E}_{x\sim P_e}[-\log p_\theta(x)].
\]
The training loss decreased, so \(\langle \nabla L_e,g_e\rangle>0\) in the
descent convention.  Validation worsened, which implies the same update has
positive first-order validation drift:
\[
  \Delta L_v \approx -\eta\langle \nabla L_v,g_e\rangle > 0.
\]
The second-difference deceleration then says increasing step count alone is
unlikely to fix the drift; the update direction needs a small component from a
validation-like row distribution, and the step size should be reduced so the
auxiliary geometry already present is not overwritten by easy-stream fitting.

### Plot Review

Reviewed all nine contact sheets:

```text
outputs/post_resume_analysis/oai-bpb-seedrotate-39500-20260601T114442Z/step-00040000/plot_contact_sheets/contact_sheet_01.png
...
outputs/post_resume_analysis/oai-bpb-seedrotate-39500-20260601T114442Z/step-00040000/plot_contact_sheets/contact_sheet_09.png
```

Specific reads:

| plot family | behavior |
|---|---|
| simplex/tetrahedra | points stay interior/right; not pulled to low-BPB vertex |
| 3D trajectories | diverse and nonblank, with terminal contacts but no stable low-energy basin |
| Ramachandran phase plots | phase activity present, broad rather than collapsed |
| energy landscapes | rugged; no consistent descent valley |
| toric winding collections | irrational winding and embedded torus projections present |
| noncommutative heatmaps | directed adjacency and skew structure visible |
| persistence morphisms | exact maps and inclusion checks numerically sane |
| Slepian/PSWF audits | concentration `1.0`, leakage `0.0` |

### Decision

Do not continue from step `40000`.  Restart from the last checkpoint that still
owned the primary validation gate:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00039500.pt
```

Implemented the smallest scalar correction:

| control | old | new | reason |
|---|---:|---:|---|
| `phase/lr_multiplier` | `0.88` | `0.52` | reduce validation drift and train-floor bounce |
| `phase/medium_mix_ratio` | `0.00` | `0.03` | add a small validation-like component without reintroducing the hard shelf |
| `phase/hard_mix_ratio` | `0.00` | `0.00` | unchanged |
| `phase/complex_mix_ratio` | `0.00` | `0.00` | unchanged |

No architecture change is made.  Random-order autoregressive graph decoding,
tropical ring/hybrid attention, toric memory, dense contest weights,
embedding-space GFlowNet graph-of-thought, Kolmogorov diagnostics,
GraphCG-style directions, directed persistence, Koszul audits, and toric
geometry probes all remain active or diagnostic as before.

Next review target: `40000`.

Acceptance criteria for the next 500-step gate:

1. `val/bpb <= 4.933094`, or at minimum no worse than `4.936` with clear
   complexity improvement;
2. `complexity/val/bpb <= 4.832355` and preferably below `4.824`;
3. train-stream BPB median stays below `3.8`;
4. mean branch BPB below `4.754877` or best branch BPB below `4.029133`;
5. topology inclusion violation remains `0.0`, HDBSCAN stability above `0.90`,
   and Slepian leakage `0.0`.

## 2026-06-01 Automated Review: Step 40,000

Training was paused by the watcher before this review.  No active training tmux
session remained and the GPU was idle, so the correct action was an explicit
restart/continue decision.

Analyzed checkpoint:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00040000.pt
```

W&B run:

```text
https://wandb.ai/amelie-iska-math/toricgt-parameter-golf/runs/oai37500r2-20260601T024939Z
```

Analysis outputs:

```text
outputs/post_resume_analysis/oai-bpb-continue-39500-20260601T094346Z/step-00040000/
```

Reviewed plots included all generated contact sheets plus the full core metric
time-series plot, simplex/tetrahedron panels, 3D graph-of-thought trajectories,
energy landscapes, Ramachandran-style phase plots, irrational toric phase
winding collections, toric shadow audits, Slepian/PSWF torus audits, directed
filtration panels, noncommutative heatmaps, and exact persistence-module
morphism/commutative-algebra audit panels.

### Metric Categorization

The automatic audit reported:

| category | count |
|---|---:|
| as desired | `322` |
| as desired but too weak or slow | `109` |
| not as desired | `185` |

Primary metric behavior:

| metric | first median | last median | recent slope / 1k | category |
|---|---:|---:|---:|---|
| `train/bpb` | `3.8575` | `4.1416` | `+0.6453` | not desired |
| `train/loss` | `2.6738` | `2.8708` | `+0.4473` | not desired |
| `train/loss_ema` | `2.6737` | `2.8438` | `+0.4060` | not desired |
| `val/bpb` | `4.9348` | `4.9416` | `-0.0006` | weak/slow |
| `complexity/train/bpb` | `4.1096` | `3.5673` | `-0.1566` | desired |
| `complexity/val/bpb` | `4.8233` | `4.8308` | `-0.0017` | weak/slow |
| `train/gflownet_loss` | `3.4943` | `2.4157` | `-1.0375` | desired |
| `train/toric_memory_entropy` | `0.3546` | `0.2964` | `-0.0180` | not desired |

The local training log is more current than the W&B history export, which ends
near step `39990`.  The final validation event in the log gives:

| step | primary val BPB | complexity val BPB | controller train BPB EMA |
|---:|---:|---:|---:|
| `39750` | `4.939299` | `4.832129` | `4.251428` |
| `40000` | `4.935114` | `4.820252` | `4.495193` |

This is mixed.  Step `40000` recovers validation relative to `39750` and
improves complexity validation, but it does not beat the step-`39500` primary
validation gate (`4.933111`).  The retained checkpoint itself is not
promotable: its raw checkpoint BPB is `5.063979` and train loss is `3.510083`.

### Desired Behavior

The geometry and topology diagnostics are coherent and should be preserved.
Mean branch BPB improved slightly to `4.74385`, best branch BPB to `4.020996`,
mean answer BPB to `4.62197`, and best answer BPB to `3.60942`.  Path
smoothness improved to `0.08425`, HDBSCAN stability stayed high at `0.93783`,
directed cycle flux remained numerically zero, inclusion violation stayed
`0.0`, exact persistence morphisms computed stayed at `20`, and Slepian
leakage remained `0.0`.

The trajectory/contact-sheet review supports this reading.  The 3D
graph-of-thought branches remain noncollapsed; energy landscapes have visible
low-energy basins; Ramachandran-style phase plots show structured phase
clusters rather than uniform noise; toric winding plots show dense projected
Kronecker-style motion with local simplicial overlays; and exact
persistence-module panels show valid H0/H1 rank structure and bounded directed
edge validity.  These are useful reasoning-geometry signals, not enough to
override the primary BPB gate.

### Desired But Too Weak Or Slow

Validation recovery is too small.  The log shows a `39750 -> 40000` primary
validation BPB improvement of about `0.00418`, but the step-`39500` validation
gate remains better.  Complexity validation did improve from the step-`39500`
area (`~4.8221`) to `4.82025`, and the simplex plots place some branches closer
to the low-BPB vertex, but branch selection is still weak: most branches sit in
the reasoning/K-helper interior rather than at the low-BPB boundary.

The GFlowNet policy remains useful as an analysis and test-time-scaling
diagnostic, but under the recovery phase its training weights are zero.  The
near-uniform entropy/action-diversity traces therefore represent preserved
exploration capacity, not a terminal-law improvement:

\[
  P_F(x) \not\approx R(x)/Z
\]

inside the likelihood-only recovery window.

### Undesirable Behavior

The raw training shelf is the dominant failure.  The first roughly 60 resumed
steps from `39500` were healthy (`BPB \approx 3.5--3.7`), then a deterministic
high-entropy row-group shelf began and persisted:

| step band | median train BPB | qualitative behavior |
|---:|---:|---|
| `39500--39599` | `~3.61` | healthy early resume |
| `39600--39699` | `~4.19` | shelf begins |
| `39700--39799` | `~4.35` | shelf persists |
| `39800--39899` | `~4.82` | high-BPB regime |
| `39900--39999` | `~4.92` | high-BPB regime |
| `40000+` | `~5.10` | not promotable |

This pattern is too structured to explain as ordinary stochastic noise.  Let
\(P_t\) denote the empirical row-group distribution seen at step \(t\).  The
retry used the same stream origin and only a `500`-step burn-in, which placed
the resumed optimizer back onto a diagnosed high-entropy segment.  The observed
loss is therefore a mixture-shift effect

\[
  \mathbb{E}_{(x,y)\sim P_t}[-\log p_\theta(y\mid x)]
\]

rather than evidence that the architecture lost capacity.  The model can still
score fixed validation and complexity probes slightly better, but the training
stream is measuring a harder local data slice and the optimizer is spending
updates in that shelf.

Toric supervision is still not reliable as a stronger recovery signal:
active-face margin is negative (`-1.0518`), shadow minimum margin is tiny
(`2.42e-4`), and phase-leaf residual is near `1.0`.  The tropical stability
condition

\[
  \Delta_{ic} > 2(\epsilon_S+\epsilon_V)
\]

is not satisfied, so fan/bend/phase losses should remain diagnostic-only until
the byte objective leaves the shelf.

### Decision

Restart from the step-`39500` checkpoint, not from step `40000`:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00039500.pt
```

Initial scalar hypothesis rejected:

- `stream_burnin_steps=1000` landed immediately on the same bad shelf
  (`first live BPB \approx 5.04`), so that retry was stopped before it could
  consume the full window.
- One-step probes over burn-ins `{0,250,500,750,1000,1250,1500,1750,2000}`
  showed that the old `500` burn-in was best for the first batch (`3.532`) but
  was already known to hit the shelf after roughly 60 steps; burn-ins
  `750--1750` were not acceptable.
- Seed-rotation probes with zero burn-in showed better alternatives.  A
  20-step CUDA probe from the same step-`39500` checkpoint gave:

| stream origin | burn-in | 20-step median BPB | max BPB | last BPB |
|---:|---:|---:|---:|---:|
| `37750` | `0` | `3.888` | `4.038` | `3.878` |
| `39500` | `0` | `3.6225` | `3.653` | `3.648` |
| `40000` | `0` | `3.6710` | `3.819` | `3.570` |
| `50000` | `0` | `3.6635` | `3.722` | `3.608` |

The revised minimal scalar update is therefore:

1. rotate the stream origin to `39500`;
2. set `stream_burnin_steps` to `0`;
3. extend the likelihood-only recovery and robust/shock guards through
   `40750`;
4. keep medium/hard/complex mixture ratios at `0.0`;
5. keep random-order autoregressive decoding, tropical ring/hybrid attention,
   toric memory, dense contest weights, GFlowNet diagnostics,
   GraphCG/analogy/persistence/Koszul diagnostics, and relative-K monitoring
   unchanged.

Acceptance criteria for the next step-`40000` review:

1. primary validation BPB beats the step-`39500` gate: `val/bpb < 4.933111`;
2. complexity validation remains at least as good as this review:
   `complexity/val/bpb <= 4.820252`;
3. step-band median raw train BPB over the resumed `39500--40000` interval is
   below `4.0`, or at minimum the final checkpoint BPB is below `4.5`;
4. branch mean BPB stays below `4.74385` or best branch BPB beats `4.020996`;
5. topology invariants remain intact: inclusion violation `0.0`, boundary
   residual `0.0`, HDBSCAN stability above `0.90`, directed cycle flux near
   zero, and Slepian leakage `0.0`.

## 2026-06-01 Automated Review: Step 38,500

Training was paused by the watcher before this review.  The GPU was idle after
analysis, so the run needed an explicit continue/restart decision.

Analyzed checkpoint:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00038500.pt
```

W&B run:

```text
https://wandb.ai/amelie-iska-math/toricgt-parameter-golf/runs/oai37500r2-20260601T024939Z
```

Analysis outputs:

```text
outputs/post_resume_analysis/oai-bpb-continue-38000-20260601T041332Z/step-00038500/
```

Reviewed plots included the metric time series, recent slopes, selected
correlations, reasoning/K-helper/BPB simplex, toric-GFlowNet-BPB tetrahedron,
3D graph-of-thought trajectories, energy landscapes, toric shadow audits, and
exact directed persistence-module morphism panels.

### Metric Categorization

The automatic audit reported:

| category | count |
|---|---:|
| as desired | `353` |
| as desired but too weak or slow | `123` |
| not as desired | `140` |

Core W&B history remains favorable:

| metric | first median | last median | recent slope / 1k | category |
|---|---:|---:|---:|---|
| `train/bpb` | `3.8531` | `3.5668` | `-1.3253` | desired |
| `train/loss` | `2.6707` | `2.4723` | `-0.9186` | desired |
| `train/loss_ema` | `2.6652` | `2.5003` | `-0.7551` | desired |
| `train/total_loss` | `2.6708` | `2.4724` | `-0.9187` | desired |

Retained checkpoint metrics show a noisy but net-improving segment:

| checkpoint | train BPB | train loss |
|---:|---:|---:|
| `37500` | `4.907855` | `3.401866` |
| `37750` | `3.829687` | `2.654536` |
| `38000` | `3.633273` | `2.518393` |
| `38250` | `3.751553` | `2.600379` |
| `38500` | `3.592812` | `2.490348` |

The `38250` checkpoint is an upward impulse, but `38500` recovers below
`38000`.  The equal-interval differences around this impulse are

\[
  \Delta_{38000\to38250}=+0.118280,\qquad
  \Delta_{38250\to38500}=-0.158741.
\]

That is not a persistent floor-bounce basin.  It is a recoverable stochastic
stream impulse under the current shock guard.  The 500-step net difference is

\[
  \Delta_{38000\to38500}=-0.040461,
\]

which is slower than the prior recovery, but still negative.

### Desired Behavior

The byte objective keeps descending in the W&B median and in the retained
checkpoint.  The robust microbatch guard improved materially:

| metric | 38,000 summary | 38,500 summary |
|---|---:|---:|
| `train/robust_micro_loss_guard_fraction` | `0.1875` | `0.0` |
| `train/robust_micro_loss_guard_scale` | `0.9930` | `1.0` |
| `train/grad_norm` | `0.4223` | `0.1803` |

This is the behavior the recovery controller was designed to produce: preserve
the high-entropy examples in the metric stream while preventing them from
dominating optimizer updates.  In statistical terms, the guard is behaving like
a Huberized stochastic-gradient estimator: the logged loss remains the true
cross-entropy sample, while the influence function for outlying microbatches is
bounded during the active recovery window.

Complexity validation improved slightly: `complexity/val/bpb` moved from
`4.8233` to `4.8226`, and validation NCD diagnostics improved in the plotted
window.  This is small, but it means the BPB recovery has not broken the
relative-K/helper-string monitors.

Directed topology remains a coherent audit layer.  Inclusion violation is
`0.0`; boundary residual is `0.0`; exact morphisms computed remains `20.0`;
edge validity is `0.9768`; directed edge validity is `0.9546`; triangle
validity improved slightly to `0.8207`; HDBSCAN stability is `0.9448`; and
Slepian leakage is `0.0`.  These are exactly the invariants that must remain
stable while the byte model is repaired.

### Desired But Too Weak Or Slow

Validation and branch-level inference-time scaling are not improving quickly
enough:

| diagnostic | step 38,000 | step 38,500 | read |
|---|---:|---:|---|
| `val/bpb` | `4.9348` | `4.9384` | slight regression, still under the `0.01` guard |
| `val/score_first_bpb` | `4.9328` | `4.9372` | slight regression |
| `complexity/val/bpb` | `4.8233` | `4.8226` | slight improvement |
| mean branch BPB | `4.7592` | `4.7638` | weaker |
| best branch BPB | `4.0135` | `4.0180` | weaker |
| mean answer BPB | `4.6467` | `4.6512` | weaker |
| best answer BPB | `3.6025` | `3.6060` | nearly flat |
| MST efficiency | `0.5082` | `0.5106` | slightly better |
| path smoothness | `0.0886` | `0.0897` | slightly rougher |

The simplex plot still places one branch near the low-BPB vertex, while most
branches remain in the reasoning/K-helper interior.  The toric/GFlowNet/BPB
tetrahedron remains clustered near toric entropy and exploration rather than
the low-BPB vertex.  This means branch diversity exists, but the current
diagnostic sampler is not yet a good selector.  Formally, the marginal over
terminal graph-of-thought branches is still closer to high-entropy exploration
than to a reward-proportional terminal law:

\[
  P_F(x) \not\propto R_{\mathrm{BPB}}(x)
\]

in the current recovery phase.  Since GFlowNet weights are intentionally zero
in `bpb_recovery_37500_39250`, this remains a diagnostic weakness rather than
a reason to restart.

### Undesirable Behavior

Toric memory entropy is still decaying:

```text
train/toric_memory_entropy: 0.3604 -> 0.3030 median, latest 0.3027
```

This is not ideal, but it is also not surprising while toric and GFlowNet
losses are disabled.  The model is using the dense byte path for recovery and
is not being rewarded for preserving phase diversity.  Do not increase the
toric-memory weight inside the recovery segment; the active-face margins are
still too small for a reliable toric training signal.

Toric geometry remains unstable as supervision:

| toric diagnostic | value |
|---|---:|
| active-face margin | `-0.9922` |
| shadow mean margin | `0.0283` |
| shadow minimum margin | `1.71e-4` |
| binomial residual | `0.5169` |
| phase-leaf residual | `0.99998` |

The tropical argmax stability condition

\[
  \Delta_{ic} > 2(\epsilon_S+\epsilon_V)
\]

is not close to being satisfied.  This explains why the toric shadow plots show
many wall hops and bend spikes: the empirical path is near fan walls.  A strong
fan or bend loss here would amplify unstable wall-crossing noise.

Hessian probes remain unavailable because the config has `hessian.enabled:
false`.  The active curvature evidence is therefore checkpoint finite
differences plus the robust-gradient/guard statistics.

### Plot Review

`core_metric_timeseries.png` shows a clean moving-average drop after the
38,250 impulse.  The latest local values are near the lower envelope of the
window, not at the spike top.  `recent_metric_slopes.png` shows the strongest
recent changes are in relative-K training diagnostics and LR/guard controls;
core BPB is not among the pathological positive slopes.  The selected
correlation plot again shows train BPB/loss tightly coupled, while kinetic
energy and viscous dissipation are negatively correlated with BPB.  The
trajectory is active rather than collapsing.

The 3D reasoning trajectories show a broader branch fan than at 38,000,
including a long branch excursion away from the main solution cloud.  This is
useful exploration, but the energy landscape remains rugged.  Low-energy
basins are visible; terminal markers are not consistently seated in the best
basins.  This explains the weak branch BPB improvement.

The toric shadow audit shows nontrivial fan occupancy and slight improvement in
binomial residual, but the phase-leaf residual remains near `1.0` and the
minimum fan margin remains effectively zero.  The exact persistence morphism
plot is healthy: H0/H1 ranks are nonzero, edge and triangle validity remain
high, and directed edge validity stays bounded.  This supports preserving the
current topology diagnostics unchanged.

### Decision

Continue from:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00038500.pt
```

No code or config scalar change is warranted yet.  The 38,250 impulse recovered
by 38,500, and 38,500 is the best retained checkpoint in the current recovery
sequence.  Validation slipped by only about `0.0036` BPB, below the configured
decision guard, while complexity validation and robust guard behavior improved.

Schedule the next interrupting review at step `39000`.  If `39000` shows
another upward checkpoint impulse without recovery, or if validation regresses
by more than `0.01` relative to `4.9348`, then the next scalar adjustment should
be conservative: reduce the recovery LR multiplier from `0.95` to `0.88` and
keep auxiliary toric/GFlowNet/topology losses at zero until after step `39250`.

Acceptance criteria for `39000`:

1. checkpoint `train_bpb < 3.5928`;
2. recent median `train/bpb < 3.5668`, or no positive 250-step derivative
   larger than `0.02`;
3. `val/bpb < 4.9384`, or no regression above `4.9448`;
4. `complexity/val/bpb <= 4.8226`;
5. robust guard fraction remains below `0.10`;
6. best branch BPB beats `4.0180`, or mean branch BPB moves below `4.7638`;
7. topology inclusion violation remains `0.0`, exact edge validity remains
   above `0.95`, HDBSCAN stability remains above `0.90`, and Slepian leakage
   remains `0.0`.

## 2026-06-01 Automated Review: Step 38,000

Training was paused by the watcher before this review.  GPU utilization was
idle, so the analysis below is a stop-and-resume decision rather than a
background-only observation.

Analyzed checkpoint:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00038000.pt
```

W&B run:

```text
https://wandb.ai/amelie-iska-math/toricgt-parameter-golf/runs/oai37500r2-20260601T024939Z
```

Analysis outputs:

```text
outputs/post_resume_analysis/oai-bpb-recovery-37500-20260601T024939Z/step-00038000/
```

Reviewed plots included `metrics/core_metric_timeseries.png`,
`metrics/recent_metric_slopes.png`, `metrics/selected_metric_correlations.png`,
`simplex/reasoning_k_bpb_triangle.png`,
`geometry/tetrahedra/toric_gfn_bpb.png`,
`geometry/trajectories/*_trajectory_3d.png`,
`geometry/trajectories/*_energy_landscape.png`,
`geometry/trajectories/*_toric_phase_simplicial_trajectory.png`,
`geometry/trajectories/*_toric_phase_winding_collection.png`, and the
directed-topology audit panels under `geometry/topology/`.

### Metric Categorization

The automatic metric audit reported:

| category | count |
|---|---:|
| as desired | `210` |
| as desired but too weak or slow | `84` |
| not as desired | `98` |

Core likelihood is now moving in the right direction:

| metric | first median | last median | recent slope / 1k | category |
|---|---:|---:|---:|---|
| `train/bpb` | `3.8575` | `3.5320` | `-0.9867` | desired |
| `train/loss` | `2.6738` | `2.4482` | `-0.6839` | desired |
| `train/loss_ema` | `2.6947` | `2.4810` | `-0.5688` | desired |
| `train/gflownet_loss` | `3.4943` | `3.3684` | `-0.0439` | desired but weak |
| `train/gflownet_entropy` | `2.77253` | `2.77250` | `-1.05e-4` | weak/flat |
| `train/gflownet_action_diversity` | `0.999982` | `0.999971` | `-4.36e-5` | weak/flat |

Checkpoint finite differences support continuation:

| checkpoint | train BPB | train loss |
|---:|---:|---:|
| `37500` | `4.907855` | `3.401866` |
| `37750` | `3.829687` | `2.654536` |
| `38000` | `3.633273` | `2.518393` |

The first differences are still negative:

\[
  \Delta \mathrm{BPB}_{37500\to37750}=-1.078168,\qquad
  \Delta \mathrm{BPB}_{37750\to38000}=-0.196413.
\]

The second difference is positive, about `+0.881755`, so descent is
decelerating, but it has not yet become a floor-bounce or rollback signal.
The correct rule here is to keep the run while monitoring curvature closely:
a positive second difference only becomes actionable when the first derivative
also turns positive or when validation/guard metrics break.

### Desired Behavior

The byte objective recovered after the 37,500 rollback.  Cross-entropy and BPB
are the same signal up to a constant byte normalization, so their matching
slopes are expected:

\[
  \mathrm{BPB}=\frac{\mathcal L_{\mathrm{nats}}}{\log 2}
  \cdot \frac{\text{tokens}}{\text{bytes}}.
\]

The current descent has no detected robust-difference spikes in the analyzed
W&B interval.  Gradient norm is bounded, and the robust microbatch guard
fraction is near but still below the operating ceiling (`0.1875` latest W&B
summary).  The artifact budget is stable at `14,816,001` deployable parameters
under the 16,000,000 byte competition cap.

Directed topology is also healthy as an audit object.  The nested complexes
have zero inclusion violation, exact persistence morphisms are being computed,
edge validity is about `0.9787`, directed edge validity about `0.9565`,
triangle validity about `0.8119`, boundary residual is `0.0`, and HDBSCAN
stability is `0.9450` with low noise (`0.0550`).  These values mean the
radius-parametrized directed complexes remain coherent chain-level objects
while likelihood recovery proceeds.

### Desired But Too Weak Or Slow

The geometry suite shows useful but under-exploited inference-time scaling:

| diagnostic | value | read |
|---|---:|---|
| mean branch BPB | `4.7592` | weak |
| best branch BPB | `4.0135` | useful branch variance |
| mean answer BPB | `4.6467` | weak |
| best answer BPB | `3.6025` | useful but not enough |
| mean MST efficiency | `0.5082` | partially organized |
| mean path smoothness | `0.0886` | bounded but not funnel-like |
| analogical map loss | `0.0985` | bounded but loose |
| directed map loss | `0.1428` | bounded but loose |

The simplex and tetrahedron plots show low-BPB pockets, but the branch clouds
remain closer to the toric/GFlowNet-diversity side than to the low-BPB vertex.
Mathematically, entropy near \(\log 16\) and action diversity near `1.0` mean
the action policy is almost uniform over the 16 graph-of-thought actions.  That
is good coverage, but it is not yet a reward-calibrated sampler:

\[
  P_F(\tau)\approx \text{uniform over actions}
  \quad\not\approx\quad
  \frac{R(x_\tau)}{Z}.
\]

Because the recovery phase deliberately keeps GFlowNet and topology weights at
zero, this is acceptable for this 500-step window.  The branch sampler should
remain diagnostic until the byte model stops recovering.

### Undesirable Behavior

Validation remains too high relative to training.  W&B summary values at this
review were `val/bpb = 4.9348`, `val/score_first_bpb = 4.9328`, and
`complexity/val/bpb = 4.8233`, while the latest training BPB summary was
`3.5436`.  That gap indicates the current recovery window is still mainly
repairing the train stream likelihood rather than transferring to validation.
This is a known consequence of the likelihood-first recovery phase; it should
not be hidden, but it is not by itself a rollback criterion while train BPB is
falling sharply.

The toric phase/active-face diagnostics are not ready to become training
pressure.  Geometry summaries show:

| toric diagnostic | value | read |
|---|---:|---|
| active-face margin | `-0.9565` | unstable argmax faces |
| shadow mean margin | `0.0283` | too close to walls |
| shadow minimum margin | `1.93e-4` | effectively on a wall |
| binomial residual | `0.5461` | weak toric ideal consistency |
| phase-leaf residual | `0.99996` | weak Kronecker-leaf organization |

The tropical stability condition is

\[
  \Delta_{ic} > 2(\epsilon_S+\epsilon_V).
\]

The observed margins are far too small, so increasing toric or tropical
regularization now would likely push noisy wall-crossings rather than stable
Newton-fan decisions.  The toric plots are valuable audits, not promotion
signals for this interval.

Hessian probes are unavailable because `hessian.enabled` is currently false.
The finite-difference curvature above is therefore the active curvature proxy
for this decision.

### Plot Review

The metric time-series plot shows jagged but monotone recovery in the moving
average of train BPB/loss.  The slope plot is dominated by stream-composition
and complexity-distribution terms, so the robust median and checkpoint
differences are more reliable than any single raw recent slope.

The selected-correlation plot shows train BPB/loss tightly correlated, as
expected.  Trajectory kinetic energy and viscous dissipation are negatively
correlated with train BPB in this window, which means the current descent is
not coming from collapsing the reasoning trajectory; the hidden trajectory is
remaining active while likelihood improves.

The 3D graph-of-thought trajectories are branch-diverse and noncollapsed, but
the energy landscapes are rugged rather than funnel-shaped.  Terminal markers
are not consistently at the deepest local minima, so the inference-time
scaling controller is not yet reliably selecting the best basin.

The toric phase winding plots now include both the flat irrational torus shadow
and the embedded torus with local simplicial edges.  They show dense,
chord-rich Kronecker winding rather than smooth motion along a few coherent
leaves.  This matches the high phase-leaf residual and supports keeping phase
foliation as an audit until margins improve.

### Decision

Continue from:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00038000.pt
```

No code or config scalar change is warranted at this checkpoint.  The rollback
rule prefers the last checkpoint before the first derivative turns positive;
the first derivative is still negative at step 38,000.  The only action is to
resume training and schedule the next interrupting analysis at step 38,500
with a fresh checkpoint mtime cutoff.

Acceptance criteria for the next review around `38500`:

1. checkpoint `train_bpb < 3.6333`;
2. recent median `train/bpb < 3.5320`, or no positive 250-step derivative
   larger than `0.02`;
3. robust microbatch guard fraction stays below `0.20`;
4. `val/bpb < 4.9348` or no validation regression larger than `0.01`;
5. `complexity/val/bpb < 4.8233`;
6. best branch BPB beats `4.0135`, and mean branch BPB moves below `4.75`;
7. topology inclusion violation remains `0.0`, HDBSCAN stability remains above
   `0.90`, and Slepian leakage remains `0.0`.

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

## Live Recovery Intervention: Step 36,750

Run analyzed: `oai-slepian-koszul-resume-20260531T192540Z`

Checkpoint analyzed:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00036750.pt
```

Analysis directory:

```text
outputs/reasoning_geometry_suite/oai-live-step36750-detailed-20260531T213114Z
```

### Metric Read

Validation still moved in the correct direction, but only marginally:

| metric | recent values | category |
|---|---:|---|
| `val/bpb` | `4.9318708 -> 4.9309084 -> 4.9305737` | desired but too slow |
| `complexity/val/bpb` | `4.8146830 -> 4.8127899 -> 4.8126030` | desired but too slow |
| `complexity/val/loss` | `3.3372841 -> 3.3359718 -> 3.3358421` | desired but too slow |
| recent train BPB slope, last 500 steps | about `+0.37 BPB / 1k steps` | undesirable |
| Hessian probe values | `NaN` trace/curvature/grad norm | undesirable as a decision signal |

The controller did not enter recovery because its checkpoint-level
`controller/train_bpb_drift` was still negative at the eval point, even though
the shorter-window train BPB slope had turned positive.  This is a timescale
mismatch: the EMA is too coarse for the observed local ricochet.

### Geometry Read

The architecture is producing meaningful structure:

| diagnostic | value | category |
|---|---:|---|
| exact persistence morphisms computed | `20` | desired |
| inclusion violation | `0.0` | desired |
| exact edge validity | `0.9895` | desired |
| exact triangle validity | `0.9618` | desired |
| HDBSCAN stability | `0.9386` | desired |
| Slepian concentration/leakage | `1.0 / 0.0` | desired |
| analogical/directed map losses | `0.1266 / 0.1648` | desired but too weak |
| empirical toric shadow margin | mean `0.0297`, min `0.00015` | desired but too weak |
| toric active-face margin | `-0.9375` | undesirable |
| phase-leaf residual | `1.0295` | undesirable |

The plots show real branching and coherent directed topology, but the toric
decision boundaries are too low-margin to justify stronger toric supervision
while BPB is locally worsening.

### Implemented Recovery Controls

The current intervention is deliberately scalar-only.  The architecture,
optimizer state, random-order autoregressive graph decoding, tropical/hybrid
attention, toric memory, GraphCG frame, and topology diagnostics remain intact.

Implemented config changes:

| control | new value | reason |
|---|---:|---|
| `phase bpb_recovery_36750_38500` | active from `36000` to `38500` | force a likelihood-first recovery window |
| `gflownet_loss_weight` in recovery | `0.0` | keep GFlowNet diagnostic, remove auxiliary pressure |
| `toric_geometry_loss_weight` in recovery | `0.0` | do not optimize low-margin toric probes during BPB recovery |
| `koszul_persistence_loss_weight` in recovery | `0.0` | keep topology for diagnostics only |
| `trajectory_flow_loss_weight` in recovery | `0.0` | remove non-BPB trajectory pressure |
| `qat_loss_weight` in recovery | `0.0` | avoid quantization pull during likelihood recovery |
| `mtp_loss_weight` in recovery | `0.004` | retain a small next-token stabilizer without dominating BPB |
| `medium_mix_ratio` in recovery | `0.20` | retain moderate validation-like rows |
| `hard/complex_mix_ratio` in recovery | `0.0` | remove hard graph rows until descent resumes |
| `lr_multiplier` in recovery | `1.05` | slightly raise the very low late-cosine LR without returning to the bounce band |
| `grad_clip_norm` in recovery | `0.65` | cap sharp updates while keeping real descent |
| Hessian probes | disabled | NaN probes must not steer restarts |
| adaptive controller `gflownet_loss_min` | `0.0` | allow true recovery to zero auxiliary GFlowNet weight |
| adaptive controller `warmup_updates` | `0` | let the controller react immediately after restart |

### Acceptance Criteria

For the next review near `37250`--`37500`:

1. last-100 train BPB slope should be negative;
2. last-100 train BPB mean should fall below the pre-restart `4.88` band;
3. validation BPB should improve by at least `0.002`;
4. `complexity/val/bpb` should continue descending;
5. toric/topology metrics may be diagnostic-only, but inclusion violation must
   remain `0.0` and HDBSCAN stability should remain above `0.85`.

## Automated Review: Step 37,750

Run analyzed: `amelie-iska-math/toricgt-parameter-golf/oai37250r1`

Checkpoint analyzed:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00037750.pt
```

Analysis directory:

```text
outputs/post_resume_analysis/oai-bpb-recovery-37250-20260601T011539Z/step-00037750
```

Contact sheets reviewed:

```text
outputs/post_resume_analysis/oai-bpb-recovery-37250-20260601T011539Z/step-00037750/contact_sheets/contact_metrics.png
outputs/post_resume_analysis/oai-bpb-recovery-37250-20260601T011539Z/step-00037750/contact_sheets/contact_simplex_tetra.png
outputs/post_resume_analysis/oai-bpb-recovery-37250-20260601T011539Z/step-00037750/contact_sheets/contact_trajectories.png
outputs/post_resume_analysis/oai-bpb-recovery-37250-20260601T011539Z/step-00037750/contact_sheets/contact_topology.png
```

### Metric Categories

The automated categorizer produced:

| category | count |
|---|---:|
| desired | `198` |
| desired but too weak/slow | `90` |
| undesirable | `104` |

The checkpoint-local scalar read is:

| step | train BPB | train loss | best validation BPB |
|---:|---:|---:|---:|
| `37250` | `4.955313` | `3.434761` | `4.696106` |
| `37500` | `4.907855` | `3.401866` | `4.696106` |
| `37750` | `4.912290` | `3.404940` | `4.696106` |

Finite differences therefore changed sign after step `37500`:

\[
  \Delta \mathrm{BPB}_{37250\to37500}=-0.047458,\qquad
  \Delta \mathrm{BPB}_{37500\to37750}=+0.004435.
\]

The correct rollback point is the last checkpoint before that positive
difference, namely:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00037500.pt
```

### Desired Behavior

The directed topology and persistence stack is behaving as intended.

| diagnostic | value | read |
|---|---:|---|
| exact directed edge validity | `0.9564` | desired |
| exact undirected edge validity | `0.9830` | desired |
| exact triangle validity | `0.8145` | acceptable, though improvable |
| exact persistence morphisms | `20` | desired |
| inclusion violation | `0.0` | desired |
| cycle flux | `6.6e-18` | desired |
| HDBSCAN stability | `0.9409` | desired |
| Slepian concentration/leakage | `1.0 / 0.0` | desired |
| Buchsbaum-Eisenbud residual | `0.00547` | desired |
| variety and boundary residuals | `0.0 / 0.0` | desired |

Mathematically, the persistence and Koszul diagnostics show that the sampled
reasoning complexes are still coherent chain-level objects.  Boundary
residuals near zero indicate \(\partial^2=0\) is respected in the audit
complexes, and zero inclusion violation means the radius-parametrized complexes
form a genuine filtration rather than unrelated snapshots.  This is exactly the
structure we want to preserve while repairing likelihood training.

### Desired But Too Weak Or Slow

The branch/test-time-scaling diagnostics show useful structure but weak
selection.

| diagnostic | value | read |
|---|---:|---|
| mean branch BPB | `4.7592` | weak |
| best branch BPB | `4.0440` | useful but not enough |
| mean answer BPB | `4.6459` | weak |
| best answer BPB | `3.6416` | useful but not enough |
| MST efficiency | `0.5092` | weak |
| path smoothness | `0.0888` | weak |
| topology analogical map loss | `0.0994` | weak |
| directed map loss | `0.1450` | weak |
| binomial residual | `0.5476` | weak |
| toric fan entropy | `0.6653` | noncollapsed but not decisive |
| occupied fan cells | `11.92` | noncollapsed |

The simplex and tetrahedron plots show that low-BPB branches exist, but the
branch distribution remains clumped around diversity and toric entropy rather
than the low-BPB vertex.  GFlowNet action diversity is essentially saturated,
and entropy is near \(\log 16\), so the policy is exploring almost uniformly.
That is good for coverage, but weak for exploitation.  Since the active
recovery phase already set GFlowNet loss weights to zero, the right action is
not to remove the GFlowNet machinery; it is to keep it diagnostic until the
byte objective is descending again.

The MST value means only about half of the branch geometry is captured by a
short connecting skeleton.  In practical terms, reasoning trajectories are
diverse but not yet organized into a clean low-energy funnel.  The energy plots
confirm this: several local minima are visible, but terminal solution markers
are not consistently seated in the lowest basins.

### Undesirable Behavior

The core likelihood metrics are not improving enough in the analyzed window.

| metric | read |
|---|---|
| `train/bpb` | last median is about `2.05%` worse than the first median |
| `train/loss` | same shape as BPB, because BPB is cross-entropy rescaled by \(\log 2\) and byte length |
| `train/total_loss` | worsens with the base likelihood signal |
| `val/bpb` | no fresh improvement after the local recovery segment |
| `controller/val_bpb` | stuck near `4.9276` |
| `gflownet_entropy` | saturated near uniform, not calibrated toward low-BPB branches |
| `toric_active_face_margin` | negative, about `-0.98` |
| toric shadow minimum margin | about `1.2e-4` |
| phase-leaf residual | about `1.00` |
| Hessian probe | unavailable; probes were disabled |

For BPB, the statistical signal is a high-variance stochastic cross-entropy
estimate.  The recent ordinary least-squares slope is not a reliable promotion
signal because the median over the same interval worsened and checkpoint
finite differences turned positive.  The finite-difference rule is more
appropriate here:

\[
  \operatorname{promote}(t)
  \quad\Longleftrightarrow\quad
  \Delta \mathrm{BPB}_{t-h\to t}<0
  \text{ and }
  \Delta^2 \mathrm{BPB}_{t-2h,t-h,t}\le 0
\]

with rejection when the first difference turns positive after a local descent.
Step `37750` fails this rule.

For tropical/toric behavior, the margin problem is decisive.  A tropical
argmax is stable only when the active-face margin exceeds the perturbation
scale:

\[
  \Delta_{ic} > 2(\epsilon_S+\epsilon_V).
\]

Here empirical fan margins are tiny and the active-face margin is negative, so
toric geometry should not be made a training pressure during BPB recovery.  It
should remain an audit shadow until likelihood descent resumes.

For conditional Kolmogorov metrics, the compressor proxies are useful monitors
but not exact \(K(x\mid y)\), which is incomputable.  The observed movements are
best read as changes in conditional compressibility under the current helper
strings and graph projections, not as proof that true algorithmic complexity
has improved.  They should not override BPB and validation loss during a
competition recovery window.

### Plot Review

The reviewed figures support the scalar decision:

1. `core_metric_timeseries.png` shows jagged BPB/loss with repeated impulses;
   the last window does not establish a robust downward trend.
2. `recent_metric_slopes.png` shows stronger slopes in complexity-distribution
   diagnostics than in BPB itself, so the optimizer is seeing stream-composition
   variance.
3. `selected_metric_correlations.png` shows GFlowNet entropy and action
   diversity positively correlated with BPB in this window, which means
   exploration is not yet translating to byte likelihood.
4. `reasoning_k_bpb.png` and related simplex plots show low-BPB branch pockets
   but weak movement toward the low-BPB vertex.
5. `toric_gfn_bpb.png` shows branches clustered near toric entropy and
   GFlowNet diversity rather than low BPB.
6. `*_trajectory_3d.png` plots show real branching but no clean terminal
   solution funnel.
7. `*_energy_landscape.png` plots are rugged rather than funnel-shaped.
8. `*_phase_energy.png` plots show phase clusters with broad scatter, so phase
   is organizing but not yet selecting solutions.
9. `*_exact_persistence_morphisms.png` plots verify that directed
   persistence-module maps remain healthy.
10. `*_toric_phase_winding_collection.png` now correctly plots flat irrational
    phase winding beside the embedded torus shadow.  The winding is dense and
    chord-rich, not smooth along a small number of Kronecker leaves, which is
    acceptable as an audit but not a reason to add toric loss in the recovery
    segment.

### Decision

Do not continue from `37750`.  Restart from:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00037500.pt
```

Implemented scalar controls:

| control | old | new | reason |
|---|---:|---:|---|
| `data.stream_origin_step` | `3500` | `37750` | avoid replaying the analyzed row order after rollback |
| recovery phase | `bpb_recovery_36750_38500` | `bpb_recovery_37500_39250` | align the recovery window with the chosen checkpoint |
| recovery LR multiplier | `1.05` | `0.95` | damp the floor-bounce without freezing descent |
| recovery grad clip | `0.65` | `0.50` | reduce sharp high-loss updates |
| recovery medium mix | `0.20` | `0.10` | keep validation-like rows, reduce stream variance |
| recovery GraphCG weight | `0.00002` | `0.0` | make geometry diagnostic-only for 500-step recovery |
| recovery contrastive weight | `0.00025` | `0.00010` | retain weak representation pressure only |
| recovery MTP weight | `0.004` | `0.0` | remove the remaining auxiliary next-token pressure |
| recovery toric/topology/GFlowNet/QAT weights | `0.0` | `0.0` | keep diagnostic-only |
| shock guard window | `1500--2200` | `37500--39250` | move bounded-influence guard to the active recovery |
| robust micro guard window | `1500--1850` | `37500--39250` | same |

Acceptance criteria for the next review around `38000`:

1. checkpoint `train_bpb` should be below `4.907855`;
2. last-window `train/bpb` median should be below `4.90`;
3. `controller/val_bpb` should be below `4.9276` or not regress by more than
   `0.01`;
4. `complexity/val/bpb` should stay at or below `4.8128`;
5. robust guard fraction should stay below `0.20`;
6. best branch BPB should beat `4.044`, and mean branch BPB should move below
   `4.75` if the geometry suite is run;
7. inclusion violation must remain `0.0`, HDBSCAN stability should remain above
   `0.85`, and Slepian leakage should remain `0.0`.

## 2026-06-01 Automated Review: Step 39,000

Analysis package:

```text
outputs/post_resume_analysis/oai-bpb-continue-38500-20260601T053330Z/step-00039000
```

Checkpoint:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00039000.pt
```

W&B run:

```text
amelie-iska-math/toricgt-parameter-golf/oai37500r2-20260601T024939Z
```

### Metric Categories

The automated categorizer produced:

| category | count |
|---|---:|
| desired | `341` |
| desired but too weak/slow | `117` |
| undesirable | `158` |

Checkpoint finite differences over the recovery window are:

| step | train BPB | train loss | best validation BPB |
|---:|---:|---:|---:|
| `37500` | `4.907855` | `3.401866` | `4.696106` |
| `37750` | `3.829687` | `2.654536` | `4.696106` |
| `38000` | `3.633273` | `2.518393` | `4.696106` |
| `38250` | `3.751553` | `2.600379` | `4.696106` |
| `38500` | `3.592812` | `2.490348` | `4.696106` |
| `38750` | `3.750200` | `2.599440` | `4.696106` |
| `39000` | `3.592368` | `2.490040` | `4.696106` |

The local second-difference pattern is an alternating floor-bounce:

\[
\Delta_{38500\to 38750}=+0.157388,\qquad
\Delta_{38750\to 39000}=-0.157831,
\]

so the final checkpoint recovered the 38.75k impulse but did not materially
improve beyond 38.5k:

\[
\mathrm{BPB}_{39000}-\mathrm{BPB}_{38500}=-4.44\times 10^{-4}.
\]

### Desired Behavior

Core training cross-entropy is still descending in the W&B export.  The recent
OLS slopes are negative:

| metric | first median | last median | recent slope / 1k |
|---|---:|---:|---:|
| `train/bpb` | `3.8625` | `3.5891` | `-0.8102` |
| `train/loss` | `2.6773` | `2.4878` | `-0.5616` |
| `train/loss_ema` | `2.6726` | `2.5275` | `-0.4643` |
| `complexity/train/bpb` | `4.2204` | `3.7158` | `-0.5322` |
| `train/gflownet_loss` | `3.4991` | `3.3208` | `-0.1907` |

The topology stack is also healthy:

| diagnostic | value | read |
|---|---:|---|
| exact edge validity | `0.9752` | desired |
| exact directed edge validity | `0.9534` | desired |
| exact triangle validity | `0.8218` | acceptable |
| exact persistence morphisms | `20.0` | desired |
| inclusion violation | `0.0` | desired |
| boundary residual | `0.0` | desired |
| directed cycle flux | `5.85e-18` | desired |
| HDBSCAN stability | `0.9449` | desired |
| Slepian concentration/leakage | `1.0 / 0.0` | desired |
| Buchsbaum-Eisenbud residual | `0.00422` | desired |
| variety-complex residual | `0.0` | desired |

Mathematically, the chain-level audits still look coherent: zero boundary
residual means the sampled complexes satisfy the \(\partial^2=0\) constraint,
zero inclusion violation means the radius filtration is nested, and high
HDBSCAN stability means the local point-cloud structure is not breaking under
small radius changes.  These are structural diagnostics; they should remain
mostly passive while the byte likelihood recovers.

### Desired But Too Weak Or Slow

Validation and geometry improved too slowly.

| metric | value / trend | read |
|---|---:|---|
| `val/bpb` | `4.9384 -> 4.9416` in W&B export | weak/slight regression |
| `val/score_first_bpb` | `4.9372 -> 4.9376` | weak |
| `val/gflownet_bpb` | `4.9397 -> 4.9432` | weak |
| `complexity/val/bpb` | `4.8233 -> 4.8320` | weak/regressed |
| mean branch BPB | `4.7681` | weak |
| best branch BPB | `4.0204` | useful but worse than 38.5k |
| mean answer BPB | `4.6545` | weak |
| best answer BPB | `3.6046` | useful pocket |
| MST efficiency | `0.5123` | modest |
| path smoothness | `0.0899` | modest |
| topology analogical map loss | `0.0980` | modest |
| directed map loss | `0.1423` | modest |

The simplex plots show the same geometry as the 38.5k review: a low-BPB branch
pocket exists, but most branches remain in the reasoning/\(K(x\mid y)\)
interior rather than moving sharply toward the low-BPB vertex.  The
toric/GFlowNet/BPB tetrahedron is clustered near toric entropy and exploration
instead of the low-BPB corner.  This is useful evidence that the graph-of-
thought machinery is noncollapsed, but not evidence that it is improving the
competition metric during recovery.

### Undesirable Behavior

The following signals should be corrected with scalar controls:

| metric | behavior |
|---|---|
| checkpoint BPB | oscillates `38500 -> 38750 -> 39000` with near-zero net gain |
| validation BPB | small upward drift, not a promotion signal |
| complexity validation BPB | upward drift after 38.25k |
| branch mean/best BPB | slightly worse than 38.5k |
| `train/toric_memory_entropy` | median decayed `0.3569 -> 0.3101` |
| toric active-face margin | remains negative, around `-1.02` in geometry |
| toric shadow min margin | tiny, about `1.42e-4` |
| phase-leaf residual | about `1.0` |

The BPB/loss plateau is best understood as stochastic optimization near a
high-curvature floor, not as a capacity limit.  The 250-step first differences
alternate sign with nearly equal magnitude, while the longer W&B slope remains
negative.  That means there is a real descent component plus a periodic stream
or curvature impulse.  A full rollback would throw away a recovered checkpoint;
an architecture change would be disproportionate.  The correct response is a
smaller effective step size and a longer diagnostic-only recovery window.

For tropical/toric behavior, the max-plus stability condition still fails:

\[
  \Delta_{ic} > 2(\epsilon_S+\epsilon_V)
\]

is not supported when observed active-face margins are negative and empirical
fan minimum margins are near zero.  Therefore toric, topology, and GFlowNet
losses should stay zero-weighted in the recovery window; their diagnostics are
informative, but they should not perturb the likelihood optimizer yet.

### Plot Review

Reviewed plots:

```text
metrics/core_metric_timeseries.png
metrics/recent_metric_slopes.png
metrics/selected_metric_correlations.png
simplex/reasoning_k_bpb_triangle.png
geometry/tetrahedra/toric_gfn_bpb.png
geometry/trajectories/R2_gss1147-got_math_500k_got_math_trajectory_3d.png
geometry/trajectories/R2_gss1147-got_math_500k_got_math_energy_landscape.png
geometry/trajectories/R2_gss1147-got_math_500k_got_math_toric_phase_simplicial_trajectory.png
geometry/trajectories/R2_gss1147-got_math_500k_got_math_toric_phase_winding_collection.png
geometry/topology/R2_gss1147-got_math_500k_got_math_exact_persistence_morphisms.png
geometry/topology/R2_gss1147-got_math_500k_got_math_toric_shadow_audit.png
```

The 3D trajectory plot shows broad branching and an actual solution-span cloud,
but the energy landscape is still rugged: terminal branches traverse several
basins rather than falling into a single low-energy/low-BPB funnel.  The toric
phase plots now include both the flat irrational winding and torus embedding
with local simplicial edges; they show dense phase coverage, but not smooth
leaf-following selection.  The persistence morphism plots remain structurally
valid, so no topology implementation change is warranted.

### Decision

Continue from the recovered `39000` checkpoint, but do not leave the old scalar
schedule unchanged.  Implemented the smallest high-impact adjustment:

| control | old | new | reason |
|---|---:|---:|---|
| `data.stream_origin_step` | `37750` | `39000` | avoid replaying the analyzed post-rollback order |
| shock guard end | `39250` | `39750` | keep bounded-influence updates through the next two reviews |
| robust micro guard end | `39250` | `39750` | same |
| recovery phase end | `39250` | `39750` | prevent immediate return to auxiliary-heavy compression phase |
| recovery LR multiplier | `0.95` | `0.88` | reduce the floor-bounce amplitude |

No architecture changes were made.  Random-order autoregressive graph decoding,
tropical ring/hybrid attention, dense contest weights, toric memory,
embedding-space GFlowNet graph-of-thought diagnostics, persistence/Koszul
diagnostics, and conditional Kolmogorov monitors remain in place.

Next review target: `39500`.

Acceptance criteria:

1. checkpoint `train_bpb < 3.59237`;
2. no 250-step positive BPB impulse above `+0.05`;
3. `val/bpb <= 4.9416` or no regression above `4.9486`;
4. `complexity/val/bpb <= 4.8320`;
5. best branch BPB beats `4.0204` or mean branch BPB moves below `4.7681`;
6. inclusion violation remains `0.0`, HDBSCAN stability remains above `0.90`,
   and Slepian leakage remains `0.0`.

## Step 39,500 Handoff Review

Analysis directory:

```text
outputs/post_resume_analysis/oai-bpb-continue-39000-20260601T065718Z/step-00039500
```

The 39,500 checkpoint is not promotable.  The key checkpoint finite differences
are:

| step | train BPB | train loss | first difference | second difference |
|---:|---:|---:|---:|---:|
| 38,500 | `3.592812` | `2.490348` | n/a | n/a |
| 38,750 | `3.750200` | `2.599440` | `+0.157388` | n/a |
| 39,000 | `3.592368` | `2.490040` | `-0.157831` | `-0.315219` |
| 39,250 | `4.227429` | `2.930231` | `+0.635061` | `+0.792892` |
| 39,500 | `4.363518` | `3.024560` | `+0.136089` | `-0.498972` |

The first large positive derivative appears immediately after 39,000.  The
large positive second difference at 39,250 is the floor-bounce signature: the
optimizer left the good local byte-compression basin and entered a harder
row-group shelf.  This is corroborated by the W&B trace after the 39k restart:
robust guard fractions repeatedly sat near `0.5`, and microbatch BPB spiked
above `5.0`.  The failure is therefore a data-order shock, not an architectural
failure.

### Categories

| group | count | read |
|---|---:|---|
| desired | `335` | structural diagnostics and many long-window slopes remain healthy |
| desired but too weak or slow | `121` | validation and branch selection are not improving fast enough |
| not desired | `160` | checkpoint BPB/loss, toric entropy decay, and hard-shelf guards require correction |

### Desired

Topology and geometric diagnostics remain coherent:

| diagnostic | value | read |
|---|---:|---|
| exact edge validity | `0.9896` | desired |
| exact directed edge validity | `0.9598` | desired |
| exact triangle validity | `0.8373` | acceptable |
| exact persistence morphisms | `20.0` | desired |
| inclusion violation | `0.0` | desired |
| boundary residual | `0.0` | desired |
| directed cycle flux | `5.26e-18` | desired |
| HDBSCAN stability | `0.9320` | desired |
| Slepian concentration/leakage | `1.0 / 0.0` | desired |
| Buchsbaum-Eisenbud residual | `0.00499` | desired |
| variety-complex residual | `0.0` | desired |

The chain-complex checks still satisfy the algebraic invariants we care about:
the sampled complexes are nested, the directed edge maps are mostly valid, and
the toric Slepian audit is numerically clean.  This justifies preserving the
current ToricGT diagnostic stack.

The long-window W&B slopes also still contain useful descent:

| metric | first median | last median | recent slope / 1k |
|---|---:|---:|---:|
| `train/bpb` | `3.862507` | `3.589069` | `-0.329946` |
| `train/loss` | `2.677286` | `2.487753` | `-0.228701` |
| `complexity/train/bpb` | `4.220424` | `3.715799` | `-0.595038` |
| `train/gflownet_loss` | `3.499146` | `3.316718` | `-0.574711` |

These are not enough to promote 39,500, because the checkpoint-local BPB is the
competition gate, but they indicate that rollback should be local rather than
resetting training.

### Desired But Too Weak Or Slow

| metric | value / trend | read |
|---|---:|---|
| `val/bpb` | median `4.934834 -> 4.941551` | weak regression |
| `val/loss` | median `3.420566 -> 3.425222` | weak regression |
| `complexity/val/bpb` | median `4.823268 -> 4.832008` | weak regression |
| `complexity/val/argmax_byte_accuracy` | `0.200684 -> 0.200195` | weak |
| mean branch BPB | `4.743205` | weak |
| best branch BPB | `4.020677` | useful pocket, not improving |
| mean answer BPB | `4.626401` | weak |
| best answer BPB | `3.618824` | useful pocket, worse than 39k |
| MST efficiency | `0.501350` | modest |
| path smoothness | `0.086616` | modest |

The simplex plots show high-reasoning branches and low-\(K(x\mid y)\) structure,
but the branch cloud does not move sharply toward the low-BPB vertex.  In the
toric/GFlowNet/BPB tetrahedron, points cluster near toric entropy and
exploration rather than the low-BPB corner.  The graph-of-thought search is
alive; it is not yet a strong validation-likelihood improver in this window.

### Undesirable

| signal | behavior |
|---|---|
| checkpoint BPB | `3.592368 -> 4.227429 -> 4.363518` after 39k |
| checkpoint loss | `2.490040 -> 2.930231 -> 3.024560` after 39k |
| robust guard fraction | often near `0.5` after stream rotation |
| `train/grad_norm` | still categorized not desired; occasional spikes |
| `train/toric_memory_entropy` | median `0.356932 -> 0.306739` |
| toric active-face margin | negative, `-0.9857` |
| toric shadow min margin | tiny, `2.03e-4` |
| phase-leaf residual | about `1.0003` |

Mathematically, the checkpoint differences indicate an impulse response rather
than smooth curvature-limited descent.  If \(b_t\) is checkpoint BPB, then
\(\Delta b_{39250}=b_{39250}-b_{39000}\approx0.635\) and
\(\Delta^2 b_{39250}\approx0.793\).  That positive second difference is too
large to interpret as ordinary minibatch noise.  Combined with high robust
guard activation, it says the stochastic gradient estimate is being dominated
by a hard stream segment.  A lower LR alone did not fix it because the stream
origin changed the sampled row-group shelf.

The toric/tropical margins also explain why auxiliary geometry should remain a
diagnostic rather than the primary correction.  The observed active-face
margin does not satisfy the stability condition

\[
  \Delta_{ic} > 2(\epsilon_S+\epsilon_V),
\]

so increasing toric or tropical auxiliary pressure during the hard shelf would
risk optimizing unstable faces instead of improving byte likelihood.

### Plot Review

Reviewed plots:

```text
metrics/core_metric_timeseries.png
simplex/reasoning_k_bpb_triangle.png
geometry/triangles/reasoning_k_bpb.png
geometry/tetrahedra/toric_gfn_bpb.png
geometry/trajectories/R2_gss1147-got_math_500k_got_math_trajectory_3d.png
geometry/trajectories/R2_gss1147-got_math_500k_got_math_energy_landscape.png
geometry/trajectories/R2_gss1147-got_math_500k_got_math_toric_phase_winding_collection.png
geometry/topology/R2_gss1147-got_math_500k_got_math_exact_persistence_morphisms.png
```

The 3D trajectory plot shows broad branch coverage and solution-span contact,
but the local energy landscape remains rugged and fragmented rather than a
single low-energy funnel.  The toric phase winding collection now contains the
flat irrational winding, torus embedding, simplicial edges, and cocycle trace;
it verifies dense phase coverage, but branch selection still jumps rather than
following smooth leaves.  The exact persistence morphism audit remains useful
and structurally valid.

### Decision

Pause the bad 39,500 continuation and resume from
`checkpoints/parameter_golf_oai_dense/random_order_step_00039000.pt`.

Implemented the smallest high-impact scalar correction:

| control | old | new | reason |
|---|---:|---:|---|
| `data.stream_origin_step` | `39000` | `37750` | restore the healthier stream origin used by the descending 38.5k->39k segment |
| `data.stream_burnin_steps` | `0` | `500` | continue that stream path from its post-39k position without replaying the 38.5k->39k rows |
| recovery LR multiplier | `0.88` | `0.88` | keep damped updates; no extra architecture change |
| shock/robust guard end | `39750` | `39750` | keep bounded-influence updates through the next review |

This preserves random-order autoregressive graph decoding, tropical
ring/hybrid attention, toric memory, dense contest weights, embedding-space
GFlowNet graph-of-thought diagnostics, conditional Kolmogorov monitors, and
persistence/Koszul diagnostics.

Next review target: `39500`, using a fresh checkpoint modification-time floor
so the watcher ignores the old bad 39,500 file.

Acceptance criteria:

1. checkpoint `train_bpb < 3.59237`;
2. no immediate 250-step positive BPB impulse above `+0.05`;
3. robust guard median below `0.25`;
4. `val/bpb <= 4.9416` or no regression above `4.9486`;
5. `complexity/val/bpb <= 4.8320`;
6. best branch BPB beats `4.0204` or mean branch BPB moves below `4.7432`;
7. inclusion violation remains `0.0`, HDBSCAN stability remains above `0.90`,
   and Slepian leakage remains `0.0`.

## Step 39,500 Retry Review

Analysis directory:

```text
outputs/post_resume_analysis/oai-bpb-retry-39000-20260601T082148Z/step-00039500
```

This review is not a simple rollback case.  The raw checkpoint train BPB is
bad, but deterministic validation improved.  Therefore the 39,500 checkpoint is
usable as a continuation point if the next 500-step window remains
likelihood-only and removes row-difficulty confounds.

### Metric Categories

| group | count | read |
|---|---:|---|
| desired | `323` | topology, complexity-train, GFlowNet-loss, and several validation diagnostics remain coherent |
| desired but too weak or slow | `111` | validation improves only slightly and graph-of-thought branch BPB remains weak |
| not desired | `182` | raw train BPB/loss and toric memory entropy are not acceptable |

### Checkpoint Finite Differences

| step | train BPB | train loss | first difference |
|---:|---:|---:|---:|
| 39,000 | `3.592368` | `2.490040` | n/a |
| 39,250 | `4.156423` | `2.881013` | `+0.564055` |
| 39,500 | `5.000765` | `3.466266` | `+0.844342` |

The training log shows the same pattern at microbatch scale:

| step band | median raw BPB | read |
|---:|---:|---|
| 39,000--39,099 | `3.5780` | healthy |
| 39,100--39,199 | `4.1915` | hard shelf begins |
| 39,200--39,299 | `4.1595` | still hard |
| 39,300--39,399 | `4.3265` | worsening |
| 39,400--39,499 | `4.8245` | severe raw-data shelf |

The derivative and second-derivative signal says not to promote raw train BPB.
However, raw train BPB here is confounded by row difficulty: validation moved
in the right direction.

### Desired Behavior

| metric | value / trend | read |
|---|---:|---|
| `val/bpb` | `4.935401 -> 4.933111` from 39,250 to 39,500 | desired |
| `complexity/val/bpb` | `4.826620 -> 4.822091` | desired |
| `complexity/val/prediction_target_ncd_lzma_mean` | last `0.917515` in W&B summary window | desired |
| `complexity/train/bpb` | median `4.165032 -> 3.481313` | desired |
| `train/gflownet_loss` | median `3.494279 -> 2.415694` | desired as diagnostic |
| HDBSCAN stability | `0.9365` | desired |
| inclusion violation | `0.0` | desired |
| Slepian leakage | `0.0` | desired |
| exact persistence morphisms | `20.0` | desired |

The validation movement is the reason to continue from 39,500 instead of
rolling back to 39,000 again.  The controller saw two consecutive validation
improvements, and complexity validation improved with it.  This is the correct
promotion signal for the Parameter-Golf objective, whereas the raw train BPB is
the loss of the current row group.

### Desired But Too Weak Or Slow

| diagnostic | value | read |
|---|---:|---|
| mean branch BPB | `4.745292` | weak |
| best branch BPB | `4.021306` | weak; not better than 39k |
| mean answer BPB | `4.624957` | small improvement |
| best answer BPB | `3.611019` | useful but not enough |
| MST efficiency | `0.502619` | modest |
| path smoothness | `0.086438` | modest |
| topology analogical map loss | `0.094604` | modest |
| directed map loss | `0.145575` | modest |

The simplex/tetrahedron plots show live branch diversity and valid topology,
but not enough movement toward the low-BPB vertices.  Geometry is healthy as an
audit, not yet a strong inference-time scaling gain.

### Undesirable

| metric | behavior |
|---|---|
| `train/bpb` | median `3.857518 -> 4.128221`, recent slope `+0.658454 / 1k` |
| `train/loss` | median `2.673828 -> 2.861465`, recent slope `+0.456406 / 1k` |
| `train/loss_ema` | median `2.673666 -> 2.843844` |
| `train/grad_norm` | still categorized not desired |
| `train/toric_memory_entropy` | median `0.354629 -> 0.296443` |
| toric active-face margin | `-1.0252` |
| toric binomial residual | `0.5086`, worse than the prior retry |
| toric shadow min margin | `1.78e-4` |

Mathematically, the raw train curve is a mixture distribution effect.  Let
\(L_t=\mathbb{E}_{x\sim P_t}[-\log p_\theta(x)]\).  Across 39.1k--39.5k,
the sampling distribution \(P_t\) is changing toward higher-entropy medium
rows, while \(\theta\) still improves on the fixed validation distribution.
The observed increase in \(L_t\) is therefore not pure optimizer deterioration.
It is a row-group covariate shift inside the recovery window.

This explains the apparent contradiction:

\[
  \Delta L_{\mathrm{train}}>0,\qquad
  \Delta L_{\mathrm{val}}<0.
\]

The correct intervention is not an architecture change and not a rollback to
an earlier model.  It is to hold the recovery distribution fixed for one more
gate so that the next derivative estimates model progress rather than
curriculum difficulty.

### Plot Review

Reviewed the generated summaries and a six-sheet contact review covering all
70 PNG artifacts, including:

```text
metrics/core_metric_timeseries.png
metrics/recent_metric_slopes.png
metrics/selected_metric_correlations.png
simplex/reasoning_k_bpb_triangle.png
simplex/reasoning_k_bpb_mst_tetrahedron.png
geometry/triangles/*.png
geometry/tetrahedra/*.png
geometry/trajectories/*trajectory_3d.png
geometry/trajectories/*phase_energy.png
geometry/trajectories/*energy_landscape.png
geometry/trajectories/*toric_phase_winding_collection.png
geometry/topology/*exact_persistence_morphisms.png
geometry/topology/*noncommutative_heatmaps.png
geometry/topology/*toric_shadow_audit.png
geometry/topology/*toric_slepian_audit.png
```

The energy landscapes remain rugged, with branch trajectories contacting
solution spans but not settling into one low-BPB basin.  The toric phase
winding plots are structurally correct: flat irrational winding, embedded torus
projection, local simplicial edges, and cocycle traces are present.  The
persistence-module morphism plots remain nested and mostly valid.  None of
these plots justify an architectural change during the BPB recovery window.

### Decision

Continue from
`checkpoints/parameter_golf_oai_dense/random_order_step_00039500.pt`, not from
39,000.  The decisive reason is validation: `val/bpb` and `complexity/val/bpb`
both improved at 39,500 despite bad raw train BPB.

Implemented the smallest scalar correction:

| control | old | new | reason |
|---|---:|---:|---|
| recovery phase end | `39750` | `40250` | keep the next gate likelihood-only |
| shock guard end | `39750` | `40250` | preserve bounded updates |
| robust micro guard end | `39750` | `40250` | preserve bounded updates |
| recovery `medium_mix_ratio` | `0.10` | `0.0` | remove row-difficulty confound |
| recovery `hard_mix_ratio` | `0.0` | `0.0` | unchanged |
| recovery `complex_mix_ratio` | `0.0` | `0.0` | unchanged |

Architecture stays fixed.  Random-order autoregressive graph decoding,
tropical ring/hybrid attention, toric memory, dense contest weights,
embedding-space GFlowNet graph-of-thought diagnostics, conditional Kolmogorov
monitors, and persistence/Koszul diagnostics all remain active.

Next review target: `40000`.

Acceptance criteria:

1. `val/bpb < 4.933111`;
2. `complexity/val/bpb < 4.822091`;
3. raw train BPB median for 39.5k--40k below `4.0` after medium mix is removed;
4. no checkpoint BPB above `4.5`;
5. best branch BPB below `4.021306` or mean answer BPB below `4.624957`;
6. inclusion violation remains `0.0`, HDBSCAN stability above `0.90`,
   Slepian leakage `0.0`.

## Step 40000 Review: Valmix18 Restart Decision

Analysis directory:

```text
outputs/post_resume_analysis/oai-bpb-valmix18-39500-20260601T160202Z/step-00040000
```

Analyzed checkpoint:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00040000.pt
```

The watcher paused the training session before this review.  No active
Parameter-Golf training tmux remained; GPU memory was idle except for the
review session.  The W&B run history is partially stale because repeated
39.5k--40k restarts reused the same W&B run and W&B rejected non-monotone
step logs.  Therefore the local training log is the authoritative source for
this gate.

### Numeric Summary

Metric categories:

| class | count |
|---|---:|
| as desired | `304` |
| as desired but too weak or slow | `114` |
| not as desired | `201` |

Validation records from the local log:

| step | deterministic BPB | score-first BPB | complexity BPB | controller best |
|---:|---:|---:|---:|---:|
| 39750 | `4.936550` | `4.935478` | `4.822938` | `4.933094` |
| 40000 | `4.937589` | `4.934494` | `4.825294` | `4.933094` |

Train BPB over the retry stayed mostly in the `3.5--3.7` median band, but
medium-row impulses recurred:

| band | median BPB | max BPB |
|---:|---:|---:|
| 39500 | `3.6360` | `4.3150` |
| 39700 | `3.6300` | `4.4770` |
| 39850 | `3.5640` | `4.4010` |
| 39950 | `3.6390` | `4.4080` |

Across the recovery sequence, medium validation-like exposure is still moving
the fixed validation set in the right direction:

```text
easy-only     primary 4.9461, complexity 4.8324
3% medium     primary 4.9415, complexity 4.8279
8% medium     primary 4.9392, complexity 4.8259
18% medium    primary 4.9376, complexity 4.8253
```

The derivative is improving, but it remains above the 39.5k deterministic
gate.  Inside the 18% retry the local finite difference turned positive:

\[
  \Delta_{\mathrm{39750\to40000}}\operatorname{BPB}
  =4.937589-4.936550
  =1.04\times10^{-3}>0.
\]

This is the floor-bounce signature: the direction helps globally relative to
easy-only recovery but overshoots locally before crossing the promotion gate.

### Desired

| diagnostic | value | interpretation |
|---|---:|---|
| checkpoint train BPB | `3.498351` | strong local likelihood improvement |
| exact persistence morphisms | `20.0` | exact module maps computed |
| boundary residual | `0.0` | chain-complex consistency preserved |
| inclusion violation | `0.0` | nested filtration maps remain valid |
| HDBSCAN stability | `0.937744` | radius-parametrized clusters are stable |
| exact edge validity | `0.978670` | local complex construction valid |
| exact directed edge validity | `0.950114` | directed topology remains coherent |
| Slepian concentration/leakage | `1.0 / 0.0` | toric projection audit is numerically clean |

The topology, persistence, Koszul, and toric projection audits do not show a
structural failure.  They should stay active as diagnostics but should not be
given new loss weight during BPB recovery.

### Desired But Too Weak Or Slow

| diagnostic | value | interpretation |
|---|---:|---|
| mean branch BPB | `4.749627` | slight improvement, still too high |
| best branch BPB | `4.031180` | weaker than earlier best branches |
| mean answer BPB | `4.630701` | close to prior but not a promotion signal |
| best answer BPB | `3.617439` | useful but not enough |
| MST efficiency | `0.498880` | roughly flat |
| path smoothness | `0.082728` | improved smoothness, not enough BPB gain |
| topology analogical map loss | `0.095687` | acceptable but not decisive |
| directed map loss | `0.145265` | acceptable but not decisive |
| toric shadow mean margin | `0.029378` | positive but thin |

The simplex and tetrahedron plots show meaningful branches, but the points are
still too interior: they do not migrate consistently toward the low-BPB vertex
or toward a low-BPB/low-relative-K/high-MST boundary.  That means
test-time-scaling structure exists, but is not yet translating into a better
validation gate.

### Not As Desired

| diagnostic | value | issue |
|---|---:|---|
| deterministic validation BPB | `4.937589` | above the `4.933094` gate |
| 39750 -> 40000 validation derivative | `+0.001038` | local bounce |
| active-face margin | `-0.979736` | tropical face confidence remains inverted |
| toric binomial residual | `0.508865` | algebraic relation probe still weak |
| toric leaf residual | `0.998852` | noncommutative phase leaf audit remains poor |
| toric shadow min margin | `1.88e-4` | walls are too thin |

Mathematically, this is a distributional directional-derivative problem rather
than an architecture failure.  Let the recovery stream be

\[
  P_\alpha=(1-\alpha)P_{\mathrm{easy}}+\alpha P_{\mathrm{medium}}.
\]

For the fixed validation distribution \(Q\), the observed values imply

\[
  \frac{\partial}{\partial \alpha}
  \mathbb{E}_{Q}[-\log p_{\theta_\alpha}(x)] < 0
\]

over the easy-to-18% sweep, but the optimizer step size is still high enough
that the finite-difference trajectory in parameter space curves back upward
inside the last half-gate.  The high-BPB train impulses are the same signal in
the training stream: medium rows are needed for the validation direction, but
the update must be more damped.

### Plot Review

Reviewed the generated contact sheets covering 70 PNG artifacts:

```text
metrics/core_metric_timeseries.png
metrics/recent_metric_slopes.png
metrics/selected_metric_correlations.png
simplex/*.png
geometry/triangles/*.png
geometry/tetrahedra/*.png
geometry/trajectories/*trajectory_3d.png
geometry/trajectories/*phase_energy.png
geometry/trajectories/*energy_landscape.png
geometry/trajectories/*toric_phase_winding_collection.png
geometry/topology/*commutative_algebra_audit.png
geometry/topology/*directed_filtration.png
geometry/topology/*exact_persistence_morphisms.png
geometry/topology/*noncommutative_heatmaps.png
geometry/topology/*step_radius_hierarchy.png
geometry/topology/*toric_shadow_audit.png
geometry/topology/*toric_slepian_audit.png
```

Visual conclusions:

1. The 3D graph-of-thought trajectories still branch and terminate near
   solution-span markers, but the terminal clusters are not yet concentrated
   in low-BPB basins.
2. Ramachandran-style phase plots are stable and diverse; they do not show
   collapse.
3. Energy landscapes remain rugged rather than funnel-shaped, so increasing
   reasoning loss weights would likely add variance before improving BPB.
4. Toric phase-winding collections correctly show the flat irrational winding,
   embedded torus with local simplicial edges, and phase-coordinate traces.
5. Exact persistence-module morphism and nested simplex plots remain coherent.

### Decision

Do not promote the 40,000 checkpoint.  Restart from:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00039500.pt
```

Implemented the smallest high-impact scalar change:

| control | old | new |
|---|---:|---:|
| recovery `lr_multiplier` | `0.30` | `0.24` |
| recovery `medium_mix_ratio` | `0.18` | `0.35` |
| recovery `hard_mix_ratio` | `0.0` | `0.0` |
| recovery `complex_mix_ratio` | `0.0` | `0.0` |

This keeps the intervention faithful to the current ToricGT Parameter-Golf
architecture.  Random-order autoregressive graph decoding, tropical
ring/hybrid attention, toric memory, dense contest weights, embedding-space
GFlowNet graph-of-thought, relative Kolmogorov diagnostics, persistence
modules, Koszul audits, and toric phase-winding plots remain unchanged.

The next restart should use a fresh W&B run in the same project.  Reusing the
old run made W&B reject repeated checkpoint-step logs, so the next watcher must
point at the fresh run path.

Next review target: `40000`.

Acceptance criteria:

1. deterministic `val/bpb < 4.933094`;
2. `complexity/val/bpb < 4.822938`, preferably below `4.822091`;
3. local derivative from 39750 to 40000 non-positive;
4. train BPB band max below `4.45` despite higher medium mix;
5. mean branch BPB below `4.745`;
6. inclusion violation `0.0`, HDBSCAN stability above `0.90`, Slepian leakage
   `0.0`.

## Step 1500 Review: Early Step-1000 Valmix Replay

Date: 2026-06-01 UTC.

Run:

```text
amelie-iska-math/toricgt-parameter-golf/oai01000valmix3520260601T190357Z
```

Analyzed checkpoint:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00001500.pt
```

Analysis directory:

```text
outputs/post_resume_analysis/oai-bpb-valmix35-01000-20260601T190357Z/step-00001500
```

### Metric Categories

The automatic category pass reported:

| category | count | interpretation |
|---|---:|---|
| as desired | 184 | primary train likelihood and many topology proxies move in the right direction |
| as desired but too weak/slow | 86 | useful geometry/search structure exists, but does not yet dominate BPB |
| not as desired | 122 | validation BPB, toric active-face margin, binomial/leaf residuals, and some complexity symmetry terms remain weak |

### As Desired

| metric | value / behavior | explanation |
|---|---:|---|
| checkpoint train BPB | `5.2174 -> 3.5454` from step 1000 to 1500 | likelihood descent is real, not only logging noise |
| W&B train BPB median | `4.5279 -> 3.6060`, `-20.36%` | recovery controls sharply improve byte likelihood |
| W&B train loss median | `3.1385 -> 2.4995`, `-20.36%` | loss and BPB agree, so this is not a unit conversion artifact |
| GFlowNet diagnostic loss | `4.0121 -> 2.7193`, `-32.22%` | policy head remains compatible even with zero GFlowNet training weight |
| topology inclusion residual | `0.0` | nested complexes preserve filtration inclusions |
| exact directed-edge validity | `0.9364` | directed topology is mostly coherent |
| HDBSCAN stability | `0.8896` | relation neighborhoods are stable enough for diagnostics |
| Slepian concentration / leakage | `1.0 / 0.0` | toric spectral windowing is numerically clean |

Mathematically, the first 250 steps approximate a steep descent in
\(\mathbb E_{P_\alpha}[-\log p_\theta(x)]\) for the mixed stream
\(P_\alpha=(1-\alpha)P_{\rm easy}+\alpha P_{\rm medium}\) with
\(\alpha=0.35\).  The fact that BPB and NLL move together shows the descent is
in the actual prequential likelihood objective, not only in an auxiliary
regularizer.

### As Desired But Too Weak Or Slow

| metric | value / behavior | issue |
|---|---:|---|
| best branch BPB | `4.0452` | search finds branches better than mean, but not enough for promotion |
| mean branch BPB | `4.9275` | still far above target and close to validation weakness |
| best answer BPB | `3.5865` | local answer spans can improve, but full-context byte modeling is weak |
| MST efficiency | `0.7065` | hidden trajectories are organized, but not yet predictive enough |
| path smoothness | `0.0552` | smooth trajectories exist; low energy does not yet imply low BPB |
| GFlowNet entropy | `2.767 -> 2.758` | high but drifting downward; acceptable while GFlowNet weight is zero |
| toric memory entropy | `0.284 -> 0.288` | slightly healthier, but still low |
| simplex/tetrahedron plots | clustered in interior | reasoning geometry is alive but not landing on low-BPB boundary |

The simplex plots show a geometric gap: increasing reasoning budget moves
points outward in the reasoning/K/BPB triangle, but the low-BPB coordinate is
not the dominant attractor.  The GFlowNet branches therefore provide
test-time-scaling diversity, not yet a reliable likelihood improvement.

### Not As Desired

| metric | value / behavior | issue |
|---|---:|---|
| validation BPB | `5.3285` at step 1250 | worse than the inherited checkpoint gate `4.6961`; do not promote |
| score-first BPB | `5.3094` | score-first bias adaptation does not repair validation weakness |
| complexity validation BPB | `5.0305` | complexity probe agrees validation is weak |
| train BPB quadratic terminal slope | positive in all windows | descent is flattening and starting to bend upward |
| train BPB second derivative | positive over 1000-1500 | local curvature predicts another floor-bounce if unchanged |
| toric active-face margin | `-1.7017` geometry mean, `-1.3691` train last | tropical face confidence remains inverted |
| toric binomial residual | `1.7119` geometry mean; train binomial loss rose | toric relation probes are not ready to train against |
| toric leaf residual | `0.9988` | noncommutative phase leaf consistency remains poor |
| information symmetry gap | train LZMA gap rises in median | forward compression is improving faster than reverse explanation |

Finite-difference summary:

| metric/window | terminal slope per 1k | second derivative sign |
|---|---:|---|
| train BPB, 1000-1250 | `+5.07` | positive |
| train BPB, 1250-1500 | `+0.94` | positive |
| loss EMA, 1250-1500 | `-0.41` | slightly negative |
| complexity train BPB, 1250-1500 | `-2.50` | negative |

The raw BPB series fell sharply but then entered a curved basin.  The EMA and
complexity probe are less pessimistic, so this is not a hard failure; it is a
step-size/mix problem.  In Hessian language, we do not need a new architecture:
we need a smaller step in the locally sharp mixed-data direction, with enough
medium data to keep the validation derivative visible.

### Plot Review

Reviewed:

```text
plots_review_contact_sheet.png
metrics/core_metric_timeseries.png
metrics/recent_metric_slopes.png
simplex/reasoning_k_bpb_triangle.png
simplex/reasoning_k_bpb_*_tetrahedron.png
geometry/triangles/*.png
geometry/tetrahedra/*.png
geometry/trajectories/*trajectory_3d.png
geometry/trajectories/*toric_phase_winding_collection.png
geometry/trajectories/*energy_landscape.png
geometry/trajectories/*phase_energy.png
geometry/topology/*step_radius_hierarchy.png
geometry/topology/*exact_persistence_morphisms.png
geometry/topology/*noncommutative_heatmaps.png
geometry/topology/*toric_shadow_audit.png
```

Visual conclusions:

1. Core metric curves show a strong initial drop, a bounce near the validation
   point, then a shallow plateau.
2. The reasoning/K/BPB simplex places some branches near useful reasoning and
   compression directions, but low-BPB concentration is absent.
3. The 3D graph-of-thought trajectories branch coherently; solution-span
   markers are reachable, but terminal likelihood basins remain broad.
4. The toric phase winding collection now renders both the flat \(T^2\) winding
   and the embedded torus with local simplicial edges; the plot is functioning.
5. Ramachandran-style phase plots show structured pseudo-periodic bands rather
   than collapse.
6. Nested simplex hierarchy, exact persistence morphism, and noncommutative
   heatmap plots are coherent; topology is diagnostically useful.
7. Toric shadow plots show many occupied fan cells but thin margins and weak
   active-face stability.

### Decision

Do not continue the exact 1000-1500 controls.  Resume from the analyzed
step-1500 checkpoint with only scalar changes:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00001500.pt
```

Implemented changes in:

```text
config/train.parameter_golf_random_order_dense_valmix35_from1000.yaml
```

| control | 1000-1500 | 1500-2000 |
|---|---:|---:|
| phase name | `bpb_valmix35_recovery_01000_1500` | `bpb_curvature_damped_1500_2000` |
| LR multiplier | `0.24` | `0.18` |
| medium mix | `0.35` | `0.18` |
| hard mix | `0.0` | `0.0` |
| complex mix | `0.0` | `0.0` |
| GFlowNet/topology/toric/QAT loss weights | `0.0` | `0.0` |
| contrastive weight | `1e-4` | `5e-5` |
| shock guard ratio/delta | `1.03 / 0.07` | `1.025 / 0.06` |
| shock guard update scale | `0.08` | `0.06` |
| robust micro guard ratio/delta | `1.12 / 0.20` | `1.10 / 0.16` |
| robust micro guard min scale | `0.50` | `0.45` |

This is the smallest high-impact intervention consistent with the evidence:
reduce curvature and validation mismatch without touching the dense contest
model, random-order graph decoding, tropical/hybrid attention, toric memory,
GFlowNet policy, Kolmogorov diagnostics, persistence audits, or export path.

Next review target: `2000`.

Acceptance criteria:

1. `val/bpb < 5.3285`, preferably below `5.0`;
2. `complexity/val/bpb < 5.0305`;
3. train BPB terminal quadratic slope non-positive or materially smaller;
4. no shock-guard cascade after step 1500;
5. mean branch BPB below `4.90` and best branch BPB below `4.04`;
6. HDBSCAN stability above `0.88`, inclusion violation `0.0`, and Slepian
   leakage `0.0`;
7. toric active-face margin no worse than this checkpoint.

## Step 2250 GraphCG Restart Gate

Date: 2026-06-02 UTC.

Analysis directory:

```text
outputs/manual_analysis/oai-bpb-postbounce-02000_step2250_20260602T002659Z
```

Analyzed checkpoint:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00002250.pt
```

The active run had reached roughly step `2395` in the log, but the latest fresh
durable replay checkpoint was step `2250`, so the gate uses that checkpoint.

### Metric Summary

The W&B export ended at step `2380`.  Core metrics:

| metric | category | first median | last median | recent slope / 1k |
|---|---:|---:|---:|---:|
| `train/bpb` | desired | `3.8067` | `3.5449` | `-1.9207` |
| `train/loss` | desired | `2.6386` | `2.4571` | `-1.3313` |
| `train/gflownet_loss` | desired | `2.9575` | `2.5909` | `-5.2094` |
| `train/gflownet_entropy` | desired but slow | `2.7564` | `2.7567` | `+0.0057` |
| `train/gflownet_action_diversity` | desired but slow | `0.9946` | `0.9947` | `+0.0036` |

The scalar objective is improving, but the core panel shows a bounce around
steps `2240-2290` and flattening afterward.  The log tail still contains
useful low local BPB values near `3.49-3.60`, so the correct action is not a
large rollback.

### Geometry And Plot Review

The latest generated plots include:

```text
metrics/core_metric_timeseries.png
metrics/recent_metric_slopes.png
simplex/reasoning_k_bpb_triangle.png
geometry/trajectories/*_toric_phase_simplicial_trajectory.png
geometry/trajectories/*_toric_phase_winding_collection.png
geometry/topology/*_exact_persistence_morphisms.png
geometry_interactive/trajectories/*_toric_phase_simplicial_trajectory.png
geometry_interactive/trajectories/*_toric_phase_winding_collection.png
geometry_interactive/topology/*_exact_persistence_morphisms.png
```

The new analysis script now also writes interactive HTML overlays:

```text
geometry_interactive/trajectories/*_toric_phase_simplicial_trajectory.html
```

These HTML plots place reasoning trajectories directly on the embedded torus
projection, with local simplicial edges, GraphCG-margin marker sizes, NLL
colors, and analogical transport arrows.

Step-2250 geometry summary:

| metric | value |
|---|---:|
| records / branches | `3 / 9` |
| mean branch BPB | `4.8040` |
| best branch BPB | `3.4762` |
| mean answer BPB | `4.7765` |
| best answer BPB | `3.4762` |
| MST efficiency | `0.6674` |
| path smoothness | `0.0480` |
| directed asymmetry | `0.3449` |
| HDBSCAN stability | `0.9314` |
| exact directed-edge validity | `0.9486` |
| exact triangle validity | `0.7418` |
| toric shadow mean margin | `0.0227` |
| toric shadow min margin | `0.000122` |
| toric active-face margin | `-1.8464` |
| Slepian concentration / leakage | `1.0 / 0.0` |

Branch search is useful: best branch BPB is `1.33` below mean branch BPB.  The
simplex still places longer reasoning budgets away from the low-BPB corner, so
GFlowNet should be promoted only as a tiny branch-learning signal, not as a
dominant auxiliary objective.  Toric margins remain too thin for heavy
toric/Koszul pressure.

### Decision

Resume from step `2250` with a small real GraphCG chart-learning term and a
small GFlowNet trajectory-balance/entropy term.  This is the minimal correction
because the active 2000-2500 phase had GraphCG diagnostic metrics but zero
GraphCG optimization, while the geometry suite found a large branch gap that
should be learned cautiously rather than ignored.

Implemented in:

```text
config/train.parameter_golf_random_order_dense_valmix35_from1000.yaml
```

| control | previous 2000-2500 | new 2000-2500 |
|---|---:|---:|
| LR multiplier | `0.10` | `0.16` |
| clip norm | `0.36` | `0.42` |
| medium mix | `0.35` | `0.35` |
| GraphCG loss | `0.0` | `0.00006` |
| analogy lattice loss | `0.0` | `0.00001` |
| contrastive loss | `0.00002` | `0.00008` |
| GFlowNet loss | `0.0` | `0.00025` |
| GFlowNet entropy loss | `0.0` | `0.00005` at target `2.0` |
| toric/Koszul/flow/QAT losses | `0.0` | `0.0` |

The proposal helper

```text
scripts/propose_training_adjustments.py
```

agrees with this decision: BPB/loss slopes are still strong enough to avoid a
large rollback, GraphCG covariance requires a small chart-learning loss, branch
search should receive a tiny GFlowNet learning signal, and toric/Koszul losses
should stay off until active-face margins improve.

Next review target: `2750`, with the watcher required to use
`--pause-training-before-analysis` and
`--codex-review-hook scripts/codex_training_review_resume.sh`.

## Step 2750 GraphCG/GFlowNet Gate

Date: 2026-06-02 UTC.

Analysis directory:

```text
outputs/post_resume_analysis/oai-bpb-graphcg-gfn-02250-20260602T011416Z/step-00002750
```

Analyzed checkpoint:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00002750.pt
```

W&B run:

```text
amelie-iska-math/toricgt-parameter-golf/oai02250graphcggfn20260602T011416Z
```

### Metric Categorization

The automatic classifier found:

| category | count |
|---|---:|
| desired | `187` |
| desired but weak/slow | `92` |
| undesirable | `113` |

Core likelihood metrics are still broadly desired:

| metric | first median | last median | relative change | recent slope / 1k |
|---|---:|---:|---:|---:|
| `train/bpb` | `3.9243` | `3.6496` | `-7.00%` | `-0.1060` |
| `train/loss` | `2.7201` | `2.5297` | `-7.00%` | `-0.0735` |
| `train/total_loss` | `2.7213` | `2.5313` | `-6.98%` | `-0.0734` |
| `train/gflownet_loss` | `3.0693` | `2.8567` | `-6.93%` | `+0.1479` |
| `train/gflownet_entropy` | `2.7640` | `2.7724` | `+0.31%` | `+0.00019` |
| `train/gflownet_action_diversity` | `0.9958` | `0.9987` | `+0.29%` | `-0.00025` |

The checkpoint itself reports:

| checkpoint metric | value |
|---|---:|
| `train_bpb` | `3.63849` |
| `train_loss` | `2.52201` |
| `best_val_bpb` | `4.69611` |

The step-2500 W&B export showed `val/bpb=5.4630`, but the durable checkpoint
metadata still carries the better historical validation gate `4.6961`.  The
step-2750 checkpoint improves train BPB over step 2500 while preserving that
same best validation gate, so rolling back to step 2500 would discard useful
likelihood progress without evidence of validation improvement.

### Plot And Geometry Review

Reviewed plot set:

```text
metrics/core_metric_timeseries.png
metrics/recent_metric_slopes.png
simplex/reasoning_k_bpb_triangle.png
simplex/efficiency_triangle.png
geometry/trajectories/R2_gss1147-got_math_500k_got_math_energy_landscape.png
geometry/trajectories/R2_gss1147-got_math_500k_got_math_trajectory_3d.png
geometry/trajectories/R2_gss1147-got_math_500k_got_math_toric_phase_simplicial_trajectory.png
geometry/topology/R2_gss1147-got_math_500k_got_math_toric_shadow_audit.png
geometry/topology/R2_gss1147-got_math_500k_got_math_exact_persistence_morphisms.png
```

Geometry summary:

| metric | value | behavior |
|---|---:|---|
| mean branch BPB | `4.8281` | weak/slow |
| best branch BPB | `3.4624` | desired |
| mean answer BPB | `4.8014` | weak/slow |
| best answer BPB | `3.4624` | desired |
| MST efficiency | `0.6662` | weak/slow |
| path smoothness | `0.0895` | desired |
| directed asymmetry | `0.3573` | desired |
| directed cycle flux | `~0` | desired |
| HDBSCAN stability | `0.9235` | desired |
| exact edge validity | `0.9734` | desired |
| exact directed-edge validity | `0.9298` | desired |
| exact triangle validity | `0.7043` | weak/slow |
| topology analogical map loss | `0.1440` | weak/slow |
| toric active-face entropy | `0.6751` | weak/slow |
| toric active-face margin | `-1.8707` | undesirable |
| toric binomial residual | `0.9746` | undesirable |
| toric phase-leaf residual | `0.9754` | undesirable |
| toric shadow min margin | `0.0000945` | undesirable |
| Slepian concentration / leakage | `1.0 / 0.0` | desired |

The core metric plot shows a real early descent into a local minimum around
step `2440`, followed by a rebound and noisy plateau.  The last local
difference in train BPB is positive (`2740 - 2730 = +0.06994`) and the last
second difference is also slightly positive, but the full-window trend from
2250 to 2740 remains negative.  This is a weak floor-bounce basin, not
catastrophic divergence.

The simplex plots are useful but statistically underpowered: four records and
a narrow BPB color range.  They confirm the scoring and plotting path, but they
should not drive checkpoint selection.  The 3D trajectory and energy plots show
a dense local search cloud plus a few long terminal excursions.  This is valid
GFlowNet exploration, but the branch search has not yet made lower-BPB
terminals typical.  The toric phase/simplicial plot is rich but dense; the
numerical issue is not plotting, it is weak active-face margin and high
binomial/leaf residuals.

### Mathematical Interpretation

The likelihood objective is estimating a byte-level cross-entropy.  The
negative full-window slope means the stochastic gradient is still aligned with
compression, but the positive local first and second finite differences show
that the current optimizer scale and auxiliary terms are occasionally pushing
updates across a shallow basin floor.  Since the best validation gate is not
improved by the 2500 evaluation, validation BPB remains the promotion metric
and the train descent must be made smoother before heavier reasoning losses
are trusted.

GraphCG is behaving as a weak chart-regularizer: covariance and basis losses
improve, but basis coherence and orthogonal loss drift upward.  In geometric
terms, the learned chart is beginning to resolve concept directions, but the
frame is still poorly conditioned enough that strong auxiliary gradients can
rotate the chart faster than the byte-likelihood objective can absorb.

GFlowNet is noncollapsed: entropy is near the target and action diversity is
very high.  However, high entropy at this stage means broad exploration, not
yet calibrated high-reward sampling.  The best branch BPB is much better than
mean branch BPB, so the policy contains useful candidates, but the trajectory
balance signal should remain light until branch improvements become typical.

The nested directed topology terms are mostly healthy.  Inclusion violations
are zero, edge validity is high, HDBSCAN stability is high, and directed cycle
flux is essentially zero.  Triangle validity and analogical map loss remain
weak, so topology should remain a diagnostic and light shaping signal.

The toric audit is the strongest reason not to activate heavier toric/Koszul
losses yet.  Active-face margins are too thin and sometimes negative under the
audit convention, while binomial and phase-leaf residuals are near one.  In
tropical terms, the model is crossing chamber walls without stable margins; in
noncommutative-toric terms, the projected phase leaves are not yet organizing
latent search strongly enough to deserve a large gradient weight.

Relative Kolmogorov metrics are diagnostic only at this window.  The compressor
proxies have small sample counts and are heavily affected by input difficulty
changes, so their recent slopes should not override deterministic validation
BPB.  They remain useful for detecting whether analogical helper strings reduce
conditional description length, but they are not checkpoint gates yet.

### Decision

Resume from step `2750`, not from step `2500`.  The model improved train BPB
from `3.7255` to `3.6385` while preserving the same best validation BPB, and
there is no saved checkpoint closer to the raw step-2440 local minimum.

Implemented in:

```text
config/train.parameter_golf_random_order_dense_valmix35_from1000.yaml
```

The active 2500-3250 phase is changed from a sprint to a likelihood hold:

| control | previous 2500-3000 | new 2500-3250 |
|---|---:|---:|
| LR multiplier | `0.24` | `0.16` |
| grad clip | `0.50` | `0.42` |
| medium mix | `0.35` | `0.35` |
| GFlowNet loss | `0.00035` | `0.00025` |
| GFlowNet entropy loss | `0.00008` | `0.00005` |
| GraphCG loss | `0.00008` | `0.00005` |
| analogy lattice loss | `0.000015` | `0.00001` |
| contrastive loss | `0.00010` | `0.00006` |
| toric/Koszul/flow/QAT losses | `0.0` | `0.0` |

The following 3250-6000 stabilization phase is also damped so the next watcher
cannot accidentally enter a high-LR auxiliary-heavy regime before review:

| control | previous 3000-6000 | new 3250-6000 |
|---|---:|---:|
| LR multiplier | `0.72` | `0.24` |
| grad clip | `0.90` | `0.50` |
| GFlowNet loss | `0.0` | `0.00020` |
| GraphCG loss | `0.00005` | `0.00005` |
| analogy lattice loss | `0.00003` | `0.00001` |
| toric/Koszul losses | `0.00002 / 0.000005` | `0.0 / 0.0` |
| contrastive loss | `0.0005` | `0.00008` |

Next review target: `3250`.  The watcher should pause training before
analysis, run on CUDA/bf16, and call
`scripts/codex_training_review_resume.sh`.

Acceptance criteria:

1. step-3250 checkpoint train BPB below `3.60`;
2. validation BPB no worse than `5.4630`, with any improvement below `4.6961`
   treated as promotion-grade;
3. recent train BPB slope negative with no positive second-difference cluster;
4. mean branch BPB below `4.80` and best branch BPB below `3.45`;
5. HDBSCAN stability above `0.90`, exact edge validity above `0.95`, exact
   triangle validity above `0.72`;
6. toric active-face margin not worse than `-1.87` and toric shadow minimum
   margin above `1e-4`;
7. GraphCG basis coherence stops its rapid upward drift.

## Step 3250 Handoff Review: Damped Restart From Step 3000

Analysis directory:

```text
outputs/post_resume_analysis/oai-bpb-graphcg-gfn-02750-20260602T024036Z/step-00003250
```

Checkpoint:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00003250.pt
```

The requested training tmux session
`toricgt_oai_graphcg_gfn_02750_20260602T024036Z` was not present when the
review began.  The watcher log confirms that its attempt to pause the pane
failed with `can't find pane`, so this review treats training as already
stopped and performs a clean restart decision.

### Metric Categorization

| family | observed behavior | category |
|---|---|---|
| train BPB/loss | median improved from the first window, but recent slope is positive: `train/bpb` recent slope `+0.0697` BPB/1k and checkpoint BPB rose from `3.5026` at step `3000` to `3.6468` at step `3250` | desired but too weak/slow, locally undesirable |
| validation BPB/loss | validation worsened at the observed gates: `5.4630` at step `2500`, `5.5287` at step `3000`, while the stored best gate remains `4.6961` | undesirable |
| GFlowNet loss | decreased overall and no collapse was visible | desired |
| GFlowNet entropy/diversity | entropy is high and action diversity is near saturated; exploration is present but not translating to validation/BPB improvement | desired but too weak/slow |
| GraphCG and analogy metrics | chart losses and topology diagnostics are active, but recent slopes are dominated by high-variance complexity terms; no evidence justifies increasing their weight | desired but too weak/slow |
| relative Kolmogorov proxies | helper-string and analogical-transfer rewards are directionally useful, but sample count is too small and input-difficulty variance dominates many `stable` metrics | diagnostic only |
| trajectory geometry | mean MST efficiency improved to `0.6981`, but path smoothness worsened relative to the prior best window and directed asymmetry rose to `0.3900` | mixed; too weak/slow |
| nested directed topology | inclusion violations are zero, edge validity is high (`0.9769` undirected, `0.9305` directed), but triangle validity is only `0.6563` and HDBSCAN stability fell to `0.8725` | desired but too weak/slow |
| toric/tropical geometry | fan entropy improved (`0.7291`), but active-face margin remains weak (`-1.8092` audit convention), bend spikes are large, and binomial residual is high (`1.2960`) | undesirable as a training gate |
| Slepian/phase-band audit | concentration/leakage remain `1.0/0.0` | desired |

Metric counts from the automatic classifier were:

```text
as_desired: 329
as_desired_but_not_strong_or_fast_enough: 136
not_as_desired: 151
```

### Statistical Interpretation

The checkpoint sequence in the current run has a clean local likelihood
minimum at step `3000`:

```text
step 2500: train BPB 3.7255
step 2750: train BPB 3.6385
step 3000: train BPB 3.5026
step 3250: train BPB 3.6468
```

The finite difference from `3000 -> 3250` is positive, and the W&B recent
window also has positive BPB/loss slope.  This is the same floor-bounce
signature as earlier runs: the model finds a shallow compression basin, then
auxiliary exploration and optimizer scale push it across the basin wall before
validation catches up.  Mathematically, the byte cross-entropy gradient is
still useful, but the local curvature is high enough that a step of size
roughly `5e-6` with active GFlowNet/GraphCG steering overshoots the local
quadratic approximation.  The lack of validation improvement means the
overshoot is not buying a better generalizing basin.

The trajectory plots explain why this is not a reason to disable reasoning.
GFlowNet branches exist and MST efficiency improved, so search is not
collapsed.  The failure is calibration: branch exploration is producing long
latent jumps and noisy tropical wall crossings before the active-face margins
are stable.  In tropical terms, bend magnitudes are large while margins are
near zero, so the model crosses chamber walls without a robust argmax face.
In toric terms, fan occupancy and entropy are useful, but binomial and
phase-leaf residuals are still too large to use as heavy gradient signals.

The nested-simplicial diagnostics are also not failure signals.  Zero
inclusion violation and high edge validity show the directed filtration code
is coherent; weak triangle validity and HDBSCAN stability below `0.90` show
that the learned chart has not yet stabilized enough for stronger topological
losses.  Keep those terms light.

### Decision

Restart from:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00003000.pt
```

This is the last current-run checkpoint before the local derivative turns
positive.  Do not restart from step `3250`, and ignore the stale step-`3500`
checkpoint because its mtime is from `2026-05-27`, not this run.

Implemented scalar-only control changes in:

```text
config/train.parameter_golf_random_order_dense_valmix35_from1000.yaml
```

| control | previous 2500-3250 | new 2500-3250 |
|---|---:|---:|
| LR multiplier | `0.16` | `0.10` |
| grad clip | `0.42` | `0.36` |
| medium mix | `0.35` | `0.35` |
| GFlowNet loss | `0.00025` | `0.00012` |
| GFlowNet entropy loss | `0.00005` | `0.000025` |
| GraphCG loss | `0.00005` | `0.00004` |
| analogy lattice loss | `0.00001` | `0.000005` |
| contrastive loss | `0.00006` | `0.00004` |

| control | previous 3250-6000 | new 3250-6000 |
|---|---:|---:|
| LR multiplier | `0.24` | `0.10` |
| grad clip | `0.50` | `0.36` |
| medium mix | `0.18` | `0.35` |
| GFlowNet loss | `0.00020` | `0.00008` |
| GFlowNet entropy loss | `0.00004` | `0.00002` |
| GraphCG loss | `0.00005` | `0.00004` |
| analogy lattice loss | `0.00001` | `0.000005` |
| contrastive loss | `0.00008` | `0.00004` |

All architecture-level choices remain unchanged: dense contest weights,
random-order autoregressive graph decoding, hybrid tropical ring attention,
toric memory, GraphCG chart regularization, embedding-space GFlowNet
graph-of-thought, and Kolmogorov diagnostics.

### Next Review Gate

Restart from step `3000` and analyze the first fresh checkpoint at or above
step `3500` using a fresh `--min-mtime-unix`, because an old step-3500 file
already exists.  The watcher should pause training before analysis and run the
full CUDA/bf16 suite with Codex review hook.

Startup stability note: the first damped restart reached a finite first
training step, then the complexity diagnostic crashed because sampled GFlowNet
policy logits became non-finite during auxiliary metric evaluation.  The
checkpoint policy weights were finite, so this was handled as a runtime
diagnostic guard rather than a training-method change.  The model now
sanitizes non-finite GFlowNet policy/flow inputs and logits at the auxiliary
policy boundary before categorical sampling.  The architecture, loss weights,
and score-first random-order decoder are unchanged; this prevents monitoring
from terminating a valid training run when an auxiliary sampled-policy path
encounters a bad diagnostic batch.

Second startup stability note: the next restart survived policy sampling, but
the following step produced `nan` BPB/loss immediately after a Koszul SVD
warning.  The active phase had zero Koszul/toric/flow weights, so the likely
mechanism was disabled diagnostic losses entering the raw sum as `0 * NaN`.
The training script now skips zero-weight auxiliary losses explicitly and
finite-clamps only positive-weight auxiliary scalars before they enter the
optimizer.  This is not a relaxation of the primary objective: byte NLL remains
the base loss, and disabled topology/toric diagnostics remain logged rather
than optimized.

Acceptance criteria for the next gate:

1. train BPB below `3.50`, or at least negative recent BPB slope with no large
   positive second-difference cluster;
2. validation BPB below the step-3000 value `5.5287`, with any move toward
   the stored `4.6961` best gate treated as strong evidence;
3. mean branch BPB below `4.85` and best branch BPB below `3.50`;
4. HDBSCAN stability back above `0.90`, directed edge validity above `0.93`,
   and triangle validity above `0.68`;
5. toric shadow minimum margin above `1e-4`, fan entropy not below `0.67`, and
   bend spikes not increasing relative to this review.

## Step 3500 Handoff Review: Curvature-Capture Restart From Step 3000

Analysis directory:

```text
outputs/post_resume_analysis/oai-bpb-graphcg-gfn-03000-damped-auxguard-20260602T041537Z/step-00003500
```

Analyzed checkpoint:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00003500.pt
```

The watcher completed the CUDA/bf16 analysis suite and attempted to pause
`toricgt_oai_graphcg_gfn_03000_damped_auxguard_20260602T041537Z`.  The tmux
pane was already gone by the time the pause command ran, so training is treated
as stopped at the step-3500 gate.  No checkpoint files were deleted.

### Metric Categorization

| family | observed behavior | category |
|---|---|---|
| train BPB/loss | automatic report marks median BPB/loss as improved (`train/bpb` first median `3.8876`, last median `3.6790`, relative change `-5.37%`), but the raw trace has a shallow U-shape: sampled BPB reaches `3.3902` near step `3280` and rebounds before the saved step-3500 checkpoint (`3.6656`) | desired overall, but too weak/slow locally |
| checkpoint finite differences | checkpoint BPB sequence for the fresh run is `3000: 3.5026`, `3250: 3.6714`, `3500: 3.6656`; the saved checkpoints do not capture the around-3280 local minimum | undesirable as a promotion trajectory |
| validation/complexity validation | controller validation remained at `5.5431`, while the small complexity-val probe reported BPB around `5.24-5.27`; both remain above the stored historical best `4.6961` | undesirable as a checkpoint gate |
| GFlowNet loss | median decreased (`3.0596 -> 2.8559`), but the trace is strongly correlated with train BPB and spikes after the around-3280 low | desired but too coupled to BPB |
| GFlowNet entropy/diversity | entropy stays near maximum and action diversity remains around `0.9987`; no collapse, but the policy is too uniform to provide selective low-BPB search pressure | desired but too weak/slow |
| GraphCG | basis and covariance losses are stable and active; GraphCG does not explain the BPB bounce | desired |
| analogy/topology losses | lattice, HDBSCAN, and barcode losses improve in aggregate, but directed map and step-topology losses drift upward in the recent window | mixed; keep light |
| relative Kolmogorov proxies | prediction-target NCD improves and analogical-transfer rewards are active, but sample counts are small and many compressor statistics are dominated by row difficulty | diagnostic only |
| trajectory geometry | mean branch BPB `4.9971`, best branch BPB `4.0310`, answer-span best `3.6156`, MST efficiency `0.6946`, path smoothness `0.0772`; branch search is rich but not yet aligned with low-BPB terminals | desired but too weak/slow |
| nested directed topology | inclusion violation is zero, undirected edge validity is high (`0.9803`), directed edge validity is high (`0.9274`), triangle validity is moderate (`0.7040`), and directed cycle flux is effectively zero | desired, with triangle transport still weak |
| toric/tropical geometry | fan entropy is healthy (`0.7338`), occupied cells are broad (`13.21`), but mean active-face margin is tiny (`0.0239` in the toric shadow; `-1.821` in the signed task convention), bend magnitude is large, and leaf residual is near `1.0` | undesirable as a training gate |
| phase/Ramachandran plots | coherent phase islands are present, but local energy is not concentrated tightly inside them | desired structure, too weak/slow |
| Hessian/sharpness | disabled in this run; curvature inference comes from first/second finite differences and gradient-norm/BPB correlation | unavailable |

Automatic category counts:

```text
as_desired: 228
as_desired_but_not_strong_or_fast_enough: 87
not_as_desired: 86
```

### Mathematical and Statistical Interpretation

The useful signal is the local finite-difference pattern, not the aggregate
classifier count.  Over W&B steps `3001-3490`, train BPB has negative average
slope, but the raw series reaches a local minimum around step `3280` and then
returns toward the old floor.  This is a classic high-curvature stochastic
basin: the local quadratic approximation to byte cross-entropy is good enough
to descend, but gradient noise plus auxiliary search pressure pushes the
iterate across a shallow wall before a checkpoint captures the minimum.

The selected Spearman matrix supports this reading.  `train/gflownet_loss` is
positively correlated with `train/bpb`, and `train/grad_norm` is also
positively correlated with BPB/loss.  Thus the GFlowNet branch objective is
not broken, but in this window it is acting like an energy injection term
rather than a selective low-BPB refinement.  In variational language, the
policy entropy is high and the action distribution is broad; the trajectory
balance residual improves, but the induced samples are not yet concentrated
on low-loss terminal reasoning traces.

The toric shadow explains why stronger toric or topology weights would be the
wrong response.  Active Newton fan cells are occupied broadly, but tropical
margins are close to zero while bend magnitudes spike.  A small-margin
piecewise-linear region is precisely where chamber crossings are unstable: a
tiny parameter update can change the active face and make the local affine
model invalid.  Heavy toric/Koszul losses should wait until the byte model
forms a more stable likelihood basin.

The topology plots are better behaved than the BPB curve.  The directed
filtration has zero inclusion violation, high edge validity, nontrivial
directed asymmetry, and negligible cycle flux.  That means the
noncommutative/persistence machinery is coherent.  Triangle validity and
branch BPB are not yet strong enough to promote these terms.

### Decision

Restart again from the best available saved checkpoint before the rebound:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00003000.pt
```

The around-3280 local low is not saved.  The next run must save dense
checkpoints every `50` steps so the review can promote the actual local low
instead of the later rebound.  Keep the model architecture and method stack
unchanged: dense contest weights, random-order autoregressive graph decoding,
hybrid/tropical ring attention, toric memory, GraphCG, embedding-space
GFlowNet graph-of-thought, directed persistence/Koszul diagnostics, and
relative Kolmogorov metrics all remain active.

Implemented config-only changes:

| control | previous | new |
|---|---:|---:|
| warmup steps | `3500` | `4500` |
| shock guard grad norm | `0.36` | `0.32` |
| shock guard update scale | `0.04` | `0.025` |
| checkpoint interval | `250` | `50` |
| adaptive controller GFlowNet max | `0.006` | `0.001` |
| adaptive controller GFlowNet increase step | `0.00025` | `0.0` |
| 2500-3250 LR multiplier | `0.10` | `0.08` |
| 2500-3250 grad clip | `0.36` | `0.32` |
| 2500-3250 GFlowNet loss | `0.00012` | `0.00008` |
| 2500-3250 GFlowNet entropy loss | `0.000025` | `0.000015` |
| 2500-3250 analogy lattice loss | `0.000005` | `0.000003` |
| 2500-3250 contrastive loss | `0.00004` | `0.00003` |
| 3250-6000 LR multiplier | `0.10` | `0.08` |
| 3250-6000 grad clip | `0.36` | `0.32` |
| 3250-6000 GFlowNet loss | `0.00008` | `0.00004` |
| 3250-6000 GFlowNet entropy loss | `0.00002` | `0.00001` |
| 3250-6000 analogy lattice loss | `0.000005` | `0.000002` |
| 3250-6000 contrastive loss | `0.00004` | `0.00003` |

This is a curvature-capture change, not a retreat from ToricGT.  It reduces
only scalar pressure during the BPB consolidation window and saves more
checkpoints.

### Next Review Gate

Resume from step `3000`, analyze the first fresh checkpoint at or above step
`3500`, and use a fresh `--min-mtime-unix` so existing checkpoint filenames do
not contaminate the watcher.  The next gate should promote any fresh
checkpoint in `[3200,3500]` whose checkpoint BPB beats step `3000`, or whose
validation/complexity-validation BPB improves without worsening branch BPB.

Acceptance criteria:

1. saved checkpoint BPB below `3.50`, preferably capturing the local low before
   the rebound;
2. controller/validation BPB below `5.5431` and moving toward the historical
   `4.6961` gate;
3. best branch BPB below `4.0` and answer-span best below `3.60`;
4. GFlowNet entropy still noncollapsed but with lower correlation to BPB;
5. directed edge validity above `0.93`, triangle validity above `0.70`, and
   inclusion violation still zero;
6. toric shadow minimum margin not collapsing and bend magnitude not
   increasing.

## Step 4700 Handoff Review: Restart From the Step-3600 Curvcap Basin

Analysis directory:

```text
outputs/post_resume_analysis/oai-bpb-graphcg-gfn-04200-basinlock-20260602T131148Z/step-00004700
```

Analyzed checkpoint:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00004700.pt
```

The watcher paused the `04200_basinlock` run for review.  At review time no
active `train_parameter_golf_random_order.py` process remained, so the correct
action was to make an explicit continuation/restart decision and then start a
fresh tmux run.  The W&B screenshot and local logs agree: later replays from
`3600`, `3800`, `4000`, and `4200` all orbit the same shallow train-BPB floor
near `3.55-3.70`, while the `03300_curvcap` run contains the strongest early
drop and the lowest raw local samples around steps `3590-3610`.

### Run Name Clarification

The two requested 3k curvcap runs are not independent successful runs.

```text
toricgt_oai_graphcg_gfn_03000_curvcap_20260602T053947Z
toricgt_oai_graphcg_gfn_03000_curvcap_20260602T054043Z
```

The first launch failed immediately because the command used an unsupported
CLI argument:

```text
--wandb-run-id oai03000curvcap20260602T053947Z
```

The training script accepts W&B run ids through the `WANDB_RUN_ID` environment
variable, not a `--wandb-run-id` flag.  The second run, at `054043Z`, is the
actual resumed W&B-backed run from step `3000`; it analyzed at step `3500`.
The later `03300_curvcap` run is the relevant one for the current rollback
decision because it gives the cleanest low-BPB basin evidence.

### Metric Categorization

Automatic category counts for the step-4700 analysis:

```text
as_desired: 361
as_desired_but_not_strong_or_fast_enough: 137
not_as_desired: 127
```

The category count is not sufficient for the training decision.  The automatic
classifier marks `train/bpb` and `train/loss` as desired because the whole
`4200-4690` window has improved medians:

```text
train/bpb: 3.8633 -> 3.5885
train/loss: 2.6778 -> 2.4874
```

However, the recent slope for both is positive:

```text
train/bpb recent slope / 1k: +0.3557
train/loss recent slope / 1k: +0.2465
```

That is a late-interval basin exit.  For checkpoint promotion, recent slope
and finite differences dominate full-window median improvement.

| family | observed behavior | category |
|---|---|---|
| train BPB/loss | full-window medians improve, but recent slopes turn positive and the step-4700 checkpoint BPB is `3.5830` | desired overall, but locally undesirable |
| deterministic validation | W&B `val/bpb` remains around `5.5545`, and the checkpoint still carries historical `best_val_bpb=4.6961`; no validation promotion | undesirable |
| complexity validation | small probe `complexity/val/bpb=5.2570`, above the historical gate and not enough to justify continuation | undesirable |
| GFlowNet | entropy and action diversity remain noncollapsed, but `train/gflownet_loss` is still coupled to BPB and does not select lower-BPB terminals reliably | desired but too weak |
| GraphCG/analogical K metrics | relative-K and analogical-transfer diagnostics remain active and stable; they are not the cause of the floor | desired |
| nested topology | inclusion violation is zero, directed edge validity is high (`0.9264`), exact edge validity is high (`0.9779`), and exact morphisms are computed | desired |
| topology strength | exact triangle validity is `0.6724`, cycle rank is low, and branch map losses remain nonzero | desired but too weak |
| toric shadow | fan entropy `0.7256`, occupied cells `12.9`, and recurrence `0.0280` show noncollapsed toric structure | desired |
| tropical/toric margins | active-face margin is still signed-negative in the task convention (`-1.8164`), toric-shadow minimum margin is tiny (`1.92e-4`), and binomial residual is `1.242` | undesirable as an optimizer target |
| branch/test-time scaling | mean branch BPB `4.9930`, best branch BPB `4.0321`, mean answer BPB `4.8765`, best answer BPB `3.6111`; search finds better terminals but not a reliable low-BPB basin | desired but too weak |
| trajectory geometry | MST efficiency `0.6964`, smoothness `0.0732`, HDBSCAN stability `0.8720`, directed asymmetry `0.3888`, cycle flux near zero | desired structure, too weak for promotion |
| Hessian probes | disabled; curvature inference comes from checkpoint and W&B finite differences | unavailable |

### Plot Review

The contact sheet at

```text
outputs/post_resume_analysis/oai-bpb-graphcg-gfn-04200-basinlock-20260602T131148Z/step-00004700/review_contact_sheet.png
```

was reviewed.  The core metric plot shows exactly the statistical mismatch:
global descent followed by a late positive-slope bounce.  The energy landscape
and 3D GoT trajectory plots are nonblank, structured, and still show reachable
solution regions, but terminals are spread over shallow basins rather than
settling into a low-BPB attractor.  The toric phase simplicial trajectory plot
is dense but coherent; the directed-filtration and exact-persistence plots
show bounded nested-complex structure rather than topology collapse.  The
reasoning/K/BPB triangle places several high-reasoning points away from the
low-BPB edge, and the toric/GFlowNet/BPB tetrahedron shows toric entropy and
GFlowNet diversity preserved while low BPB remains the limiting coordinate.

Therefore the plots agree with the metric diagnosis: geometry is present and
noncollapsed; likelihood capture is the active failure mode.

### Finite-Difference Evidence

The `03300_curvcap` log gives the cleanest early basin:

```text
3300-3400: min BPB 3.580 at step 3372, median 3.748
3400-3500: min BPB 3.547 at step 3465, median 3.705
3500-3600: min BPB 3.364 at step 3590, median 3.619
3600-3650: min BPB 3.401 at step 3610, median 3.575
3650-3700: min BPB 3.566 at step 3651, median 3.697
```

The raw local minimum is step `3590`, and the nearby low highlighted by the
W&B screenshot is step `3610`.  There is no exact step-3610 checkpoint, so the
usable checkpoint is:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00003600.pt
```

Its checkpoint metadata is:

```text
train_bpb = 3.5206793755398187
train_loss = 2.440348982810974
best_val_bpb = 4.69610598173399
```

The discrete curvature is positive around the basin.  In notation
\(\Delta b_t=b_t-b_{t-h}\) and
\(\Delta^2b_t=b_{t+h}-2b_t+b_{t-h}\), the observed low has
\(\Delta b<0\) entering `3590-3610` and then \(\Delta b>0\) after it.  This is
not a capacity wall.  It is a shallow likelihood basin with stale optimizer
momentum and residual auxiliary/data-mixture curvature pushing the iterate
across the basin wall.

### Decision

Do not continue from step `4700`.  Restart from:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00003600.pt
```

with optimizer reset.  Resetting Adam moments is justified because the model
weights at step `3600` are good, but the previous momentum vector is exactly
the vector that carried subsequent runs through the low-BPB basin.

Implemented config-only changes:

| control | old active 3250-6000 value | new value |
|---|---:|---:|
| split historical capture | one phase `3250-6000` | keep `3250-3600`, add `3600-4300`, add `4300-6000` |
| 3600-4300 `lr_multiplier` | `0.016` | `0.036` |
| 3600-4300 `grad_clip_norm` | `0.18` | `0.16` |
| 3600-4300 `medium_mix_ratio` | `0.06` | `0.00` |
| 3600-4300 `gflownet_loss_weight` | `5e-7` | `1e-7` |
| 3600-4300 `graphcg_loss_weight` | `5e-6` | `1e-6` |
| 3600-4300 analogy/toric/Koszul/flow/memory/MTP/contrastive | diagnostic or tiny | `0` except GFlowNet/GraphCG dust |
| 4300-6000 `lr_multiplier` | `0.016` | `0.028` |
| 4300-6000 `medium_mix_ratio` | `0.06` | `0.02` |

This keeps the ToricGT Parameter-Golf architecture intact: dense contest
weights, random-order autoregressive graph decoding, hybrid/tropical ring
attention, toric memory, GraphCG frame, GFlowNet head, relative Kolmogorov
diagnostics, directed persistence/Koszul analyses, and plotting remain enabled.
Only scalar optimizer and auxiliary-loss controls changed.

### New Run

An initial recapture launch at `20260602T142041Z` was stopped before it reached
the next checkpoint because the effective LR under the current `4500`-step
warmup was only about `3.35e-7`, too low to recapture the basin aggressively.
The config was corrected to target about `1.1e-6` effective LR at step `3600`.

Started:

```text
tmux: toricgt_oai_graphcg_gfn_03600_recapture_20260602T142041Z
W&B:  amelie-iska-math/toricgt-parameter-golf/oai03600recapture20260602T142041Z
log:  logs/training/oai-bpb-graphcg-gfn-03600-recapture-20260602T142041Z.log
```

That launch was intentionally stopped before a checkpoint.  The active restart
is:

```text
tmux: toricgt_oai_graphcg_gfn_03600_recapture_20260602T142446Z
W&B:  amelie-iska-math/toricgt-parameter-golf/oai03600recapture20260602T142446Z
log:  logs/training/oai-bpb-graphcg-gfn-03600-recapture-20260602T142446Z.log
```

Startup logs should confirm:

```text
optimizer_state_loaded = false
resume checkpoint = random_order_step_00003600.pt
```

### Next Review Gate

Watcher:

```text
tmux: toricgt_watch_03600_recapture_20260602T142446Z
target checkpoint: >= 4100
output root: outputs/post_resume_analysis/oai-bpb-graphcg-gfn-03600-recapture-20260602T142446Z
```

It uses a fresh `--min-mtime-unix` captured at restart time, pauses training
before analysis, runs the CUDA/bf16 analysis suite, and hands the results back
through `scripts/codex_training_review_resume.sh`.

Acceptance criteria for the step-4100 review:

1. checkpoint train BPB below `3.5207`, with raw samples below `3.36` treated
   as evidence that the old basin has been recaptured;
2. recent train BPB/loss slope nonpositive after the first 100 fresh steps;
3. validation and complexity-validation BPB not worse than the step-4700
   analysis;
4. GFlowNet entropy/diversity noncollapsed even though its loss weight is dust;
5. GraphCG basis coherence and axis variance stable;
6. exact topology inclusion violation still zero, directed edge validity at or
   above `0.926`, and triangle validity moving toward `0.70`;
7. toric shadow fan entropy above `0.70`, occupied cells not collapsing, and
   bend/leaf residuals not worsening materially;
8. branch best BPB below `4.0` or answer-span best below `3.60` before
   reintroducing medium rows or heavier geometry losses.

## 2026-06-02 Step-2250 BPB-Cliff Watcher Review

Analysis directory:

```text
outputs/post_resume_analysis/oai-bpb-cliff-02000-20260602T143707Z/step-00002250
```

Analyzed checkpoint:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00002250.pt
```

The watcher paused the step-2000 BPB-cliff recovery and exported W&B metrics,
checkpoint metrics, simplex summaries, reasoning geometry summaries, and plot
artifacts.  The evidence does not support continuing from step `2250`.

### Metric Categorization

| metric family | category | evidence |
|---|---|---|
| `train/bpb`, `train/loss` | undesirable | checkpoint BPB was `4.1042`; W&B recent median rose from roughly `4.25` to `4.29`, with positive recent slope.  This is well above earlier 1190/1690/2190 historical lows near `3.37`--`3.42`. |
| validation BPB | undesirable | best validation BPB was still about `4.6961`, so the replay did not produce a validation-aligned likelihood basin. |
| GFlowNet loss | undesirable for this window | recent GFlowNet loss increased by roughly `18.8%`; because its scalar weight is zero/dust in the BPB capture window, the increase is diagnostic rather than directly causal, but it confirms unstable hidden search geometry. |
| reasoning geometry BPB | desired but too weak | branch/answer probes found best answer BPB around `3.56`, showing recoverable local structure, but mean branch BPB remained high (`4.82`) and did not pull the main likelihood down. |
| MST efficiency and trajectory smoothness | desired but too weak | MST efficiency around `0.706` and smoothness around `0.042` indicate nonchaotic local geometry; however, this geometry is not translating into stronger byte likelihood. |
| toric/topological diagnostics | desired but too weak | no evidence of catastrophic collapse was reported, but active geometry is not yet a strong BPB control signal in this early compression phase. |

### Mathematical Reading

The step-2250 behavior is a floor-bounce basin, not ordinary slow descent.  The
first finite difference of BPB is positive over the recent W&B window, and the
historical 1000--1700 traces show the same pattern: a strong negative slope is
followed by positive curvature once the optimizer reaches a narrow early basin.
Under SGD/Adam noise, this looks like a shallow likelihood trough with
high-variance microbatches repeatedly kicking the state across the local tangent
cone of descent.  Heavy reasoning auxiliaries are not the direct cause in this
window because their weights are already zero or tiny; the remaining cause is
that the neural LM is trying to learn byte-frequency and local-neighbor
statistics through dense hidden weights that are too slow relative to the BPB
cliff window.

The remedy should therefore be a legal compression-side intervention rather
than a larger architecture change.  Random-order graph decoding exposes a set of
already revealed vertices at every step.  A Dirichlet-smoothed prequential
prefix distribution

```text
p_t(b) = (n_t(b) + alpha) / (t + alpha V)
```

is a lawful score-before-update model because `n_t` only counts revealed tokens
from prior reveal steps.  A first additive product-of-experts trial showed the
danger of overconfidence: at step `1510`, `train/neural_bpb` was `4.2182` while
the context-corrected `train/bpb` rose to `4.7690`.  The corrected design mixes
the prior in probability space,

```text
q_t = (1 - lambda) q_neural + lambda p_t,
```

which caps the worst-case penalty from a weak prior by `-log(1-lambda)` while
still allowing entropy reduction when revealed-prefix statistics are predictive.
A zero-initialized trainable revealed-neighbor head then learns local graph
potentials over positions `p +/- r` when those vertices have already been
revealed.  This stays faithful to the ToricGT contest adapter: random-order
autoregressive graph decoding, dense packed weights, hybrid tropical ring
attention, toric memory, GraphCG, and GFlowNet heads are preserved.

### Decision

Action: `EDIT_AND_RESTART`.

Implemented controls:

| control | new setting |
|---|---:|
| resume checkpoint | `random_order_step_00001500.pt` |
| revealed Dirichlet prefix prior | enabled as probability mixture |
| prior alpha | `0.25` |
| prior mixture mass | `0.10` |
| trainable revealed-neighbor context | enabled |
| neighbor radius | `2` |
| neighbor logit weight | `0.45` |
| stream origin step | `1500` |
| shock/robust guard window | `1500`--`1750` |
| phase split | `1500`--`1600`, `1600`--`1700`, fallback `1700`--`1750` |
| next watcher target | `1700` |

Acceptance criteria for the next review:

1. fresh `train/bpb` should fall below the current 2250 value quickly and
   ideally return to the `3.37`--`3.55` historical early-basin band;
2. `train/revealed_context_prior_mixture_weight` should be nonzero and
   `train/revealed_neighbor_context_norm` finite;
3. future-token causal audit remains below tolerance;
4. first finite difference near 1650--1700 is nonpositive or only weakly
   positive with better validation behavior;
5. GFlowNet and topology diagnostics remain noncollapsed while byte likelihood
   owns the optimizer direction.

## 2026-06-02 Step-1700 Revealed-Context BPB Review

Analysis directory:

```text
outputs/post_resume_analysis/oai-bpb-revealed-context-01500-20260602T153522Z/step-00001700
```

Analyzed checkpoint:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00001700.pt
```

The watcher correctly paused the revealed-context replay at step `1700`.  The
run is not acceptable to continue from this checkpoint.  The live trace found a
useful local byte-likelihood valley around steps `1545--1550`, but the state
left that valley before step `1700`.

### Metric Categorization

| metric family | category | evidence |
|---|---|---|
| `train/bpb`, `train/loss`, `train/total_loss` | not as desired | the automatic report classifies all three as `not_as_desired`; first median BPB `4.19618` rose to last median `4.55003`, a relative worsening of `8.43%`; recent BPB slope is positive at `11.3025/1k` with `t=3.55`. |
| checkpoint BPB | not as desired | saved checkpoint metrics rose from the useful local `random_order_step_00001550.pt` value `3.682350` to `4.796419` at `1700`. |
| validation BPB | not as desired / insufficiently observed | `best_val_bpb` remains `4.69610598173399`; the old `eval_interval=250` did not give a fixed validation read inside the 1500--1700 rebound band. |
| GFlowNet loss | desired but not causal | recent GFlowNet loss fell about `20.22%`, but its training weight was zero in this window, so it is a diagnostic improvement rather than a byte-likelihood driver. |
| GFlowNet entropy/diversity | desired but too weak | entropy and diversity are noncollapsed but almost flat; they do not explain the BPB deterioration. |
| geometry branch BPB | desired but too weak | branch analysis found `best_answer_bpb=3.582888`, so local reasoning branches still contain recoverable signal, but mean geometry BPB is `4.912794`, too high to treat the checkpoint as good. |
| topology and toric diagnostics | desired but too weak | nested topology is stable (`mean_topology_inclusion_violation=0`, HDBSCAN stability about `0.894`), but toric active-face margins remain negative and leaf/binomial residuals are not yet useful BPB controls. |

### Mathematical Reading

This is a rebound from a narrow stochastic likelihood basin.  The relevant
finite-difference signal is not a single noisy batch: the report's median BPB
increases across the recent window, checkpoint BPB worsens materially by step
`1700`, and the live trace shows the same sustained high-BPB band after about
step `1653`.  The GFlowNet and topology metrics are not the cause, because
their loss weights are zero or diagnostic-only in this band.  The failure is an
optimizer-geometry problem: Adam is still allowed enough nonzero motion that
high-loss microbatches move the dense byte model across the local tangent cone
of the early descent basin.

The previous shock guard was also too permissive.  At the bad checkpoint, W&B
reports `train/shock_guard_active=1`, but the update scale was still `0.01` and
microbatch guard scaling averaged about `0.8904`; this observed the shock but
did not sufficiently project it away.  The correct local control is therefore a
trust-region-like floor lock: if the microbatch exceeds the running loss cap,
log it, but send no optimizer impulse from that shock.  This is equivalent to a
Huberized stochastic gradient estimator with a zero update on out-of-trust-region
samples during the early recovery band.

### Decision

Action: `EDIT_AND_RESTART`.

Restart from:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00001500.pt
```

Implemented scalar controls:

| control | old | new |
|---|---:|---:|
| fixed validation interval | `250` | `50` |
| fixed validation batches | `20` | `10` |
| shock guard end | `1750` | `1800` |
| shock loss ratio | `1.020` | `1.006` |
| shock loss delta | `0.040` | `0.012` |
| shock grad threshold | `0.18` | `0.00` |
| shock update scale | `0.010` | `0.000` |
| micro guard ratio | `1.020` | `1.006` |
| micro guard delta | `0.018` | `0.008` |
| micro guard min scale | `0.08` | `0.02` |
| 1500--1600 LR multiplier | `0.150` | `0.075` |
| 1500--1600 grad clip | `0.30` | `0.16` |
| 1600--1700 LR multiplier | `0.105` | `0.030` |
| 1600--1700 grad clip | `0.22` | `0.10` |
| recovery medium ratio override | `0.35` | `0.0` |
| GraphCG/contrastive dust in recovery | nonzero | `0.0` |
| next watcher target | `1700` | `1600` |

Acceptance criteria for the next review:

1. fixed validation BPB appears at step `1550` and `1600` and does not worsen
   relative to the step-1700 run;
2. median train BPB after step `1525` remains below `4.0`, with any local dip
   below `3.6` preserved by the shock guard;
3. no positive sustained first difference across 1550--1600;
4. shock guard may activate, but shock update scale must be zero in the recovery
   band;
5. GFlowNet/topology/toric metrics remain noncollapsed and diagnostic-only.

## 2026-06-02 Step-2125 Plateau-Control Update

The step-2250 automated review correctly selected `EDIT_AND_RESTART` from the
step-2075 checkpoint, because the checkpoint finite differences showed a
floor-bounce pattern:

```text
2075 train_bpb=3.688517
2100 train_bpb=3.887838
2125 train_bpb=4.268689
2150 train_bpb=4.021380
2175 train_bpb=3.975372
2200 train_bpb=4.482183
2225 train_bpb=4.689205
2250 train_bpb=4.647771
```

The first guarded replay recovered another fresh low at step `2125`:

```text
2075 train_bpb=3.688517
2100 train_bpb=4.058194
2125 train_bpb=3.683005
```

but the live per-step trace after 2125 was already oscillating around
`3.6--3.8` with the same high-loss impulses visible.  The active phase still
used `lr_multiplier=0.140` until step 2160, which is too large for the observed
basin width.  The replay also reused the old W&B run id while stepping backward
from 2250 to 2075, so W&B ignored logs below the existing run step and made the
next analysis partially blind to the fresh replay.

Action: `EDIT_AND_RESTART`.

Implemented controls:

| control | old | new |
|---|---:|---:|
| shock guard ratio | `1.018` | `1.010` |
| shock guard delta | `0.045` | `0.025` |
| shock guard grad threshold | `0.18` | `0.10` |
| shock guard update scale | `0.006` | `0.002` |
| micro guard ratio | `1.018` | `1.010` |
| micro guard delta | `0.020` | `0.010` |
| micro guard min scale | `0.08` | `0.05` |
| 2000--2160 LR multiplier | `0.140` | `0.060` |
| 2000--2160 grad clip | `0.30` | `0.16` |
| 2000--2160 medium mix | `0.35` | `0.25` |
| 2160--2225 LR multiplier | `0.110` | `0.045` |
| 2160--2225 grad clip | `0.22` | `0.12` |
| 2160--2225 medium mix | `0.35` | `0.20` |
| 2225--2500 LR multiplier | `0.075` | `0.030` |
| 2225--2500 grad clip | `0.18` | `0.10` |
| 2225--2500 medium mix | `0.30` | `0.12` |
| GraphCG/contrastive anchors in cliff band | nonzero dust | `0.0` |
| default rollback W&B id | inherited old id | fresh run id by default |
| default launcher naming | hard-coded `01500` | derived from `START_STEP` |

Restarted from:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00002125.pt
```

Fresh run:

| item | value |
|---|---|
| training tmux | `toricgt_oai_bpb_revealed_02125_20260602T195844Z` |
| watcher tmux | `toricgt_watch_bpb_revealed_02125_20260602T195844Z` |
| W&B run path | `amelie-iska-math/toricgt-parameter-golf/oai-revealed-20260602T195844Z` |
| training log | `logs/training/oai-bpb-revealed-context-02125-20260602T195844Z.log` |
| watcher log | `logs/training/oai-bpb-revealed-context-02125-20260602T195844Z.watcher.log` |
| next target | `2175` |
| optimizer state | preserved |

Acceptance criteria for the step-2175 review:

1. checkpoint BPB should remain near the fresh 2125 low instead of rebounding
   above `4.0`;
2. shock/micro guards should activate on high-loss impulses without letting
   those impulses dominate the optimizer update;
3. validation and complexity validation should be read from the fresh W&B run,
   not the old monotonic-step-filtered run;
4. Toric BGG, Koszul, toric, topology, memory, and GFlowNet losses remain
   diagnostic-only in this cliff band;
5. if BPB still plateaus, the next action should be an even shorter 25-step
   trust-region replay from the best checkpoint, not a continuation to 2325.

## 2026-06-02 Step-2150 Trust-Region Replay Update

The tightened step-2125 replay did not stabilize the basin.  The run used a
fresh W&B id and lower scalar step, so the signal is no longer an artifact of
out-of-order W&B logging.  It still rebounded:

```text
2125 train_bpb=3.683005
2150 train_bpb=4.051360
```

The live trace from 2125 to 2145 sat mostly around `4.0--4.29` BPB.  This means
the remaining `medium_mix_ratio=0.25` and nonzero motion are enough to leave the
byte-likelihood basin even under the stricter 2000--2500 guard.  The correct
next test is therefore not another 50--250 step continuation.  It is a
25-step trust-region replay from the best checkpoint with no medium rows and no
shock updates.

Action: `EDIT_AND_RESTART`.

Additional controls:

| control | old | new |
|---|---:|---:|
| shock guard ratio | `1.010` | `1.006` |
| shock guard delta | `0.025` | `0.008` |
| shock guard grad threshold | `0.10` | `0.00` |
| shock guard update scale | `0.002` | `0.000` |
| micro guard ratio | `1.010` | `1.006` |
| micro guard delta | `0.010` | `0.006` |
| micro guard min scale | `0.05` | `0.02` |
| 2000--2160 LR multiplier | `0.060` | `0.035` |
| 2000--2160 grad clip | `0.16` | `0.08` |
| 2000--2160 medium mix | `0.25` | `0.0` |
| 2160--2225 LR multiplier | `0.045` | `0.025` |
| 2160--2225 medium mix | `0.20` | `0.0` |
| 2225--2500 LR multiplier | `0.030` | `0.018` |
| 2225--2500 medium mix | `0.12` | `0.0` |

Restart target:

```text
resume: checkpoints/parameter_golf_oai_dense/random_order_step_00002125.pt
next gate: 2150
```

Acceptance criteria:

1. step-2150 checkpoint BPB must not rebound above `4.0`;
2. if step-2150 still rebounds, stop training and evaluate whether the
   checkpoint at 2125 should be promoted as the local low rather than trying to
   push through this cliff with the current data order;
3. do not activate Toric BGG, toric geometry, topology, GraphCG, memory, QAT,
   or GFlowNet losses in this band.

## 2026-06-02 Periodic BPB Codex Loop Activation

The training review process is now an explicit BPB optimization loop rather than
an ad hoc watcher handoff.  `scripts/codex_training_review_resume.sh` records a
persistent loop state, increments one review iteration for every completed
analysis gate, and launches Codex reviews until one of these conditions holds:

```text
BPB target: <= 1.2
maximum review iterations: 100
better-strategy sentinel: outputs/bpb_codex_loop_stop
state file: outputs/bpb_codex_loop_state.json
```

Every review prompt now includes the current primary BPB signal, the best BPB
seen in the loop, the iteration count, and the same resume/restart/reset action
contract.  Future launches through `scripts/launch_oai_bpb_cliff_recovery.sh`
propagate the loop variables into the watcher environment so restarts preserve
the same objective.  The loop remains BPB-first: Toric BGG and other structural
metrics stay diagnostic-only until they improve BPB or a later gate explicitly
activates them.

### Review 1 Decision: Better BPB Surface Required

The first loop review analyzed
`checkpoints/parameter_golf_oai_dense/random_order_step_00002150.pt` from
`outputs/post_resume_analysis/oai-bpb-revealed-context-02125-20260602T200545Z/step-00002150`.
The result is not promotable:

```text
checkpoint train_bpb: 4.052447
checkpoint best_val_bpb: 4.696106
geometry mean_bpb: 4.824010
geometry best_bpb: 4.055717
geometry best_answer_bpb: 3.622722
```

The core metric plot shows BPB, loss, and gradient norm rising together through
the measured window, despite the byte-only trust-region replay.  The simplex
plot does not show a branch allocation near the requested BPB regime either.
The correct action is therefore `EDIT_AND_RESTART`, but not another ordinary
hard-shard cliff replay.  The requested `<=1.2` target is the Parameter-Golf
FineWeb BPB regime; the current hard ToricGT validation BPB is an auxiliary
reasoning difficulty metric and is not comparable to the leaderboard target.

The better-strategy sentinel has been written to:

```text
outputs/bpb_codex_loop_stop
```

Next action: start the official-style FineWeb BPB scaffold, log `val_bpb`
periodically, and use that as the primary BPB gate while keeping the ToricGT hard
reasoning diagnostics as a secondary gate.

## 2026-06-03 Native All-Phases Step-1000 Review

Run:

```text
wandb: amelie-iska-math/toricgt-parameter-golf/toricgt-all-phases-20260602T214351Z
checkpoint: checkpoints/parameter_golf_all_phases/random_order_step_00001000.pt
analysis: outputs/post_resume_analysis/toricgt-all-phases-20260602T214351Z/step-00001000
```

Status:

```text
checkpoint train_bpb: 3.772965
checkpoint best_val_bpb: 8.238147
geometry mean_bpb: 7.948619
geometry best_bpb: 4.092317
geometry best_answer_bpb: 3.563895
metric categories: desired=305, weak/slow=121, undesirable=207
```

Decision: `CONTINUE` from the analyzed checkpoint with the same
`config/train.parameter_golf_all_phases.yaml`.  This is not yet a plateau:
training BPB fell from the random-init range to about `3.77` by step 1000, and
the local train curve is still descending.  Validation/geometry BPB remain high,
which is expected this early for the native hard-reasoning surface and should
not trigger BGG/topology activation.  Toric BGG remains instantiated for
diagnostics only; `toric_bgg_loss_weight=0.0` until the late Category O phase.

Operational note: the watcher successfully paused training and produced the
analysis, but the launched Codex review tmux exited before resuming.  A manual
continuation was started from step 1000 with optimizer state preserved:

```text
training tmux: toricgt_all_phases_continue_1000_20260603T004919Z
watcher tmux: toricgt_all_phases_continue_1000_20260603T004919Z_watcher
next gate: 1500
```

## 2026-06-03 FineWeb Full-Rank GraphCG Step-500 Review

Run:

```text
wandb: amelie-iska-math/toricgt-parameter-golf/toricgt-fineweb-fullrank-graphcg-from0-20260603T023620Z
checkpoint: checkpoints/parameter_golf_all_phases_fineweb_from0/random_order_step_00000500.pt
analysis: outputs/post_resume_analysis/toricgt-fineweb-fullrank-graphcg-from0-20260603T023620Z/step-00000500
loop: fineweb_fullrank_graphcg_from0_nonblocking, iteration 2 / 100, target BPB <= 1.2
```

Status:

```text
checkpoint train_bpb: 4.8141105017623
checkpoint train_loss: 3.3368871212005615
checkpoint best_val_bpb: 5.852094936744996
official-style OAI/FineWeb BPB: 4.734427155999241
official-style OAI/FineWeb loss: 3.2816548347473145
metric categories: desired=391, weak/slow=161, undesirable=222
geometry records: 4 selected records, 24 branches
geometry mean_bpb: 6.496695836385091
geometry best_bpb: 4.983981132507324
geometry best_answer_bpb: 4.9085289770859815
mean MST efficiency: 0.20736046289367974
mean path smoothness: 0.4451148548712142
```

Two-gate read:

- Desired: FineWeb/OAI BPB improved from `5.60212` at step 250 to `4.73566`
  at step 500, curated validation BPB improved from `6.24077` to `5.85209`,
  train BPB fell from `8.14068` at step 5 to `4.81411` at step 500, and the
  latest checkpoint finite difference for train BPB is still negative
  (`d1=-0.04883`, `d2=-0.03018`).  This is not a floor-bounce signal.
- Desired but too weak or slow: validation has only two samples, so no
  reliable second-difference test exists yet; OAI eval used only 8 batches;
  complexity validation BPB remains `5.47885`; Argmax byte accuracy is only
  `0.173828`; GraphCG is stable but barely moving (`graphcg_loss` slope about
  `-0.00591` per 1k steps, lattice margin down to `0.01452`); simplex records
  are all high BPB (`9.36--9.46`) with only modest MST efficiency
  (`0.205--0.241`).
- Undesirable: GFlowNet entropy fell from `2.37496` to `1.43440`, action
  diversity fell from `0.94055` to `0.52161`, toric memory entropy fell from
  `0.92286` to `0.70924`, active-face entropy fell from `0.64972` to
  `0.32966`, active-face margin worsened to about `-3.49`, BGG standard
  leakage rose to `0.46325`, and BGG resolution consistency slipped to
  `0.95309`.

Mathematical/statistical interpretation:

- The BPB gate is improving on both calibration streams.  In bits per byte,
  the FineWeb delta over steps 250--500 is `-0.86647`, or about `-3.47` BPB
  per 1k steps, while curated validation changes by `-0.38868`, or about
  `-1.55` BPB per 1k steps.  Both first derivatives are negative.  The train
  second difference is also negative at step 500, so there is no evidence yet
  for positive-curvature floor bounce.
- The reasoning gate is partially present but not strong.  Branch replay has a
  useful mean-vs-best gap (`mean_bpb - best_bpb ~= 1.5127`), but the GFlowNet
  policy is collapsing in entropy/diversity while the branch search itself is
  still only reaching `4.98` best branch BPB.  This supports continued branch
  diagnostics and later tiny GFlowNet training, not a step-500 restart.
- Kolmogorov proxies improved on validation NCD: prediction-target NCD fell to
  `0.92453` with lzma and `1.00867` with zlib, but zlib is still above 1.0 and
  byte accuracy is low.  The relative-K gains are therefore weak compression
  evidence, not a promotion signal.
- GraphCG basis behavior is valid and full-rank but undertrained.  Axis entropy
  remains near `0.99781`, basis coherence is low and stable, but basis loss and
  lattice margin do not yet show strong alignment.  Keeping the existing tiny
  `graphcg_loss_weight=5e-5` is enough for this early BPB-first phase.
- Persistence/simplex/Koszul topology is coherent but not decisive.  Koszul
  exactness residual improved to `0.01225`; chart coverage is `1.0`; directed
  cycle flux is numerically zero; HDBSCAN stability is about `0.53` in W&B and
  `0.745` in the branch geometry suite.  Persistence loss and topology losses
  are too high to activate as primary gradients.
- Toric BGG diagnostics are mixed: `d2_residual=0.04923`,
  `gale_dual_consistency=0.08107`, `koszul_linearity_residual=0`, and
  signature smoothness is low, but standard leakage is high.  BGG remains a
  diagnostic, not a training loss, until BPB is closer to the target.
- The phase/energy/trajectory plots are nonblank and structurally valid.  All
  70 PNG plots decoded with non-flat pixel variance.  The inspected bundle
  includes metrics plots, simplex triangles/tetrahedra, 3D trajectories,
  Ramachandran-style phase-energy and winding views, energy landscapes, and
  topology audits.  Trajectory length is still large (`mean path_length
  ~= 2906`, `trajectory_tokens=2048`), smoothness is moderate, and MST
  efficiency remains low; this is usable geometry, not yet efficient reasoning.
- No OOD transfer claim is warranted for this run.  Phase 0 has
  `fineweb_mix_ratio=1.0`, so a good FineWeb result here is mostly direct
  FineWeb exposure.  Any future OOD-transfer claim must control against
  tokenizer convention, n-gram/context-tree baselines, dataset easiness, and
  equal-budget FineWeb-only versus hard-data-only ablations.
- Hessian sharpness did not provide rollback evidence.  `probe_loss` improved
  from `4.01434` at step 250 to `3.36377` at step 500, while trace and dominant
  curvature returned NaN, so there is no measured sharp basin or positive
  curvature alarm.

The adjustment proposal from `scripts/propose_training_adjustments.py` was run
on the analysis directory.  It recommended tiny GraphCG/GFlowNet activation
because branch replay has a clear gap and GraphCG is slow, while also warning
that BPB/loss have strong negative slopes and large resets should be avoided.
I treat that as evidence for the next review, not as authority for an immediate
restart: at step 500 the BPB gate is still descending, and activating new
gradient terms would trade a live negative BPB derivative for a speculative
reasoning repair.

Decision: `CONTINUE`.

No code or config scalar was changed.  The better-strategy stop sentinel was
not written.  Training was explicitly unpaused with `SIGCONT` and handed back to
the active supervisor.  The live process had already advanced beyond the
analyzed checkpoint, preserving optimizer state and W&B continuity.

Operational handoff:

```text
supervisor tmux: toricgt_supervisor_fineweb_fullrank_graphcg
training tmux: toricgt_fineweb_fullrank_graphcg_live
requested training tmux alias from prompt: toricgt_pg_oai was not present
training log: logs/parameter_golf_all_phases/toricgt-fineweb-fullrank-graphcg-from0-20260603T023620Z/resume_00000250_20260603T031530Z/train.log
active W&B run path: amelie-iska-math/toricgt-parameter-golf/toricgt-fineweb-fullrank-graphcg-from0-20260603T023620Z
checkpoint selected for decision: checkpoints/parameter_golf_all_phases_fineweb_from0/random_order_step_00000500.pt
next watcher tmux: toricgt_fineweb_fullrank_graphcg_watcher
next watcher target: 750
next watcher device/precision: cpu / fp32
next watcher log: logs/parameter_golf_all_phases/toricgt-fineweb-fullrank-graphcg-from0-20260603T023620Z/supervisor/analysis_target_00000750_20260603T035310Z/watcher.log
next watcher pause behavior: non-interrupting; no --pause-training-before-analysis
loop env: BPB_TARGET=1.2, BPB_MAX_REVIEW_ITERATIONS=100,
          BPB_LOOP_STATE=/home/iska/Documents/amelie/bio/ToricGT/outputs/bpb_codex_loop_state_fineweb_fullrank_graphcg_from0.json,
          BPB_LOOP_STOP_FILE=outputs/bpb_codex_loop_stop_fineweb_fullrank_graphcg_from0,
          BPB_LOOP_NAME=fineweb_fullrank_graphcg_from0_nonblocking
```

## 2026-06-03 FineWeb Full-Rank GraphCG Step-750 Review

Run:

```text
wandb: amelie-iska-math/toricgt-parameter-golf/toricgt-fineweb-fullrank-graphcg-from0-20260603T023620Z
checkpoint: checkpoints/parameter_golf_all_phases_fineweb_from0/random_order_step_00000750.pt
analysis: outputs/post_resume_analysis/toricgt-fineweb-fullrank-graphcg-from0-20260603T023620Z/step-00000750
loop: fineweb_fullrank_graphcg_from0_nonblocking, iteration 3 / 100, target BPB <= 1.2
```

Status:

```text
checkpoint train_bpb: 4.67823010500331
checkpoint train_loss: 3.242702007293701
checkpoint best_val_bpb: 5.852094936744996
checkpoint best_oai_competition_bpb: 4.60303929004609
official-style OAI/FineWeb BPB: 4.603772624069369
official-style OAI/FineWeb loss: 3.191092014312744
metric categories: desired=394, weak/slow=144, undesirable=236
geometry records: 4 selected records, 24 branches
geometry mean_bpb: 6.699534634749095
geometry best_bpb: 4.897327899932861
geometry best_answer_bpb: 4.811350524181253
mean MST efficiency: 0.28028703928160065
mean path smoothness: 0.3348794286786749
```

Two-gate read:

- Desired: official-style FineWeb/OAI BPB improved across review gates
  (`5.60212` at step 250, `4.73566` at step 500, `4.60377` at step 750);
  train BPB improved from early `8.14` to `4.6735` in W&B history and
  `4.6782` in checkpoint metadata; train/loss fell similarly; GraphCG basis
  coherence stayed small (`0.000576`); Koszul exactness stayed good
  (`0.01205`); Toric BGG resolution consistency stayed usable (`0.95433`);
  topology boundary residual and directed cycle flux stayed essentially zero.
- Desired but too weak or slow: hard curated validation is still far from the
  target (`best_val_bpb=5.85209`), score-first validation is only slightly
  better (`5.83782`), OAI evaluation uses only 8 batches, and validation has
  only two logged W&B points.  Geometry branches have useful isolated best
  BPB (`4.8973`) but poor mean BPB (`6.6995`).  MST efficiency is coherent but
  modest and trajectory smoothness is not high enough to make branch replay a
  strong test-time-scaling mechanism.
- Undesirable: GFlowNet entropy collapsed from `2.47766` at step 250 to
  `0.30816` at step 745, action diversity fell from `0.92402` to `0.11152`,
  toric memory entropy fell from `0.92075` to `0.24652`, GraphCG axis
  variance collapsed to `4.67e-05`, active-face entropy fell to `0.29784`,
  and gradients show local variability.  Hessian trace/dominant curvature/HVP
  were NaN, so no sharpness rollback signal was available.

Mathematical/statistical interpretation:

- The BPB gate is still improving.  FineWeb/OAI BPB has negative first
  differences over both observed intervals: `-0.86647` from 250 to 500 and
  about `-0.13188` from 500 to the analyzed CPU eval at 750.  The second
  difference is positive, meaning improvement is slowing, but the first
  derivative is not positive.  This is not a rollback basin by itself.
- The hard-reasoning gate disagrees with the BPB gate.  The model is learning
  FineWeb byte prediction, but the learned branch policy is losing entropy and
  diversity.  In the geometry suite, per-record branch BPB standard deviation
  is only about `0.001--0.003`, so branch replay is nearly invariant inside
  each record rather than exploring distinct useful continuations.
- Kolmogorov and relative-K proxies are mixed.  Prediction and analogical
  transfer rewards exist, and some validation lzma NCD metrics are stable, but
  argmax byte accuracy is only about `0.17` and zlib NCD remains near or above
  `1.0`.  These are weak compression-proxy signals, not promotion signals.
- GraphCG is valid but underactive.  Full-rank basis coherence is stable and
  low, while axis variance has collapsed; this supports a tiny chart-learning
  lift rather than a topology-heavy intervention.
- Persistence, simplex, and Koszul topology are coherent.  Exact persistence
  has no boundary residual; directed cycle flux is numerical zero; HDBSCAN
  stability is reasonable (`0.7424` in geometry); Slepian concentration is
  `1.0` with zero leakage.  These diagnostics should remain active, but their
  loss weights should stay zero during this BPB-first window.
- Toric BGG diagnostics are mixed but consistent enough to keep as probes:
  `d2_residual ~= 0.04786`, `gale_dual_consistency ~= 0.09833`,
  `koszul_linearity_residual=0`, `signature_smoothness ~= 0.000188`, and
  `standard_leakage ~= 0.44602`.  BGG should not be promoted to a training
  loss yet.
- The plot review covered metric plots, simplex triangles/tetrahedra, geometry
  triangles/tetrahedra, 3D trajectories, Ramachandran-style phase-energy plots,
  toric winding plots, energy landscapes, and topology audits.  The visuals
  show coherent but compact/noisy branch geometry: lower BPB aligns more with
  low energy and terminal confidence than with high diversity, toric entropy,
  or long smooth reasoning trajectories.
- No OOD-transfer claim is justified for this phase.  `fineweb_mix_ratio=1.0`,
  so the FineWeb improvement is direct calibration exposure.  Future transfer
  claims still need tokenizer, n-gram/context-tree, dataset-easiness,
  FineWeb-only, hard-data-only, and equal-budget mixed controls.

The adjustment proposal from `scripts/propose_training_adjustments.py` was run
on the analysis directory and saved as:

```text
outputs/post_resume_analysis/toricgt-fineweb-fullrank-graphcg-from0-20260603T023620Z/step-00000750/PROPOSED_TRAINING_ADJUSTMENTS.md
outputs/post_resume_analysis/toricgt-fineweb-fullrank-graphcg-from0-20260603T023620Z/step-00000750/proposed_training_adjustments.json
```

I treated it as evidence, not authority.  Its recommendation for tiny
GFlowNet, entropy, GraphCG, analogy, and contrastive activation matches the
two-gate conflict: BPB is still descending, so the edit must be small; the
reasoning gate is collapsing, so doing nothing risks locking in a degenerate
branch policy.

Decision: `EDIT_AND_RESTART`.

Minimal scalar/config changes:

```text
byte_warmup gflownet_loss_weight: 0.0 -> 0.00025
byte_warmup gflownet_entropy_weight: 0.0 -> 0.00005
byte_warmup graphcg_loss_weight: 0.00005 -> 0.00008
byte_warmup analogy_lattice_loss_weight: 0.0 -> 0.00001
byte_warmup contrastive_loss_weight: 0.00025 -> 0.00040
base training weights mirrored to the same values
unchanged: lr schedule, FineWeb mix ratio, hard/complex mix ratios,
           toric_geometry_loss_weight, koszul_persistence_loss_weight,
           toric_bgg_loss_weight, trajectory/memory/topology losses
```

Process-control change:

```text
scripts/supervise_parameter_golf_training.py now honors BPB_LOOP_STATE,
BPB_LOOP_STOP_FILE, and BPB_LOOP_NAME from the supervisor environment when it
launches non-blocking watcher sidecars.
```

Operational handoff:

```text
supervisor tmux: toricgt_supervisor_fineweb_fullrank_graphcg
training tmux: toricgt_fineweb_fullrank_graphcg_live
watcher tmux: toricgt_fineweb_fullrank_graphcg_watcher
restart checkpoint: checkpoints/parameter_golf_all_phases_fineweb_from0/random_order_step_00000750.pt
training log: logs/parameter_golf_all_phases/toricgt-fineweb-fullrank-graphcg-from0-20260603T023620Z/supervisor/restart_000_20260603T044231Z/train.log
watcher target: 1000
watcher log: logs/parameter_golf_all_phases/toricgt-fineweb-fullrank-graphcg-from0-20260603T023620Z/supervisor/analysis_target_00001000_20260603T044231Z/watcher.log
watcher device/precision: cpu / fp32
watcher pause behavior: non-interrupting; no --pause-training-before-analysis
W&B run path: amelie-iska-math/toricgt-parameter-golf/toricgt-fineweb-fullrank-graphcg-from0-20260603T023620Z
loop env: BPB_TARGET=1.2, BPB_MAX_REVIEW_ITERATIONS=100,
          BPB_LOOP_STATE=/home/iska/Documents/amelie/bio/ToricGT/outputs/bpb_codex_loop_state_fineweb_fullrank_graphcg_from0.json,
          BPB_LOOP_STOP_FILE=outputs/bpb_codex_loop_stop_fineweb_fullrank_graphcg_from0,
          BPB_LOOP_NAME=fineweb_fullrank_graphcg_from0_nonblocking
```

Note: the old live process had advanced past step 900 without a newer
checkpoint.  The controlled restart intentionally selected the analyzed step
750 checkpoint so the config edit applies from a saved state.  W&B may ignore
log rows below the previously reached W&B step until the restarted process
passes that step, but local training, checkpointing, and the non-blocking
watcher are active.

## 2026-06-03 Revealed FineWeb BPB Recovery Step-4500 Review

Run:

```text
wandb: amelie-iska-math/toricgt-parameter-golf/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z
checkpoint: checkpoints/parameter_golf_all_phases_fineweb_from0/random_order_step_00004500.pt
analysis: outputs/post_resume_analysis/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/step-00004500
loop: all_phases_supervised_watchdog, iteration 3 / 100, target BPB <= 1.2
```

Status:

```text
checkpoint train_bpb: 4.640970192680871
checkpoint train_loss: 3.2168754041194916
checkpoint best_val_bpb: 5.359212953324989
checkpoint best_oai_competition_bpb: 4.5321342827483715
official-style OAI/FineWeb BPB: 4.531955764756776
official-style OAI/FineWeb loss: 3.14131236076355
metric categories: desired=444, weak/slow=192, undesirable=138
geometry records: 4 selected records, 24 branches
geometry mean_bpb: 7.249248186747233
geometry best_bpb: 5.000315189361572
geometry best_answer_bpb: 4.882333955567765
mean MST efficiency: 0.4691183015454703
mean path smoothness: 0.13947036705111682
hessian trace estimate: -13.477617263793944
hessian dominant curvature: 8.59504234540509e-06
```

Two-gate read:

- Desired: official-style FineWeb/OAI BPB and validation BPB improved across
  the review loop.  The loop history moved from `best_val_bpb=5.382641461921426`
  at step 4000 to `5.373151803212118` at step 4250 and `5.359212953324989` at
  step 4500.  OAI BPB moved from `4.54670499689477` at step 4000 to
  `4.540763684238064` in W&B at step 4250 and `4.531955764756776` in the
  analyzed checkpoint eval.  Future-permutation audit stayed zero; FineWeb
  calibration was active; artifact size and parameter count were stable.
- Desired but too weak or slow: train BPB and loss are mostly flat/noisy over
  steps 3755--4495.  W&B recent train BPB slope is positive
  (`+0.22511659058030803` per 1k steps), while validation/OAI slopes remain
  negative but sparse.  Hard curated validation is still far from the BPB target
  (`best_val_bpb=5.3592`), and OAI eval covers only 8 batches.
- Undesirable: robust micro-loss guard activity rose (`fraction=0.125` at the
  latest W&B row), toric binomial loss/residual showed the strongest adverse
  recent slopes, toric memory entropy is low (`0.029866` in W&B), and some
  analogical/relative-K stability proxies moved too much.  These are reasoning
  gate warnings, not yet evidence to abandon the FineWeb-first warmup.

Mathematical/statistical interpretation:

- The BPB gate has not reached the target, but its validation derivative is
  still negative.  The finite difference for W&B `val/bpb` from step 4000 to
  4250 is about `-3.80e-5` BPB/step; OAI `bpb` over the same interval is about
  `-2.38e-5` BPB/step.  The checkpoint metadata improves again at step 4500.
  This rejects rollback: there is no positive validation first derivative and
  no measured floor-bounce basin.
- Train BPB is noisy: minimum `4.33499` at step 4325, maximum `5.04374` at
  step 4050, and latest `4.57025` at W&B step 4495.  The positive recent train
  slope should be monitored, but it is dominated by microbatch variance and is
  weaker evidence than deterministic validation.
- Hessian probes do not justify rollback.  Dominant curvature is tiny positive
  (`8.6e-6`) and trace is negative, so the checkpoint does not look like a
  sharp positive-curvature trap.
- The phase is still `byte_warmup`: `fineweb_mix_ratio=1.0` and hard/complex
  microbatch fractions are zero.  Therefore the current FineWeb result is direct
  FineWeb calibration, not an OOD-transfer claim.  Any future transfer claim
  still needs tokenizer, n-gram/context-tree, dataset-easiness, FineWeb-only,
  hard-data-only, and equal-budget mixed controls.
- GFlowNet branch replay is alive but weak.  Entropy and action diversity are
  high in training (`entropy ~= 2.771`, diversity `~= 0.9986`), but geometry
  branch BPB has a large mean-best gap (`7.2493 - 5.0003 ~= 2.249`), so the
  policy is not reliably selecting low-BPB branches.
- Kolmogorov and relative-K proxies are mixed.  Prediction target NCD remains
  high (`complexity/val/prediction_target_ncd_lzma_mean ~= 0.9269`), argmax byte
  accuracy is low (`0.1738` on curated validation), and several relative-K
  metrics are categorized undesirable.  These proxies do not override the BPB
  gate.
- GraphCG basis behavior is acceptable for the current phase.  Basis coherence
  decreased from `0.0791` to `0.07275`; graphcg loss weight is still the tiny
  phase value (`5e-5`).  The proposal's GraphCG-lift suggestion is noted for
  later but is not a reason to restart while validation BPB is descending.
- Persistence/simplex/Koszul topology is coherent but not promotion-grade.
  Directed cycle flux is numerical zero, exact edge validity is high
  (`0.9753`), exact triangle validity is moderate (`0.4858`), chart coverage is
  `1.0`, and HDBSCAN stability is usable (`0.7739` in geometry).  Koszul and
  topology losses should remain probes in this BPB-first window.
- Toric BGG diagnostics are mixed but stable enough as probes:
  `resolution_consistency=0.94384`, `d2_residual=0.05985`,
  `gale_dual_consistency=0.18488`, `koszul_linearity_residual=0`,
  `standard_leakage=0.46279`, and signature smoothness is low but drifting.
  BGG remains diagnostic-only.
- Plot review covered metric plots, recent-slope bars, simplex
  triangles/tetrahedra, geometry triangles/tetrahedra, 3D GoT trajectories,
  Ramachandran-style phase-energy plots, toric shadow/topology audits, and
  energy landscapes.  The plots are structured and non-collapsed; lower BPB is
  not yet well aligned with stronger MST efficiency, toric entropy, or smooth
  trajectory flow.

The adjustment proposal from `scripts/propose_training_adjustments.py` was run
on the analysis directory and saved as:

```text
outputs/post_resume_analysis/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/step-00004500/training_adjustment_proposal.md
outputs/post_resume_analysis/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/step-00004500/training_adjustment_proposal.json
```

It recommended lower LR or rollback, tiny GraphCG/GFlowNet activation, and
continued zero toric/Koszul loss while toric margins are weak.  I treated that
as evidence, not authority.  The active config already has FineWeb-only warmup,
tiny GraphCG/GFlowNet weights, zero toric/Koszul losses, and improving
validation/OAI BPB, so a restart would add process risk without a validated
gain.

Decision: `CONTINUE`.

No code or config scalar was changed.  The better-strategy stop sentinel was
not written.  Training was already active again when inspected, so the correct
handoff is to preserve the active supervisor and live training process rather
than force a restart from the step-4500 checkpoint.

Operational handoff:

```text
supervisor tmux: toricgt_supervisor_fineweb_bpb_recovery
training tmux: toricgt_fineweb_bpb_recovery_live
training config: config/train.parameter_golf_all_phases.yaml
active checkpoint for decision: checkpoints/parameter_golf_all_phases_fineweb_from0/random_order_step_00004500.pt
training log: logs/parameter_golf_all_phases/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/supervisor/restart_001_20260603T124401Z/train.log
W&B run path: amelie-iska-math/toricgt-parameter-golf/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z
watcher tmux: toricgt_fineweb_bpb_recovery_watcher
next watcher target: 4750
next watcher log: logs/parameter_golf_all_phases/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/supervisor/analysis_target_00004750_20260603T143906Z/watcher.log
watcher device/precision: cpu / fp32
watcher pause behavior: non-interrupting; no --pause-training-before-analysis
loop env: BPB_TARGET=1.2, BPB_MAX_REVIEW_ITERATIONS=100,
          BPB_LOOP_STATE=/home/iska/Documents/amelie/bio/ToricGT/logs/parameter_golf_all_phases/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/supervisor/bpb_codex_loop_state.json,
          BPB_LOOP_STOP_FILE=logs/parameter_golf_all_phases/toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z/supervisor/bpb_codex_loop_stop,
          BPB_LOOP_NAME=all_phases_supervised_watchdog
```
## 2026-06-04 R79 Low-Train-BPB Trigger Capture And R80 Rollback

R79 replayed the R77-style low train-BPB dips from the step-3600 checkpoint and
now saves forced checkpoints whenever `train_bpb <= 1.13`.

Captured forced checkpoints:

- step 3608: `train_bpb=1.1149`, trigger val BPB `1.2180`,
  sampled OAI BPB `1.215465`;
- step 3641: `train_bpb=1.1270`, trigger val BPB `1.2177`,
  sampled OAI BPB `1.212380`;
- step 3658: `train_bpb=1.1273`, trigger val BPB `1.2173`,
  sampled OAI BPB `1.206637`;
- step 3674: `train_bpb=1.0955`, trigger val BPB `1.2171`,
  sampled OAI BPB `1.210648`.

Conclusion: the dips are real and useful for checkpoint capture, but they are
train-side/generalization-gap events, not validation breakthroughs.  The best
deterministic validation BPB in this branch remains step 3600 at `1.2169`.
W&B now exposes the trigger rows through `trigger/train_bpb`,
`trigger/val_bpb`, `trigger/low_train_bpb`,
`trigger/low_train_bpb_threshold`, and
`checkpoint/reason_low_train_bpb`.

Implementation update: `scripts/watch_seq4096_analysis.py` and
`scripts/mirror_fineweb_full_diagnostics_to_wandb.py` now parse
`low_train_bpb_trigger_val` rows, checkpoint-scoped analysis logs are written
for replayed check-ins, and sparse interval watchers jump to the next actual
forced checkpoint instead of waiting on nonexistent intermediate steps.

Operational decision: R79 served its capture purpose.  The active optimizer
handoff is R80, a rollback from the step-3600 best checkpoint with W&B enabled,
non-interrupting analysis, forced low-train-BPB checkpoints still active, and
the BPB target unchanged at `<= 1.2`.

```text
active run: toricgt_seq4096_4k_recovery_r80_rollback3600_20260604T190756Z
resume checkpoint: amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_4k_recovery_r75_20260604T182013Z/toricgt_seq4096_4k_recovery_r75_20260604T182013Z_step_003600.pt
W&B: amelie-iska-math/toricgt-parameter-golf/toricgt_seq4096_4k_recovery_r80_rollback3600_20260604T190756Z
training tmux: toricgt_seq4096_4k_recovery_r80_rollback3600_20260604T190756Z
analysis tmux: toricgt_seq4096_4k_analysis_r80_rollback3600_20260604T190756Z
gate tmux: toricgt_seq4096_4k_gate_r80_rollback3600_20260604T190756Z
```

Next review rule: if R80 repeats the R79 pattern where low train-BPB dips do
not reduce deterministic validation BPB by the next scheduled checkpoint,
prefer a damped-auxiliary replay from step 3600 over another high-aux replay:
lower `ADVANCED_LOSS_SCALE`, `GRAPHCG_LOSS_WEIGHT`, `SLEPIAN_LOSS_WEIGHT`, and
`ADVANCED_LOSS_MAX_CE_RATIO`, or make the advanced losses log-only until the
validation BPB slope turns negative again.
