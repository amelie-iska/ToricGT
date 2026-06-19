# BPB Ideas 1-4 Implementation Plan

Date: 2026-06-19

Scope: implement ideas 1-4 from `BPB-FOCUSED-ADVANCED-TRAINING-IDEAS-20260619.md`.

Do **not** implement idea 5.  The next campaign should remain exploratory.  We have only run a small number of short runs, fewer than twenty with the newer FoT path and the `BPB < 1.19 by 1K` target.  The controller should not prematurely collapse onto one hyperparameter family.

## Goals

1. Add a BPB-first staged auxiliary gate.
2. Add metric/family-level gradient conflict controls, with practical low-overhead functionality in the OAI baseline training path.
3. Reground FoT reward toward score-defining per-byte likelihood improvement.
4. Add reward-coupled FoT entropy/diversity control.
5. Update the 25+10 one-thousand-step campaign loop so review prompts, reports, and hyperparameter proposals use these mechanisms.
6. Run tests, then stop the old loop only when ready to GPU-smoke the new path.
7. Restart a 25-run exploratory `BPB < 1.19 by 1K` campaign, followed by 10 more runs after a meta-review if the target is not reached.

## Research Notes Used

- PCGrad projects conflicting task gradients away from each other and leaves non-conflicting gradients alone.  This matches our problem better than globally lowering every advanced loss family.
- FoT-style methods rely on multiple trees, sparse activation, dynamic correction, and consensus.  For BPB optimization, these must be tied to byte-level log-likelihood improvements rather than just structural rewards.
- Subtrajectory balance and trajectory balance both support reward-proportional exploration, but short one-thousand-step fresh starts need low-variance, local rewards.
- Small-model MTP and reasoning auxiliaries benefit from staged curricula.  Early BPB descent must be protected before auxiliary complexity ramps.

## Implementation Design

### A. BPB-First Staged Auxiliary Gate

Add environment-controlled staged gates in `amelie-iska/parameter-golf/train_gpt.py`.

New knobs:

- `BPB_FIRST_AUX_STAGING=1`
- `BPB_FIRST_CORE_STEPS=250`
- `BPB_FIRST_RAMP_STEPS=500`
- `BPB_FIRST_MIN_AUX_MULT=0.02`
- `BPB_FIRST_REQUIRE_NEGATIVE_SLOPE=1`
- `BPB_FIRST_CURVATURE_GUARD=1`
- `BPB_FIRST_BAD_SLOPE_MULT=0.25`
- `BPB_FIRST_BAD_CURVATURE_MULT=0.35`

Runtime behavior:

- During the core phase, keep NTP CE, first-class graphification, graph-output flattening, and score-correction active.
- Damp sidecar, graph-LM, FoT, GFlowNet, MTP, and advanced certificate losses by a shared stage multiplier unless explicitly marked as score-path-safe.
- During ramp phase, interpolate from `MIN_AUX_MULT` to `1`.
- If train BPB slope is not improving or curvature rebounds, apply an additional temporary multiplier.

### B. Family-Level Conflict Controls

Full exact PCGrad over the entire model every step would be expensive.  Implement two layers:

1. **Always-on scalar conflict controller**:
   - Track detached per-family loss changes, train BPB slope, train BPB curvature, and family-to-BPB correlation proxies.
   - Produce family multipliers for:
     - graph-LM
     - sidecar aggregate
     - GraphCG
     - TokenGT graph
     - memory/analogy
     - toric geometry
     - 1D-cone vector-bundle/sheaf
     - BGG/Koszul/CCA/derived
     - FoT
     - GFlowNet
     - MTP
   - Expose metrics to W&B and logs.

2. **Sampled gradient conflict audit**:
   - At configurable intervals, compute dot/cosine conflict on a small shared parameter subset or selected auxiliary head parameters.
   - Use this for reports and next-run hyperparameter proposals.

New knobs:

- `AUX_CONFLICT_CONTROLLER=1`
- `AUX_CONFLICT_EMA=0.90`
- `AUX_CONFLICT_DAMP_MIN=0.10`
- `AUX_CONFLICT_BOOST_MAX=1.35`
- `AUX_CONFLICT_BAD_CORR_THRESHOLD=0.05`
- `AUX_CONFLICT_GOOD_CORR_THRESHOLD=-0.02`
- `AUX_CONFLICT_AUDIT_EVERY=100`
- `AUX_CONFLICT_AUDIT_PARAMS=tokengt,graph_flatten,fot,gfn`

Training behavior:

- Families whose losses rise while BPB improves are not automatically penalized.
- Families whose losses worsen while BPB worsens or whose short-run proxy correlation is positive receive temporary dampening.
- Families whose losses improve while BPB improves can receive a bounded boost.

### C. FoT Reward Regrounding in Byte Likelihood

Modify `src/toricgt/embedding_forest_of_thought.py` and training integration.

New knobs:

- `OAI_FOT_REWARD_MODE=bpb_delta`
- `OAI_FOT_BPB_DELTA_WEIGHT=1.0`
- `OAI_FOT_REWARD_GRAPH_WEIGHT=0.10`
- `OAI_FOT_REWARD_CONSENSUS_WEIGHT=0.20`
- `OAI_FOT_REWARD_COMPLEXITY_WEIGHT=0.02`
- `OAI_FOT_REWARD_FLOOR=1e-4`

Desired behavior:

- Compute FoT local correction logits or branch scores.
- Estimate local CE improvement from corrected logits on the same target positions.
- Use positive byte-likelihood delta as the dominant reward signal.
- Keep graph/toric/consensus rewards as bounded bonuses.
- Report both raw structural reward and BPB-delta reward separately.

### D. Reward-Coupled FoT Entropy/Diversity Controller

New knobs:

- `OAI_FOT_ADAPTIVE_CONTROL=1`
- `OAI_FOT_REWARD_TARGET=0.08`
- `OAI_FOT_DIVERSITY_TARGET=0.12`
- `OAI_FOT_ENTROPY_HIGH=0.985`
- `OAI_FOT_ENTROPY_LOW=0.75`
- `OAI_FOT_TEMP_MIN=0.45`
- `OAI_FOT_TEMP_MAX=1.10`
- `OAI_FOT_UCB_MIN=0.40`
- `OAI_FOT_UCB_MAX=1.80`

Runtime behavior:

- If entropy is high and reward is below target, reduce FoT temperature and UCB exploration.
- If reward is healthy but diversity collapses, increase UCB and sparse diversity pressure.
- If reward is poor and diversity is poor, reduce FoT loss weight instead of forcing exploration.
- Emit adaptive multipliers and suggested next-run FoT settings.

### E. Campaign Loop and Review Agent

Update `scripts/run_oai_sidecar_bpb_campaign.py`, `scripts/run_oai_sidecar_full_iteration_analysis.py`, and `scripts/adaptive_bpb_annealing.py`.

Requirements:

- Keep `max-runs=25`, `steps-per-run=1000`, `target-bpb=1.19`.
- If no run hits target in 25 runs, write a comprehensive meta-review into `training_notes/<campaign>/META-REVIEW-25.md`.
- Also write a planning document in `planning/` from the meta-review with:
  - theories for each run behavior;
  - aggregate evidence;
  - per-family win/loss table;
  - proposed hyperparameter changes for the next 10 runs.
- Then launch 10 additional exploratory runs using those changes.
- Review prompt must explicitly avoid lumping advanced losses together.
- Review prompt must encourage inventive but evidence-based experiments.

## Checklist

- [x] Inspect current FoT, GFlowNet, sidecar, campaign, and adaptive controller implementation.
- [x] Implement staged auxiliary gate.
- [x] Implement scalar family-level conflict controller.
- [x] Implement optional sampled gradient conflict audit if practical without large runtime overhead.
- [x] Implement FoT BPB-delta reward mode.
- [x] Implement reward-coupled FoT adaptive controller.
- [x] Update campaign environment profiles and exploratory proposal logic.
- [x] Update review/meta-review prompts.
- [x] Add tests for stage multiplier, conflict multiplier, FoT reward mode, and campaign env generation.
- [x] Stop old loop only when ready to GPU smoke test.
- [x] Run CPU tests.
- [x] Run GPU smoke training with a few steps and verify metrics appear.
- [x] Restart 25+10 campaign.
- [ ] Update docs and push.

## Progress Log

- 2026-06-19: Plan created.  Implementation not started yet.
- 2026-06-19: Implemented train-time BPB-first auxiliary staging, per-family gradient route scaling, FoT BPB-delta reward mode through corrected LM-head CE, reward-coupled FoT runtime control, campaign default environment updates, direct full-analysis fallback overrides, and primary/final meta-analysis planning output.
- 2026-06-19: Added regression tests for FoT BPB-delta reward metrics/backpropagation and adaptive campaign environment generation.  CPU tests passed.  A short CUDA smoke run completed train/validation/int8+zlib packing/final roundtrip with graphification, graph-output flattening, graph-LM, sidecar, GFlowNet, FoT, MTP, BPB-first staging, and conflict controls enabled.
- 2026-06-19: Restarted detached tmux campaign `toricgt_bpb119_ideas14_1k` with campaign id `tg-bpb119-1k-ideas14-20260619T152338Z`, 25 primary one-thousand-step attempts, `BPB < 1.19` target, strict full analysis, Codex review, and 10 follow-up attempts after the 25-run meta-review if the target is missed.
