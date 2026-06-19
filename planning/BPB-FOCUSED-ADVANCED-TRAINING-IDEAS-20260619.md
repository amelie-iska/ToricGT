# BPB-Focused Advanced Training Ideas

Date: 2026-06-19

This document collects 30 training adaptations for the ToricGT OAI Parameter-Golf baseline adaptation.  The primary objective is still OpenAI FineWeb BPB after legal score-first evaluation and int8+zlib export.  The advanced graph, tropical, toric, topological, sheaf, BGG, Koszul, FoT, GFlowNet, GraphCG, and memory systems should be used only when they improve BPB directly, improve a metric that is empirically predictive of BPB, or provide a controlled route to reasoning quality without damaging BPB.

The ideas below are ranked twice:

1. **Impact rank**: expected usefulness for lowering BPB or stabilizing early BPB descent in our present runs.
2. **Invasiveness rank**: implementation risk and disruption, where rank 1 is least invasive.

## Current Evidence

Recent campaign `tg-fot-bpb119-1k-20260618T223154Z` has completed 12 one-thousand-step fresh-start runs with FoT active.  The best completed run is run 8:

- best validation BPB: `1.3298`
- best int8+zlib roundtrip BPB: `1.33158715`
- best train BPB: `1.3526`
- artifact size: about `15.13 MB`

The target `BPB < 1.19 by 1K steps` has not been reached.  The prior 5K baseline remains stronger at about `1.2638` roundtrip BPB, so the one-thousand-step target is substantially more aggressive than our empirical data currently supports.

The evidence from runs 1-12 is consistent with the following:

- The `gate1500_fast_main_lr_aux_conflict_recovery` family is consistently stronger than the light-GraphCG exploration family.
- FoT is active, but reward is low, entropy remains high, and diversity often collapses after early steps.  FoT is therefore functioning but not yet acting as a strong BPB-improving search policy.
- TokenGT graphification and OAI FineWeb flattening are viable under the 16 MB cap, but their pressure must be ramped in a BPB-first way.
- GraphCG, memory, toric, BGG, Koszul, CCA, derived, and vector-bundle/sheaf losses are too heterogeneous to treat as one "advanced loss" group.  The controller needs family-level and preferably metric-level decisions.

## Research Anchors

- TokenGT treats graph nodes and edges as tokens and shows that pure Transformers can be powerful graph learners when token embeddings encode graph structure: <https://arxiv.org/abs/2207.02505>
- Graphormer demonstrates the importance of explicit structural encodings for graph Transformers: <https://arxiv.org/abs/2106.05234>
- Simple Path Structural Encoding suggests richer path-count edge encodings than random-walk-only features: <https://arxiv.org/html/2502.09365v1>
- Continuous GFlowNets justify continuous or hybrid state-space flow objectives: <https://arxiv.org/abs/2301.12594>
- Forest-of-Thought uses multiple reasoning trees, sparse path activation, self-correction, and consensus to improve reasoning efficiency: <https://arxiv.org/abs/2412.09078>
- Multi-token prediction can improve sample efficiency and induction-style reasoning, but small models need a curriculum: <https://arxiv.org/abs/2404.19737> and <https://arxiv.org/abs/2505.22757>
- Any-order autoregressive models and sigma-GPTs support the distinction between sequence order and autoregressive factorization order: <https://arxiv.org/abs/2205.13554> and <https://arxiv.org/html/2404.09562v1>
- PCGrad motivates explicit mitigation of destructive multi-objective gradient interference: <https://arxiv.org/abs/2001.06782>
- Language modeling is compression, and tokenizer compression rate changes BPB-relevant scaling behavior: <https://arxiv.org/html/2309.10668v2> and <https://arxiv.org/html/2605.01188v1>
- Tropical transformer expressivity and toric ReLU geometry support our fan, Newton-polytope, active-face, divisor, and toric-audit machinery: <https://arxiv.org/abs/2604.14727> and <https://arxiv.org/abs/2509.05894>

## The 30 Ideas

### 1. BPB-First Staged Auxiliary Gate

Run the first 150-300 steps as a BPB-dominant phase: NTP CE, graphification embeddings, and flattening score correction stay active, but all non-BPB auxiliary losses use near-zero weights.  Then ramp auxiliary families only when train BPB slope is negative and curvature is not rebounding.

- Why it may help: the current early target is harsh; auxiliary objectives should not steal the initial descent direction.
- Implementation: add phase gates in the campaign controller and train script.
- Impact rank: 1
- Invasiveness rank: 4

### 2. Metric-Level PCGrad or Projected Auxiliary Updates

Do not merely scale advanced losses.  Compute gradient cosine between BPB CE and each family: GraphCG, graph-LM, TokenGT graph, memory, FoT, GFlowNet, toric geometry, 1D-cone vector-bundle/sheaf, BGG, Koszul, CCA, derived.  Project or damp only conflicting components.

- Why it may help: present analysis already suggests auxiliary conflict, but the correction is too coarse.
- Implementation: sample gradients every N steps or on microbatches; route conflicting families through PCGrad-style projection.
- Impact rank: 2
- Invasiveness rank: 21

### 3. FoT Reward Regrounding in Per-Byte Delta Log-Likelihood

Replace the current FoT reward with a reward dominated by local improvement in score-defining byte log-probability, plus small bonuses for graph consistency and consensus.  FoT should not be rewarded for structural elegance unless it predicts bytes better.

- Why it may help: current FoT reward is low and does not clearly correlate with best BPB runs.
- Implementation: for sampled branches, evaluate cheap local delta CE over selected positions and use that as the reward scale.
- Impact rank: 3
- Invasiveness rank: 16

### 4. FoT Entropy-Diversity Controller with Reward Coupling

High entropy is useful only when reward improves.  If FoT entropy is high and reward low, reduce temperature/UCB exploration; if reward rises and diversity collapses, increase branch diversity.  Control diversity by reward-conditioned targets rather than a fixed entropy target.

- Why it may help: current FoT has high entropy and often low diversity, which is exploration without enough BPB lift.
- Implementation: controller rule using `fot_R`, `fot_H`, `fot_div`, train BPB slope, and val/roundtrip BPB.
- Impact rank: 4
- Invasiveness rank: 8

### 5. Consolidate Around the Best Profile with Small Orthogonal Sweeps

Use `gate1500_fast_main_lr_aux_conflict_recovery` as the center of the sweep because it dominates current 1K results.  Do not alternate equally with weaker profiles unless the sweep explicitly tests one variable.

- Why it may help: run 8, run 10, and run 12 all indicate the conflict-recovery profile is the stronger basin.
- Implementation: change campaign proposal prior so 60-70 percent of runs are local perturbations of the best profile.
- Impact rank: 5
- Invasiveness rank: 2

### 6. Graph-Output Flattening as a Learned Residual Mixture

Keep graph-in/graph-out internally, but score FineWeb through a learned mixture of sequential logits and graph-flattened logits:

`logits = logits_seq + gate(context) * delta_logits_graph_flattened`

Regularize the gate toward zero early and let it open only where it lowers train BPB.

- Why it may help: graph structure becomes a residual improvement rather than a mandatory perturbation.
- Implementation: add a scalar or low-rank gate conditioned on local graph confidence and BPB residual.
- Impact rank: 6
- Invasiveness rank: 13

### 7. Adaptive Graph Radius 2-6 with BPB-Safe Gates

Use radius 2-3 early, radius 4 only when local graph confidence is high, and radius 5-6 for sparse high-value regions selected by entropy, repeated-token motifs, or graph-LM certainty.  Store no extra deployable parameters beyond existing bucket tables.

- Why it may help: user wants radius up to 6, but dense radius can inject noise.  A gate makes it local and BPB-safe.
- Implementation: adaptive radius schedule tied to graph confidence and CE residual.
- Impact rank: 7
- Invasiveness rank: 10

### 8. MTP Curriculum Instead of Constant MTP

For a small 18M-parameter model, start with NTP-only or very weak MTP, then ramp offsets 2, 3, 4 after CE stabilizes.  Test reverse curriculum as an exploration branch, but default to forward curriculum.

- Why it may help: literature suggests MTP helps, but small models can struggle without a curriculum.
- Implementation: schedule `OAI_MTP_LOSS_WEIGHT`, `OAI_MTP_OFFSETS`, and `OAI_MTP_EVERY`.
- Impact rank: 8
- Invasiveness rank: 3

### 9. Byte-Entropy/Difficulty Curriculum

Bucket FineWeb samples by byte entropy, token rarity, punctuation/code density, and graphification complexity.  Warm up on medium-entropy examples, then expand to hard examples.  Avoid overtraining on easy graph-LM rows while FineWeb BPB is high.

- Why it may help: early BPB is highly sensitive to optimizer basin; a smoother first thousand steps can matter.
- Implementation: shard sampler weights and W&B reporting by difficulty bucket.
- Impact rank: 9
- Invasiveness rank: 17

### 10. Checkpoint-Soup and EMA Under Export Size

Keep EMA weights and test checkpoint soup among the best early checkpoints in a run.  Only serialize one final parameter set.  Measure roundtrip BPB after soup and EMA.

- Why it may help: reduces variance without increasing artifact bytes.
- Implementation: maintain EMA in training memory and optional final interpolation.
- Impact rank: 10
- Invasiveness rank: 7

### 11. Distill from the Best 5K Checkpoint

Use the prior stronger 5K checkpoint as a teacher for early logits, while keeping legal data and no validation leakage.  Distill only on training data, with a temperature and BPB-first CE still dominant.

- Why it may help: the 5K baseline is already better than 1K FoT runs; it can teach the new graphified model the old good basin.
- Implementation: load teacher checkpoint, train KL on selected positions, and ablate teacher weight.
- Impact rank: 11
- Invasiveness rank: 14

### 12. Any-Order Graph Decoding with Inference-Frequency Weighting

For graph rows, train several legal factorization orders, but upweight the conditionals actually used by the BPB flattening path.  For causal DAGs, use topological order; for non-DAG graph data, use sampled any-order factorization.

- Why it may help: it keeps graph-in/graph-out honest while aligning training with the scored sequential factorization.
- Implementation: order-policy sampler plus loss weights from observed inference frequency.
- Impact rank: 12
- Invasiveness rank: 18

### 13. Tokenizer Compression Audit and Alternate Tokenizer Sweep

Evaluate whether the SentencePiece 1024 tokenizer is at the right compression rate for the 18M-parameter, 16MB artifact regime.  Sweep a small set of vocabulary sizes or byte/subword hybrids if rules allow.

- Why it may help: BPB is explicitly byte-normalized; tokenizer compression rate has non-monotonic effects.
- Implementation: train short probes with identical byte budgets and measure true BPB plus artifact cost.
- Impact rank: 13
- Invasiveness rank: 24

### 14. Graphification Dropout

Randomly drop graph structural features on a fraction of FineWeb batches.  The model learns to use graph features when helpful but keeps the raw sequential path strong.

- Why it may help: prevents graphification from becoming a brittle always-on perturbation.
- Implementation: dropout over structural, edge, endpoint, torus, identifier, and edge-token features with separate rates.
- Impact rank: 14
- Invasiveness rank: 5

### 15. Local Toric/Tropical Regularization Only Where CE Is Uncertain

Apply fan-margin, active-face, Newton-polytope, and tropical entropy losses only to positions with high CE, low top-token margin, or high graph uncertainty.  Do not regularize easy bytes.

- Why it may help: toric/tropical structure should resolve ambiguous decisions, not distort already-correct local distributions.
- Implementation: multiply toric/tropical losses by detached uncertainty weights.
- Impact rank: 15
- Invasiveness rank: 9

### 16. GraphCG as BPB-Orthogonal Basis Discovery

GraphCG should discover full-rank concept axes, but the axes should be trained in subspaces orthogonal or low-conflict to BPB CE.  Use it to make memory/FoT retrieval cleaner, not as a global high-pressure objective.

- Why it may help: current GraphCG pressure can correlate with worse BPB in some runs, but full-rank structure remains valuable.
- Implementation: orthogonal projection, covariance damping, and retrieval-only GraphCG loss windows.
- Impact rank: 16
- Invasiveness rank: 11

### 17. Retrieval Memory as a Score-First kNN Interpolator

Use trajectory memory retrieval to produce a legal, training-set-only kNN logit correction.  Mix it with base logits through a tiny gate and enforce the interpolation bound so it cannot catastrophically harm BPB.

- Why it may help: memory can improve rare byte or motif prediction without expanding the model substantially.
- Implementation: compact datastore or batch-local memory, gated logit interpolation, exact byte-score audit.
- Impact rank: 17
- Invasiveness rank: 22

### 18. BGG/Koszul/CCA as Late Sparse Certificates

Keep BGG category O, Koszul, CCA, and derived losses nonzero but tiny early.  Activate stronger weights only on graph-LM/certificate batches after FineWeb BPB is below a moving threshold.

- Why it may help: exactness losses are semantically rich but can be off-manifold for byte prediction early.
- Implementation: thresholded curriculum by train BPB EMA and graph-LM BPB.
- Impact rank: 18
- Invasiveness rank: 6

### 19. FoT Consensus Distillation to the Main Logits

When several FoT branches agree and local BPB improves, distill the consensus into the main logits.  When branches disagree, treat FoT as an exploration diagnostic rather than a strong loss.

- Why it may help: the current FoT branch machinery needs a sharper path into the score-defining model.
- Implementation: top-K tree consensus target with confidence threshold and detached teacher distribution.
- Impact rank: 19
- Invasiveness rank: 15

### 20. GFlowNet Subtrajectory Balance for Local Byte Motifs

Use subtrajectory balance on short byte/graph motifs instead of full long-trajectory objectives early.  Reward terminal states by local CE improvement and graph consistency.

- Why it may help: long-horizon sparse rewards are difficult in 1K-step fresh starts.
- Implementation: motif windows, local TB/SubTB loss, reward normalization per batch.
- Impact rank: 20
- Invasiveness rank: 12

### 21. Sidecar Family Shapley or Knockout Analysis

At each review interval, run cheap ablation replays on the checkpoint: zero one family at a time during evaluation and measure delta train/validation BPB on a small fixed byte slice.

- Why it may help: correlation alone is not enough; knockouts tell which family is helping or hurting.
- Implementation: eval-only masks for each auxiliary head.
- Impact rank: 21
- Invasiveness rank: 19

### 22. Adaptive LR by BPB Curvature

Monitor train BPB EMA, slope, and second difference.  If BPB curvature turns positive before target, reduce main LR or enter warmdown; if BPB slope is too slow with stable aux metrics, raise embed/matrix LR slightly.

- Why it may help: current fresh-start runs still converge too slowly relative to the 1K target.
- Implementation: controller rule producing LR overrides for next run.
- Impact rank: 22
- Invasiveness rank: 1

### 23. Structured Data Mixture Controller

Treat FineWeb, graph-LM shards, certificate rows, and reasoning/FoT rows as separate streams with byte-equivalent weights.  If graph-LM BPB is easy and FineWeb is poor, reduce graph-LM mixture or keep only its structural feature contribution.

- Why it may help: graph rows can become a side objective that the model solves while FineWeb BPB lags.
- Implementation: mixture weights controlled by metric deltas.
- Impact rank: 23
- Invasiveness rank: 10

### 24. Quantization-Aware Stability Window

Run fake int8 or QAT only in a late window after BPB descent stabilizes.  Track float BPB versus roundtrip BPB every run.  If the gap grows, add fake quant earlier in the next run.

- Why it may help: the competition score is exported artifact BPB, not float checkpoint BPB.
- Implementation: existing export path plus scheduled fake quant.
- Impact rank: 24
- Invasiveness rank: 8

### 25. Simple Path Structural Encoding for FineWeb Graphification

Add approximate simple-path counts for radius-limited graphified FineWeb rows.  Keep as tiny bucket features.  Emphasize local repeated motifs and cycles only where graphification confidence is high.

- Why it may help: SPSE is stronger than random-walk-only features for local graph patterns, and text graphification has repeated motif/cycle structure.
- Implementation: approximate path-count buckets up to radius 3 early, 6 late.
- Impact rank: 25
- Invasiveness rank: 20

### 26. Tropical Active-Face Capacity Matching

Use tropical active-face diagnostics to decide head specialization: if too many heads share the same active face, increase diversity or route only uncertain tokens to tropical heads; if active faces are unstable on easy bytes, reduce tropical pressure.

- Why it may help: tropical transformer theory frames attention partitions as polyhedral cells; we should manage that capacity.
- Implementation: active-face entropy target conditioned on local CE and token class.
- Impact rank: 26
- Invasiveness rank: 13

### 27. 1D-Cone Vector-Bundle/Sheaf Retrieval Gate

Use vector-bundle/sheaf consistency mainly as a gate for retrieval and analogy memory, not a global loss.  A memory item should be trusted only if its 1D-cone filtrations and local sheaf restrictions are compatible.

- Why it may help: sheaf consistency is a good retrieval validator but may be too indirect for direct BPB gradients.
- Implementation: retrieval gate and W&B metrics, small optional auxiliary loss.
- Impact rank: 27
- Invasiveness rank: 16

### 28. CAS-Cached Exactness Teacher

Use Sage/Macaulay2 to precompute exact free resolutions, syzygies, divisors, and BGG/Koszul certificates offline.  Distill cached labels during training; do not call CAS in the hot BPB loop.

- Why it may help: keeps exact mathematics real while protecting step time and BPB.
- Implementation: cache schema, data loader, certificate distillation heads.
- Impact rank: 28
- Invasiveness rank: 23

### 29. Score-First Test-Time Adaptation as Candidate Selection

Keep TTA non-committing by default, but use it to rank candidate decoding/calibration settings on an allowed prefix.  Only commit if competition rules and score-first audit allow it.

- Why it may help: can recover local distribution shifts without changing serialized model weights.
- Implementation: prefix-only adaptation audit with explicit no-future-token checks.
- Impact rank: 29
- Invasiveness rank: 18

### 30. Review-Agent Upgrade: Metric-Semantics-Aware Run Proposals

The 1K/1.5K review agent should include a metric dictionary and family-specific hypotheses.  It must propose changes at the family level, not lump all advanced losses together.

- Why it may help: prevents lazy conclusions like "advanced metrics hurt BPB" when only one family is conflicting.
- Implementation: add metric descriptions, knockout deltas, correlations, and proposed next-run deltas to reports.
- Impact rank: 30
- Invasiveness rank: 5

## Impact Ranking

1. BPB-first staged auxiliary gate
2. Metric-level PCGrad or projected auxiliary updates
3. FoT reward regrounding in per-byte delta log-likelihood
4. FoT entropy-diversity controller with reward coupling
5. Consolidate around the best profile with small orthogonal sweeps
6. Graph-output flattening as a learned residual mixture
7. Adaptive graph radius 2-6 with BPB-safe gates
8. MTP curriculum instead of constant MTP
9. Byte-entropy/difficulty curriculum
10. Checkpoint-soup and EMA under export size
11. Distill from the best 5K checkpoint
12. Any-order graph decoding with inference-frequency weighting
13. Tokenizer compression audit and alternate tokenizer sweep
14. Graphification dropout
15. Local toric/tropical regularization only where CE is uncertain
16. GraphCG as BPB-orthogonal basis discovery
17. Retrieval memory as a score-first kNN interpolator
18. BGG/Koszul/CCA as late sparse certificates
19. FoT consensus distillation to the main logits
20. GFlowNet subtrajectory balance for local byte motifs
21. Sidecar family Shapley or knockout analysis
22. Adaptive LR by BPB curvature
23. Structured data mixture controller
24. Quantization-aware stability window
25. Simple Path Structural Encoding for FineWeb graphification
26. Tropical active-face capacity matching
27. 1D-cone vector-bundle/sheaf retrieval gate
28. CAS-cached exactness teacher
29. Score-first test-time adaptation as candidate selection
30. Review-agent upgrade: metric-semantics-aware run proposals

## Invasiveness Ranking

1. Adaptive LR by BPB curvature
2. Consolidate around the best profile with small orthogonal sweeps
3. MTP curriculum instead of constant MTP
4. BPB-first staged auxiliary gate
5. Graphification dropout
6. BGG/Koszul/CCA as late sparse certificates
7. Checkpoint-soup and EMA under export size
8. FoT entropy-diversity controller with reward coupling
9. Local toric/tropical regularization only where CE is uncertain
10. Adaptive graph radius 2-6 with BPB-safe gates
11. GraphCG as BPB-orthogonal basis discovery
12. GFlowNet subtrajectory balance for local byte motifs
13. Graph-output flattening as a learned residual mixture
14. Tropical active-face capacity matching
15. Distill from the best 5K checkpoint
16. FoT reward regrounding in per-byte delta log-likelihood
17. Byte-entropy/difficulty curriculum
18. Any-order graph decoding with inference-frequency weighting
19. Score-first test-time adaptation as candidate selection
20. Sidecar family Shapley or knockout analysis
21. Simple Path Structural Encoding for FineWeb graphification
22. Metric-level PCGrad or projected auxiliary updates
23. Retrieval memory as a score-first kNN interpolator
24. CAS-cached exactness teacher
25. Tokenizer compression audit and alternate tokenizer sweep
26. Structured data mixture controller
27. FoT consensus distillation to the main logits
28. 1D-cone vector-bundle/sheaf retrieval gate
29. Review-agent upgrade: metric-semantics-aware run proposals
30. Quantization-aware stability window

Note: the least-invasive ranking is not the same as the safest ranking.  For example, review-agent upgrades are not mathematically risky, but they touch the campaign/reporting system broadly.  Quantization-aware training is ranked invasive because it can perturb the primary optimization loop if introduced too early.

## Recommended First Batch

The first implementation batch should prioritize high impact with moderate or low invasiveness:

1. BPB-first staged auxiliary gate.
2. Consolidate the sweep around the best `aux_conflict_recovery` profile.
3. MTP curriculum.
4. FoT entropy-diversity controller with reward coupling.
5. Adaptive graph radius 2-6 with BPB-safe gates.

The second batch should add:

1. FoT reward regrounding in delta byte log-likelihood.
2. Graph-output residual mixture gate.
3. Local toric/tropical uncertainty gating.
4. GraphCG BPB-orthogonal basis discovery.
5. Sidecar family knockout analysis.

The third batch should address the heavier but potentially valuable changes:

1. Metric-level PCGrad.
2. Distillation from the best 5K checkpoint.
3. Retrieval kNN logit interpolation.
4. Tokenizer compression audit.
5. CAS-cached exactness teacher.

## Practical Decision Rule

For each new run proposal, the controller should produce:

1. expected BPB mechanism;
2. exact changed hyperparameters;
3. family-level BPB correlation and knockout evidence;
4. whether the change affects artifact bytes;
5. rollback condition.

A change should be promoted only if it improves at least one of:

- train BPB at matched step 250, 500, 750, or 1000;
- validation BPB at the gate;
- final int8+zlib roundtrip BPB;
- BPB-neutral reasoning or retrieval metric that has historical negative correlation with later BPB.

If a change helps graph, toric, topology, or FoT metrics but does not improve any BPB-linked metric after two controlled trials, it should remain diagnostic or late-phase only.
