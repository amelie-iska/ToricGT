# Adaptive BPB Annealing Controller

Generated: 2026-06-17

Objective: keep OAI FineWeb BPB as the primary score while using graph-in/graph-out
ToricGT structure aggressively but safely.  Every 1.5K-step gate run restarts
from step 0 after full analysis, visualization, W&B metric review, and a fresh
hyperparameter decision.  The controller below is intentionally family-specific:
it must not collapse GraphCG, graph LM, tropical/toric geometry, BGG category O,
Koszul/persistent homology, vector-bundle/sheaf, analogy, memory, and
combinatorial toric algebra into one auxiliary bucket.

## Implemented Items

- [x] Add a reusable adaptive annealing planner.
  - Inputs: parsed train/validation BPB metrics, detailed `toricgt_sidecar/*`
    W&B review, strict-analysis status, artifact size, graph-LM BPB, and train
    metric correlations.
  - Outputs: `next_profile_hint`, concrete `env_overrides`, mathematical
    rationale, family decisions, score context, and sweep tags.

- [x] Add family-specific evidence rules.
  - Train BPB slope controls how quickly graph-LM and graph-structured pressure
    can rise.
  - Positive correlation with train BPB decays a family weight.
  - Negative correlation with train BPB permits a bounded increase.
  - PCGrad conflict metrics reduce pressure for the conflicting family.
  - Retrieval gates determine analogy/memory pressure.
  - Uncertainty gates localize toric/BGG/topological pressure to high-NLL
    regions.

- [x] Keep graph structure first-class.
  - FineWeb graphification and TokenGT-style structural embeddings remain on.
  - Low-rank node identifiers, endpoint-pair features, virtual local edge-token
    folding, and sequence score-correction weights are separate adaptive knobs.
  - OAI FineWeb output flattening remains scoped to BPB scoring only.
  - General graph data remains graph structured and unflattened.
  - Graph-LM is primary training data, but its weight is annealed to avoid
    harming early FineWeb BPB.

- [x] Wire adaptive decisions into full-analysis reports.
  - `next_profile_decision.json` now includes direct environment overrides.
  - Reports explain why each family was increased, held, or reduced.

- [x] Wire adaptive decisions into fresh campaign restarts.
  - The campaign still chooses a base profile, but then applies
    analysis-derived overrides on top.
  - Reports record both the base profile and applied overrides.

- [x] Add a handoff watcher for already-running campaigns.
  - The active campaign process cannot reload edited Python code.
  - The watcher waits for the current 1.5K run to finish, kills the old campaign
    sessions only after the run has written its final checkpoint/roundtrip line,
    runs the patched full-analysis pass, then launches a patched adaptive
    campaign from step 0 using the analysis result as prior evidence.

## Controller Rules

1. If FineWeb train BPB is still high or its slope is not improving, decrease
   graph-LM peak weight and delay graph-LM ramp.  Graph data still trains, but
   BPB remains dominant.
2. If graph-LM BPB is already easy while FineWeb BPB is poor, reduce graph-LM
   pressure and use graph structure through lower-risk TokenGT adapters.
3. If an advanced family has strong positive correlation with train BPB, lower
   that family for the next run unless strict validation shows it helped final
   BPB.
4. If an advanced family has weak or negative BPB correlation and its residual
   remains meaningful, increase it within conservative bounds.
5. Analogy and memory are raised only when retrieval evidence is present; weak
   retrieval keeps diagnostics active but lowers gradient pressure.
6. GraphCG remains full rank from step 0.  Orthogonality remains active; covariance
   pressure is damped when BPB-critical directions conflict.
7. Toric/tropical, BGG, vector-bundle/sheaf, Koszul persistence, and combinatorial
   toric algebra remain nonzero in exploratory runs, but use high-NLL uncertainty
   localization and PCGrad routing to avoid global BPB damage.
8. The sweep remains creative: the controller chooses profile neighborhoods and
   applies bounded deterministic perturbations to under-tested/helpful families,
   while preserving evidence-backed safety bounds.

## Active Gate Contract

- Gate interval: 1.5K steps.
- Target: BPB `< 1.19` before or at the 1.5K gate.
- Hardware target: approximately 20GB VRAM.
- If target is missed:
  - run full analysis and all visualizations;
  - review every W&B metric, especially `toricgt_sidecar/*`;
  - write report and adaptive decision;
  - restart from step 0 with the base profile plus adaptive env overrides.
