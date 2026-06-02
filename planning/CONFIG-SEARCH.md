# Codex-Gated Hyperparameter Search

Status: active planning and lightweight tooling for the `oai` branch.

The training loop should not treat every metric bounce as a reason for manual
trial-and-error.  The right control problem is a multi-fidelity search under
the Parameter-Golf constraints: BPB is the primary objective, but GraphCG,
GFlowNet, toric, topology, Kolmogorov, and artifact metrics are constraints or
secondary monitors.

## Objective

For a candidate phase control vector

```text
u = (lr_multiplier, grad_clip, medium_mix, hard_mix,
     lambda_graphcg, lambda_analogy, lambda_contrastive,
     lambda_gfn, lambda_toric, lambda_koszul, lambda_flow)
```

optimize a gated score

```text
J(u) = Delta BPB_val + alpha Delta BPB_train
       + beta max(0, geometry_regression)
       + gamma max(0, artifact_bytes - budget)
       + eta instability_penalty.
```

The candidate is acceptable only if:

- score-before-update and random-order validity remain true;
- artifact bytes remain below `16,000,000`;
- train BPB and loss have nonpositive recent slopes;
- validation or held-out branch BPB does not regress beyond the gate tolerance;
- GraphCG/topology metrics do not improve by damaging BPB.

## Search Policy

Use successive-halving with Codex review gates:

1. Generate 4-8 nearby phase candidates from the last accepted config.
2. Run cheap micro-evaluations for 50-100 optimizer steps or a fixed subset of
   validation branches.
3. Keep the best 2 by BPB slope and instability penalty.
4. Run the kept candidates for another 150-250 steps.
5. Accept one candidate only after the full metrics/simplex/geometry suite and
   Codex review.

The proposal helper

```bash
conda run -n tokengt env PYTHONPATH=src python scripts/propose_training_adjustments.py \
  --analysis-dir outputs/manual_analysis/<gate-dir>
```

converts an existing analysis directory into a conservative phase delta.  It is
not allowed to edit configs automatically.  The review loop can use the JSON
output as one input alongside visual plot inspection.

## Current Step-2250 Decision

The current gate showed:

- BPB/loss improving but flattening after a local bounce;
- GraphCG covariance and analogical-map residual drifting;
- branch search finding better terminals than the mean branch;
- weak toric active-face margins.

Therefore the next accepted candidate is:

- resume from `random_order_step_00002250.pt`;
- activate `graphcg_loss_weight: 0.00006`;
- activate `analogy_lattice_loss_weight: 0.00001`;
- raise `contrastive_loss_weight` to `0.00008`;
- activate a tiny GFlowNet branch-learning term,
  `gflownet_loss_weight: 0.00025`, with
  `gflownet_entropy_weight: 0.00005` at target entropy `2.0`;
- raise `lr_multiplier` to `0.16` with `grad_clip_norm: 0.42`;
- keep toric, Koszul, flow, QAT, and Soft-MoE losses off until the next
  500-step gate.

## Why This Is Principled

GraphCG is a chart-learning loss.  It should be promoted when the chart itself
is drifting, not merely when BPB is bad.  GFlowNet is promoted only at a tiny
weight in this gate because branch search already finds much lower-BPB
terminals than the mean branch; the point is to begin learning that branch gap
without allowing trajectory balance to dominate next-byte likelihood.  Toric
and Koszul losses are chamber and algebra losses.  They should be promoted only
after fan margins and branch quality are stable enough that the losses sharpen
the geometry rather than fighting the byte likelihood objective.
