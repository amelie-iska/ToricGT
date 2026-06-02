# BPB Cliff Recovery Plan

## Purpose

The repeated ToricGT Parameter-Golf runs show the same pathology: BPB finds a
narrow low-loss basin, then rebounds and settles into an unfavorable
approximately `3.5--3.7` BPB floor.  The earliest strong basin is not the later
`3.3k--3.8k` basin.  It appears in the run
`oai-bpb-postbounce-02000-20260601T232845Z`, where the local BPB minimum occurs
around step `2189`:

```text
step 2160: bpb ~= 3.405
step 2182: bpb ~= 3.399
step 2189: bpb ~= 3.368
step 2202: bpb ~= 3.408
step 2210: bpb ~= 3.554
```

The saved checkpoints do not include step `2189` or `2210`.  The robust restart
point is therefore:

```text
checkpoints/parameter_golf_oai_dense/random_order_step_00002000.pt
```

The goal is not to change the random-order autoregressive graph decoder.  The
decoder remains the contest-facing ToricGT adaptation.  The intervention is a
training-control intervention: make byte likelihood own the gradient long enough
to cross the first plateau, then reintroduce reasoning auxiliaries only after
the lower BPB floor is held.

## Diagnosis

Let

```text
L_total = L_BPB + sum_i lambda_i L_i
```

where the auxiliary terms include GFlowNet graph-of-thought, GraphCG frame
disentanglement, toric probes, topology/Koszul persistence, trajectory flow,
trajectory memory, and contrastive terms.  A BPB descent step requires

```text
< grad L_BPB, grad L_total > > 0.
```

The observed floor-bounce is consistent with three coupled mechanisms:

1.  **Auxiliary-gradient interference.**  Reasoning auxiliaries improve latent
    structure, but if their weights are nontrivial before byte likelihood
    stabilizes, they rotate the update direction away from BPB descent.

2.  **High-loss microbatch impulses.**  The useful low is narrow.  A handful of
    high-entropy rows can dominate a step and kick Adam out of the basin.  Those
    rows should remain visible in logs, but their gradients need Huber-like
    damping during the cliff window.

3.  **Positive curvature after the first low.**  The BPB first derivative turns
    weakly negative, then the second difference becomes positive.  Ordinary
    uninterrupted training keeps moving through the low instead of treating it
    as a line-search boundary.

## Implemented Mitigations

The active config `config/train.parameter_golf_random_order_dense_valmix35_from1000.yaml`
now has a dedicated step-2000 cliff region.

### Dense Checkpointing

Checkpoints are saved every `25` steps.  This preserves recoverable states in
the 2160--2225 floor region rather than forcing rollbacks to coarse 250-step
boundaries.

### Shock Guard

The shock guard is active only for `2000 <= step < 2500`.  It leaves raw metrics
unchanged but scales gradients for updates whose loss and gradient norm indicate
a high-loss impulse:

```text
ratio >= 1.018 or delta >= 0.045, with grad_norm >= 0.18
```

The resulting update scale is `0.006`, which is effectively a near-no-op for
detected shocks.

### Robust Microbatch Guard

The microbatch guard applies a Huber-style cap inside each accumulated step.  It
does not remove data; it only prevents a single high-loss microbatch from
dominating the accumulated gradient:

```text
cap = min(running_loss * 1.018, running_loss + 0.020)
min_scale = 0.08
```

### Auxiliary Quarantine

GFlowNet, topology, toric, Koszul, analogy, trajectory-flow, and trajectory-memory
losses are held at zero through the cliff window.  A tiny GraphCG/contrastive
anchor remains early so the learned steerable frame does not drift, but it is
kept orders of magnitude below the BPB gradient.

### Piecewise BPB-First Phases

The 2000--2500 window is split into three phases:

```text
2000--2160: bpb_cliff_entry_2000_2160
2160--2225: bpb_cliff_floor_capture_2160_2225
2225--2500: bpb_cliff_consolidation_2225_2500
```

The effective LR is high enough to keep the fast descent, then damped before the
known rebound.  The random-order autoregressive decoder and tropical/hybrid
attention architecture are unchanged.

## Monitoring And Codex Review

The launch script starts a watcher at target step `2250`.  The watcher:

1. waits for a fresh checkpoint at or above `2250`;
2. pauses the training tmux before analysis;
3. runs W&B/statistical metrics, simplex diagnostics, and geometry diagnostics
   on GPU;
4. writes a synopsis;
5. launches a Codex review tmux via `scripts/codex_training_review_resume.sh`.

The Codex handoff must decide one of:

```text
continue: resume from the analyzed checkpoint with the same config;
rollback: resume from the last dense checkpoint before positive curvature;
edit: make minimal scalar/config changes, then resume.
```

After any decision, another watcher should be scheduled approximately `250`
steps later while still inside `2000--2500`, and approximately `500` steps later
after the cliff window.  The user-requested invariant is that every analysis
pause triggers a Codex review and an explicit resume/restart decision.

## Promotion Rule

Do not promote a checkpoint just because instantaneous train BPB dips.  Promotion
requires:

```text
deterministic validation BPB not worse;
train BPB EMA lower than the old floor;
no large positive second difference over the last dense-checkpoint window;
GraphCG basis coherence stable;
GFlowNet entropy not collapsed in eval;
toric/topology diagnostics not exploding.
```

During the cliff window, deterministic validation BPB and train BPB curvature are
the gates.  GFlowNet and geometry diagnostics are audits, not primary promotion
metrics.

## Launch

Run:

```bash
scripts/launch_oai_bpb_cliff_recovery.sh
```

This script pauses existing OAI training/watch tmux sessions, resumes from step
`2000`, sets a deterministic W&B run id for the watcher, and starts the first
interrupting analysis gate at step `2250`.
