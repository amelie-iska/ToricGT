# ToricGT Configs

These YAML files provide script defaults. Command-line flags still override the
values loaded from `--config`.

## Training

### Parameter-Golf Random-Order Dense Track

`train.parameter_golf_random_order_dense.yaml` is the default competition
configuration. It keeps the research model's ToricGT bias while adapting to the
16,000,000 byte artifact cap:

`train.parameter_golf_random_order_dense_valmix35_from1000.yaml` is the
step-1000 recovery replay. It resumes from
`checkpoints/parameter_golf_oai_dense/random_order_step_00001000.pt`, rotates
the data stream from step 1000, and places the validation-mixed BPB recovery
phase first in the phase list so the `medium_mix_ratio: 0.35`, damped
`lr_multiplier: 0.24`, and likelihood-first auxiliary-loss shutoff are active
immediately. The step-2000 handoff showed the lower-medium `0.18/0.18` retry
entered a shallow floor-bounce basin. The active rollback resumes from the
step-1500 checkpoint and switches to
`bpb_valmix_curvature_damped_1500_2000` with `medium_mix_ratio: 0.35`,
`lr_multiplier: 0.12`, `grad_clip_norm: 0.40`, and diagnostic-only auxiliary
geometry losses. The revealed-context replay then found a usable local valley
near step 1545--1550 but left it before the step-1700 watcher. The active
config now treats 1500--1700 as a strict BPB floor-lock: auxiliary losses are
diagnostic-only, effective LR is reduced, shock updates are skipped, fixed
validation probes run every 50 steps, and the watcher interrupts at 1600 before
the historical rebound band.

| Component | Default |
| --- | --- |
| Tokenization | byte-level, tied input/output embedding |
| Order | content-independent random target-position order per chunk |
| Seed policy | new derived seed per chunk/sample; fixed base seed only for reproducible runs |
| Stored blocks | 7 dense Transformer blocks |
| Effective depth | 14 block applications via 2 recurrent passes |
| Attention | lower softmax, upper tropical-ring attention |
| PolarQuant | 8-bit KV perturbation in eval/export checks |
| Graph data | compact `graph_json` node/edge projection is delayed until the hard-composite phase |
| Complex-row curriculum | text-first restart at step 1500; medium rows delayed until after the 1500--1700 BPB floor-lock, hard graph-technical rows at 6000 |
| Domain tags | math, code, graph, Hebrew, biomed, biochem, biophysics, toric |
| GFlowNet | 16-action prefix-visible embedding policy with TB surrogate |
| Cheap byte features | BigramHash, CaseOps byte classes, SmearGate confidence |
| Toric memory | 32 compact irrational clock/shift/cocycle slots |
| GraphCG | auto-sized lattice basis, resolving to 256 directions on the 24 GB 4090 under the 10% VRAM guard |
| Directed topology | nested scale-normalized simplex-tree analogies with noncommutative skew maps |
| Complexity diagnostics | compressor-tagged conditional-K, NCD, order-program, and GFlowNet action-trace metrics |
| HF best checkpoint | promotes `parameter_golf_oai_best.pt` to `AmelieSchreiber/toricgt-checkpoints` only when BPB plus complexity score improves |
| Auxiliary heads | 2 offset multi-token heads plus contrastive hidden regularization, stripped from export |
| Evaluation scaling | random orders, GFlowNet samples, and score-first bias adaptation |
| Export | bit-packed 6-bit row quantization with LZMA |
| Soft-MoE | off for the contest track; still on by default in the graph research model |
| Artifact target | `15,600,000` bytes, below the `16,000,000` byte cap |

The full all-phases OAI config is now the BPB-first default.  It sets
`model.order_mode: sequential`, so the trained likelihood is ordinary
left-to-right byte prediction from step 0 rather than random target-position
completion.  The same trainer can still run random order with
`--order-mode random`, but FineWeb/OAI BPB branches should use the sequential
default.  The config also adds a native TokenGT graph-token path on top of the
byte stream.  `use_tokengt_graph_fusion: true` builds a prefix-causal graph over
reveal steps, tokenizes node/edge/endpoint records, runs rank-causal graph
attention, and fuses node contexts into logits from step 0.
`use_tokengt_causal_graph: true` keeps the audited auxiliary graph losses active
in parallel.  The trajectory-memory teacher also uses vectorized persistence
landscapes/images through `trajectory_memory_persistence_*` settings, while the
report-grade exact persistence and CAS resolutions remain in the GUDHI/Sage/M2
analysis sidecars.  The supervised all-phases launcher now creates run-specific
checkpoint and BPB-loop state paths and passes that checkpoint directory through
to the trainer, preventing accidental reuse of stale all-phases checkpoints.

Run:

```bash
conda run --no-capture-output -n tokengt env PYTHONPATH=src \
  python scripts/train_parameter_golf_random_order.py \
  --config config/train.parameter_golf_random_order_dense.yaml
```

The configured run uses 50,000 optimizer steps. With `batch_size: 2`,
`grad_accum_steps: 16`, and `max_seq_len: 1024`, one optimizer step consumes
32 byte chunks or 32,768 supervised byte targets. The full run therefore sees
about 1.638B supervised byte targets, plus validation passes every 500 steps.
Full optimizer checkpoints are retained every 250 steps with no automatic
pruning, so earlier resume points remain available for ablations and recovery.
There is no separate unsupervised-only phase in this config. GFlowNet-style
training runs as an auxiliary objective on all 50,000 steps with an initial
recovery value of `gflownet_loss_weight: 0.010`; validation uses
`eval_gflownet_samples: 2` by
default, plus score-first output-bias adaptation at `0.025`; inference config
can raise random-order and GFlowNet samples to 4+ when runtime permits. The
full graph model's richer embedding-space GFlowNet remains in `train.full_30m_*`.

The dense config enables a bounded checkpoint-level adaptive controller. It
updates only after eval windows and only adjusts scalar knobs such as GFlowNet
entropy target, GFlowNet loss weight, and complex-row mix. For the current
`oai` restart, the controller state is fresh from step `1500`, the first
post-restart window uses a reduced LR multiplier of `0.90`, and graph-heavy
rows remain delayed so the byte model first regains a strong negative BPB slope.
Stale controller state is ignored if the resume checkpoint step does not match
the controller state.

GraphCG and analogy topology are auxiliary objectives. GraphCG disentangles
hidden transitions into an auto-sized lattice basis; repeated byte-relation
arrows are regularized to behave like analogical functors. The directed
topology term builds a scale-normalized filtered complex over hidden relation
arrows, adds an antisymmetric toric skew form, and logs nested inclusion,
triangle-density, directed-chain, and noncommutative cycle-flux metrics. The
same relation clouds also run a radius-parametrized HDBSCAN surrogate based on
core distances and mutual reachability; only persistent density-stable
neighbors are pulled together, which protects BPB training from noisy analogy
groups. These terms are phased in gently from step `1500`, starting at
`analogy_lattice_loss_weight: 0.00003`.

Complexity diagnostics run every 50 optimizer steps by default on a tiny sample.
They are logged under `complexity/train/*` and `complexity/val/*` and do not
change the BPB objective.

Periodic analyses now also produce directed nested-simplicial plots under
`outputs/post_resume_analysis/<run>/step-*/geometry/topology/`. These include
per-branch filtration curves and heatmaps of normalized distances,
mutual-reachability distances, density-persistence adjacency, antisymmetric
toric skew, and directed adjacency at multiple radii.

The same config also enables best-checkpoint publishing. Promotion uses
`val_bpb + 0.05 * complexity/val/prediction_target_ncd_lzma_mean`, replacing
the previous HF checkpoint only when the composite score improves.

`train.parameter_golf_random_order_packed_2048.yaml` is the longer-context
throughput probe. It keeps the dense ToricGT architecture, extends packed
chunks to 2048 bytes, filters toward larger technical records, and resumes from
1024-token checkpoints by resizing only the learned position table. Use it for
challenge-equivalent sample-efficiency experiments, not as a drop-in local
replacement unless VRAM and wallclock are acceptable.

### Graph Research Track

The configured 30M-class training plan has two executable training YAMLs.
Both use `batch_size: 2` and `grad_accum_steps: 16`, so one optimizer step
consumes 32 graph records.

| Phase | Config | Optimizer steps | Approx. train epochs | GFlowNet TB steps | Unsupervised-only steps |
| --- | --- | ---: | ---: | ---: | ---: |
| Warmup | `train.full_30m_warmup.yaml` | 2,000 | 0.013812 | 2,000 at weight 0.01 | 0 |
| Braided experts | `train.full_30m_cyclic.yaml` | 98,000 | 0.676798 | 98,000 at weight 0.05 | 0 |
| Total | both | 100,000 | 0.690610 | 100,000 | 0 |

The train split has 4,633,582 graph records and about 11.607B estimated source
text tokens. These epoch counts are pass-fraction estimates over the curated
train split, not strict full-dataset epochs. The current implementation does
not have a separate unsupervised-only phase; the base objective is supervised
or self-supervised graph reconstruction from curated graph records, with
GFlowNet trajectory balance as an auxiliary objective when enabled.

Run the warmup:

```bash
conda run --no-capture-output -n tokengt env PYTHONPATH=src WANDB_PROJECT=toricgt \
  python scripts/train.py --config config/train.full_30m_warmup.yaml
```

Then resume into braided expert training:

```bash
conda run --no-capture-output -n tokengt env PYTHONPATH=src WANDB_PROJECT=toricgt \
  python scripts/train.py --config config/train.full_30m_cyclic.yaml
```

To continue from a newer checkpoint, override only `--resume`:

```bash
conda run --no-capture-output -n tokengt env PYTHONPATH=src WANDB_PROJECT=toricgt \
  python scripts/train.py \
  --config config/train.full_30m_cyclic.yaml \
  --resume checkpoints/toricgt_full_30m/toricgt_step_00008000.pt
```

## Inference-Time Scaling

```bash
conda run --no-capture-output -n tokengt env PYTHONPATH=src \
  python scripts/test_time_scaling.py --config config/inference.test_time_scaling.yaml
```

Use `--device cpu` for a non-GPU check, or `--synthetic --batches 2 --budgets 1 2`
for a tiny local validation run that does not auto-curate held-out data.
