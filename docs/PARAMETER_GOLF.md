# Parameter-Golf Adaptation

The Parameter-Golf track is a compatibility layer, not a replacement for the
research graph model. The goal is to keep the ToricGT inductive biases that fit
inside the OpenAI Model Craft Challenge constraints with minimal invasive
changes.

## Defaults

- Architecture: dense random-order autoregressive byte model.
- Artifact target: `15,600,000` bytes, leaving margin under the `16,000,000`
  decimal-byte challenge cap.
- Stored size: 7 dense blocks at width 384, recurrently applied twice.
- Attention: hybrid lower softmax and upper tropical-ring attention.
- Toric features: fixed sine/cosine phase channels on target positions.
- Order policy: content-independent random permutation per byte chunk/sample.
- Graph projection: curated `graph_json` rows are serialized to compact
  node/edge traces and appended to the byte stream.
- GFlowNet: compact prefix-visible embedding action policy with sixteen latent
  graph-of-thought actions, a trajectory-balance surrogate, entropy logging,
  and multi-sample evaluation.
- Competition feature path: BigramHash strict-prefix context embeddings,
  CaseOps byte-class features, SmearGate confidence, stripped multi-token
  auxiliary heads, coprime row striding, score-first output-bias adaptation,
  contrastive hidden-state regularization, compact noncommutative-toric memory,
  compact domain tags for math/code/graph/Hebrew/biomed/biochem/biophysics
  records, QAT grid regularization, and causal future-byte audits.
- Complexity diagnostics: compressor-tagged Kolmogorov-style proxies for
  conditional byte programs, graph projections, random-order permutations, and
  GFlowNet action traces.
- GraphCG/analogy geometry: auto-sized hidden lattice basis, scale-normalized
  nested simplex-tree maps, directed flag-complex topology, and analogical
  functor diagnostics.
- Soft-MoE: off for the contest track; still default for the graph research
  encoder.
- Export: bit-packed 6-bit row quantization with LZMA by default; auxiliary
  heads are excluded from the artifact.
- PolarQuant: optional 8-bit KV perturbation in evaluation/export checks.

## Why Random-Order AR

A byte chunk is treated as a small ordered graph whose target nodes are byte
positions. A random permutation chooses the order in which positions are
revealed. At step `k`, the model sees only BOS and the tokens revealed at steps
`< k`, plus the current target position. It does not see the token being scored
or any later revealed token. This is the score-before-update rule:

```text
previous_tokens[k] = BOS                         if k = 0
previous_tokens[k] = target_tokens[k - 1]        otherwise
target_tokens[k]  = bytes[permutation[k]]
```

The permutation seed is derived from the run seed, sample id, pass id, and
length. It is independent of byte content, so it cannot leak validation bytes.

## Compact GFlowNet Scaling

The full graph model keeps the richer embedding-space GFlowNet over graph
states. The Parameter-Golf adapter uses the byte-safe subset of that idea. For
each score row, the hidden state is prefix-visible, so a small policy can sample
a latent graph-of-thought action without seeing the current target byte. The
action embedding is scaled by a small residual coefficient before the output
projection. Training adds a one-step trajectory-balance surrogate whose reward
proxy is the detached next-byte log likelihood, plus entropy and action
diversity diagnostics. Evaluation can average several sampled action
trajectories and several random orders; this is test-time scaling over legal
internal randomness, not validation adaptation.

The `oai` branch also exposes a score-first output-bias adapter for validation
and challenge-style evaluation. The base model first scores a token from a
prefix-causal distribution, records the loss, and only then updates a tiny
per-sequence byte bias from the just-scored target. This gives a legal
test-time adaptation mechanism without changing model weights or reading
future bytes. The causal audit mutates future bytes and verifies that current
logits are unchanged.

Curated graph data is used by serializing `graph_json` into compact records:
`node id=... type=... text=...` and `edge source->target type=...`. This keeps
graph structure visible to the byte model while preserving the self-contained
artifact and avoiding a second evaluator-side graph dependency.

## BPB Targets

Lower BPB is better. Use these target bands when judging local runs before
doing official challenge-style reproduction:

| Validation BPB | Interpretation |
| ---: | --- |
| `>1.35` | Debugging only. |
| `1.25-1.35` | Functional but not yet competitive. |
| `1.20-1.22` | Reasonable first target; roughly the naive baseline range reported by OpenAI. |
| `1.16-1.19` | Strong candidate. |
| `1.13-1.15` | Excellent and near top-tier. |
| `<=1.12` | Exceptional/SOTA-class target based on OpenAI's published recap. |
| `<1.10` | Breakthrough-class; require strict leakage, tokenizer, and scoring audits. |

For record-quality claims, require multiple runs and enough evidence that the
improvement is larger than run-to-run variance. Treat single-run changes below
about `0.007 BPB` as noise unless confirmed independently.

## Local Wallclock Equivalence

The current workstation is an RTX 4090 24GB with a Ryzen 9 9950X3D.  A
challenge budget of `10 min` on an `8xH100` node is estimated as about `4 h`
locally, with a realistic range of `3-6 h` depending on H100 form factor,
parallel efficiency, data loading, and kernel shape.  The current dense
random-order exploratory run is slower by design: observed throughput is about
`4.64 s/step`, so `50,000` local steps is roughly `64-65 h` before allowing for
validation/checkpoint overhead.  A local challenge-equivalent probe is therefore
about `2.3k-4.7k` steps on this machine.

## Curated Split Usage

The local mirror of `AmelieSchreiber/toricgt-curated-splits` is used through
Parquet shard globs. The default supervised training path consumes only
`data/curated_hf_shards/train/*.parquet`. Validation consumes
`data/curated_hf_shards/validation/*.parquet`. Test shards are reserved for
held-out score-first evaluation and GFlowNet test-time scaling. This is the
competition-safe split: do not train on validation/test bytes, and do not let
GFlowNet adaptation or score-first bias updates see a byte before its loss has
been recorded.

The Parquet loader performs multi-record packing: rows are appended into a byte
buffer separated by a configurable delimiter until a full sequence is available.
This lets one training example contain several problems or one large graph
reasoning trace. The default `oai` config uses this as a curriculum: the early
BPB-capture stream remains text-first, medium-length rows are introduced after
step 1650, and larger graph-projected technical examples are delayed until
step 6000. This prevents graph scaffolding from destabilizing the early byte
compressor while still letting later phases exploit tropical-ring context. For
runs meant to reduce optimizer steps under the challenge wallclock, use
`config/train.parameter_golf_random_order_packed_2048.yaml`. It extends context
to 2048 bytes, keeps tropical-ring upper layers, preserves graph projections,
and can resume from a 1024-token checkpoint by resizing the position embedding.
The run also exposes `min_estimated_tokens`, `task_family_keywords`, and
`dataset_keywords` so a packed-context phase can focus on larger math, coding,
graph, physics, biomedical, biochemical, and biophysical records. This is a
sample-efficiency tradeoff: longer context reduces the number of optimizer
steps needed to expose comparable byte volume only when the GPU batch geometry
stays efficient.

## Kolmogorov-Style Diagnostics

True Kolmogorov complexity is uncomputable, so the implementation reports
estimator-tagged proxies instead of one canonical score. The `oai` trainer logs
small-sample metrics such as:

```text
complexity/train/target_cond_k_lzma_mean
complexity/train/order_program_k_zlib_mean
complexity/train/prediction_target_ncd_lzma_mean
complexity/train/gflownet_action_trace_k_lzma_mean
complexity/val/target_cond_k_lzma_mean
```

These metrics are diagnostic by default. They help detect whether BPB
improvements come with more compact, robust reasoning programs or only local
byte-pattern modeling. The BPB objective and causal scoring contract remain
unchanged.

## GraphCG And Directed Topological Analogies

The `oai` branch includes a low-byte GraphCG-style hidden basis for analogical
reasoning. `graphcg_num_directions: auto` chooses the largest configured
multiple of eight under a small activation-memory budget with a ten percent
safety margin; on the current 24GB RTX 4090 this resolves to 256 directions.
Old checkpoints with no basis or a narrower basis can still resume: missing
rows are initialized and stale optimizer moments are dropped only for changed
parameters.

The analogy loss is not just vector arithmetic. For repeated coarse byte
relations, normalized hidden arrows are grouped into small point clouds. Each
group builds nested soft Vietoris-Rips complexes at several radii, giving
scale-insensitive simplex-tree maps. The trainer logs inclusion penalties,
chain-map commutators, 0D-persistence/MST-style barcode proxies, edge density,
and triangle density. A directed noncommutative extension adds an antisymmetric
form on relation vectors, so `i -> j` and `j -> i` can differ. The resulting
directed flag complex logs transitive-closure pressure, directed cycle/holonomy
balance, directed chain-map diagnostics, asymmetry, and skew magnitude. These
metrics should improve reasoning geometry without replacing the BPB objective.

The same point clouds now use a radius-parametrized HDBSCAN surrogate. For each
relation group, the trainer computes core distances, mutual-reachability
distances, and a radius sweep over the existing filtration grid. Only
persistent high-stability mutual-reachability neighbors receive pullback weight;
unstable and outlier arrows are mostly ignored rather than collapsed into a
wrong analogy class. W&B reports `train/analogy_hdbscan_*` metrics for the
auxiliary loss, stability, persistent edge density, outlier score, and core
radius. This makes the topological analogy objective more relevant to training
quality because it reinforces density-stable relation families while limiting
damage from noisy early hidden states.

The periodic geometry suite visualizes the same nested complexes. Each
analysis checkpoint writes `geometry/topology/*_directed_filtration.png` with
radius-indexed edge density, soft triangle density, directed asymmetry, and
noncommutative cycle flux, radius-HDBSCAN cluster count, and HDBSCAN outlier
fraction, plus `*_noncommutative_heatmaps.png` for scale-normalized hidden-arrow
distance, mutual-reachability distance, density-persistence adjacency,
antisymmetric toric skew, and directed adjacency at low and middle filtration
radii. These plots are computed from model hidden states and branch losses, not
hand-drawn diagrams.

## Best Checkpoint Publishing

The `oai` trainer promotes checkpoints to
`AmelieSchreiber/toricgt-checkpoints` only when the validation candidate beats
the previous published manifest. The default lower-is-better promotion score is

```text
val_bpb + 0.05 * complexity/val/prediction_target_ncd_lzma_mean
```

so BPB stays primary while the prediction-target compression distance is part
of the gate. The uploaded checkpoint path is `parameter_golf_oai_best.pt` and
the manifest path is `parameter_golf_oai_best.json`. Local state is kept at
`checkpoints/parameter_golf_oai_dense/hf_best_publish_state.json`; this file is
not a training dependency and can be regenerated from the HF manifest. Failed
uploads log `hf_publish/error` to W&B and do not interrupt training.

## Commands

Train:

```bash
conda run --no-capture-output -n tokengt env PYTHONPATH=src \
  python scripts/train_parameter_golf_random_order.py \
  --config config/train.parameter_golf_random_order_dense.yaml
```

Probe the long-context packed curriculum from an existing checkpoint:

```bash
conda run --no-capture-output -n tokengt env PYTHONPATH=src \
  python scripts/train_parameter_golf_random_order.py \
  --config config/train.parameter_golf_random_order_packed_2048.yaml \
  --resume checkpoints/parameter_golf_oai_dense/best.pt
```

Project early loss and BPB to any requested checkpoint:

```bash
conda run --no-capture-output -n tokengt env PYTHONPATH=src \
  python scripts/project_pg_loss.py \
  --wandb-run amelie-iska-math/toricgt-parameter-golf/gbmw7z3a \
  --fit-through-step 2000 \
  --target-step 50000 \
  --metrics train/loss train/bpb \
  --output-dir outputs/projections/oai-step2000-to-50000
```

Evaluate complexity over a held-out sample without training:

```bash
conda run --no-capture-output -n tokengt env PYTHONPATH=src \
  python scripts/evaluate_complexity.py \
  --data-glob 'data/curated_hf_shards/validation/*.parquet' \
  --samples 512 \
  --output-dir outputs/complexity/oai-validation
```

Evaluate the reasoning-simplex visual diagnostics:

```bash
conda run --no-capture-output -n tokengt env PYTHONPATH=src \
  python scripts/evaluate_reasoning_simplex.py \
  --checkpoint checkpoints/parameter_golf_oai_dense/best.pt \
  --data-glob 'data/curated_hf_shards/validation/*.parquet' \
  --samples 16 \
  --budgets 1 2 4 8 \
  --output-dir outputs/reasoning_simplex/oai-best
```

The script writes three triangle heatmaps and two tetrahedron views. The main
triangle has vertices reasoning budget, K-proxy, and low BPB. The tetrahedra
add either hidden-trajectory MST efficiency or GFlowNet diversity. The MST
metric is computed from model hidden states, not source graph metadata: each
revealed-position hidden state is a node, pairwise Euclidean distances are edge
weights, and the minimum spanning tree measures how compactly the reasoning
trajectory can be summarized.

The default training config has graph projection and GFlowNet sampling enabled:

```yaml
data:
  include_graph_projection: false
  coprime_row_stride: true
  document_separator: "\n\n"
  medium_start_step: 1650
  medium_min_estimated_tokens: 129
  complex_start_step: 6000
  complex_min_estimated_tokens: 385
training:
  ckpt_interval: 250
  gflownet_loss_weight: 0.001
  gflownet_entropy_weight: 0.0005
  graphcg_loss_weight: 0.0002
  analogy_lattice_loss_weight: 0.0002
  eval_gflownet_samples: 2
  mtp_loss_weight: 0.01
  eval_score_first_bias_lr: 0.025
complexity:
  enabled: true
  eval_every: 50
  eval_samples: 2
checkpoint_publishing:
  enabled: true
  repo_id: AmelieSchreiber/toricgt-checkpoints
  checkpoint_filename: parameter_golf_oai_best.pt
  manifest_filename: parameter_golf_oai_best.json
  complexity_metric: complexity/val/prediction_target_ncd_lzma_mean
  complexity_weight: 0.05
model:
  use_gflownet_policy: true
  gflownet_num_actions: 16
  use_graphcg: true
  graphcg_num_directions: auto
  use_analogy_lattice: true
  analogy_topology_directed: true
  use_bigram_hash: true
  use_caseops_features: true
  use_smear_gate: true
  use_toric_memory: true
  aux_mtp_offsets: 2
  contrastive_temperature: 0.2
export:
  bits: 6
  quantization_mode: row
  compression: lzma
```

Export:

```bash
conda run -n tokengt env PYTHONPATH=src python scripts/export_parameter_golf_artifact.py \
  --checkpoint checkpoints/parameter_golf_oai_dense/best.pt \
  --output outputs/parameter_golf/toricgt_artifact.zip \
  --bits 6 \
  --quantization-mode row \
  --compression lzma
```

Minimal local validation:

```bash
conda run --no-capture-output -n tokengt env PYTHONPATH=src \
  python scripts/train_parameter_golf_random_order.py \
  --synthetic --steps 2 --batch-size 2 --grad-accum-steps 1 \
  --seq-len 32 --d-model 32 --num-heads 4 --num-layers 1 \
  --recurrent-passes 1 --device cpu --precision fp32 --no-wandb
```

## Compliance Notes

- The training script does not read validation rows during optimization.
- Evaluation derives fresh content-independent random orders for each batch.
- GFlowNet actions are sampled from prefix-visible hidden states only.
- Score-first bias adaptation updates only after a token loss is recorded.
- Auxiliary multi-token heads are training-only and stripped from exports.
- The artifact audit runs before training and fails if the compressed export is
  above the challenge cap.
- `keys.txt`, checkpoints, and logs are not intended for git commits.
