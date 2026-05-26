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

## Commands

Train:

```bash
conda run --no-capture-output -n tokengt env PYTHONPATH=src \
  python scripts/train_parameter_golf_random_order.py \
  --config config/train.parameter_golf_random_order_dense.yaml
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

The default training config has graph projection and GFlowNet sampling enabled:

```yaml
data:
  include_graph_projection: true
  coprime_row_stride: true
training:
  gflownet_loss_weight: 0.01
  gflownet_entropy_weight: 0.001
  eval_gflownet_samples: 2
  mtp_loss_weight: 0.05
  eval_score_first_bias_lr: 0.025
model:
  use_gflownet_policy: true
  gflownet_num_actions: 16
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
