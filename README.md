# ToricGT

Author: Amelie Schreiber

ToricGT is a research prototype for TokenGT-style graph-to-graph modeling with tropical ring attention, default 4-expert Soft-MoE feed-forward blocks, finite noncommutative-torus features, and embedding-space GFlowNet fine-tuning.

![ToricGT architecture and training paradigm](assets/toricgt_architecture_and_training_diagram.png)

![Dark-mode ToricGT torus reasoning animation](assets/toricgt_torus_reasoning_dark.gif)

<p align="center">
  <a href="./assets/toricgt_paper_pg_softmoe_final.tex"><img src="https://img.shields.io/badge/arXiv-94133F?style=for-the-badge&logo=arxiv" alt="arXiv"/></a>
  <a href="https://github.com/amelie-iska/ToricGT/"><img src="https://img.shields.io/badge/📝%20Blog-007A87?style=for-the-badge&logoColor=white" alt="GitHub"/></a>
  <a href="https://huggingface.co/blog/AmelieSchreiber/toricgt"><img src="https://img.shields.io/badge/HuggingFace-DE9B35.svg?style=for-the-badge&logo=HuggingFace" alt="HF"/></a>
</p>

Current validated status:

- CPU tests: `pytest -q tests` passes.
- CPU implementation validation: default Soft-MoE, tropical-ring attention, embedding-space GFlowNet trajectory balance, and finite rotation-algebra checks run without meaningful VRAM use.
- CUDA capacity validation: `d=384`, 8 layers, 8 heads, 29.8M parameters, 1,280 graph tokens, bf16, default Soft-MoE, and embedding-space GFlowNet loss completed one optimizer step at about 6.0GB peak VRAM for batch 1 and 11.9GB for batch 2.
- Parameter-Golf dense random-order scaffold: the default 13.0M-parameter byte model exports as a 12.94MB int8 compressed artifact, below the 16,000,000 byte cap, with compact embedding-space GFlowNet action sampling enabled.
- The full curation job writes leakage-controlled train/validation/test Parquet splits under `data/curated/`; validate it with `scripts/inspect_curated_data.py` before launching long training.

## What Is Here

Public upstream clones are under `amelie-iska/`:

- `amelie-iska/tokengt`: TokenGT baseline.
- `amelie-iska/ringattention`: official Ring Attention implementation linked from the paper.
- `amelie-iska/Tropical-Attention`: official Tropical Attention implementation.

Local implementation:

- `src/toricgt/toric_algebra.py`: clock/shift operators, finite torus approximants, braid and affine Coxeter helpers.
- `src/toricgt/tropical_attention.py`: softmax, tropical, and exact blockwise tropical-ring attention.
- `src/toricgt/soft_moe.py`: graph-token Soft-MoE, prefix-causal Soft-MoE, and routing diagnostics.
- `src/toricgt/graph_tokenizer.py`: node/edge TokenGT-style graph tokenization.
- `src/toricgt/model.py`: graph-to-graph ToricTokenGT encoder with default embedding-space GFlowNet policy head.
- `src/toricgt/gflownet.py`: embedding-space GFlowNet policy and trajectory-balance loss.
- `src/toricgt/graph_dataset.py`: Parquet/JSONL graph-record streaming into padded graph batches.
- `src/toricgt/synthetic.py`: synthetic rotation-algebra and tropical shortest-path curriculum records.
- `src/toricgt/polar_cache.py`: recursive polar encode/decode utilities for optional KV-cache compression experiments.
- `src/toricgt/parameter_golf_export.py`: byte accounting and compressed artifact export helpers.
- `src/toricgt/random_order_lm.py`: dense random-order autoregressive ToricGT adapter for the OpenAI Parameter Golf track, including compact prefix-visible GFlowNet action routing.
- `src/toricgt/music.py`: dark analog-synth algorithmic music from torus orbits, tropical active faces, and Soft-MoE-style routing.
- `src/toricgt/datasets.py`: dataset manifest and leakage-controlled splitting.
- `scripts/`: curation, training, evaluation, visualization, publication, and validation entrypoints.
- `assets/toricgt_torus_reasoning_dark.gif`: README animation for toric phase, tropical active-face, Soft-MoE, and GFlowNet flow intuition.
- `assets/toricgt_paper_pg_softmoe_final.tex`: research paper source.
- `assets/toricgt_paper_pg_softmoe_final.pdf`: compiled paper.
- `planning/IMPLEMENTATION-PLAN.md`: detailed implementation plan.
- `planning/DATA.md`: dataset research, curation, and segmentation plan.
- `docs/PARAMETER_GOLF.md`: dense random-order Parameter-Golf adaptation notes.

## Setup

Use Conda only. The working environment used for this repository is `tokengt`.

```bash
conda create -n tokengt python=3.11 -y
conda activate tokengt
python -m pip install -e .
```

If the machine already has a managed PyTorch installation in the Conda environment, install the remaining packages first and then install this package with `--no-deps`.

```bash
python -m pip install -e . --no-deps
```

## Credentials

Do not paste API tokens into shell history, notebooks, docs, or issue comments.

Configure services locally:

```bash
gh auth login
hf auth login
wandb login
```

Authenticated forking and Hugging Face pushes are intentionally not hard-coded in this repo.

## Dataset Curation

The full data research and segmentation plan is in [planning/DATA.md](/home/iska/Documents/amelie/bio/ToricGT/planning/DATA.md). The selected corpus includes:

- `AI-MO/NuminaMath-CoT`
- `open-r1/OpenR1-Math-220k`
- `open-r1/codeforces-cots`
- `openai/gsm8k`
- `openai/frontierscience` (eval-only)
- `openai/healthbench` (eval-only)
- `openai/healthbench-professional` (eval-only)
- `openai/graphwalks` (test-time scaling)
- `EleutherAI/hendrycks_math`
- `HuggingFaceH4/MATH-500`
- `terrycraddock/Tree_Of_Thoughts_BASE_24k`
- `gss1147/Got_Math_500K`
- `unimorph/universal_morphologies`
- `Sefaria/Rabbinic-Hebrew-English-Pairs`
- `Sefaria/hebrew_library`
- `Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b`
- `Jackrong/GPT-OSS-120B-Distilled-Reasoning-math`
- `reasoning-degeneration-dev/algorithmic-sft-training-data-v1`
- `lamm-mit/graph-reasoning-messages-11K`
- `sequelbox/DAG-Reasoning-DeepSeek-R1-0528`
- `Gryphe/Opus-4.6-Reasoning-24k`
- `nvidia/Nemotron-RL-ReasoningGym-v1`
- `nvidia/Nemotron-Content-Safety-Reasoning-Dataset`
- `nvidia/PhysicalAI-Traffic-Anomaly-Reasoning`

The Hebrew/Jewish-text slice intentionally uses Sefaria and UniMorph Hebrew sources, and excludes Christian-branded biblical-language datasets.

The frontier-reasoning slice prioritizes open or permissively licensed public traces, especially gpt-oss-120b text distillations. Logprob-only gpt-oss sidecar data is reserved for optional reward/GFlowNet work after tokenizer alignment.

The NVIDIA/Nemotron slice adds procedurally verifiable reasoning, safety-label justification traces, and physical/temporal scene reasoning. These newly added sources are forced into the ToricGT `test` split for held-out test-time scaling. Larger NVIDIA PhysicalAI spatial and PhysicsNeMo CFD datasets are documented in `planning/CYCLIC-EXPERT-GFLOWNET-PLAN.md` as opt-in graph-adapter sources rather than default text curation.

Additional OpenAI benchmark and graph-reasoning notes, plus the proposed
SauravMaheshkar higher-order simplicial augmentation plan, are in
[planning/DATA-II.md](/home/iska/Documents/amelie/bio/ToricGT/planning/DATA-II.md).

Create a bounded sample curation:

```bash
conda run -n tokengt env PYTHONPATH=src python scripts/curate_datasets.py \
  --output-dir data/curated_sample \
  --raw-dir data/raw/hf \
  --max-records-per-source 1000 \
  --num-workers 4 \
  --normalize-batch-size 128
```

Run full curation and leakage-controlled 80/10/10 segmentation:

```bash
conda run -n tokengt env PYTHONPATH=src python scripts/curate_datasets.py \
  --output-dir data/curated \
  --raw-dir data/raw/hf \
  --chunk-size 20000 \
  --num-workers 16 \
  --normalize-batch-size 256
```

Inspect curated outputs:

```bash
conda run -n tokengt env PYTHONPATH=src python scripts/inspect_curated_data.py \
  --curated-dir data/curated \
  --with-token-counts
```

Annotate Hebrew rows for niqqud coverage before public upload or training:

```bash
conda run -n tokengt env PYTHONPATH=src python scripts/enrich_hebrew_niqqud.py \
  --curated-dir data/curated \
  --in-place \
  --batch-size 32768
```

Then write the coverage audit:

```bash
conda run -n tokengt env PYTHONPATH=src python scripts/report_hebrew_niqqud.py \
  --curated-dir data/curated
```

This preserves existing niqqud and marks Hebrew rows with `has_niqqud`, `has_hebrew_without_niqqud`, and related counts in `quality_flags_json`. It does not fabricate vowels for unpointed Hebrew; use `--drop-unpointed-hebrew` only for a strict pointed-Hebrew subset. Add `--update-metadata` only when you explicitly need the same niqqud payload embedded into `metadata_json`; that is slower for large Sefaria rows. Current audit: `3,564,756` rows contain Hebrew, `2,868` are pointed from upstream source text, and `3,561,888` are unpointed upstream rows that are explicitly flagged for filtering or later vetted restoration.

The current full curation contains `5,790,736` records and about `14.52B` estimated whitespace tokens. The split is `4,633,582 / 578,319 / 578,835` rows for train/validation/test. The repaired GoT Math shard contributes `518,439` graph-of-thought records, and Hebrew rows carry niqqud coverage flags.

The public curated split repository is `AmelieSchreiber/toricgt-curated-splits`.
It currently contains the dataset card, manifest, split reports, niqqud audit,
and architecture image. The full local Parquet splits are large
(`train.parquet` is about 40GB; validation and test are about 5.1GB each), so
publish or resume them with `hf upload-large-folder` before expecting the
download command below to fetch Parquet files. The dataset repo is listed with
the checkpoint repo in the ToricGT collection:
<https://huggingface.co/collections/AmelieSchreiber/toricgt>.
After the Parquet files are present on the Hub, download the splits with the
current Hugging Face CLI:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 \
conda run --no-capture-output -n tokengt hf download \
  AmelieSchreiber/toricgt-curated-splits \
  --repo-type dataset \
  --local-dir data/curated \
  --max-workers 8 \
  --include "README.md" \
  --include "train.parquet" \
  --include "validation.parquet" \
  --include "test.parquet" \
  --include "manifest.json" \
  --include "split_report.json" \
  --include "split_report.md" \
  --include "niqqud_report.json"
```

Publish split Parquet files to a public Hugging Face dataset repo using the local credential store. For large uploads, prefer the resumable CLI path:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 HF_XET_HIGH_PERFORMANCE=1 \
conda run --no-capture-output -n tokengt hf upload-large-folder \
  AmelieSchreiber/toricgt-curated-splits \
  data/curated \
  --type dataset \
  --no-private \
  --num-workers 8 \
  --include "README.md" \
  --include "train.parquet" \
  --include "validation.parquet" \
  --include "test.parquet" \
  --include "manifest.json" \
  --include "split_report.json" \
  --include "split_report.md" \
  --include "niqqud_report.json"
```

The Python helper performs the same logical upload for smaller runs:

```bash
conda run -n tokengt env PYTHONPATH=src python scripts/publish_hf_dataset.py \
  --curated-dir data/curated \
  --repo-id AmelieSchreiber/toricgt-curated-splits \
  --public
```

The splitter uses task-family keys, exact content hashes, and SimHash prefixes before assigning train/validation/test to reduce leakage. The target split is 80/10/10. Outputs are Parquet files under `data/curated/`, with per-dataset shards under `data/curated/by_dataset/`.

Publish a checkpoint to the public Hugging Face model repo:

```bash
conda run -n tokengt env PYTHONPATH=src python scripts/publish_hf_model.py \
  --checkpoint checkpoints/<run>/toricgt_final.pt \
  --repo-id AmelieSchreiber/toricgt-checkpoints
```

For large checkpoint files, git-lfs is more reliable than the HTTP helper:

```bash
git lfs install
git clone https://huggingface.co/AmelieSchreiber/toricgt-checkpoints /tmp/toricgt-checkpoints
cp checkpoints/<run>/toricgt_final.pt /tmp/toricgt-checkpoints/
cd /tmp/toricgt-checkpoints
git add .
git commit -m "Upload ToricGT checkpoint"
git push origin main
```

## Implementation Validation

CPU-only validation:

```bash
conda run -n tokengt env PYTHONPATH=src python scripts/validate_cpu.py
```

This checks model shapes, default Soft-MoE routing, tropical-ring attention, embedding-space GFlowNet trajectory balance, and finite rotation-algebra commutator consistency without using meaningful GPU memory.

## Training

Generate local synthetic curriculum records:

```bash
conda run -n tokengt env PYTHONPATH=src python scripts/generate_synthetic_data.py \
  --output data/synthetic/toricgt_synthetic.jsonl \
  --count 10000 \
  --seed 17
```

Small synthetic training run:

```bash
conda run -n tokengt env PYTHONPATH=src python scripts/train.py --steps 100 --attention tropical_ring --device cpu
```

One-step CUDA capacity check for the primary 30M-class model:

```bash
conda run -n tokengt env PYTHONPATH=src python scripts/validate_cuda_capacity.py \
  --device cuda \
  --attention hybrid \
  --d-model 384 \
  --num-heads 8 \
  --num-layers 8 \
  --max-nodes 256 \
  --max-edges 1024 \
  --batch-size 1 \
  --precision bf16 \
  --gflownet-loss-weight 0.01
```

Soft-MoE is enabled by default in the upper half of the encoder with four experts and two slots per expert. Use `--no-soft-moe` for dense feed-forward ablations, or `--soft-moe-all-layers` for the heavier all-layer setting.

Curated Parquet training run:

```bash
conda run -n tokengt env PYTHONPATH=src python scripts/train.py \
  --data-path data/curated/train.parquet \
  --steps 1000 \
  --attention hybrid \
  --device cuda \
  --d-model 256 \
  --num-heads 8 \
  --num-layers 6 \
  --max-nodes 256 \
  --max-edges 1024 \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --precision bf16 \
  --lr-schedule cosine \
  --warmup-steps 2000 \
  --gflownet-loss-weight 0.01 \
  --gflownet-space embedding \
  --val-data-path data/curated/validation.parquet \
  --eval-every 1000 \
  --eval-batches 20 \
  --checkpoint-every 1000 \
  --wandb
```

Resume from a checkpoint by adding:

```bash
--resume checkpoints/<run>/toricgt_step_00001000.pt
```

When `--wandb` is enabled, the trainer reports online metrics for train loss, supervised loss, GFlowNet trajectory-balance loss, GFlowNet loss weight, graph tokens per microbatch, LR, grad norm, validation masked MSE, VRAM, and Soft-MoE routing diagnostics.

YAML configs are available in [config/](/home/iska/Documents/amelie/bio/ToricGT/config). CLI flags override YAML values:

```bash
conda run --no-capture-output -n tokengt env PYTHONPATH=src WANDB_PROJECT=toricgt \
  python scripts/train.py --config config/train.full_30m_warmup.yaml

conda run --no-capture-output -n tokengt env PYTHONPATH=src WANDB_PROJECT=toricgt \
  python scripts/train.py --config config/train.full_30m_cyclic.yaml \
  --resume checkpoints/toricgt_full_30m/toricgt_step_00008000.pt
```

The configured full plan is 2,000 warmup optimizer steps plus 98,000 braided-expert optimizer steps. With batch size 2 and gradient accumulation 16, this is 3.2M graph records, about 0.6906 pass-equivalent epochs over the 4,633,582-record train split. GFlowNet trajectory-balance supervision runs as an auxiliary objective on all 100,000 steps; there is no separate unsupervised-only phase in the current training scripts.

Braided expert-curriculum training is available with `--expert-cyclic-curriculum`. It partitions curated Parquet rows into stable hash-disjoint subsets, trains one active Soft-MoE expert at a time, rotates experts through subsets in a cyclic or braided order, then enables inter-expert distillation and GFlowNet reward shaping after full coverage. The full plan is in `planning/CYCLIC-EXPERT-GFLOWNET-PLAN.md`.

Resume the current 30M-class run from step 2000 with the braided curriculum:

```bash
tmux new-session -d -s toricgt_train_cyclic_experts '
cd /home/iska/Documents/amelie/bio/ToricGT &&
conda run --no-capture-output -n tokengt env PYTHONPATH=src WANDB_PROJECT=toricgt python scripts/train.py \
  --data-path data/curated/train.parquet \
  --val-data-path data/curated/validation.parquet \
  --resume checkpoints/toricgt_full_30m/toricgt_step_00002000.pt \
  --checkpoint-dir checkpoints/toricgt_full_30m \
  --steps 98000 \
  --attention hybrid \
  --device cuda \
  --d-model 384 \
  --num-heads 8 \
  --num-layers 8 \
  --max-nodes 256 \
  --max-edges 1024 \
  --batch-size 2 \
  --grad-accum-steps 16 \
  --precision bf16 \
  --lr 3e-4 \
  --lr-schedule cosine \
  --warmup-steps 2000 \
  --gflownet-loss-weight 0.05 \
  --gflownet-space embedding \
  --expert-cyclic-curriculum \
  --expert-curriculum-subsets 4 \
  --expert-curriculum-phase-steps 500 \
  --expert-curriculum-start-step 2000 \
  --expert-curriculum-order braid \
  --expert-curriculum-distill-weight 0.05 \
  --eval-every 1000 \
  --eval-batches 20 \
  --checkpoint-every 2000 \
  --wandb \
  --log-interval 20 \
  2>&1 | tee -a logs/toricgt_train_cyclic_experts.log
'
```

Watch it with:

```bash
tmux attach -t toricgt_train_cyclic_experts
tail -f logs/toricgt_train_cyclic_experts.log
```

Additional W&B metrics in this mode are `expert_curriculum/phase`, `expert_curriculum/round`, `expert_curriculum/active_expert`, `expert_curriculum/subset_id`, `expert_curriculum/full_coverage_complete`, `expert_curriculum/teacher_expert`, `train/expert_distill_loss`, and `train/expert_teacher_supervised_loss`.

Run the full 30M-class job inside tmux:

```bash
tmux new -s toricgt_train
conda run --no-capture-output -n tokengt env PYTHONPATH=src python scripts/train.py \
  --data-path data/curated/train.parquet \
  --val-data-path data/curated/validation.parquet \
  --steps 100000 \
  --attention hybrid \
  --device cuda \
  --d-model 384 \
  --num-heads 8 \
  --num-layers 8 \
  --max-nodes 256 \
  --max-edges 1024 \
  --batch-size 2 \
  --grad-accum-steps 16 \
  --precision bf16 \
  --lr 3e-4 \
  --lr-schedule cosine \
  --warmup-steps 2000 \
  --gflownet-loss-weight 0.01 \
  --gflownet-space embedding \
  --eval-every 1000 \
  --eval-batches 20 \
  --checkpoint-every 1000 \
  --checkpoint-dir checkpoints/toricgt_full_30m \
  --wandb \
  --log-interval 20
```

Detach with `Ctrl-b d` and reattach with:

```bash
tmux attach -t toricgt_train
```

Synthetic JSONL training uses the same `--data-path` flag:

```bash
conda run -n tokengt env PYTHONPATH=src python scripts/train.py \
  --data-path data/synthetic/toricgt_synthetic.jsonl \
  --steps 1000 \
  --attention tropical_ring \
  --device cuda \
  --precision bf16
```

Use CUDA only after checking the current GPU process:

```bash
nvidia-smi
conda run -n tokengt env PYTHONPATH=src python scripts/train.py --steps 1000 --attention hybrid --device cuda --wandb
```

Recommended 4090 defaults are in `planning/IMPLEMENTATION-PLAN.md`: bf16/fp16, microbatch 1-4, gradient accumulation 8-64, block size 128-512, and frequent checkpoints.

Measured default Soft-MoE model sizes:

- `d=256`, 6 layers, 8 heads: about 10.4M parameters.
- `d=384`, 8 layers, 8 heads: about 29.8M parameters.
- `d=384`, 9 layers, 8 heads: about 35.2M parameters; use 8 layers for a strict sub-35M run.

## Hardware Recommendation

For real training, use the current 4090 when the full 24GB is free. A 2020 13-inch M1 MacBook Pro is fine for editing, CPU validation, small data inspection, and perhaps tiny MPS experiments, but it is not a good target for this graph model: the CUDA kernels, bf16/fp16 throughput, and available sustained memory bandwidth on the 4090 dominate it for 8M-35M parameter training.

If only a very small amount of VRAM is available, keep work to CPU tests, curation, and tiny synthetic runs. For the actual 10M or 30M Soft-MoE/GFlowNet model, wait for the 4090 to be free rather than moving training to the MacBook.

## Evaluation

```bash
conda run -n tokengt env PYTHONPATH=src python scripts/evaluate.py \
  --checkpoint checkpoints/toricgt_final.pt \
  --batches 20 \
  --device cpu
```

Held-out test-time scaling over curated test rows:

```bash
conda run --no-capture-output -n tokengt env PYTHONPATH=src python scripts/test_time_scaling.py \
  --config config/inference.test_time_scaling.yaml
```

By default, this uses the new held-out OpenAI/NVIDIA test-time scaling dataset
group and auto-curates it into `data/test_time_scaling/new_external/test.parquet`
if it is not already present. GFlowNet rollouts and Monte Carlo dropout are on
by default. This is a separate evaluation process and does not interrupt the
active training tmux run. If CUDA memory is tight, run it on CPU or wait for a
checkpoint instead of stopping training.

## Parameter-Golf Dense Random-Order Track

The competition-facing path is now intentionally narrow and dense. The full
ToricGT graph encoder still uses Soft-MoE by default for graph reasoning, but
the Parameter-Golf adapter uses dense shared feed-forward blocks because this
is the better bytes-per-quality tradeoff under a 16,000,000 byte artifact cap.
It stays close to ToricGT by projecting byte chunks to random-order graph
positions, adding toric phase features, using lower softmax plus upper
tropical-ring attention, recurrently reusing blocks for extra effective depth,
and auditing int8 PolarQuant-style KV perturbations during evaluation/export.
It also projects `graph_json` records into compact node/edge byte traces so
graph-of-thought supervision participates in the same byte-level training
stream used by the competition model.

Train the default dense random-order model:

```bash
conda run --no-capture-output -n tokengt env PYTHONPATH=src \
  python scripts/train_parameter_golf_random_order.py \
  --config config/train.parameter_golf_random_order_dense.yaml
```

The default config stores 7 dense blocks at width 384 and applies them twice,
for 14 effective block applications. Random target orders are derived from a
fixed run seed plus per-chunk/sample ids, so each new input gets a new
content-independent order while reproducible runs remain possible. A small
GFlowNet-style action policy is enabled by default: each prefix-visible hidden
state samples one of sixteen latent graph-of-thought actions, adds a tiny
embedding residual before prediction, and trains with a trajectory-balance
surrogate plus entropy diagnostics. Validation/test-time scaling can average
multiple random orders and multiple GFlowNet action samples without reading
future bytes.

The `oai` competition branch adds the OpenAI-retrospective-inspired fast path:
BigramHash context embeddings, CaseOps byte-class features, SmearGate
input-dependent confidence, stripped multi-token auxiliary heads, coprime row
striding, score-first output-bias adaptation during validation, causal
future-byte audits, lightweight quantization-aware grid regularization,
contrastive hidden-state regularization in place of JEPA, compact
noncommutative-toric memory slots, compact domain tags for math, code, graph,
Hebrew, biomedicine, biochemistry, biophysics, and toric tasks, and bit-packed
6-bit row quantized LZMA exports. JEPA is deliberately excluded from this branch per the current
experiment scope.

Reference BPB target bands for the OpenAI Parameter Golf setting:

| Validation BPB | Meaning |
| ---: | --- |
| `>1.35` | Debugging only; not competitive. |
| `1.25-1.35` | Functional small LM, below serious leaderboard quality. |
| `1.20-1.22` | Reasonable first competition target, around the naive baseline range reported by OpenAI. |
| `1.16-1.19` | Strong candidate; real modeling/compression value. |
| `1.13-1.15` | Excellent, near top-tier territory. |
| `<=1.12` | Exceptional/SOTA-class target based on OpenAI's published recap. |
| `<1.10` | Breakthrough-class, requiring especially careful leakage and scoring audits. |

Hardware-time translation for the current workstation:

| Reference budget | Best current-machine estimate | Notes |
| --- | ---: | --- |
| `10 min` on `8xH100` | `4.0 h` central estimate, roughly `3-6 h` plausible range | Based on dense bf16 tensor/memory throughput ratios between an 8-H100 node and the local RTX 4090 24GB, with overhead for data loading, smaller local batch geometry, and less optimized single-GPU kernels. |
| current `50,000` local-step run | `64-65 h` at the observed `4.64 s/step`, roughly `60-72 h` with validation/checkpoint overhead | This is a long local research run, not a claim that the same schedule fits the official challenge wallclock. |
| local challenge-equivalent probe | about `2.3k-4.7k` local steps | At `4.64 s/step`, this is the local step count corresponding to the `3-6 h` estimate above. Use the projection script at these target steps and at `50k` for the long-run extrapolation. |

Watch a tmux run:

```bash
tmux attach -t toricgt_pg_oai
tail -f logs/parameter_golf_oai_random_order.log
```

Project loss and BPB from an early W&B window without interrupting training.
The target step can be any positive value, so the same script can estimate
step 5k, 10k, 25k, or 50k from the first 2k steps:

```bash
conda run --no-capture-output -n tokengt env PYTHONPATH=src \
  python scripts/project_pg_loss.py \
  --wandb-run amelie-iska-math/toricgt-parameter-golf/gbmw7z3a \
  --fit-through-step 2000 \
  --target-step 50000 \
  --metrics train/loss train/bpb \
  --output-dir outputs/projections/oai-step2000-to-50000
```

Export a compressed artifact from either a graph checkpoint or a dense
random-order checkpoint:


```bash
conda run -n tokengt env PYTHONPATH=src python scripts/export_parameter_golf_artifact.py \
  --checkpoint checkpoints/parameter_golf_oai_dense/best.pt \
  --output outputs/parameter_golf/toricgt_artifact.zip \
  --bits 6 \
  --quantization-mode row \
  --compression lzma
```

This remains a byte-accounting scaffold for local experiments. The final
competition package still needs the official `train_gpt.py` wrapper and
challenge evaluator round-trip once the candidate checkpoint is selected.

## Visualizations

```bash
conda run -n tokengt env PYTHONPATH=src python scripts/visualize.py --output-dir outputs/visualizations
```

This writes unit-circle braid frames, a tropical decision-boundary plot,
interactive and static 3D embedding-space graph-of-thought trajectories,
Ramachandran-style reasoning torsion plots, and energy/fitness landscapes
whose low-energy basins represent high-quality terminal reasoning states.

## Toric Music

Generate and play an original dark analog-synth WAV driven by irrational torus
orbits, tropical active-face selection, noncommutative-torus cocycle bias, and
four-expert Soft-MoE-style routing:

```bash
./scripts/music_gen.sh
```

For a headless render without playback:

```bash
./scripts/music_gen.sh --no-play --output outputs/music/toricgt_torus_music.wav
```

## Paper

The LaTeX source is:

```text
assets/toricgt_paper_pg_softmoe_final.tex
```

Generate figures and compile with the Conda TeX engine:

```bash
conda run -n tokengt env PYTHONPATH=src python scripts/generate_paper_figures.py
conda run -n tokengt tectonic -X compile assets/toricgt_paper_pg_softmoe_final.tex --outdir assets
```

The document contains theorem statements and proof sketches for:

- TokenGT graph-token equivariance;
- graph-to-graph universal approximation on bounded graph domains;
- exactness of tropical ring attention;
- permutation equivariance of graph-token Soft-MoE;
- prefix-causal Soft-MoE for Parameter-Golf style language-model scoring;
- recursive PolarQuant-style cache coordinates and perturbation bounds;
- tropical TokenGT approximation of max-plus graph algorithms;
- finite noncommutative-torus approximants and clock/shift regularizers.

## Scaling Target

Use Chinchilla-style token budgets as a lower bound:

- 8M parameters: at least 160M graph/text tokens;
- 15M parameters: at least 300M graph/text tokens;
- 35M parameters: at least 700M graph/text tokens.

Longer high-quality runs can target 40-100 tokens per parameter. Test-time scaling is tracked separately through graph-of-thought rollout budget, GFlowNet trajectory count, verifier calls, and budget forcing; do not inflate the training corpus solely to simulate inference-time compute.

## Current Status

Implemented:

- local source package;
- upstream public clones;
- paper source;
- generated paper figures and compiled PDF;
- detailed implementation plan;
- CPU validation script;
- CPU tests for Soft-MoE equivariance, tropical-ring exactness, algebra, model shapes, and GFlowNet loss;
- GPU validation for bf16 synthetic training, Parquet-backed training, checkpoint reload/evaluation, and synthetic JSONL training;
- curation manifest and clustered splitting logic;
- visualization script.

The active full 30M-class training run uses the documented `tmux` command, reports online metrics to W&B, and writes checkpoints under `checkpoints/toricgt_full_30m/`.
