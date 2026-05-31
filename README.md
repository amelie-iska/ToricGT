# ToricGT

Author: Amelie Schreiber

ToricGT is a research prototype for TokenGT-style graph-to-graph modeling with tropical ring attention, default 4-expert Soft-MoE feed-forward blocks, finite noncommutative-torus features, and embedding-space GFlowNet fine-tuning.

![ToricGT architecture and training paradigm](assets/toricgt_architecture_and_training_diagram.png)

![Dark-mode ToricGT torus reasoning animation](assets/toricgt_torus_reasoning_dark.gif)

<p align="center">
  <a href="./assets/toricgt_paper_pg_softmoe_final.tex"><img src="https://img.shields.io/badge/arXiv-94133F?style=for-the-badge&logo=arxiv" alt="arXiv"/></a>
  <a href="https://github.com/amelie-iska/ToricGT/"><img src="https://img.shields.io/badge/📝%20GitHub-007A87?style=for-the-badge&logoColor=grey" alt="GitHub"/></a>
  <a href="https://huggingface.co/blog/AmelieSchreiber/toricgt"><img src="https://img.shields.io/badge/HuggingFace-DE9B35.svg?style=for-the-badge&logo=HuggingFace" alt="HF"/></a>
</p>

*Note: consider PH disambiguation along decision boundaries or of words with multiple meaning*

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
- `src/toricgt/toric_geometry_tasks.py`: training-only low-rank toric probes for Newton active-face, bend, binomial, affine-Coxeter, braid, and phase-foliation signals.
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

The `oai` Parameter Golf trainer also has automatic best-checkpoint publishing
enabled in `config/train.parameter_golf_random_order_dense.yaml`. At each
validation improvement, it saves `checkpoints/parameter_golf_oai_dense/best.pt`
and uploads that file to `AmelieSchreiber/toricgt-checkpoints` as
`parameter_golf_oai_best.pt` only when the composite promotion score improves:

```text
publish_score = val_bpb + 0.05 * complexity/val/prediction_target_ncd_lzma_mean
```

The matching manifest is `parameter_golf_oai_best.json`. Worse checkpoints are
skipped, so the HF repo remains the current best candidate rather than a dump
of every interval. Auth comes from the normal Hugging Face credential store or
the local ignored `keys.txt`; tokens are never printed, logged, or stored in
checkpoints. Publish failures are reported as `hf_publish/error` in W&B and do
not stop training.

Local periodic training checkpoints are retained every 250 steps by default
under `checkpoints/parameter_golf_oai_dense/`; they are not pruned, so earlier
resume points remain available for ablations and recovery.

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

The same branch now separates GraphCG-style lattice-basis training from the
analogical-reasoning mechanism. With CUDA memory available,
`graphcg_num_directions: auto` expands the learned hidden-space concept chart
under a 10% safety margin; on the local 24GB RTX 4090 this resolves to 256
directions. Analogical maps are functor-like, but they are not only
parallelogram vector losses: they are filtered simplicial/chain maps between
collections of related thought vectors. Repeated hidden relation classes are
first expressed in the GraphCG chart, then build normalized nested
Vietoris-Rips/simplex-tree filtrations, including directed flag-complex edges
from an antisymmetric noncommutative form. W&B logs `train/analogy_*` metrics
for functor loss, filtration inclusions, chain-map commutators, directed
transitive closure, directed cycle/holonomy balance, edge/triangle densities,
and basis alignment. The step-local topology pass now constructs the
corresponding filtered complexes on the reasoning states themselves, in the
same GraphCG concept chart. Each sampled graph-of-thought window becomes a
radius-parametrized simplex-tree hierarchy with time-oriented directed edges,
Betti-0 and cycle-rank proxies, boundary residuals, Dirichlet energy, directed
chain commutators, and HDBSCAN stability/noise diagnostics. Consecutive windows
are joined by soft transport maps that induce approximate morphisms of
persistence modules; W&B reports `train/analogy_step_analogical_map_loss`,
`train/analogy_step_directed_map_loss`, and
`train/analogy_step_transport_entropy` along with the rest of
`train/analogy_step_*`. The training objective also includes a
radius-parametrized HDBSCAN surrogate over mutual-reachability distances:
persistent high-stability relation neighbors are pulled together, while
unstable/outlier relation arrows contribute little. W&B reports
`train/analogy_hdbscan_loss`, `train/analogy_hdbscan_stability`,
`train/analogy_hdbscan_persistent_edge_density`,
`train/analogy_hdbscan_outlier_score`, and
`train/analogy_hdbscan_core_radius`.
Following `assets/1508.01166v2.pdf` (Mohamed, Hirani, and Samtaney's DEC
Navier-Stokes discretization), the step-local topology pass also audits
conservative reasoning flow over the same directed simplicial windows. The
directed adjacency is treated as a discrete 1-form; its skew flow gives a
divergence/mass residual, node vorticity, kinetic energy, Hodge-balanced energy,
and a wedge/interior-product consistency proxy for convective transport. These
are not a fluid simulator. They are cheap DEC-style invariants that should make
long graph-of-thought paths less noisy while leaving BPB primary. W&B reports
`train/analogy_step_dec_conservation_loss`,
`train/analogy_step_dec_mass_residual`,
`train/analogy_step_dec_vorticity_drift`,
`train/analogy_step_dec_kinetic_energy`,
`train/analogy_step_dec_kinetic_energy_drift`,
`train/analogy_step_dec_hodge_balance`, and
`train/analogy_step_dec_wedge_interior_residual`.
The periodic analysis suite also renders these objects under
`outputs/post_resume_analysis/<run>/step-*/geometry/topology/`: per-branch
filtration curves for edge density, triangle density, directed asymmetry,
noncommutative cycle flux, and DEC conservation/mass residuals, plus HDBSCAN
stable-cluster and outlier curves.
The heatmaps include normalized hidden-arrow distances, mutual-reachability
distances, density-persistence adjacency, antisymmetric toric skew, and directed
adjacency at several radii. The new `*_step_radius_hierarchy.png` panels show
window-by-radius edge density, Betti-0, cycle-rank, DEC conservation/energy
panels, analogical map residuals, and directed map residuals for the actual
hidden reasoning states. The analysis suite also writes
`*_toric_phase_simplicial_trajectory.png`, which projects the
irrational rotation-algebra phase path onto a torus, overlays local
Vietoris-Rips edges, and draws the soft analogical maps between reasoning
windows. These plots sit next to the 3D graph-of-thought trajectories,
Ramachandran-style phase plots, energy landscapes, reasoning/K/BPB triangles,
and tetrahedral simplex diagnostics.
The condensed and full papers now make the next-iteration algebra explicit:
toric character/cocharacter lattices become Weyl-chamber coordinates once a
root datum is attached, translated root hyperplanes give affine
Weyl/Coxeter actions, and ordered wall crossings give Artin braid actions.
The same section describes noncommutative-torus phase projections as finite
audit shadows of irrational Kronecker foliations on ordinary tori: smooth
projected phase leaves are expected between verified tropical or algebraic
wall crossings, while high leaf residual indicates incoherent toric memory.
For computationally tractable sampled windows, the analysis pass also computes
exact F2 homology up to degree 1 and nearest-neighbor induced maps between
consecutive windows. It reports exact H0/H1 dimensions, induced map ranks,
minimal radius shifts needed for a simplicial map, edge/triangle validity, and
directed-edge validity. These values are written to
`*_exact_persistence_morphisms.png`, `reasoning_geometry_summary.json`, and,
when `--wandb` is passed, to the same W&B project under `analysis/*`.

The `oai` branch also turns the toric-geometry next-iteration proposal into a
training-only signal. A low-rank, 6-bit fake-quantized toric probe predicts
Newton-polytope active faces from hidden states, aligns soft moment-map
features to irrational phase teachers, regularizes Cartier-style bend
magnitudes, checks toric binomial relations, enforces affine-Coxeter reflection
and A2 braid consistency, and audits noncommutative phase-foliation leaves. The
probe is excluded from Parameter-Golf artifact export, so it can shape training
without increasing deploy bytes. W&B reports `train/toric_geometry_loss`,
`train/toric_active_face_margin`, `train/toric_bend_magnitude`,
`train/toric_binomial_residual`, `train/toric_coxeter_loss`,
`train/toric_braid_loss`, and `train/toric_leaf_residual`. The geometry suite
adds `*_toric_shadow_audit.png`, showing occupied fan cells, active-face
margins, bend magnitudes, branch fan coverage, and phase-leaf residuals.

Operational rule for the current `oai` experiments: training remains paused
until the full analysis suite has completed and the generated plot classes have
been reviewed. The requested "there be dragons" responding-onlooker ablation is
currently an absence check: repository search finds no matching observer,
prompt, or role path outside ignored outputs/checkpoints/data.

Planning notes:
[`planning/GRAPHCG-ANALOGY-TOPOLOGY-PLAN.md`](planning/GRAPHCG-ANALOGY-TOPOLOGY-PLAN.md)
separates GraphCG basis learning from analogical functor structure and defines
the filtered topological map contract.
[`planning/TOPOLOGICAL-ANALOGY-IMPLEMENTATION.md`](planning/TOPOLOGICAL-ANALOGY-IMPLEMENTATION.md)
defines the directed persistent-topology implementation contract, and
[`planning/HoTT.md`](planning/HoTT.md) gives the homotopy-type-theory analogue
with the HoTT book reference: <https://homotopytypetheory.org/book/>.
[`planning/TORIC-GEOMETRY-TRAINING-SIGNAL.md`](planning/TORIC-GEOMETRY-TRAINING-SIGNAL.md)
records the toric probe losses, metrics, resume policy, and analysis gate.
[`planning/DEC-CONSERVATIVE-REASONING.md`](planning/DEC-CONSERVATIVE-REASONING.md)
records the DEC/NSE comparison and the conservative reasoning-flow additions.

Kolmogorov-style reasoning diagnostics are enabled by default on the `oai`
branch without changing the BPB objective. The trainer periodically logs
compressor-tagged proxies such as `complexity/train/target_cond_k_lzma_mean`,
`complexity/train/order_program_k_zlib_mean`,
`complexity/train/gflownet_action_trace_k_lzma_mean`,
`complexity/train/target_helper_cond_k_lzma_mean`,
`complexity/train/information_symmetry_gap_k_lzma_mean`,
`complexity/train/analogical_transfer_relative_k_lzma_mean`, and validation
analogues. These estimate conditional reasoning/program complexity,
random-order tree-program length, prediction-target NCD, GFlowNet action-trace
complexity, and analogical transfer of `K(x|y)`. The helper side of the
conditional program includes the strict random-order prefix, the public
permutation/tree program, the original byte chunk, optional GFlowNet action
traces, and optional serialized graph/tree helper payloads. Negative
`analogical_transfer_relative_k_*` values mean a source analogy shortened the
estimated program for the target relative to direct helpers alone. Prediction
relative-K rewards are correctness-gated, so a short wrong prediction is not
treated as an improvement. They are diagnostic unless an explicit future config
gives them nonzero training weight.

The local `AmelieSchreiber/toricgt-curated-splits` mirror is used as far as is
competition-safe: supervised training streams `data/curated_hf_shards/train/*.parquet`,
validation streams `data/curated_hf_shards/validation/*.parquet`, and the test
shards are reserved for held-out score-first evaluation and GFlowNet
test-time-scaling studies. Do not train on validation/test bytes or use future
validation/test bytes for adaptation; score-first state updates are allowed only
after the current byte has been scored.

The loader already packs multiple source rows into one byte chunk with a
separator, so short problems are combined into a single random-order training
instance. The default `oai` config keeps the initial BPB capture stream
text-first, introduces medium-length rows after step `2500`, and delays the
larger technical graph-projection stream until step `6000` with
math/code/graph/reasoning/health/physics/biomed/biochem task-family filters.
The current restart is from the aligned step `1,250` checkpoint with a lower
`1500-2000` LR multiplier and zero medium/hard mixture in that capture window
to stay below the observed bounce band while the new topology metrics are
diagnostic-only. A bounded
checkpoint-level adaptive controller remains enabled for GFlowNet entropy
target, GFlowNet loss weight, and hard-row mix; it writes
`checkpoints/parameter_golf_oai_dense/adaptive_controller_state_01500_capture_schneller.json`
and logs `controller/*` metrics to W&B.

For challenge-time throughput experiments, use the long-context packed config:

```bash
conda run --no-capture-output -n tokengt env PYTHONPATH=src \
  python scripts/train_parameter_golf_random_order.py \
  --config config/train.parameter_golf_random_order_packed_2048.yaml \
  --resume checkpoints/parameter_golf_oai_dense/best.pt
```

That config extends the context to 2048 bytes, increases graph projection room,
uses `<next_problem>` separators, and begins with larger technical rows before
raising the complex-row threshold again after step 750. Position embeddings are
resized on resume, and the optimizer state is reset only if the position table
shape changes. This is intended for from-checkpoint challenge-equivalent probes;
the active 1024-token run remains the safer local long run.

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
tail -f logs/training/oai-resume-23250-recovery.log
tmux attach -t toricgt_pg_oai_analysis
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

For earlier checkpoints, keep the same command and change the target/output:

```bash
conda run --no-capture-output -n tokengt env PYTHONPATH=src \
  python scripts/project_pg_loss.py \
  --wandb-run amelie-iska-math/toricgt-parameter-golf/gbmw7z3a \
  --fit-through-step 2000 \
  --target-step 10000 \
  --metrics train/loss train/bpb \
  --output-dir outputs/projections/oai-step2000-to-10000
```

Evaluate standalone complexity diagnostics over a held-out shard sample:

```bash
conda run --no-capture-output -n tokengt env PYTHONPATH=src \
  python scripts/evaluate_complexity.py \
  --data-glob 'data/curated_hf_shards/validation/*.parquet' \
  --samples 512 \
  --output-dir outputs/complexity/oai-validation
```

Plot the resulting summary:

```bash
conda run --no-capture-output -n tokengt env PYTHONPATH=src \
  python scripts/plot_complexity.py \
  --summary-json outputs/complexity/oai-validation/complexity_summary.json \
  --output-dir outputs/complexity/oai-validation/plots
```

Evaluate reasoning-simplex diagnostics from a checkpoint. This produces
heatmapped triangles, static tetrahedra, interactive tetrahedron HTML, and
CSV/JSON records computed from actual model passes at several reasoning
budgets:

```bash
conda run --no-capture-output -n tokengt env PYTHONPATH=src \
  python scripts/evaluate_reasoning_simplex.py \
  --checkpoint checkpoints/parameter_golf_oai_dense/best.pt \
  --data-glob 'data/curated_hf_shards/validation/*.parquet' \
  --samples 16 \
  --budgets 1 2 4 8 \
  --output-dir outputs/reasoning_simplex/oai-best
```

For larger technical examples that better exploit tropical-ring context:

```bash
conda run --no-capture-output -n tokengt env PYTHONPATH=src \
  python scripts/evaluate_reasoning_simplex.py \
  --checkpoint checkpoints/parameter_golf_oai_dense/best.pt \
  --data-glob 'data/curated_hf_shards/validation/*.parquet' \
  --seq-len 1024 \
  --samples 16 \
  --budgets 1 2 4 8 \
  --min-estimated-tokens 256 \
  --task-family-keywords math code graph got reasoning health physics biomed biochem \
  --output-dir outputs/reasoning_simplex/oai-large-technical
```

Run the full reasoning-geometry suite, including toric shadow audits,
directed/persistence plots, 3D trajectories, triangles, and tetrahedra:

```bash
conda run --no-capture-output -n tokengt env PYTHONPATH=src \
  python scripts/evaluate_reasoning_geometry_suite.py \
  --checkpoint checkpoints/parameter_golf_oai_dense/best.pt \
  --data-glob 'data/curated_hf_shards/validation/*.parquet' \
  --output-dir outputs/reasoning_geometry_suite/oai-best \
  --records 2 \
  --branches 3 \
  --seq-len 384 \
  --max-files 8 \
  --scan-batches-per-file 2 \
  --scan-batch-size 256 \
  --max-pca-points 1024 \
  --max-plot-points 128 \
  --max-mst-nodes 64 \
  --device cuda \
  --precision bf16 \
  --wandb \
  --wandb-project toricgt-parameter-golf
```

The primary triangle has vertices `reasoning budget`, `K(x)`, and `low BPB`.
The heatmap is dark near the base reasoning region and fades toward lighter
blue as reasoning budget, trajectory length, and complexity increase. The
tetrahedra add either hidden-trajectory MST efficiency or GFlowNet diversity as
the fourth vertex. MST efficiency treats the hidden reasoning path as a complete
weighted graph and asks how compactly a minimum spanning tree summarizes the
trajectory; it is useful for spotting trajectories that gain BPB by organized
exploration rather than noisy wandering.

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
