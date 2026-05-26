# ToricGT Implementation Plan

Author: Amelie Schreiber

## 1. Scope

ToricGT is a TokenGT-style graph-to-graph model with:

- vertex and edge graph tokenization;
- tropical, softmax, and tropical-ring attention modes;
- default 4-expert Soft-MoE feed-forward replacement blocks in the upper encoder layers;
- finite noncommutative-torus features from clock and shift operators;
- supervised graph-to-graph training;
- default embedding-space GFlowNet policy head with an optional graph-token-space mode;
- visualization for braid, affine Coxeter, noncommutative-torus, and tropical-geometry tasks.

The current workspace contains public upstream clones under `amelie-iska/`:

- `amelie-iska/tokengt`: upstream TokenGT implementation;
- `amelie-iska/ringattention`: official Ring Attention implementation linked from the paper;
- `amelie-iska/Tropical-Attention`: official Tropical Attention implementation.

GitHub and Hugging Face authenticated push/fork operations must use locally configured credential stores. Do not paste tokens into logs, commands, code, notebooks, or docs.

## 2. Repository Layout

Keep upstream repositories untouched until local modules pass tests.

```text
.
├── amelie-iska/
│   ├── tokengt/
│   ├── ringattention/
│   └── Tropical-Attention/
├── assets/
│   ├── *.pdf
│   ├── extracted/
│   ├── toricgt_paper_pg_softmoe_final.tex
│   ├── toricgt_paper_pg_softmoe_final.pdf
│   └── toricgt_pg_softmoe_figures/
├── data/
│   ├── raw/
│   ├── curated/
│   ├── synthetic/
│   └── splits/
├── planning/
│   └── IMPLEMENTATION-PLAN.md
├── scripts/
│   ├── curate_datasets.py
│   ├── cuda_stress_test.py
│   ├── evaluate.py
│   ├── generate_paper_figures.py
│   ├── smoke_test.py
│   ├── train.py
│   └── visualize.py
└── src/
    └── toricgt/
```

Add future modules in this order:

1. `src/toricgt/synthetic.py`: synthetic algebraic graph generators.
2. `src/toricgt/train_loop.py`: real dataloader and multiphase trainer.
3. `src/toricgt/checkpointing.py`: checkpoint save/load and config hashing.
4. `scripts/publish_hf_dataset.py` and `scripts/publish_hf_model.py`: Hugging Face upload helpers using the local `huggingface_hub` credential store.
5. `src/toricgt/polar_cache.py`: optional PolarQuant-style KV-cache prototype for long-context evaluation.
6. `src/toricgt/parameter_golf_export.py`: byte accounting and export helpers for the micro-LM track.
7. `tests/`: CPU-only tests for algebra, attention exactness, tokenization equivariance, Soft-MoE equivariance, and GFlowNet loss.

Hebrew curation must be niqqud-aware: preserve pointed text, prefer same-consonant pointed variants when present, and annotate unpointed Hebrew with explicit quality flags. Run `scripts/enrich_hebrew_niqqud.py --in-place` after full curation and before public HF upload.

## 3. Upstream Baselines

### 3.1 TokenGT

Use upstream as a comparison and as a source for mature graph preprocessing patterns:

- graph tokenization and identifiers in `large-scale-regression/tokengt/modules/tokengt_graph_encoder.py`;
- synthetic equivariant basis approximation under `equivariant-basis-approximation/`;
- PCQM4Mv2 pipeline under `large-scale-regression/`.

Integration strategy:

1. Keep the standalone `src/toricgt` implementation active for research iteration.
2. Once stable, port attention modules into the TokenGT module path behind a config flag.
3. Preserve upstream softmax/Performer behavior for baseline reproducibility.
4. Add `--attention {softmax,tropical,tropical_ring,hybrid}` and `--toric-features` flags.

### 3.2 Ring Attention

The upstream implementation is JAX-oriented. The local PyTorch implementation initially simulates the exact blockwise schedule on one process:

- use blockwise loops for low-memory exactness;
- later add distributed process groups for multi-GPU ring traversal;
- verify tropical exactness against full tropical attention on CPU before CUDA work.

### 3.3 Tropical Attention

The upstream `TropicalAttention.py` and `models.py` are reference material for max-plus heads and tropical projection. The local module uses a minimal PyTorch API compatible with TokenGT hidden states:

```python
MultiHeadTropicalAttention(d_model, num_heads, mode="tropical_ring")
```

Future port:

- optionally call `tropical_gemm` when installed;
- fall back to PyTorch `amax(a.unsqueeze(-1) + b.unsqueeze(-3))`;
- expose active-face diagnostics for visualization.

## 4. Dataset Plan

### 4.1 Primary Hugging Face Datasets

The current data research, loader decisions, and segmentation plan are maintained in `planning/DATA.md`.

Use at least these nine datasets after license review:

| Dataset | Purpose |
|---|---|
| `AI-MO/NuminaMath-CoT` | large math chain-of-thought |
| `open-r1/OpenR1-Math-220k` | multi-trace math reasoning |
| `open-r1/codeforces-cots` | algorithmic/programming reasoning |
| `openai/gsm8k` | high-quality OpenAI math word problems |
| `EleutherAI/hendrycks_math` | competition math |
| `HuggingFaceH4/MATH-500` | OpenAI verifier subset benchmark |
| `terrycraddock/Tree_Of_Thoughts_BASE_24k` | explicit tree-of-thought supervision |
| `gss1147/Got_Math_500K` | graph-of-thought / CoT math mixture |
| `unimorph/universal_morphologies` | Hebrew and multilingual morphology |
| `Sefaria/Rabbinic-Hebrew-English-Pairs` | rabbinic Hebrew/Aramaic-English parallel text |
| `Sefaria/hebrew_library` | Sefaria Hebrew Jewish text library |

Add synthetic datasets for:

- irrational rotation algebra normal forms;
- higher noncommutative torus word reduction;
- braid group actions on marked unit-circle points;
- affine type-A Coxeter reflections and alcove walks;
- tropical shortest paths, longest paths, transitive closure, assignment-like recurrences;
- graph-of-thought expansions generated from existing CoT records.

Local generator:

```bash
conda run -n tokengt env PYTHONPATH=src python scripts/generate_synthetic_data.py \
  --output data/synthetic/toricgt_synthetic.jsonl \
  --count 10000 \
  --seed 17
```

### 4.2 Graph Conversion

Each record is converted to a graph:

- problem statement node;
- step nodes for reasoning steps;
- equation/formula nodes;
- answer node;
- dependency edges;
- verification edges;
- optional source-span edges;
- task-family metadata as graph-level features.

For algebraic synthetic data:

- generator nodes;
- word-position nodes;
- relation nodes;
- reduction edges;
- state-transition edges;
- phase/cocycle edge labels;
- terminal answer node.

### 4.3 Leakage-Controlled Splitting

Do not use random record-level splitting.

Algorithm:

1. Normalize text fields and graph serializations.
2. Compute word shingles, formula shingles, and graph structural hashes.
3. Cluster near-duplicates with Jaccard threshold around `0.82`.
4. Assign clusters, not individual records, to train/validation/test.
5. Use 80/10/10 target ratios.
6. Hold out entire synthetic families:
   - braid word lengths;
   - Coxeter ranks;
   - graph sizes;
   - denominator ranges for rational approximants;
   - tropical path lengths.

Artifacts:

```text
data/curated/manifest.json
data/curated/train.parquet
data/curated/validation.parquet
data/curated/test.parquet
data/synthetic/*.jsonl
data/curated/split_report.md
```

After curation, run `scripts/inspect_curated_data.py --with-token-counts` to report split ratios, file sizes, estimated token totals, and tokens-per-parameter for the 8M, 15M, and 35M regimes.

### 4.4 Token Budget

Use Chinchilla-style scaling as the lower bound:

- 8M parameters: at least 160M graph/text tokens;
- 15M parameters: at least 300M graph/text tokens;
- 35M parameters: at least 700M graph/text tokens.

Preferred high-quality regime:

- 40-100 tokens per parameter when training compute permits;
- deduplicate aggressively;
- report effective tokens after filtering, not raw downloaded tokens.

For graph tokens, count:

- one token per graph node;
- one token per graph edge;
- one token per formula symbol group when formulas are expanded;
- one token per action in GFlowNet trajectories.

## 5. Model Plan

### 5.1 Model Families

| Name | Layers | Width | Heads | Measured Params | Use |
|---|---:|---:|---:|---:|---|
| `smoke` | 2 | 64 | 4 | 0.25M | CPU and CUDA tests |
| `tiny-10m` | 6 | 256 | 8 | 10.4M | first real training |
| `small-30m` | 8 | 384 | 8 | 29.8M | primary 4090 model |
| `medium-35m-cap` | 9 | 384 | 8 | 35.2M | upper edge of requested range; reduce one layer if strict $<35$M is required |

Current CUDA validation on the available 24GB RTX 4090:

- `smoke`: default 4-expert Soft-MoE, tropical-ring attention, and GFlowNet trajectory balance pass through `scripts/smoke_test.py`;
- `tiny-10m`: curated Parquet graph records train on CUDA with `--gflownet-loss-weight 0.01`;
- `small-30m`: one bf16 optimizer step with 256 node tokens, 1024 edge tokens, default upper-half Soft-MoE, and embedding-space GFlowNet loss peaks at about 6.0GB allocated VRAM for batch 1 and 11.9GB for batch 2.
- Parameter-Golf export scaffold: a 10.2M curated-smoke checkpoint packs to an 8-bit compressed artifact under the 16,000,000 byte limit.

Recommendation: train the 10M or 30M family on the 4090 when it is free. A 2020 13-inch M1 MacBook Pro should be used only for documentation, curation inspection, and CPU/tiny smoke checks; it is not a competitive training device for this CUDA-oriented graph model.

### 5.2 Attention Modes

- `softmax`: baseline.
- `tropical`: full max-plus attention.
- `tropical_ring`: exact blockwise max-plus attention.
- `hybrid`: softmax lower layers, tropical-ring upper layers.

Default curriculum:

1. pretrain with `hybrid`;
2. ablate with pure `softmax` and pure `tropical_ring`;
3. fine-tune with GFlowNet policy head active.

### 5.3 Soft-MoE Defaults

Soft-MoE is enabled by default in the local `ModelConfig`.

- `soft_moe_num_experts=4`;
- `soft_moe_slots_per_expert=2`;
- `soft_moe_start_layer=None`, interpreted as the upper half of the encoder;
- `soft_moe_residual_scale=0.1`;
- `scripts/train.py --no-soft-moe` gives the dense feed-forward ablation;
- `scripts/train.py --soft-moe-all-layers` enables the heavier all-layer setting.
- `scripts/train.py --val-data-path ... --eval-every ...` runs bounded validation during training and logs `val/masked_mse` to wandb when enabled.
- `scripts/train.py --resume checkpoint.pt` restores model and optimizer state and continues step numbering for interrupted runs.
- `scripts/train.py --gflownet-space embedding|token` chooses embedding-space GFlowNet states by default or switches to graph-token-space trajectory states for ablations.
- `scripts/train.py --lr-schedule cosine --warmup-steps ...` provides the default warmup/cosine schedule; use `--lr-schedule constant` for ablations.

Required diagnostics:

- per-expert combine mass;
- dispatch entropy per expert;
- token combine entropy;
- gradient finiteness;
- token-permutation equivariance under random relabelings.

### 5.4 Toric Features

Use finite rational approximants:

- compute `p/q` with continued fractions;
- add sine/cosine phase channels;
- add endpoint cocycle features for edges;
- add commutator loss on selected channels.

Regularizers:

```text
L_comm = || S C H - omega C S H ||_F^2
L_theta = sum_jk || U_j U_k H - exp(2 pi i Theta_jk) U_k U_j H ||_F^2
```

Keep weights small at first: `1e-4` to `1e-3`.

## 6. Training Plan

### 6.1 Phases

Phase A: CPU and tiny CUDA smoke tests.

- Run `scripts/smoke_test.py`.
- Train 50-200 synthetic steps.
- Verify no NaNs.
- Verify tropical-ring output equals full tropical attention on small tensors.

Phase B: Synthetic algebra pretraining.

- rotation algebra normal forms;
- noncommutative torus words;
- braid and Coxeter actions;
- tropical graph algorithms.

Phase C: Reasoning graph supervised training.

- curated CoT/ToT/GoT graph records;
- node, edge, and graph labels;
- verifier outputs.

Phase D: GFlowNet fine-tuning.

- embedding-space trajectory balance;
- SubTB for long sparse tasks;
- reward from correctness, novelty, equivariance, tropical margin, and energy.

Phase E: Evaluation and visualization.

- held-out family extrapolation;
- graph-size extrapolation;
- braid-length extrapolation;
- denominator extrapolation;
- Hebrew morphology and Jewish-text held-out roots/forms/references;
- visualization package outputs.

### 6.2 4090-Safe Defaults

Before any CUDA run:

```bash
nvidia-smi
```

If another process is using substantial VRAM, run CPU smoke tests only.

Default real training:

- bf16 if supported, fp16 fallback;
- microbatch 1-4;
- gradient accumulation 8-64;
- activation checkpointing for long graph sequences;
- attention block size 128-512;
- `num_workers=2` initially;
- checkpoint every 2k steps;
- eval every 1k steps;
- log every 20 steps.

### 6.3 tqdm and Monitoring

Every long operation must use `tqdm`:

- dataset streaming;
- graph conversion;
- split writing;
- synthetic generation;
- training steps;
- validation batches;
- visualization frame rendering;
- checkpoint upload.

Wandb metrics:

- `train/loss_total`;
- `train/loss_node`;
- `train/loss_edge`;
- `train/loss_graph`;
- `train/loss_equivariance`;
- `train/loss_torus`;
- `train/tropical_margin`;
- `train/grad_norm`;
- `train/lr`;
- `eval/exact_match`;
- `eval/node_accuracy`;
- `eval/edge_accuracy`;
- `eval/equivariance_error`;
- `eval/commutator_error`;
- `gfn/tb_loss`;
- `gfn/reward_mean`;
- `gfn/reward_entropy`;
- `gfn/diversity_at_k`;
- `moe/expert_mass_min`;
- `moe/expert_mass_max`;
- `moe/dispatch_entropy`;
- `moe/combine_entropy`;
- `system/vram_allocated`;
- `system/tokens_seen`;
- `system/graph_tokens_seen`.

## 7. GFlowNet Details

### 7.1 State

Embedding-space state:

```text
s_t = (graph_embedding, node_embeddings, edge_embeddings, mask, metadata)
```

Graph-token-space state:

```text
s_t = serialized graph tokens plus valid edit masks
```

### 7.2 Actions

- add reasoning node;
- add proof edge;
- choose active tropical face;
- apply braid generator;
- apply Coxeter reflection;
- apply toric shift;
- apply toric clock;
- refine rational approximant;
- call verifier head;
- terminate.

### 7.3 Reward

Use strictly positive reward:

```text
R = exp((2*C + 0.5*N - E_eq + 0.2*M_trop - 0.1*Energy) / temperature)
```

Where:

- `C`: correctness;
- `N`: novelty;
- `E_eq`: equivariance error;
- `M_trop`: tropical margin;
- `Energy`: graph complexity or invalidity penalty.

### 7.4 Objectives

Start with trajectory balance:

```text
(logZ + sum logPF - logR - sum logPB)^2
```

Then add SubTB for longer trajectories.

## 8. Evaluation Plan

### 8.1 In-Distribution

- train/val loss;
- exact answer match;
- graph edit distance;
- node/edge F1;
- proof-step validity;
- morphology feature accuracy.

### 8.2 Out-of-Distribution

- longer braid words;
- larger Coxeter ranks;
- unseen denominator intervals for `p/q`;
- larger graph sizes;
- longer tropical dynamic-programming horizons;
- held-out Hebrew roots and morphological patterns;
- held-out Sefaria references, categories, or verse/rabbinic text ranges when permitted by dataset license.

### 8.3 Visual Outputs

Write to `outputs/visualizations/`:

- unit-circle braid frames;
- tropical 2D decision boundaries;
- 3D polytope projections;
- 3D plus time trajectories;
- graph evolution frames;
- energy/fitness landscapes;
- Ramachandran-like phase plots.

## 9. Reproducibility

Each run must save:

- git status and commit hash if repository is initialized;
- Python version;
- package versions;
- dataset manifest;
- split hash;
- model config;
- training config;
- random seeds;
- wandb run id;
- checkpoint hash;
- evaluation report.

Use deterministic settings for tests:

```python
torch.manual_seed(seed)
numpy.random.default_rng(seed)
random.seed(seed)
```

Full determinism is not guaranteed on GPU, so report deterministic flags and CUDA versions.

## 10. Publication and Push Steps

Authenticated steps must be run only after local credential stores are configured.

GitHub:

```bash
gh auth login
gh repo fork jw9730/tokengt --clone=false --remote=false
gh repo fork haoliuhl/ringattention --clone=false --remote=false
gh repo fork Baran-phys/Tropical-Attention --clone=false --remote=false
```

Hugging Face:

```bash
hf auth login
conda run -n tokengt env PYTHONPATH=src python scripts/publish_hf_dataset.py \
  --curated-dir data/curated \
  --repo-id AmelieSchreiber/toricgt-curated-splits \
  --public
```

Publish only curated metadata and allowed dataset derivatives. Do not upload restricted or license-incompatible records.

## 11. Immediate Next Engineering Tasks

1. Add unit tests for:
   - clock/shift commutator;
   - tropical-ring exactness;
   - graph tokenization equivariance;
   - GFlowNet trajectory-balance shape and gradient.
2. Keep Soft-MoE tests active for default configs:
   - graph-token permutation equivariance;
   - prefix-causal no-future-dependence;
   - tropical-ring + Soft-MoE + embedding-space GFlowNet CPU backward pass.
3. Extend the synthetic generators in `src/toricgt/synthetic.py` beyond rotation words and tropical shortest paths.
4. Extend `src/toricgt/polar_cache.py` from exact polar coordinates to packed codebooks and fused dequantization kernels.
5. Extend `src/toricgt/parameter_golf_export.py` into a complete contest micro-LM submission script after the local fork is cloned and inspected.
6. Extend the current Parquet/JSONL graph dataloader with richer supervised labels beyond the initial graph-feature reconstruction objective.
7. Add wandb media logging for visualizations.
8. Add Hugging Face parquet export after license review and local credential-store authentication.
9. Port attention into upstream TokenGT once the standalone package is verified.
