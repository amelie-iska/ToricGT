# Soft-MoE, GFlowNet GoT, and Parameter-Golf Implementation Plan

Author: Amelie Schreiber

## Current Local State

The local reference repositories are cloned under `./amelie-iska`:

- `tokengt`: TokenGT reference for node/edge graph tokenization.
- `ringattention`: ring-attention reference for exact blockwise long-context scheduling.
- `Tropical-Attention`: tropical attention reference for max-plus algorithmic reasoning.
- `parameter-golf`: local fork of the OpenAI Parameter Golf challenge scaffold.
- `soft-mixture-of-experts`: Soft-MoE reference implementation.

The curated ToricGT dataset lives in `./data/curated` with deterministic grouped splits:

- total records: `5,790,736`
- train/validation/test rows: `4,633,582 / 578,319 / 578,835`
- estimated total tokens: `14,521,960,233`
- estimated train tokens: `11,607,210,191`
- GoT Math repaired count: `518,439`
- Hebrew/Jewish text count: `3,549,020` Sefaria Hebrew-library rows, `3,708` rabbinic Hebrew/Aramaic-English pairs, and `33,177` UniMorph Hebrew rows
- niqqud flags are written to `quality_flags_json` in train/validation/test. The policy is to preserve pointed Hebrew, prefer upstream pointed variants when present, and flag unpointed Hebrew rather than inventing vowels.

## Architecture Commitments

Soft-MoE is on by default. The default research model uses:

- `use_soft_moe=True`
- `soft_moe_num_experts=4`
- `soft_moe_slots_per_expert=2`
- Soft-MoE activated in the upper half of layers unless explicitly overridden
- `use_gflownet_head=True`
- embedding-space GFlowNet actions by default, with token-space toggling available

The baseline training-ready 24 GB 4090 target is:

- `d_model=384`
- `num_layers=8`
- `num_heads=8`
- `max_nodes=256`
- `max_edges=1024`
- `batch_size=2`
- `grad_accum_steps=16`
- bf16 mixed precision

The tested CUDA stress point for this shape uses about 12 GB peak VRAM for batch size 2, leaving headroom for optimizer state, validation, tqdm, logging, and checkpoint I/O.

## Soft-MoE Integration

The graph-token Soft-MoE block must remain permutation equivariant:

- dispatch weights normalize over valid graph tokens for each expert slot
- combine weights normalize over expert slots for each token
- masks remove padded nodes and edges from dispatch
- slot diagnostics are logged to wandb
- default expert count is at least four

Required diagnostics:

- `soft_moe/layer_*/expert_mass_mean`
- `soft_moe/layer_*/expert_mass_std`
- `soft_moe/layer_*/dispatch_entropy`
- `soft_moe/layer_*/combine_entropy`
- `soft_moe/layer_*/residual_scale`

Parameter-Golf adaptation differs from the graph encoder:

- graph encoder may use bidirectional graph context
- Parameter-Golf LM must use prefix-causal Soft-MoE
- prefix-causal dispatch is score-before-update
- future validation bytes must never affect current BPB scoring
- full expert banks are too expensive under 16 MB; use shared base experts plus low-rank adapters

## GFlowNet GoT Fine-Tuning

Default GFlowNet space is embedding space:

- state: graph embedding, node embeddings, edge embeddings, masks, optional algebra metadata
- forward actions: add reasoning node, choose dependency edge, choose tropical active face, apply toric operator, choose Hebrew analysis, terminate
- backward actions: reverse the same abstract edit sequence
- reward terms: correctness, novelty, equivariance, tropical margin, toric commutator error, graph complexity

The implementation already exposes:

- forward logits
- backward logits
- learned `log_z`
- trajectory-balance style auxiliary loss in `scripts/train.py`
- `--gflownet-space embedding|token`
- `--gflownet-loss-weight`

Next implementation increment after the first full run:

- add replay buffer sampling
- add subtrajectory-balance loss
- add reward decomposition logging
- add terminal graph diversity metrics
- add graph edit-distance diversity for GoT terminal samples

## Dataset and Leakage Controls

The current split is grouped by task key, source family key, content hash, and SimHash prefix. It is better than random splitting but should be strengthened before model release claims:

- add graph Weisfeiler-Lehman hashes for graph-native records
- add algebraic normal-form cluster keys for torus, braid, Coxeter, and tropical synthetic data
- add Hebrew root-family holdouts when analyzer-root fields are available
- add Sefaria reference-range holdouts for stricter Jewish text generalization

The current public upload should include:

- `train.parquet`
- `validation.parquet`
- `test.parquet`
- `manifest.json`
- `split_report.json`
- `split_report.md`
- `niqqud_report.json`

Do not upload `keys.txt`, raw Hugging Face caches, local checkpoints, `.pytest_cache`, or tmux logs.

## Parameter-Golf Track

The Parameter-Golf track is a separate micro-LM distillation path, not the full graph-to-graph ToricGT encoder. It should use the fork in `./amelie-iska/parameter-golf` only as a compliance and artifact-accounting scaffold.

Constraints:

- artifact cap: exactly 16,000,000 decimal bytes
- no network access during evaluation
- no validation leakage
- score-before-update for any test-time adaptation
- tokenizer-agnostic BPB
- runnable `train_gpt.py`
- artifact byte accounting for code, weights, tokenizer, and metadata

Recommended micro model:

- byte or 1024/2048-token small-BPE tokenizer
- tied embeddings
- 4 to 6 unique blocks, recurrently applied to 8 to 12 effective blocks
- prefix-causal Soft-MoE only in upper unique blocks
- 2 to 4 experts, 1 to 2 slots per expert
- shared base MLP with low-rank expert adapters
- int6/int8 QAT export
- zstd/zlib round-trip evaluation

Parameter-Golf metrics:

- validation BPB
- artifact bytes
- code bytes
- compressed weight bytes
- tokenizer bytes
- wallclock
- seed
- causal audit pass/fail
- packed-vs-float BPB delta

## Training Run

Full run command target:

```bash
tmux new -d -s toricgt_train "cd /home/iska/Documents/amelie/bio/ToricGT && \
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
  --log-interval 20 2>&1 | tee -a logs/toricgt_train.log"
```

Watch with:

```bash
tmux attach -t toricgt_train
tail -f logs/toricgt_train.log
```

## Wandb Reporting

Online wandb reporting is required for the full run. Log:

- train/supervised loss
- train/GFlowNet loss
- train/total loss
- train/graph tokens per microbatch
- train/lr
- train/grad norm
- val/masked MSE
- system/VRAM allocated
- Soft-MoE expert utilization and entropy
- checkpoint path
- dataset rows and estimated token budget

Future additions:

- GFlowNet terminal diversity
- reward component histograms
- tropical active-face entropy
- toric commutator loss
- Hebrew niqqud coverage and Hebrew task metrics

## Replication Checks

Before starting a long run:

```bash
conda run -n tokengt python -m compileall -q src scripts
conda run -n tokengt env PYTHONPATH=src pytest -q tests
conda run -n tokengt env PYTHONPATH=src python scripts/cuda_stress_test.py \
  --device cuda --attention hybrid --d-model 384 --num-heads 8 --num-layers 8 \
  --max-nodes 256 --max-edges 1024 --batch-size 2 --precision bf16 \
  --gflownet-loss-weight 0.01
```

Post-run:

- upload best checkpoint to the Hugging Face weights repo
- export train/validation/test metrics
- archive wandb run URL in README or `outputs/`
- run inference smoke test from the saved checkpoint
- run eval-only validation from the saved checkpoint

