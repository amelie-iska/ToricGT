# Sequential FineWeb Pivot

Date: 2026-06-03 UTC

## Summary

The current ToricGT Parameter-Golf recovery run is a dense random-order byte model with legal revealed-prefix context, full-rank GraphCG diagnostics, compact GFlowNet policy routing, toric geometry probes, Toric BGG probes, Koszul persistence diagnostics, topology diagnostics, and trajectory-memory instrumentation.  It is scientifically useful, but it is not a credible primary route to the OpenAI Parameter-Golf BPB target.

Latest completed random-order evidence:

```text
run: toricgt-fineweb-revealed-bpb-recovery-20260603T124202Z
latest completed analysis: step 5250
checkpoint: checkpoints/parameter_golf_all_phases_fineweb_from0/random_order_step_00005250.pt
best native OAI/FineWeb BPB: 4.523199440070747
target BPB: <= 1.2
```

The scheduled step-5500 random-order gate was not reached.  On 2026-06-03 UTC the resumed trainer was only around step 5361, was still near the same raw BPB band, and W&B was rejecting some resumed metrics as out-of-order.  Because the last durable checkpoint was step 5250 and the branch was consuming the only local GPU, the random-order supervisor, watcher, and live tmux sessions were stopped and the GPU was reassigned to the sequential FineWeb branch.

The root cause is objective mismatch.  FineWeb BPB rewards strict local left-to-right context, tokenizer-agnostic byte accounting, long context, legal score-first evaluation, artifact-aware quantization, and optional legal test-time adaptation.  The random-order model is score-valid, but its factorization hides the very context that the challenge baseline uses.

## Decision

Promote a sequential FineWeb-first branch as the primary contest path.  The current random-order ToricGT recovery run has been stopped before step 5500 because the BPB signal was flat near `4.52`, the resumed logging state was inconsistent, and the only active local GPU was needed for the sequential branch.

## Primary launch candidate

Use the local OpenAI Parameter-Golf clone as the immediate sequential scaffold:

```text
repo: /home/iska/Documents/amelie/bio/ToricGT/amelie-iska/parameter-golf
data: data/datasets/fineweb10B_sp1024
tokenizer: data/tokenizers/fineweb_1024_bpe.model
```

First launch should prefer the proven long-context recipe:

```bash
RUN_ID=toricgt_seq4096_pivot_seed1337_20260603 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
TRAIN_SEQ_LEN=4096 \
TRAIN_BATCH_TOKENS=393216 \
TIED_EMBED_LR=0.030 \
MATRIX_LR=0.020 \
SCALAR_LR=0.020 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
MUON_MOMENTUM_WARMUP_START=0.92 \
WARMDOWN_ITERS=3000 \
MAX_WALLCLOCK_SECONDS=0 \
VAL_LOSS_EVERY=1000 \
TRAIN_LOG_EVERY=200 \
CHECKPOINT_EVERY=1000 \
CHECKPOINT_DIR=checkpoints/toricgt_seq4096_wandb_ckpt_seed1337_20260603T1743Z \
torchrun --standalone --nproc_per_node=1 records/track_10min_16mb/2026-03-19_TrainingOptSeq4096/train_gpt.py
```

This command runs beyond the 10-minute record cap because the available machine is a single RTX 4090, not the challenge's 8xH100 evaluation setting.  The result is a local training artifact and BPB reference, not an official record claim.

## Local compatibility patch

The local machine uses PyTorch `2.8.0+cu128`.  The standalone `TrainingOptSeq4096` record script failed after initial validation because tensors created inside `torch.inference_mode()` were reused during backward:

```text
RuntimeError: Inference tensors cannot be saved for backward.
```

The local copy of:

```text
amelie-iska/parameter-golf/records/track_10min_16mb/2026-03-19_TrainingOptSeq4096/train_gpt.py
```

was patched to use `torch.no_grad()` for validation and to clone cached cosine and sine tables on return from `Rotary.forward`.  This is a compatibility fix only; it does not change the model, optimizer, data, BPB metric, or launch hyperparameters.

The same local script now also has opt-in durability knobs:

```text
CHECKPOINT_EVERY
CHECKPOINT_DIR
RESUME_CHECKPOINT
```

When enabled, checkpoints are written at validation boundaries and include model weights, optimizer states, token-loader position, CPU/CUDA RNG state, step, and accumulated training time.

W&B logging is default-on in the local Parameter Golf fork as of commit `00ebe09`.  It defaults to:

```text
WANDB_ENTITY=amelie-iska-math
WANDB_PROJECT=toricgt-parameter-golf
WANDB=1
```

The script reads `WANDB_API_KEY` first and otherwise searches local `keys.txt` files at runtime.  `keys.txt`, `wandb/`, checkpoints, and generated model artifacts are ignored and must not be committed.

W&B metric location:

```text
canonical BPB metric: val/bpb
initial live-run value: 4.107707800251096 at _step=0
refresh cadence: every 1000 training steps
```

Commit `5050ecd` on the Parameter Golf fork adds more visible BPB aliases (`bpb`, `val_bpb`, and `openai_parameter_golf/bpb`) for the next resume or launch.  The active `toricgt_seq4096_wandb_ckpt_seed1337_20260603T1743Z` process was already launched from `00ebe09`, so it logs the canonical `val/bpb` metric until resumed.

Commit `7dc7d59` expands the next-resume metric surface further with progress, best BPB, target gap, throughput, optimizer LR, VRAM, artifact, and checkpoint metrics.  The full inventory and utilization policy are in `planning/WANDB-SWEEP.md`.

Smoke verification:

```text
run: toricgt_seq4096_pivot_smoke3
initial val_bpb: 4.1077
one-step train_loss: 6.9366
post-step val_bpb: 4.1072
post-quant round-trip val_bpb: 4.10907054
int8+zlib total submission size: 4,972,577 bytes
result: passed initial-validation -> backward -> final-validation -> quantized-roundtrip
```

Checkpoint/resume smoke verification:

```text
run A: toricgt_seq4096_resume_smoke_a
run A checkpoint: checkpoints/toricgt_seq4096_resume_smoke/toricgt_seq4096_resume_smoke_a_step_000001.pt
checkpoint bytes: 135,612,859
run B: toricgt_seq4096_resume_smoke_b
resume result: loaded step 1, trained step 2, saved step-2 checkpoint
run B val_bpb: 4.1067
run B post-quant round-trip val_bpb: 4.10907022
int8+zlib total submission size after durability patch: 4,983,519 bytes
```

## Active sequential run

```text
tmux session: toricgt_seq4096_pivot
run id: toricgt_seq4096_wandb_ckpt_seed1337_20260603T1743Z
wandb: https://wandb.ai/amelie-iska-math/toricgt-parameter-golf/runs/toricgt_seq4096_wandb_ckpt_seed1337_20260603T1743Z
log: /home/iska/Documents/amelie/bio/ToricGT/amelie-iska/parameter-golf/logs/toricgt_seq4096_wandb_ckpt_seed1337_20260603T1743Z.txt
console log: /home/iska/Documents/amelie/bio/ToricGT/amelie-iska/parameter-golf/logs/toricgt_seq4096_wandb_ckpt_seed1337_20260603T1743Z.console.txt
checkpoint dir: /home/iska/Documents/amelie/bio/ToricGT/amelie-iska/parameter-golf/checkpoints/toricgt_seq4096_wandb_ckpt_seed1337_20260603T1743Z
script: amelie-iska/parameter-golf/records/track_10min_16mb/2026-03-19_TrainingOptSeq4096/train_gpt.py
parameter-golf commit: 00ebe09
python: /home/iska/miniconda3/envs/tokengt/bin/python
status at launch check: W&B handshake complete, GPU process alive
```

Launch command actually used:

```bash
RUN_ID=toricgt_seq4096_wandb_ckpt_seed1337_20260603T1743Z \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
TRAIN_SEQ_LEN=4096 \
TRAIN_BATCH_TOKENS=393216 \
TIED_EMBED_LR=0.030 \
MATRIX_LR=0.020 \
SCALAR_LR=0.020 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
MUON_MOMENTUM_WARMUP_START=0.92 \
WARMDOWN_ITERS=3000 \
MAX_WALLCLOCK_SECONDS=0 \
VAL_LOSS_EVERY=1000 \
TRAIN_LOG_EVERY=200 \
ITERATIONS=20000 \
CHECKPOINT_EVERY=1000 \
CHECKPOINT_DIR=checkpoints/toricgt_seq4096_wandb_ckpt_seed1337_20260603T1743Z \
/home/iska/miniconda3/envs/tokengt/bin/python -m torch.distributed.run --standalone --nproc_per_node=1 \
  records/track_10min_16mb/2026-03-19_TrainingOptSeq4096/train_gpt.py
```

## ToricGT methodology gate

The sequential branch should reintroduce ToricGT-specific methodology only under these gates:

| Methodology | Early status | Promotion gate |
|---|---|---|
| GraphCG chart | diagnostic or tiny auxiliary | no BPB regression and lower covariance/coherence drift |
| GFlowNet graph-of-thought actions | diagnostic first | branch/search lift improves with flat or better BPB |
| Toric geometry probes | diagnostic only | active-face margins become stable and BPB is below `1.2` |
| Toric BGG Category O probes | diagnostic only | resolution consistency improves without BPB regression |
| Koszul persistence | diagnostic only | exactness residual improves without BPB regression |
| Slepian/toric phase audits | diagnostic only | improves recurrence/cache diagnostics or compression slices |
| Trajectory memory | off for contest artifact | enable only for graph/reasoning continuation after BPB target |

## Checkpoint and reporting policy

Every run must record:

```text
run id
git commit
exact launch command
hardware
training wallclock
step count
pre-quant val_loss
pre-quant val_bpb
post-quant round-trip val_loss
post-quant round-trip val_bpb
serialized model bytes
code bytes
total artifact bytes
log path
```

Training may continue to graph-of-thought, tree-of-thought, chain-of-thought, analogical reasoning, trajectory-memory, and ToricGT mathematical auxiliary phases only after the sequential branch produces a contest-credible BPB result.
