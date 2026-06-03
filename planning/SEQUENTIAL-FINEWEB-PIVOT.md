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

Promote a sequential FineWeb-first branch as the primary contest path.  Keep the current random-order ToricGT run only long enough to finish the scheduled step-5500 analysis.  If step 5500 is still flat near `4.52` BPB, demote or stop that run and use the GPU for sequential FineWeb training.

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

## Active sequential run

```text
tmux session: toricgt_seq4096_pivot
run id: toricgt_seq4096_pivot_seed1337_20260603T1718Z
log: /home/iska/Documents/amelie/bio/ToricGT/amelie-iska/parameter-golf/logs/toricgt_seq4096_pivot_seed1337_20260603T1718Z.txt
script: amelie-iska/parameter-golf/records/track_10min_16mb/2026-03-19_TrainingOptSeq4096/train_gpt.py
python: /home/iska/miniconda3/envs/tokengt/bin/python
status at launch check: warmup step 9/20, GPU process alive
```

Launch command actually used:

```bash
RUN_ID=toricgt_seq4096_pivot_seed1337_20260603T1718Z \
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
