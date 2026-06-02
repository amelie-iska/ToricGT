#!/usr/bin/env bash
# Launch an official-style Parameter-Golf FineWeb BPB run from the local
# scaffold.  This is the BPB target surface for <= 1.2 style checks; ToricGT
# hard-shard BPB remains an auxiliary reasoning metric.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PG_ROOT="${PG_ROOT:-$REPO_ROOT/amelie-iska/parameter-golf}"
VARIANT="${VARIANT:-sp1024}"
TRAIN_SHARDS="${TRAIN_SHARDS:-10}"
ITERATIONS="${ITERATIONS:-5000}"
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-131072}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-131072}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-250}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"
SEED="${SEED:-1337}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="${RUN_ID:-toricgt-fineweb-bpb-${STAMP}}"
SESSION="${SESSION:-toricgt_fineweb_bpb_${STAMP}}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/training}"
RUN_LOG="${LOG_DIR}/${RUN_ID}.log"
MONITOR_LOG="${LOG_DIR}/${RUN_ID}.monitor.log"
WANDB_LOG="${LOG_DIR}/${RUN_ID}.wandb.log"
DIAGNOSTICS_LOG="${LOG_DIR}/${RUN_ID}.diagnostics.log"
COMMAND_FILE="${LOG_DIR}/${RUN_ID}.command.sh"
MONITOR_COMMAND_FILE="${LOG_DIR}/${RUN_ID}.monitor.command.sh"
WANDB_COMMAND_FILE="${LOG_DIR}/${RUN_ID}.wandb.command.sh"
DIAGNOSTICS_COMMAND_FILE="${LOG_DIR}/${RUN_ID}.diagnostics.command.sh"
MONITOR_SESSION="${MONITOR_SESSION:-${SESSION}_monitor}"
WANDB_SESSION="${WANDB_SESSION:-${SESSION}_wandb}"
DIAGNOSTICS_SESSION="${DIAGNOSTICS_SESSION:-${SESSION}_diagnostics}"
BPB_TARGET="${BPB_TARGET:-1.2}"
BPB_STATE="${BPB_STATE:-$REPO_ROOT/outputs/bpb_codex_loop_state.json}"
BPB_STATUS="${BPB_STATUS:-$REPO_ROOT/outputs/fineweb_bpb_status.json}"
BPB_REPORT="${BPB_REPORT:-$REPO_ROOT/outputs/fineweb_bpb_status.md}"
FINEWEB_STOP_FILE="${FINEWEB_STOP_FILE:-$REPO_ROOT/outputs/fineweb_bpb_monitor_stop}"
FULL_DIAGNOSTICS_JSON="${FULL_DIAGNOSTICS_JSON:-$REPO_ROOT/outputs/fineweb_full_diagnostics.json}"

case "$VARIANT" in
  sp1024)
    VOCAB_SIZE="${VOCAB_SIZE:-1024}"
    DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
    TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
    ;;
  sp*)
    VOCAB_SIZE="${VOCAB_SIZE:-${VARIANT#sp}}"
    DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_${VARIANT}}"
    TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_${VOCAB_SIZE}_bpe.model}"
    ;;
  byte260)
    VOCAB_SIZE="${VOCAB_SIZE:-260}"
    DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_byte260}"
    TOKENIZER_PATH="${TOKENIZER_PATH:-}"
    ;;
  *)
    echo "unsupported VARIANT=$VARIANT; expected sp1024, sp<VOCAB>, or byte260" >&2
    exit 2
    ;;
esac

if [[ ! -d "$PG_ROOT" ]]; then
  echo "missing parameter-golf scaffold: $PG_ROOT" >&2
  exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi
if tmux has-session -t "$MONITOR_SESSION" 2>/dev/null; then
  echo "monitor tmux already exists: $MONITOR_SESSION" >&2
  exit 1
fi
if tmux has-session -t "$WANDB_SESSION" 2>/dev/null; then
  echo "wandb tmux already exists: $WANDB_SESSION" >&2
  exit 1
fi
if tmux has-session -t "$DIAGNOSTICS_SESSION" 2>/dev/null; then
  echo "diagnostics tmux already exists: $DIAGNOSTICS_SESSION" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
cat > "$COMMAND_FILE" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '$PG_ROOT'
conda run --no-capture-output -n tokengt python data/cached_challenge_fineweb.py --variant '$VARIANT' --train-shards '$TRAIN_SHARDS'
RUN_ID='$RUN_ID' \\
DATA_PATH='$DATA_PATH' \\
TOKENIZER_PATH='$TOKENIZER_PATH' \\
VOCAB_SIZE='$VOCAB_SIZE' \\
ITERATIONS='$ITERATIONS' \\
TRAIN_BATCH_TOKENS='$TRAIN_BATCH_TOKENS' \\
VAL_BATCH_SIZE='$VAL_BATCH_SIZE' \\
VAL_LOSS_EVERY='$VAL_LOSS_EVERY' \\
TRAIN_LOG_EVERY='$TRAIN_LOG_EVERY' \\
MAX_WALLCLOCK_SECONDS='$MAX_WALLCLOCK_SECONDS' \\
SEED='$SEED' \\
conda run --no-capture-output -n tokengt torchrun --standalone --nproc_per_node=1 train_gpt.py
EOF
chmod 700 "$COMMAND_FILE"

cat > "$MONITOR_COMMAND_FILE" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '$REPO_ROOT'
python3 scripts/watch_fineweb_bpb_log.py \\
  --log '$RUN_LOG' \\
  --run-id '$RUN_ID' \\
  --target-bpb '$BPB_TARGET' \\
  --state '$BPB_STATE' \\
  --status '$BPB_STATUS' \\
  --report '$BPB_REPORT' \\
  --stop-file '$FINEWEB_STOP_FILE' \\
  --poll-seconds 30
EOF
chmod 700 "$MONITOR_COMMAND_FILE"

cat > "$WANDB_COMMAND_FILE" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '$REPO_ROOT'
conda run --no-capture-output -n tokengt env PYTHONPATH=src \\
  python scripts/mirror_fineweb_log_to_wandb.py \\
  --log '$RUN_LOG' \\
  --run-id '$RUN_ID' \\
  --run-name '$RUN_ID' \\
  --target-bpb '$BPB_TARGET'
EOF
chmod 700 "$WANDB_COMMAND_FILE"

cat > "$DIAGNOSTICS_COMMAND_FILE" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '$REPO_ROOT'
conda run --no-capture-output -n tokengt env PYTHONPATH=src \\
  python scripts/mirror_fineweb_full_diagnostics_to_wandb.py \\
  --log '$RUN_LOG' \\
  --run-id '$RUN_ID' \\
  --run-name '$RUN_ID' \\
  --target-bpb '$BPB_TARGET' \\
  --output-json '$FULL_DIAGNOSTICS_JSON'
EOF
chmod 700 "$DIAGNOSTICS_COMMAND_FILE"

tmux new-session -d -s "$SESSION" "bash '$COMMAND_FILE' 2>&1 | tee '$RUN_LOG'"
tmux new-session -d -s "$MONITOR_SESSION" "bash '$MONITOR_COMMAND_FILE' 2>&1 | tee '$MONITOR_LOG'"
tmux new-session -d -s "$WANDB_SESSION" "bash '$WANDB_COMMAND_FILE' 2>&1 | tee '$WANDB_LOG'"
tmux new-session -d -s "$DIAGNOSTICS_SESSION" "bash '$DIAGNOSTICS_COMMAND_FILE' 2>&1 | tee '$DIAGNOSTICS_LOG'"

cat <<EOF
started Parameter-Golf FineWeb BPB run
tmux:       $SESSION
monitor:    $MONITOR_SESSION
wandb:      $WANDB_SESSION
diagnostic: $DIAGNOSTICS_SESSION
run id:     $RUN_ID
variant:    $VARIANT
train shards: $TRAIN_SHARDS
iterations: $ITERATIONS
val every:  $VAL_LOSS_EVERY
target bpb: $BPB_TARGET
log:        $RUN_LOG
monitor log:$MONITOR_LOG
wandb log:  $WANDB_LOG
diag log:   $DIAGNOSTICS_LOG
command:    $COMMAND_FILE
EOF
