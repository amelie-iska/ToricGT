#!/usr/bin/env bash
# Launch a fresh Seq4096 FineWeb run with cleaned W&B metrics, PolarQuant,
# and numerically guarded step-0 advanced losses. The 10K gate is trainer
# step 10000 because this run starts from scratch.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARAMETER_GOLF_ROOT="$REPO_ROOT/amelie-iska/parameter-golf"
PYTHON="${PYTHON:-/home/iska/miniconda3/envs/tokengt/bin/python}"
PROJECT="${WANDB_PROJECT:-toricgt-parameter-golf}"
ENTITY="${WANDB_ENTITY:-amelie-iska-math}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

RUN_ID="${RUN_ID:-toricgt_advdg_step0_polar_seq4096_${STAMP}}"
TRAIN_TMUX="${TRAIN_TMUX:-${RUN_ID}_train}"
ANALYSIS_TMUX="${ANALYSIS_TMUX:-${RUN_ID}_analysis}"
MIRROR_TMUX="${MIRROR_TMUX:-${RUN_ID}_mirror}"
DIAG_TMUX="${DIAG_TMUX:-${RUN_ID}_diag}"
GATE_TMUX="${GATE_TMUX:-${RUN_ID}_gate}"

ITERATIONS="${ITERATIONS:-40000}"
GATE_STEP="${GATE_STEP:-10000}"
TARGET_BPB="${TARGET_BPB:-1.09}"
CONTINUE_THRESHOLD_BPB="${CONTINUE_THRESHOLD_BPB:-1.17}"
ANALYSIS_START_STEP="${ANALYSIS_START_STEP:-250}"
ANALYSIS_INTERVAL_STEPS="${ANALYSIS_INTERVAL_STEPS:-500}"

CKPT_DIR="${CKPT_DIR:-$PARAMETER_GOLF_ROOT/checkpoints/$RUN_ID}"
LOG_PATH="${LOG_PATH:-$PARAMETER_GOLF_ROOT/logs/$RUN_ID.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/post_resume_analysis/$RUN_ID}"
STATE_PATH="${STATE_PATH:-$REPO_ROOT/outputs/$RUN_ID.10k_gate_state.json}"
DIAG_JSON="${DIAG_JSON:-$REPO_ROOT/logs/$RUN_ID.full_diag.latest.json}"
COMMAND_DIR="$REPO_ROOT/logs/$RUN_ID/supervisor"
TRAINER_PATH="$PARAMETER_GOLF_ROOT/records/track_10min_16mb/2026-03-19_TrainingOptSeq4096/train_gpt.py"

if [[ ! -x "$PYTHON" ]]; then
  echo "missing python executable: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$TRAINER_PATH" ]]; then
  echo "missing trainer: $TRAINER_PATH" >&2
  exit 1
fi
"$PYTHON" "$REPO_ROOT/scripts/patch_seq4096_advanced_loss_guard.py" \
  --trainer "$TRAINER_PATH" \
  --python "$PYTHON"
for session in "$TRAIN_TMUX" "$ANALYSIS_TMUX" "$MIRROR_TMUX" "$DIAG_TMUX" "$GATE_TMUX"; do
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    exit 1
  fi
done

mkdir -p "$CKPT_DIR" "$(dirname "$LOG_PATH")" "$OUTPUT_ROOT" "$COMMAND_DIR" "$(dirname "$STATE_PATH")"

TRAIN_CMD=(
  env
  "PYTHONPATH=$REPO_ROOT/src"
  "TORICGT_WANDB_RAW_MODE=off"
  "RUN_ID=$RUN_ID"
  "DATA_PATH=./data/datasets/fineweb10B_sp1024"
  "TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model"
  "VOCAB_SIZE=1024"
  "WANDB=1"
  "WANDB_PROJECT=$PROJECT"
  "WANDB_ENTITY=$ENTITY"
  "WANDB_RUN_ID=$RUN_ID"
  "WANDB_RUN_NAME=$RUN_ID"
  "WANDB_RESUME=allow"
  "WANDB_MODE=online"
  "TARGET_BPB=$TARGET_BPB"
  "ARTIFACT_SIZE_LIMIT_BYTES=16000000"
  "EXPORT_PRUNE_FRACTION=0.12"
  "CHECKPOINT_DIR=$CKPT_DIR"
  "RESUME_CHECKPOINT="
  "RESET_OPTIMIZER_ON_RESUME=1"
  "RESET_RNG_ON_RESUME=1"
  "RESET_LOADER_ON_RESUME=1"
  "ITERATIONS=$ITERATIONS"
  "MAX_WALLCLOCK_SECONDS=0"
  "VAL_LOSS_EVERY=250"
  "TRAIN_LOG_EVERY=10"
  "WANDB_LOG_EVERY=1"
  "CHECKPOINT_EVERY=250"
  "TRAIN_SEQ_LEN=4096"
  "VAL_BATCH_SIZE=524288"
  "TRAIN_BATCH_TOKENS=1048576"
  "WARMUP_STEPS=200"
  "WARMDOWN_ITERS=3000"
  "TIED_EMBED_LR=0.018"
  "MATRIX_LR=0.012"
  "SCALAR_LR=0.012"
  "MUON_MOMENTUM=0.99"
  "MUON_MOMENTUM_WARMUP_START=0.85"
  "MUON_MOMENTUM_WARMUP_STEPS=1500"
  "BIGRAM_BIAS=1"
  "BIGRAM_BIAS_LR=0.006"
  "BIGRAM_BIAS_SCALE=1.0"
  "POLARQUANT_KV_BITS=8"
  "POLARQUANT_TRAIN=1"
  "POLARQUANT_TRAIN_SAMPLE_TOKENS=16"
  "POLARQUANT_EVAL_SAMPLE_TOKENS=256"
  "POLARQUANT_SEED=271828"
  "POLARQUANT_TRAIN_START_STEP=0"
  "POLARQUANT_TRAIN_WARMUP_STEPS=0"
  "ADVANCED_LOSS_SCALE=0.00015"
  "GRAPHCG_LOSS_WEIGHT=0.0005"
  "TORIC_TROPICAL_LOSS_WEIGHT=0.00025"
  "SLEPIAN_LOSS_WEIGHT=0.0005"
  "KOSZUL_BGG_LOSS_WEIGHT=0.0001"
  "ANALOGY_LOSS_WEIGHT=0.000002"
  "ADVANCED_LOSS_SAMPLE_TOKENS=64"
  "TORIC_TROPICAL_FAN_BINS=8"
  "ADVANCED_LOSS_LOG_ONLY=0"
  "ADVANCED_LOSS_START_STEP=0"
  "ADVANCED_LOSS_END_STEP=0"
  "ADVANCED_LOSS_EVERY=16"
  "ADVANCED_LOSS_WARMUP_STEPS=20000"
  "ADVANCED_LOSS_MIN_BEST_VAL_BPB=0"
  "ADVANCED_LOSS_MAX_CE_RATIO=0.000005"
  "GRAD_CLIP_NORM=0.22"
  "CHECKPOINT_ON_TRAIN_BPB_BELOW=1.06"
  "CHECKPOINT_ON_TRAIN_BPB_COOLDOWN_STEPS=250"
  "CHECKPOINT_ON_TRAIN_BPB_MAX=2"
  "VAL_ON_TRAIN_BPB_CHECKPOINT=1"
  "$PYTHON"
  -u
  records/track_10min_16mb/2026-03-19_TrainingOptSeq4096/train_gpt.py
)

ANALYSIS_CMD=(
  env
  "PYTHONPATH=$REPO_ROOT/src"
  "TORICGT_WANDB_RAW_MODE=off"
  "WANDB_PROJECT=$PROJECT"
  "WANDB_ENTITY=$ENTITY"
  "BPB_TARGET=$TARGET_BPB"
  "$PYTHON"
  "$REPO_ROOT/scripts/watch_seq4096_analysis.py"
  --checkpoint-dir "$CKPT_DIR"
  --log "$LOG_PATH"
  --run-path "$ENTITY/$PROJECT/$RUN_ID"
  --output-root "$OUTPUT_ROOT"
  --start-step "$ANALYSIS_START_STEP"
  --analyze-start-step
  --interval-steps "$ANALYSIS_INTERVAL_STEPS"
  --poll-seconds 30
  --target-bpb "$TARGET_BPB"
  --gate-step "$GATE_STEP"
  --python "$PYTHON"
  --training-tmux "$TRAIN_TMUX"
  --codex-review-hook "$REPO_ROOT/scripts/codex_training_review_resume.sh"
  --codex-review-tmux-prefix "toricgt_codex_review_${RUN_ID}"
)

MIRROR_CMD=(
  env
  "PYTHONPATH=$REPO_ROOT/src"
  "TORICGT_WANDB_RAW_MODE=off"
  "WANDB_PROJECT=$PROJECT"
  "WANDB_ENTITY=$ENTITY"
  "$PYTHON"
  "$REPO_ROOT/scripts/mirror_fineweb_log_to_wandb.py"
  --log "$LOG_PATH"
  --run-id "$RUN_ID"
  --target-bpb "$TARGET_BPB"
  --poll-seconds 15
  --diagnostics-json "$DIAG_JSON"
  --gate-state-json "$STATE_PATH"
)

DIAG_CMD=(
  env
  "PYTHONPATH=$REPO_ROOT/src"
  "TORICGT_WANDB_RAW_MODE=off"
  "WANDB_PROJECT=$PROJECT"
  "WANDB_ENTITY=$ENTITY"
  "$PYTHON"
  "$REPO_ROOT/scripts/mirror_fineweb_full_diagnostics_to_wandb.py"
  --log "$LOG_PATH"
  --run-id "$RUN_ID"
  --target-bpb "$TARGET_BPB"
  --poll-seconds 60
  --output-json "$DIAG_JSON"
)

GATE_CMD=(
  env
  "PYTHONPATH=$REPO_ROOT/src"
  "$PYTHON"
  "$REPO_ROOT/scripts/watch_seq4096_bpb_gate.py"
  --log "$LOG_PATH"
  --run-id "$RUN_ID"
  --train-tmux "$TRAIN_TMUX"
  --gate-step "$GATE_STEP"
  --target-bpb "$TARGET_BPB"
  --continue-threshold-bpb "$CONTINUE_THRESHOLD_BPB"
  --state "$STATE_PATH"
  --poll-seconds 30
  --kill-on-miss
)

printf "%q " "${TRAIN_CMD[@]}" > "$COMMAND_DIR/train_command.sh"
printf "\n" >> "$COMMAND_DIR/train_command.sh"
printf "%q " "${ANALYSIS_CMD[@]}" > "$COMMAND_DIR/analysis_command.sh"
printf "\n" >> "$COMMAND_DIR/analysis_command.sh"
printf "%q " "${MIRROR_CMD[@]}" > "$COMMAND_DIR/mirror_command.sh"
printf "\n" >> "$COMMAND_DIR/mirror_command.sh"
printf "%q " "${DIAG_CMD[@]}" > "$COMMAND_DIR/full_diag_command.sh"
printf "\n" >> "$COMMAND_DIR/full_diag_command.sh"
printf "%q " "${GATE_CMD[@]}" > "$COMMAND_DIR/gate_command.sh"
printf "\n" >> "$COMMAND_DIR/gate_command.sh"
chmod 700 "$COMMAND_DIR"/*.sh

tmux new-session -d -s "$TRAIN_TMUX" "cd '$PARAMETER_GOLF_ROOT' && bash '$COMMAND_DIR/train_command.sh' 2>&1 | tee -a '$LOG_PATH'"
sleep "${SIDECAR_START_DELAY_SECONDS:-25}"
tmux new-session -d -s "$ANALYSIS_TMUX" "cd '$REPO_ROOT' && bash '$COMMAND_DIR/analysis_command.sh' 2>&1 | tee -a '$REPO_ROOT/logs/$RUN_ID.analysis_watcher.txt'"
tmux new-session -d -s "$MIRROR_TMUX" "cd '$REPO_ROOT' && bash '$COMMAND_DIR/mirror_command.sh' 2>&1 | tee -a '$REPO_ROOT/logs/$RUN_ID.wandb_mirror.txt'"
tmux new-session -d -s "$DIAG_TMUX" "cd '$REPO_ROOT' && bash '$COMMAND_DIR/full_diag_command.sh' 2>&1 | tee -a '$REPO_ROOT/logs/$RUN_ID.full_diag.txt'"
tmux new-session -d -s "$GATE_TMUX" "cd '$REPO_ROOT' && bash '$COMMAND_DIR/gate_command.sh' 2>&1 | tee -a '$REPO_ROOT/logs/$RUN_ID.10k_gate.txt'"

cat <<EOF
started fresh step-0 Seq4096 PolarQuant advanced 10K gate run
run id:        $RUN_ID
wandb:         https://wandb.ai/$ENTITY/$PROJECT/runs/$RUN_ID
train tmux:    $TRAIN_TMUX
analysis tmux: $ANALYSIS_TMUX
mirror tmux:   $MIRROR_TMUX
diag tmux:     $DIAG_TMUX
gate tmux:     $GATE_TMUX
resume:        disabled / step 0 fresh
checkpoint dir:$CKPT_DIR
log:           $LOG_PATH
analysis root: $OUTPUT_ROOT
gate state:    $STATE_PATH
target BPB:    $TARGET_BPB
continue if:   best <= $CONTINUE_THRESHOLD_BPB by trainer step $GATE_STEP
advanced:      start_step=0 every=16 warmup=20000 min_best_val_bpb=0 max_ce_ratio=0.000005 grad_clip=0.22
cca/topology:  train-time exact CCA/DG/Taylor topology active from step 0 with koszul_bgg=0.0001
polarquant:    kv_bits=8 train=1 train_start_step=0 train_warmup_steps=0 train_sample_tokens=16
EOF
