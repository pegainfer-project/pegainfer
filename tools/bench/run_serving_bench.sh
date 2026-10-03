#!/usr/bin/env bash
# One-shot serving benchmark: launches a server (pegainfer or vLLM), runs a QPS
# sweep (and optional DSpark concurrency sweep, pegainfer only), then summarizes.
#
# The script launches the server, waits for readiness, runs all sweeps, and
# kills the server on exit (trap). Results land in RESULT_DIR as JSON + a
# summary table on stdout.
#
# Usage:
#   MODEL=/data/Qwen3-4B tools/bench/run_serving_bench.sh
#
# Optional env:
#   MODEL            model path (required)
#   ENGINE           pegainfer | vllm [default: pegainfer]
#   DRAFT_MODEL      DSpark/DFlash draft model path (pegainfer only, skip spec sweep if omitted)
#   GPU              CUDA device ordinal [default: 0]
#   PORT             server port [default: 8000]
#   RESULT_DIR       output directory [default: ./bench-results]
#   DATASET          vLLM dataset: random | sharegpt | hf | custom | ... [default: random]
#   DATASET_PATH     Local dataset path or Hugging Face dataset ID
#   HF_SUBSET        Hugging Face subset (for example, main for openai/gsm8k)
#   HF_SPLIT         Hugging Face split [default: test when DATASET=hf]
#   HF_NO_STREAM     load the Hugging Face dataset locally: 0 | 1 [default: 0]
#   HF_OFFLINE       force Hugging Face Hub and datasets offline: 0 | 1 [default: 0]
#   QPS_LIST         space-separated QPS values [default: "1 2 4 8 10 12 16"]
#   CONCURRENCY_LIST space-separated concurrency values for spec sweep [default: "1 4 8"]
#   INPUT_LEN        input length [default: 1024]
#   OUTPUT_LEN       output length [default: 128]
#   NUM_WARMUPS      warm-up requests before each measured point [default: 0]
#   SAVE_DETAILED    save per-request timing/output arrays: 0 | 1 [default: 0]
#   BINARY           prebuilt pegainfer binary path
#   SEED             base random seed; each point derives its own from SEED + axis/value [default: 42]
#   SECONDS_PER_RUN  seconds per QPS run [default: 60]
#   BENCH            benchmark command; auto-detects `vllm bench serve` or vllm-bench
#   VLLM             path to vllm binary for ENGINE=vllm [default: vllm on PATH]
#   VLLM_EXTRA_ARGS  extra args passed to `vllm serve` [default: "--max-model-len 8192"]
#   LABEL            engine label for result filenames [default: $ENGINE]
#
# Examples:
#   # pegainfer Qwen3-4B QPS sweep
#   MODEL=/data/Qwen3-4B GPU=7 tools/bench/run_serving_bench.sh
#
#   # pegainfer Qwen3-4B + DSpark concurrency sweep
#   MODEL=/data/Qwen3-4B DRAFT_MODEL=/data/dspark_qwen3_4b_block7 GPU=7 \
#     QPS_LIST="" CONCURRENCY_LIST="1 4 8" tools/bench/run_serving_bench.sh
#
#   # vLLM Qwen3-4B QPS sweep
#   ENGINE=vllm MODEL=/data/Qwen3-4B GPU=7 \
#     VLLM=~/develop/xingming/.venv/bin/vllm tools/bench/run_serving_bench.sh
set -euo pipefail

MODEL=${MODEL:?MODEL (model path) is required}
ENGINE=${ENGINE:-pegainfer}
DRAFT_MODEL=${DRAFT_MODEL:-}
GPU=${GPU:-0}
PORT=${PORT:-8000}
RESULT_DIR=${RESULT_DIR:-./bench-results}
DATASET=${DATASET:-random}
DATASET_PATH=${DATASET_PATH:-}
HF_SUBSET=${HF_SUBSET:-}
HF_SPLIT=${HF_SPLIT:-}
HF_NO_STREAM=${HF_NO_STREAM:-0}
HF_OFFLINE=${HF_OFFLINE:-0}
QPS_LIST=${QPS_LIST-"1 2 4 8 10 12 16"}
CONCURRENCY_LIST=${CONCURRENCY_LIST:-"1 4 8"}
INPUT_LEN=${INPUT_LEN:-1024}
OUTPUT_LEN=${OUTPUT_LEN:-128}
NUM_WARMUPS=${NUM_WARMUPS:-0}
SAVE_DETAILED=${SAVE_DETAILED:-0}
SEED=${SEED:-42}
SECONDS_PER_RUN=${SECONDS_PER_RUN:-60}
LABEL=${LABEL:-$ENGINE}
SKIP_BUILD=${SKIP_BUILD:-0}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_LABEL=$(basename "$MODEL")

# A shared seed replays the same prompts at every point — warm for a
# prefix-cache-on server. Deriving from axis+value (not draw order) keeps a
# point's prompt stream stable no matter which sweeps are enabled.
USED_SEEDS=" "
point_seed() { # $1 axis  $2 value
  POINT_SEED=$(( SEED + $(printf '%s' "$1=$2" | cksum | cut -d' ' -f1) % 100000 ))
  case "$USED_SEEDS" in *" $POINT_SEED "*)
    echo "FATAL: seed $POINT_SEED for $1=$2 already used this run (duplicate point or hash collision); points would share a prompt stream" >&2
    exit 1;;
  esac
  USED_SEEDS="$USED_SEEDS$POINT_SEED "
}
# Summarize only this run's files; RESULT_DIR may hold stale JSONs.
RESULT_FILES=()

mkdir -p "$RESULT_DIR"

if [[ -n "${BENCH:-}" ]]; then
  read -r -a BENCH_CMD <<< "$BENCH"
elif command -v vllm >/dev/null 2>&1; then
  BENCH_CMD=(vllm bench serve)
elif command -v vllm-bench >/dev/null 2>&1; then
  BENCH_CMD=(vllm-bench)
else
  echo "FATAL: neither 'vllm bench serve' nor vllm-bench is available" >&2
  exit 1
fi

if [[ "$HF_OFFLINE" == "1" ]]; then
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
fi

DATASET_ARGS=(--dataset-name "$DATASET" --output-len "$OUTPUT_LEN")
if [[ -n "$DATASET_PATH" ]]; then
  DATASET_ARGS+=(--dataset-path "$DATASET_PATH")
fi
if [[ "$DATASET" == "hf" ]]; then
  if [[ -z "$DATASET_PATH" ]]; then
    echo "FATAL: DATASET_PATH is required when DATASET=hf" >&2
    exit 1
  fi
  DATASET_ARGS+=(--hf-split "${HF_SPLIT:-test}")
  if [[ -n "$HF_SUBSET" ]]; then
    DATASET_ARGS+=(--hf-subset "$HF_SUBSET")
  fi
  if [[ "$HF_NO_STREAM" == "1" ]]; then
    DATASET_ARGS+=(--no-stream)
  fi
fi
if [[ "$DATASET" == "random" ]]; then
  DATASET_ARGS+=(--input-len "$INPUT_LEN")
fi
COMMON_BENCH_ARGS=(--num-warmups "$NUM_WARMUPS")
if [[ "$SAVE_DETAILED" == "1" ]]; then
  COMMON_BENCH_ARGS+=(--save-detailed)
fi

# ---- launch server ----------------------------------------------------------
case "$ENGINE" in
  pegainfer)
    BINARY=${BINARY:-"${CARGO_TARGET_DIR:-$REPO_ROOT/target}/release/pegainfer"}
    if [[ "$SKIP_BUILD" != "1" ]]; then
      echo "=== building pegainfer (SKIP_BUILD=1 to skip) ==="
      (cd "$REPO_ROOT" && CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} cargo build --release -p pegainfer-server)
    fi
    SERVER_EXTRA_ARGS=()
    if [[ -n "$DRAFT_MODEL" ]]; then
      SERVER_EXTRA_ARGS+=(--dflash-draft-model-path "$DRAFT_MODEL")
      MODEL_LABEL="${MODEL_LABEL}-dspark"
    fi
    echo "=== launching pegainfer: model=$MODEL gpu=$GPU port=$PORT draft=${DRAFT_MODEL:-none} ==="
    CUDA_VISIBLE_DEVICES=$GPU "$BINARY" \
      --model-path "$MODEL" \
      --port "$PORT" \
      --served-model-name "$MODEL" \
      "${SERVER_EXTRA_ARGS[@]}" \
      > "$RESULT_DIR/server-${ENGINE}-${MODEL_LABEL}.log" 2>&1 &
    SERVER_PID=$!
    READY_TIMEOUT=120
    ;;
  vllm)
    VLLM=${VLLM:-vllm}
    VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS:-"--max-model-len 8192"}
    read -r -a VLLM_EXTRA_CMD <<< "$VLLM_EXTRA_ARGS"
    if [[ -n "$DRAFT_MODEL" ]]; then
      echo "WARN: DRAFT_MODEL is ignored for ENGINE=vllm" >&2
    fi
    echo "=== launching vLLM: model=$MODEL gpu=$GPU port=$PORT ==="
    CUDA_VISIBLE_DEVICES=$GPU "$VLLM" serve "$MODEL" \
      --port "$PORT" \
      --served-model-name "$MODEL" \
      --trust-remote-code \
      "${VLLM_EXTRA_CMD[@]}" \
      > "$RESULT_DIR/server-${ENGINE}-${MODEL_LABEL}.log" 2>&1 &
    SERVER_PID=$!
    # vLLM cold start (torch.compile) can take 70+ seconds
    READY_TIMEOUT=300
    ;;
  *)
    echo "FATAL: ENGINE must be 'pegainfer' or 'vllm', got '$ENGINE'" >&2
    exit 1
    ;;
esac

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "=== shutting down server (pid $SERVER_PID) ==="
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# ---- wait for readiness -----------------------------------------------------
echo "=== waiting for server readiness (timeout ${READY_TIMEOUT}s) ==="
for i in $(seq 1 "$READY_TIMEOUT"); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "FATAL: server process died. Server log:" >&2
    cat "$RESULT_DIR/server-${ENGINE}-${MODEL_LABEL}.log" >&2
    exit 1
  fi
  if curl -sf "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
    echo "=== server ready (after ${i}s) ==="
    break
  fi
  sleep 1
done

if ! kill -0 "$SERVER_PID" 2>/dev/null; then
  echo "FATAL: server process died during readiness wait" >&2
  cat "$RESULT_DIR/server-${ENGINE}-${MODEL_LABEL}.log" >&2
  exit 1
fi

if ! curl -sf "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
  echo "FATAL: server not ready after ${READY_TIMEOUT}s" >&2
  cat "$RESULT_DIR/server-${ENGINE}-${MODEL_LABEL}.log" >&2
  exit 1
fi

# ---- QPS sweep --------------------------------------------------------------
if [[ -n "${QPS_LIST// /}" ]]; then
  echo "=== QPS sweep: qps=[$QPS_LIST] dataset=$DATASET ==="
  for QPS in $QPS_LIST; do
    NUM_PROMPTS=$(python3 -c "print(int($QPS * $SECONDS_PER_RUN))")
    point_seed qps "$QPS"
    echo "--- $LABEL $MODEL_LABEL qps=$QPS num_prompts=$NUM_PROMPTS dataset=$DATASET seed=$POINT_SEED ---"
    "${BENCH_CMD[@]}" \
      --backend openai --model "$MODEL" --port "$PORT" \
      --base-url "http://localhost:$PORT" \
      "${DATASET_ARGS[@]}" \
      "${COMMON_BENCH_ARGS[@]}" \
      --num-prompts "$NUM_PROMPTS" \
      --request-rate "$QPS" \
      --seed "$POINT_SEED" \
      --ignore-eos --temperature 0 \
      --tokenizer "$MODEL" \
      --percentile-metrics ttft,tpot,itl,e2el \
      --save-result --result-dir "$RESULT_DIR" \
      --result-filename "${LABEL}-${MODEL_LABEL}-${DATASET}-qps${QPS}-seed${POINT_SEED}.json"
    RESULT_FILES+=("$RESULT_DIR/${LABEL}-${MODEL_LABEL}-${DATASET}-qps${QPS}-seed${POINT_SEED}.json")
  done
else
  echo "=== QPS sweep skipped (QPS_LIST is empty) ==="
fi

# ---- Concurrency sweep (pegainfer only) ------------------------------------
if [[ "${ENGINE}" == "pegainfer" && -n "${CONCURRENCY_LIST// /}" ]]; then
  echo "=== spec concurrency sweep: c=[$CONCURRENCY_LIST] dataset=$DATASET ==="
  for C in $CONCURRENCY_LIST; do
    NUM_PROMPTS=$(python3 -c "print(int($C * $SECONDS_PER_RUN))")
    point_seed c "$C"
    echo "--- $LABEL $MODEL_LABEL c=$C num_prompts=$NUM_PROMPTS dataset=$DATASET seed=$POINT_SEED ---"
    "${BENCH_CMD[@]}" \
      --backend openai --model "$MODEL" --port "$PORT" \
      --base-url "http://localhost:$PORT" \
      "${DATASET_ARGS[@]}" \
      "${COMMON_BENCH_ARGS[@]}" \
      --num-prompts "$NUM_PROMPTS" \
      --max-concurrency "$C" \
      --seed "$POINT_SEED" \
      --ignore-eos --temperature 0 \
      --tokenizer "$MODEL" \
      --percentile-metrics ttft,tpot,itl,e2el \
      --save-result --result-dir "$RESULT_DIR" \
      --result-filename "${LABEL}-${MODEL_LABEL}-${DATASET}-c${C}-seed${POINT_SEED}.json"
    RESULT_FILES+=("$RESULT_DIR/${LABEL}-${MODEL_LABEL}-${DATASET}-c${C}-seed${POINT_SEED}.json")
  done
fi

METRICS_FILE="$RESULT_DIR/${LABEL}-${MODEL_LABEL}-${DATASET}-metrics.prom"
if curl -sf "http://localhost:$PORT/metrics" > "$METRICS_FILE"; then
  echo "metrics saved to $METRICS_FILE"
else
  echo "WARN: /metrics was unavailable; no metrics snapshot saved" >&2
  rm -f "$METRICS_FILE"
fi

# ---- summary ---------------------------------------------------------------
echo ""
echo "=== results summary ==="
"$SCRIPT_DIR/summarize_qps_sweep.py" "${RESULT_FILES[@]}"
echo ""
echo "results saved to $RESULT_DIR"
