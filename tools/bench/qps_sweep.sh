#!/usr/bin/env bash
# QPS sweep with Poisson arrivals against an already-running OpenAI-compatible
# server. Drives `vllm bench serve` (random dataset, fixed seed) once per QPS
# value and saves one JSON per run.
#
# Usage:
#   MODEL=/data/Qwen3-4B PORT=8000 ENGINE=pegainfer RESULT_DIR=/data/bench \
#   QPS_LIST="1 2 4 8 16" INPUT_LEN=1024 OUTPUT_LEN=128 SEED=42 \
#   VLLM=.venv/bin/vllm tools/bench/qps_sweep.sh
set -euo pipefail

MODEL=${MODEL:?model path}
PORT=${PORT:-8000}
ENGINE=${ENGINE:?engine label for result filenames}
RESULT_DIR=${RESULT_DIR:?result dir}
QPS_LIST=${QPS_LIST:-"1 2 4 8 16"}
INPUT_LEN=${INPUT_LEN:-1024}
OUTPUT_LEN=${OUTPUT_LEN:-128}
SEED=${SEED:-42}
SECONDS_PER_RUN=${SECONDS_PER_RUN:-60}
VLLM=${VLLM:-.venv/bin/vllm}

mkdir -p "$RESULT_DIR"

for QPS in $QPS_LIST; do
  NUM_PROMPTS=$(python3 -c "print(int($QPS * $SECONDS_PER_RUN))")
  echo "=== $ENGINE qps=$QPS num_prompts=$NUM_PROMPTS in=$INPUT_LEN out=$OUTPUT_LEN seed=$SEED ==="
  "$VLLM" bench serve \
    --backend openai --model "$MODEL" --port "$PORT" \
    --dataset-name random \
    --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --request-rate "$QPS" --burstiness 1.0 \
    --seed "$SEED" \
    --ignore-eos --temperature 0 \
    --tokenizer "$MODEL" \
    --percentile-metrics ttft,tpot,itl,e2el \
    --save-result --result-dir "$RESULT_DIR" \
    --result-filename "${ENGINE}-in${INPUT_LEN}-out${OUTPUT_LEN}-qps${QPS}-seed${SEED}.json"
done
