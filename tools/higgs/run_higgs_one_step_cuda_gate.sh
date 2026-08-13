#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run the Higgs Audio one-step CUDA golden gate.

Required:
  --model-dir DIR          Higgs checkpoint directory.

Optional:
  --golden FILE            Golden safetensors fixture.
                           Default: test_data/higgs-one-step-audio-logits.safetensors
  --result-root DIR        Result directory.
                           Default: /data/results/pegainfer/higgs-audio
  --label LABEL            Output label. Default: current git short SHA.
  --sm SM                  PEGAINFER_CUDA_SM. Default: 89
  --nvcc-jobs N            PEGAINFER_NVCC_JOBS. Default: 8
  --profile                Also capture an NSYS profile for the actual dump.
  -h, --help               Show this help.

Outputs:
  <result-root>/actual/higgs-one-step-actual-cuda-bf16-auto-<label>.safetensors
  <result-root>/actual/higgs-one-step-session-cuda-bf16-auto-<label>.safetensors
  <result-root>/actual/semantic-compare-auto-<label>.txt
  <result-root>/actual/higgs-prompt-session-smoke-<label>.txt
  <result-root>/actual/semantic-compare-session-auto-<label>.txt
  <result-root>/actual/higgs-one-step-cuda-gate-<label>.txt
  <result-root>/actual/higgs-qwen3-config-view/
  <result-root>/profiles/higgs-one-step-actual-auto-<label>.nsys-rep when --profile is set
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
model_dir=""
golden="$repo_root/test_data/higgs-one-step-audio-logits.safetensors"
result_root="/data/results/pegainfer/higgs-audio"
label="$(git -C "$repo_root" rev-parse --short HEAD)"
sm="89"
nvcc_jobs="8"
profile=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir)
      model_dir="${2:?missing value for --model-dir}"
      shift 2
      ;;
    --golden)
      golden="${2:?missing value for --golden}"
      shift 2
      ;;
    --result-root)
      result_root="${2:?missing value for --result-root}"
      shift 2
      ;;
    --label)
      label="${2:?missing value for --label}"
      shift 2
      ;;
    --sm)
      sm="${2:?missing value for --sm}"
      shift 2
      ;;
    --nvcc-jobs)
      nvcc_jobs="${2:?missing value for --nvcc-jobs}"
      shift 2
      ;;
    --profile)
      profile=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$model_dir" ]]; then
  echo "--model-dir is required" >&2
  usage >&2
  exit 2
fi

if [[ ! -d "$model_dir" ]]; then
  echo "model dir not found: $model_dir" >&2
  exit 1
fi

if [[ ! -f "$golden" ]]; then
  echo "golden fixture not found: $golden" >&2
  exit 1
fi

require_nonempty_file() {
  local path="$1"
  if [[ ! -s "$path" ]]; then
    echo "required output missing or empty: $path" >&2
    exit 1
  fi
}

actual_dir="$result_root/actual"
profile_dir="$result_root/profiles"
actual="$actual_dir/higgs-one-step-actual-cuda-bf16-auto-$label.safetensors"
session_actual="$actual_dir/higgs-one-step-session-cuda-bf16-auto-$label.safetensors"
compare_log="$actual_dir/semantic-compare-auto-$label.txt"
session_smoke_log="$actual_dir/higgs-prompt-session-smoke-$label.txt"
session_compare_log="$actual_dir/semantic-compare-session-auto-$label.txt"
gate_summary="$actual_dir/higgs-one-step-cuda-gate-$label.txt"
auto_view="$actual_dir/higgs-qwen3-config-view"

mkdir -p "$actual_dir" "$profile_dir"
rm -rf "$auto_view"

export PEGAINFER_CUDA_SM="$sm"
export PEGAINFER_NVCC_JOBS="$nvcc_jobs"

echo "==> Higgs one-step CUDA gate"
echo "repo:        $repo_root"
echo "commit:      $(git -C "$repo_root" rev-parse --short HEAD)"
echo "model_dir:   $model_dir"
echo "golden:      $golden"
echo "actual:      $actual"
echo "session:     $session_actual"
echo "compare_log: $compare_log"
echo "smoke_log:   $session_smoke_log"
echo "session_log: $session_compare_log"
echo "summary:     $gate_summary"
echo "sm:          $PEGAINFER_CUDA_SM"
echo "nvcc_jobs:   $PEGAINFER_NVCC_JOBS"

cd "$repo_root"

echo "==> Checking runtime-qwen3 bins"
cargo check -p pegainfer-higgs-audio --features runtime-qwen3 --bins

echo "==> Dumping CUDA BF16 one-step actual"
cargo run --release -p pegainfer-higgs-audio --features runtime-qwen3 \
  --bin higgs_dump_one_step_actual -- \
  --model-dir "$model_dir" \
  --golden "$golden" \
  --out "$actual"
require_nonempty_file "$actual"

echo "==> Running semantic comparison"
cargo run --release -p pegainfer-higgs-audio --bin higgs_compare_one_step -- \
  --mode semantic \
  --golden "$golden" \
  --actual "$actual" | tee "$compare_log"
grep -q "higgs one-step semantic comparison: ok" "$compare_log"
require_nonempty_file "$compare_log"

echo "==> Smoke-testing retained prompt session"
cargo run --release -p pegainfer-higgs-audio --features runtime-qwen3 \
  --bin higgs_prefill_prompt_session_smoke -- \
  --model-dir "$model_dir" \
  --golden "$golden" \
  --out "$session_actual" | tee "$session_smoke_log"
grep -q "duplicate_request_id_guard: ok" "$session_smoke_log"
require_nonempty_file "$session_actual"
require_nonempty_file "$session_smoke_log"

echo "==> Running session semantic comparison"
cargo run --release -p pegainfer-higgs-audio --bin higgs_compare_one_step -- \
  --mode semantic \
  --golden "$golden" \
  --actual "$session_actual" | tee "$session_compare_log"
grep -q "higgs one-step semantic comparison: ok" "$session_compare_log"
require_nonempty_file "$session_compare_log"
require_nonempty_file "$auto_view/config.json"
require_nonempty_file "$auto_view/generation_config.json"
require_nonempty_file "$auto_view/higgs-qwen3-tensor-aliases.json"

echo "==> Auto config view"
find "$auto_view" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort

cat >"$gate_summary" <<SUMMARY
status=ok
repo=$repo_root
commit=$(git -C "$repo_root" rev-parse --short HEAD)
label=$label
model_dir=$model_dir
golden=$golden
sm=$PEGAINFER_CUDA_SM
nvcc_jobs=$PEGAINFER_NVCC_JOBS
actual=$actual
session_actual=$session_actual
compare_log=$compare_log
session_smoke_log=$session_smoke_log
session_compare_log=$session_compare_log
auto_view=$auto_view
semantic_comparison=ok
session_semantic_comparison=ok
duplicate_request_id_guard=ok
artifacts_nonempty=ok
SUMMARY
require_nonempty_file "$gate_summary"
echo "==> Gate summary"
cat "$gate_summary"
python3 "$repo_root/tools/higgs/check_higgs_gate_summary.py" "$gate_summary" \
  --expected-label "$label" \
  --expected-sm "$PEGAINFER_CUDA_SM" \
  --expected-nvcc-jobs "$PEGAINFER_NVCC_JOBS" \
  --expected-model-dir "$model_dir" \
  --expected-golden "$golden" \
  --check-files

if [[ "$profile" -eq 1 ]]; then
  if ! command -v nsys >/dev/null 2>&1; then
    echo "nsys not found; cannot capture profile" >&2
    exit 1
  fi

  profiled_actual="$actual_dir/higgs-one-step-actual-cuda-bf16-auto-$label-profiled.safetensors"
  profile_base="$profile_dir/higgs-one-step-actual-auto-$label"
  rm -f "$profile_base.nsys-rep" "$profile_base.sqlite"

  echo "==> Capturing NSYS profile"
  nsys profile --trace=cuda,nvtx,cublas --cuda-graph-trace=node \
    --force-overwrite=true -o "$profile_base" \
    cargo run --release -p pegainfer-higgs-audio --features runtime-qwen3 \
      --bin higgs_dump_one_step_actual -- \
      --model-dir "$model_dir" \
      --golden "$golden" \
      --out "$profiled_actual"

  echo "==> NSYS artifacts"
  ls -lh "$profile_base.nsys-rep" "$profiled_actual"

  stats_base="$profile_dir/higgs-one-step-actual-auto-$label-stats"
  if nsys stats --report cuda_gpu_kern_sum --format csv --output "$stats_base" \
    "$profile_base.nsys-rep"; then
    stats_csv="${stats_base}_cuda_gpu_kern_sum.csv"
    if [[ -f "$stats_csv" ]]; then
      echo "==> Top CUDA kernels"
      head -20 "$stats_csv"
    fi
  else
    echo "nsys stats failed; raw report is still available at $profile_base.nsys-rep" >&2
  fi
fi

echo "==> Higgs one-step CUDA gate: ok"
