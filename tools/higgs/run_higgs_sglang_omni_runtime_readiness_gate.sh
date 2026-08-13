#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run the Higgs Audio SGLang-Omni full-runtime readiness gate.

This is a fail-closed precondition for claiming SGLang-Omni runtime parity. It
does not compare tensors by itself; it proves that the local Python environment
can import both the direct Higgs source modules and the full Higgs model module.

Required:
  --sglang-omni-src DIR    SGLang-Omni source tree containing sglang_omni/.

Optional:
  --python PYTHON          Python executable. Default: python3
  --result-root DIR        Result directory.
                           Default: /data/results/pegainfer/higgs-audio
  --label LABEL            Output label. Default: current git short SHA.
  -h, --help               Show this help.

Outputs:
  <result-root>/actual/sglang-omni-runtime-readiness-<label>.txt
  <result-root>/actual/higgs-sglang-omni-runtime-readiness-gate-<label>.txt
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
sglang_omni_src=""
python_bin="python3"
result_root="/data/results/pegainfer/higgs-audio"
label="$(git -C "$repo_root" rev-parse --short HEAD)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sglang-omni-src)
      sglang_omni_src="${2:?missing value for --sglang-omni-src}"
      shift 2
      ;;
    --python)
      python_bin="${2:?missing value for --python}"
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

if [[ -z "$sglang_omni_src" ]]; then
  echo "--sglang-omni-src is required" >&2
  usage >&2
  exit 2
fi

if [[ ! -f "$sglang_omni_src/sglang_omni/models/higgs_tts/text_tokenizer.py" ]]; then
  echo "SGLang-Omni Higgs tokenizer not found under: $sglang_omni_src" >&2
  exit 1
fi

if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "python executable not found: $python_bin" >&2
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
readiness_log="$actual_dir/sglang-omni-runtime-readiness-$label.txt"
gate_summary="$actual_dir/higgs-sglang-omni-runtime-readiness-gate-$label.txt"

mkdir -p "$actual_dir"
cd "$repo_root"

echo "==> Higgs SGLang-Omni full-runtime readiness gate"
echo "repo:             $repo_root"
echo "commit:           $(git -C "$repo_root" rev-parse --short HEAD)"
echo "python:           $python_bin"
echo "sglang_omni_src:  $sglang_omni_src"
echo "readiness_log:    $readiness_log"
echo "summary:          $gate_summary"

"$python_bin" -m py_compile tools/higgs/check_higgs_sglang_omni_imports.py
"$python_bin" -m py_compile tools/higgs/check_higgs_sglang_omni_runtime_readiness_summary.py

"$python_bin" tools/higgs/check_higgs_sglang_omni_imports.py \
  --sglang-omni-src "$sglang_omni_src" | tee "$readiness_log"
require_nonempty_file "$readiness_log"

sglang_omni_commit="$(grep '^sglang_omni_commit=' "$readiness_log" | cut -d= -f2-)"
python_version="$(grep '^python.version=' "$readiness_log" | cut -d= -f2-)"
pyproject_torch="$(grep '^pyproject.dependency.torch=' "$readiness_log" | cut -d= -f2-)"
pyproject_sglang="$(grep '^pyproject.dependency.sglang=' "$readiness_log" | cut -d= -f2-)"
torch_version="$(grep '^package.torch.version=' "$readiness_log" | cut -d= -f2-)"
torch_cuda="$(grep '^package.torch.cuda=' "$readiness_log" | cut -d= -f2-)"
torch_has_cuda_pool_api="$(grep '^package.torch.has_cuda_begin_allocate_current_thread_to_pool=' "$readiness_log" | cut -d= -f2-)"
sglang_version="$(grep '^package.sglang.version=' "$readiness_log" | cut -d= -f2-)"
transformers_version="$(grep '^package.transformers.version=' "$readiness_log" | cut -d= -f2-)"
direct_imports="$(grep '^direct_higgs_imports=' "$readiness_log" | cut -d= -f2-)"
full_model_import="$(grep '^full_higgs_model_import=' "$readiness_log" | cut -d= -f2-)"
status="fail"
runtime_ready="fail"
if [[ "$direct_imports" == "ok" && "$full_model_import" == "ok" ]]; then
  status="ok"
  runtime_ready="ok"
fi

cat >"$gate_summary" <<SUMMARY
status=$status
repo=$repo_root
commit=$(git -C "$repo_root" rev-parse --short HEAD)
label=$label
python=$python_bin
python_version=$python_version
sglang_omni_src=$sglang_omni_src
sglang_omni_commit=$sglang_omni_commit
readiness_log=$readiness_log
pyproject_torch=$pyproject_torch
pyproject_sglang=$pyproject_sglang
torch_version=$torch_version
torch_cuda=$torch_cuda
torch_has_cuda_pool_api=$torch_has_cuda_pool_api
sglang_version=$sglang_version
transformers_version=$transformers_version
sglang_omni_direct_imports=$direct_imports
sglang_omni_full_model_import=$full_model_import
runtime_ready=$runtime_ready
artifacts_nonempty=ok
SUMMARY
require_nonempty_file "$gate_summary"

echo "==> Gate summary"
cat "$gate_summary"
"$python_bin" "$repo_root/tools/higgs/check_higgs_sglang_omni_runtime_readiness_summary.py" \
  "$gate_summary" \
  --expected-status "$status" \
  --expected-label "$label" \
  --expected-python "$python_bin" \
  --expected-sglang-omni-src "$sglang_omni_src" \
  --expected-sglang-omni-commit "$sglang_omni_commit" \
  --check-files

if [[ "$status" != "ok" ]]; then
  echo "==> Higgs SGLang-Omni full-runtime readiness gate: fail" >&2
  echo "full_higgs_model_import=$full_model_import" >&2
  exit 1
fi

echo "==> Higgs SGLang-Omni full-runtime readiness gate: ok"
