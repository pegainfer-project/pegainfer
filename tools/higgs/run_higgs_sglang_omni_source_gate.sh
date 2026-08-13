#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run the Higgs Audio SGLang-Omni source-reference golden gate.

This gate imports SGLang-Omni's Higgs tokenizer/head modules directly from a
source tree, regenerates the one-step reference safetensors, and compares that
file against the committed Higgs golden fixture.

Required:
  --model-dir DIR          Higgs checkpoint directory.
  --sglang-omni-src DIR    SGLang-Omni source tree containing sglang_omni/.

Optional:
  --golden FILE            Golden safetensors fixture.
                           Default: test_data/higgs-one-step-audio-logits.safetensors
  --result-root DIR        Result directory.
                           Default: /data/results/pegainfer/higgs-audio
  --label LABEL            Output label. Default: current git short SHA.
  --device DEVICE          Torch device for reference generation. Default: cuda:0
  -h, --help               Show this help.

Outputs:
  <result-root>/actual/higgs-one-step-sglang-omni-src-reference-<label>.safetensors
  <result-root>/actual/sglang-omni-src-reference-compare-<label>.txt
  <result-root>/actual/sglang-omni-import-readiness-<label>.txt
  <result-root>/actual/higgs-sglang-omni-source-gate-<label>.txt
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
model_dir=""
sglang_omni_src=""
golden="$repo_root/test_data/higgs-one-step-audio-logits.safetensors"
result_root="/data/results/pegainfer/higgs-audio"
label="$(git -C "$repo_root" rev-parse --short HEAD)"
device="cuda:0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir)
      model_dir="${2:?missing value for --model-dir}"
      shift 2
      ;;
    --sglang-omni-src)
      sglang_omni_src="${2:?missing value for --sglang-omni-src}"
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
    --device)
      device="${2:?missing value for --device}"
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

if [[ -z "$model_dir" || -z "$sglang_omni_src" ]]; then
  echo "--model-dir and --sglang-omni-src are required" >&2
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

if [[ ! -f "$sglang_omni_src/sglang_omni/models/higgs_tts/text_tokenizer.py" ]]; then
  echo "SGLang-Omni Higgs tokenizer not found under: $sglang_omni_src" >&2
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
reference="$actual_dir/higgs-one-step-sglang-omni-src-reference-$label.safetensors"
compare_log="$actual_dir/sglang-omni-src-reference-compare-$label.txt"
readiness_log="$actual_dir/sglang-omni-import-readiness-$label.txt"
gate_summary="$actual_dir/higgs-sglang-omni-source-gate-$label.txt"

mkdir -p "$actual_dir"
cd "$repo_root"

echo "==> Higgs SGLang-Omni source-reference gate"
echo "repo:             $repo_root"
echo "commit:           $(git -C "$repo_root" rev-parse --short HEAD)"
echo "model_dir:        $model_dir"
echo "sglang_omni_src:  $sglang_omni_src"
echo "golden:           $golden"
echo "reference:        $reference"
echo "compare_log:      $compare_log"
echo "readiness_log:    $readiness_log"
echo "summary:          $gate_summary"
echo "device:           $device"

python3 -m py_compile tools/accuracy/dump_higgs_one_step_golden.py
python3 -m py_compile tools/higgs/check_higgs_sglang_omni_imports.py
python3 -m py_compile tools/higgs/check_higgs_sglang_omni_source_gate_summary.py

python3 tools/higgs/check_higgs_sglang_omni_imports.py \
  --sglang-omni-src "$sglang_omni_src" \
  --require-direct | tee "$readiness_log"
grep -q "^direct_higgs_imports=ok$" "$readiness_log"
require_nonempty_file "$readiness_log"

python3 tools/accuracy/dump_higgs_one_step_golden.py \
  --snapshot-dir "$model_dir" \
  --sglang-omni-src "$sglang_omni_src" \
  --require-sglang-omni-source \
  --out "$reference" \
  --device "$device"
require_nonempty_file "$reference"

cargo run --release -p pegainfer-higgs-audio --bin higgs_compare_one_step -- \
  --golden "$golden" \
  --actual "$reference" | tee "$compare_log"
grep -q "higgs one-step strict comparison: ok" "$compare_log"
require_nonempty_file "$compare_log"

metadata_tmp="$(mktemp)"
python3 - <<'PY' "$reference" >"$metadata_tmp"
import sys
from safetensors import safe_open

path = sys.argv[1]
with safe_open(path, framework="pt", device="cpu") as f:
    md = f.metadata()

required = {
    "sglang_omni_source_dir": None,
    "sglang_omni_source_commit": None,
    "sglang_omni_direct_imports": "text_tokenizer.py;modeling.py",
    "sglang_omni_full_model_imported": "false",
}
for key, expected in required.items():
    value = md.get(key)
    if not value:
        raise SystemExit(f"missing metadata {key}")
    if expected is not None and value != expected:
        raise SystemExit(f"metadata {key}={value!r}, expected {expected!r}")
    print(f"{key}={value}")
PY
cat "$metadata_tmp"

sglang_omni_commit="$(grep '^sglang_omni_source_commit=' "$metadata_tmp" | cut -d= -f2-)"
full_model_import="$(grep '^full_higgs_model_import=' "$readiness_log" | cut -d= -f2-)"
cat >"$gate_summary" <<SUMMARY
status=ok
repo=$repo_root
commit=$(git -C "$repo_root" rev-parse --short HEAD)
label=$label
model_dir=$model_dir
sglang_omni_src=$sglang_omni_src
sglang_omni_commit=$sglang_omni_commit
golden=$golden
reference=$reference
compare_log=$compare_log
readiness_log=$readiness_log
sglang_omni_direct_imports=ok
sglang_omni_full_model_import=$full_model_import
source_reference_strict_comparison=ok
artifacts_nonempty=ok
SUMMARY
rm -f "$metadata_tmp"
require_nonempty_file "$gate_summary"

echo "==> Gate summary"
cat "$gate_summary"
python3 "$repo_root/tools/higgs/check_higgs_sglang_omni_source_gate_summary.py" "$gate_summary" \
  --expected-label "$label" \
  --expected-model-dir "$model_dir" \
  --expected-sglang-omni-src "$sglang_omni_src" \
  --expected-sglang-omni-commit "$sglang_omni_commit" \
  --expected-golden "$golden" \
  --check-files
echo "==> Higgs SGLang-Omni source-reference gate: ok"
