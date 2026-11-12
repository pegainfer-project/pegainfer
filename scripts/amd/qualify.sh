#!/usr/bin/env bash
# Run inside the locked ROCm10 environment. Logs stay outside the source tree.
set -euo pipefail
output=${1:-"$PWD/../artifacts/amd/qualification/$(date -u +%Y%m%dT%H%M%SZ)"}
mkdir -p "$output"
output=$(cd "$output" && pwd)
[[ ! -e "$output/passed-checks.txt" ]] || { echo 'Choose a fresh qualification output directory' >&2; exit 1; }
export PEGAINFER_QUALIFICATION_DIR="$output"
run_check() {
  local name=$1
  shift
  printf '%s\n' "Running $name"
  if ! "$@" > "$output/$name.log" 2>&1; then
    tail -n 70 "$output/$name.log"
    return 1
  fi
  printf '%s\n' "$name" >> "$output/passed-checks.txt"
}
run_check build cargo build --locked --release -p pegainfer-server --no-default-features --features rocm,qwen3
run_check device cargo test --locked --release -p pegainfer-device --features rocm --lib -- --ignored --test-threads=1
run_check native bash scripts/amd/check_native.sh
run_check host cargo test --locked --release -p pegainfer-core -p pegainfer-qwen3 -p pegainfer-frontend --no-default-features --features pegainfer-core/rocm,pegainfer-qwen3/rocm --lib
run_check server cargo test --locked --release -p pegainfer-server --no-default-features --features rocm,qwen3 --bin pegainfer
run_check cli python scripts/amd/check_cli.py
for size in 0.6B 4B; do
  run_check "golden-$size" env "PEGAINFER_TEST_MODEL_PATH=$PWD/../models/Qwen3-$size" \
    cargo test --locked --release -p pegainfer-qwen3 --no-default-features --features rocm --test rocm_golden -- --ignored --nocapture --test-threads=1
  context=512
  if [[ "$size" == 4B ]]; then context=4096; fi
  run_check "http-$size" python scripts/amd/check_http.py --model-path "../models/Qwen3-$size" --context "$context" --output "$output/http-$size"
done
run_check fmt cargo fmt --all --check
run_check clippy cargo clippy --locked --release -p pegainfer-server -p pegainfer-device --no-default-features --features pegainfer-server/rocm,pegainfer-server/qwen3,pegainfer-device/rocm
python scripts/amd/record_qualification.py
printf '%s\n' "Qualification passed: $output"
