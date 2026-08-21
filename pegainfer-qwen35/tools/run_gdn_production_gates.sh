#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "required environment variable is missing: $name" >&2
    exit 2
  fi
}

require_env PEGAINFER_QWEN35_GDN_AOT_BUNDLE
require_env PEGAINFER_TEST_MODEL_PATH
require_env PEGAINFER_TEST_MODEL_REVISION
require_env PEGAINFER_TRITON_PYTHON
require_env PEGAINFER_CUDA_SM
require_env CARGO_TARGET_DIR

expected_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
expected_config_sha="ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670"
bundle="$PEGAINFER_QWEN35_GDN_AOT_BUNDLE"
model="$PEGAINFER_TEST_MODEL_PATH"
python="$PEGAINFER_TRITON_PYTHON"
log_root="${PEGAINFER_GDN_GATE_LOG_DIR:-$repo_root/target/gdn-production-gate-logs}"

mkdir -p "$log_root"
echo "GDN gate log root: $log_root"

for command in \
  git nvidia-smi nvcc rustc cargo protoc cc c++ clang cmake ninja pkg-config \
  rg sha256sum awk sed tee timeout; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "required GDN gate command is missing: $command" >&2
    exit 2
  fi
done

if [[ "$PEGAINFER_CUDA_SM" != "120" ]]; then
  echo "GDN production gates require PEGAINFER_CUDA_SM=120, got $PEGAINFER_CUDA_SM" >&2
  exit 2
fi

if [[ -n "${PEGAINFER_GDN_EXPECT_BRANCH:-}" ]]; then
  actual_branch="$(git branch --show-current)"
  if [[ "$actual_branch" != "$PEGAINFER_GDN_EXPECT_BRANCH" ]]; then
    echo "GDN gate branch mismatch: expected $PEGAINFER_GDN_EXPECT_BRANCH, got $actual_branch" >&2
    exit 2
  fi
fi

if [[ -n "$(git status --short --untracked-files=no)" ]]; then
  echo "GDN production gates require a clean tracked working tree" >&2
  exit 2
fi

gpu_compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed -n '1p' | tr -d '[:space:]')"
if [[ "$gpu_compute_cap" != "12.0" ]]; then
  echo "GDN production gates require compute capability 12.0, got $gpu_compute_cap" >&2
  exit 2
fi

if [[ "$PEGAINFER_TEST_MODEL_REVISION" != "$expected_revision" ]]; then
  echo "model revision mismatch: expected $expected_revision, got $PEGAINFER_TEST_MODEL_REVISION" >&2
  exit 2
fi

for required in \
  "$model/config.json" \
  "$bundle/manifest.json" \
  "$bundle/kernel.o" \
  "$python"; do
  if [[ ! -f "$required" ]]; then
    echo "required GDN gate input is missing: $required" >&2
    exit 2
  fi
done
if [[ ! -x "$python" ]]; then
  echo "GDN gate Python is not executable: $python" >&2
  exit 2
fi

actual_config_sha="$(sha256sum "$model/config.json" | awk '{print $1}')"
if [[ "$actual_config_sha" != "$expected_config_sha" ]]; then
  echo "model config SHA mismatch: expected $expected_config_sha, got $actual_config_sha" >&2
  exit 2
fi

"$python" pegainfer-kernels/tools/flashinfer_gdn/artifact_contract.py \
  validate-bundle "$(dirname "$bundle")" \
  --flashinfer-dir pegainfer-kernels/third_party/flashinfer

commit_sha="$(git rev-parse HEAD)"
submodule_sha="$(git -C pegainfer-kernels/third_party/flashinfer rev-parse HEAD)"
manifest_sha="$(sha256sum "$bundle/manifest.json" | awk '{print $1}')"
object_sha="$(sha256sum "$bundle/kernel.o" | awk '{print $1}')"
{
  echo "commit_sha=$commit_sha"
  echo "branch=$(git branch --show-current)"
  echo "expected_branch=${PEGAINFER_GDN_EXPECT_BRANCH:-not-enforced}"
  echo "flashinfer_submodule_sha=$submodule_sha"
  echo "model_revision=$PEGAINFER_TEST_MODEL_REVISION"
  echo "model_config_sha256=$actual_config_sha"
  echo "manifest_sha256=$manifest_sha"
  echo "object_sha256=$object_sha"
  echo "gpu_compute_cap=$gpu_compute_cap"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  nvcc --version
  rustc --version
  cargo --version
  protoc --version
  clang --version | sed -n '1p'
  cmake --version | sed -n '1p'
  ninja --version
  "$python" --version
} | tee "$log_root/provenance.log"

timeout 90m cargo build --release --locked \
  -p pegainfer-server \
  --no-default-features \
  --features qwen35 \
  --bin pegainfer 2>&1 | tee "$log_root/production-build.log"

timeout 60m cargo test --release --locked \
  -p pegainfer-kernels --features qwen35 --lib --no-run \
  2>&1 | tee "$log_root/kernels-tests-build.log"
timeout 60m cargo test --release --locked \
  -p pegainfer-qwen35 --features qwen35 --lib --tests --no-run \
  2>&1 | tee "$log_root/qwen35-default-tests-build.log"
timeout 60m cargo test --release --locked \
  -p pegainfer-qwen35 --features qwen35,gdn-validation --lib --tests --no-run \
  2>&1 | tee "$log_root/qwen35-validation-tests-build.log"

run_exact_gate() {
  local label="$1"
  local exact_name="$2"
  shift 2
  local list_log="$log_root/$label-list.log"
  local run_log="$log_root/$label.log"

  timeout 60m cargo test --release --locked "$@" "$exact_name" \
    -- --ignored --exact --list >"$list_log" 2>&1
  local listed
  listed="$(rg -c "^${exact_name}: test$" "$list_log" || true)"
  if [[ "$listed" != "1" ]]; then
    echo "$label exact filter matched $listed tests, expected 1" >&2
    sed -n '1,160p' "$list_log" >&2
    exit 3
  fi

  timeout 60m cargo test --release --locked "$@" "$exact_name" \
    -- --ignored --exact --nocapture 2>&1 | tee "$run_log"
  local passed
  passed="$(rg -c "^test ${exact_name} \.\.\. ok$" "$run_log" || true)"
  if [[ "$passed" != "1" ]]; then
    echo "$label executed-pass count was $passed, expected 1" >&2
    exit 3
  fi
}

run_exact_gate \
  gate1-real-aot-boundary \
  ops::qwen35::tests::sm120_stable_abi_alias_and_separate_state_are_bitwise_identical \
  -p pegainfer-kernels --features qwen35 --lib

run_exact_gate \
  gate2-native-prepare-cpu-oracle \
  recurrent::tests::native_prepare_hv32_dynamic_t_and_non_finite_inputs \
  -p pegainfer-qwen35 --features qwen35,gdn-validation --lib

run_exact_gate \
  gate3-production-hf-golden \
  production_flashinfer_gdn_matches_hf_short_golden \
  -p pegainfer-qwen35 --features qwen35,gdn-validation --test hf_golden_gate

run_exact_gate \
  gate4-chunk-continuation \
  prefill::tests::flashinfer_gdn_chunk_continuation_and_model_outputs_match \
  -p pegainfer-qwen35 --features qwen35,gdn-validation --lib

run_exact_gate \
  gate5-scheduler-cuda-graph \
  test_e2e_qwen35_scheduler_flashinfer_gdn \
  -p pegainfer-qwen35 --features qwen35,gdn-validation --test e2e_scheduler

echo "all five Qwen3.5 GDN production gates passed for $commit_sha object $object_sha"
