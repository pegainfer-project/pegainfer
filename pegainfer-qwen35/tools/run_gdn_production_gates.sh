#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

require_env() {
  if [[ -z "${!1:-}" ]]; then
    echo "required environment variable is missing: $1" >&2
    exit 2
  fi
}
for name in PEGAINFER_QWEN35_GDN_AOT_BUNDLE PEGAINFER_TEST_MODEL_PATH \
  PEGAINFER_TEST_MODEL_REVISION PEGAINFER_TRITON_PYTHON PEGAINFER_GDN_AOT_PYTHON \
  PEGAINFER_GDN_FLASHINFER_DIR PEGAINFER_CUDA_SM CARGO_TARGET_DIR; do
  require_env "$name"
done
for command in git nvidia-smi nvcc rustc cargo protoc cc c++ clang cmake ninja \
  pkg-config rg sha256sum awk sed tee timeout realpath tr mktemp cp; do
  command -v "$command" >/dev/null || { echo "missing command: $command" >&2; exit 2; }
done
if [[ -n "${PEGAINFER_GDN_NSYS_REPORT:-}" ]]; then
  command -v nsys >/dev/null || { echo "requested Nsight capture needs nsys" >&2; exit 2; }
fi

bundle="$(realpath "$PEGAINFER_QWEN35_GDN_AOT_BUNDLE")"
gdn_flashinfer="$(realpath "$PEGAINFER_GDN_FLASHINFER_DIR")"
model="$(realpath "$PEGAINFER_TEST_MODEL_PATH")"
python="$PEGAINFER_TRITON_PYTHON"
aot_python="$PEGAINFER_GDN_AOT_PYTHON"
target_root="$(realpath -m "$CARGO_TARGET_DIR")"
log_root="$(realpath -m "${PEGAINFER_GDN_GATE_LOG_DIR:-$repo_root/target/gdn-production-gate-logs}")"
expected_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
expected_config_sha="ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670"

[[ "$PEGAINFER_CUDA_SM" == 120 ]] || { echo "gates require SM120" >&2; exit 2; }
[[ "$PEGAINFER_TEST_MODEL_REVISION" == "$expected_revision" ]] || {
  echo "model revision mismatch" >&2; exit 2;
}
[[ -z "$(git status --short --untracked-files=no)" ]] || {
  echo "gates require a clean tracked tree" >&2; exit 2;
}
if [[ -n "${PEGAINFER_GDN_EXPECT_BRANCH:-}" ]]; then
  [[ "$(git branch --show-current)" == "$PEGAINFER_GDN_EXPECT_BRANCH" ]] || {
    echo "gate branch mismatch" >&2; exit 2;
  }
fi
[[ ! -e "$target_root/qwen3" && ! -e "$target_root/stock" && ! -e "$target_root/candidate" ]] || {
  echo "qwen3, stock and candidate targets must be absent before acceptance: $target_root" >&2; exit 2;
}
for input in "$model/config.json" "$bundle/manifest.json" "$bundle/kernel.o" \
  "$python" "$aot_python"; do
  [[ -f "$input" ]] || { echo "missing input: $input" >&2; exit 2; }
done
[[ -x "$python" && -x "$aot_python" ]] || { echo "Python inputs must be executable" >&2; exit 2; }
generation_flashinfer_sha="$("$aot_python" - "$gdn_flashinfer" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, "pegainfer-kernels/tools/flashinfer_gdn")
from artifact_contract import verify_flashinfer_base
print(verify_flashinfer_base(Path(sys.argv[1])))
PY
)"
gpu_compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed -n '1p' | tr -d '[:space:]')"
[[ "$gpu_compute_cap" == 12.0 ]] || { echo "real SM120 GPU required" >&2; exit 2; }
config_sha="$(sha256sum "$model/config.json" | awk '{print $1}')"
[[ "$config_sha" == "$expected_config_sha" ]] || { echo "model config SHA mismatch" >&2; exit 2; }
mkdir -p "$log_root" "$target_root"
export PEGAINFER_TEST_MODEL_PATH="$model"
export PEGAINFER_TEST_QWEN35_GDN_BACKEND=flashinfer-candidate
export PEGAINFER_QWEN35_GDN_LAYOUT_REFERENCE="$log_root/layout-reference"
# The committed oracle must be used; per-developer fixture overrides are not acceptance inputs.
unset PEGAINFER_QWEN35_HF_GOLDEN PEGAINFER_QWEN35_HF_LONG_GOLDEN

commit_sha="$(git rev-parse HEAD)"
tree_sha="$(git rev-parse HEAD^{tree})"
object_sha="$(sha256sum "$bundle/kernel.o" | awk '{print $1}')"
{
  echo "commit_sha=$commit_sha"
  echo "tree_sha=$tree_sha"
  echo "branch=$(git branch --show-current)"
  echo "flashinfer_submodule_sha=$(git -C pegainfer-kernels/third_party/flashinfer rev-parse HEAD)"
  echo "gdn_generation_flashinfer_sha=$generation_flashinfer_sha"
  echo "model_revision=$expected_revision"
  echo "model_config_sha256=$config_sha"
  echo "candidate_object_sha256=$object_sha"
  sha256sum "$model"/*.safetensors "$model/tokenizer.json"
  if [[ -f "$model/model.safetensors.index.json" ]]; then
    sha256sum "$model/model.safetensors.index.json"
  fi
  sha256sum "$bundle/manifest.json" pegainfer-kernels/tools/flashinfer_gdn/source-lock.json
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  nvcc --version
  rustc --version
  cargo --version
  protoc --version
  "$python" --version
  "$aot_python" --version
} | tee "$log_root/provenance.log"

server_pid=""
stop_server() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=""
  fi
}
trap stop_server EXIT

http_request() {
  local label="$1" binary="$2" expected="$3"
  shift 3
  local max_prefill_tokens=8192 prompt_tokens=0
  if [[ "$expected" == flashinfer-candidate ]]; then
    max_prefill_tokens=20000
    prompt_tokens=20000
  fi
  local port
  port="$("$python" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"
  timeout --kill-after=10s 5m "$binary" --model-path "$model" --port "$port" --max-batch 4 \
    --max-prefill-tokens "$max_prefill_tokens" --decode-overlap stream "$@" \
    >"$log_root/$label-server.log" 2>&1 &
  server_pid=$!
  "$python" - "$port" "$server_pid" "$prompt_tokens" <<'PY' | tee "$log_root/$label-request.log"
import json, os, sys, time, urllib.error, urllib.request
port, pid, expected_prompt_tokens = map(int, sys.argv[1:])
base = f"http://127.0.0.1:{port}"
deadline = time.monotonic() + 180
while True:
    os.kill(pid, 0)
    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=5) as response:
            model = json.load(response)["data"][0]["id"]
        break
    except (urllib.error.URLError, TimeoutError):
        if time.monotonic() >= deadline:
            raise RuntimeError("production model did not become ready")
        time.sleep(0.25)
prompt = ([100 + (index * 17 % 1000) for index in range(expected_prompt_tokens)]
          if expected_prompt_tokens else
          "Explain why a paged KV cache preserves earlier tokens during resumed prefill.")
payload = {"model": model, "prompt": prompt,
           "max_tokens": 8, "temperature": 0, "ignore_eos": True}
request = urllib.request.Request(base + "/v1/completions", json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
with urllib.request.urlopen(request, timeout=60) as response:
    result = json.load(response)
if expected_prompt_tokens:
    assert result["usage"]["prompt_tokens"] == expected_prompt_tokens, result
assert result["usage"]["completion_tokens"] == 8, result
assert result["choices"][0]["finish_reason"] == "length", result
assert result["choices"][0]["text"].strip(), result
print(json.dumps(result, sort_keys=True))
PY
  stop_server
  rg -F "Qwen3.5 GDN: requested=$expected resolved=$expected" "$log_root/$label-server.log"
  if [[ "$expected" == flashinfer-candidate ]]; then
    rg -F "object_sha256=$object_sha" "$log_root/$label-server.log"
  fi
}

# Distinct empty build directories prove that an environment cache cannot supply a bundle to stock.
env -u PEGAINFER_QWEN35_GDN_AOT_BUNDLE -u PEGAINFER_TRITON_PYTHON \
  CARGO_TARGET_DIR="$target_root/qwen3" timeout 90m cargo build \
  --release --locked -p pegainfer-server --bin pegainfer \
  2>&1 | tee "$log_root/default-qwen3-build.log"
env -u PEGAINFER_QWEN35_GDN_AOT_BUNDLE -u PEGAINFER_TRITON_PYTHON \
  CARGO_TARGET_DIR="$target_root/qwen3" timeout 90m cargo clippy \
  --release --locked -p pegainfer-server --bin pegainfer -- -D warnings \
  2>&1 | tee "$log_root/default-qwen3-clippy.log"
env -u PEGAINFER_QWEN35_GDN_AOT_BUNDLE CARGO_TARGET_DIR="$target_root/stock" \
  timeout 90m cargo build --release --locked -p pegainfer-server \
  --no-default-features --features qwen35 --bin pegainfer \
  2>&1 | tee "$log_root/gate1-stock-build.log"
http_request gate1-stock-default "$target_root/stock/release/pegainfer" triton

expect_startup_rejection() {
  local label="$1" binary="$2" message="$3"
  shift 3
  local status=0
  timeout 180 "$binary" --model-path "$model" --max-batch 4 \
    --qwen35-gdn-backend flashinfer-candidate "$@" \
    >"$log_root/$label.log" 2>&1 || status=$?
  [[ "$status" -gt 0 && "$status" -lt 124 ]] || {
    echo "$label did not reject the explicit candidate at startup" >&2; exit 3;
  }
  rg -F "$message" "$log_root/$label.log"
}
expect_startup_rejection gate1-missing-candidate "$target_root/stock/release/pegainfer" \
  "no AOT candidate was linked"

"$aot_python" pegainfer-kernels/tools/flashinfer_gdn/layout_reference.py \
  --flashinfer-dir "$gdn_flashinfer" --candidate "$bundle" \
  --output "$PEGAINFER_QWEN35_GDN_LAYOUT_REFERENCE" \
  2>&1 | tee "$log_root/gate1-layout-generation.log"
export CARGO_TARGET_DIR="$target_root/candidate"
export PEGAINFER_QWEN35_GDN_AOT_BUNDLE="$bundle"
timeout 90m cargo build --release --locked -p pegainfer-server \
  --no-default-features --features qwen35 --bin pegainfer \
  2>&1 | tee "$log_root/gate1-candidate-build.log"
sha256sum "$target_root/stock/release/pegainfer" "$CARGO_TARGET_DIR/release/pegainfer" \
  | tee "$log_root/binary-sha256.log"
http_request gate1-linked-default "$CARGO_TARGET_DIR/release/pegainfer" triton
http_request gate1-explicit-candidate "$CARGO_TARGET_DIR/release/pegainfer" flashinfer-candidate \
  --qwen35-gdn-backend flashinfer-candidate
expect_startup_rejection gate1-unsupported-tp "$CARGO_TARGET_DIR/release/pegainfer" \
  "Qwen3.5 --qwen35-gdn-backend=flashinfer-candidate requires TP world_size=1" --tp-size 2

# One real-file mutation exercises the same build.rs boundary; the field matrix is a host test.
corrupt_bundle="$(mktemp -d "$log_root/corrupt-candidate.XXXXXX")"
cp "$bundle/manifest.json" "$bundle/kernel.h" "$bundle/kernel.o" \
  "$bundle/libcuda_dialect_runtime_static.a" "$corrupt_bundle/"
"$python" - "$corrupt_bundle/kernel.o" <<'PY'
from pathlib import Path
import sys
with Path(sys.argv[1]).open("ab") as artifact:
    artifact.write(b"\x00")
PY
set +e
PEGAINFER_QWEN35_GDN_AOT_BUNDLE="$corrupt_bundle" timeout 90m cargo build \
  --release --locked -p pegainfer-server --no-default-features --features qwen35 --bin pegainfer \
  >"$log_root/gate1-corrupt-candidate.log" 2>&1
corrupt_status=$?
set -e
[[ "$corrupt_status" != 0 && "$corrupt_status" != 124 ]] || {
  echo "corrupted candidate was not rejected before link" >&2; exit 3;
}
rg 'kernel\.o.*(size|hash|SHA|digest).*mismatch' "$log_root/gate1-corrupt-candidate.log"

timeout 90m cargo clippy --release --locked -p pegainfer-server \
  -p pegainfer-qwen35 -p pegainfer-kernels --no-default-features --features qwen35 \
  -- -D warnings 2>&1 | tee "$log_root/candidate-clippy.log"
timeout 60m cargo test --release --locked -p pegainfer-kernels --features qwen35 --lib --no-run \
  2>&1 | tee "$log_root/kernels-tests-build.log"
timeout 60m cargo test --release --locked -p pegainfer-qwen35 --features qwen35 \
  --lib --test e2e_scheduler --test chunked_prefill --no-run \
  2>&1 | tee "$log_root/qwen35-tests-build.log"

run_exact_gate() {
  local label="$1" exact_name="$2" mode="$3"
  shift 3
  local extra=()
  local profiler=()
  [[ "$mode" != ignored ]] || extra=(--ignored)
  if [[ "$label" == gate5-shared-sm && -n "${PEGAINFER_GDN_NSYS_REPORT:-}" ]]; then
    profiler=(nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none
      --output "${PEGAINFER_GDN_NSYS_REPORT%.nsys-rep}")
  fi
  timeout 60m cargo test --release --locked "$@" "$exact_name" \
    -- --exact "${extra[@]}" --list >"$log_root/$label-list.log" 2>&1
  [[ "$(rg -Fxc "$exact_name: test" "$log_root/$label-list.log" || true)" == 1 ]] || {
    echo "$label did not list exactly one test" >&2; exit 3;
  }
  timeout 60m "${profiler[@]}" cargo test --release --locked "$@" "$exact_name" \
    -- --exact "${extra[@]}" --nocapture 2>&1 | tee "$log_root/$label.log"
  if rg -i '(^|[[:space:]:])skip(ping|ped)?([[:space:]:]|$)' "$log_root/$label.log"; then
    echo "$label skipped required coverage" >&2; exit 3;
  fi
  [[ "$(rg -c '^test result: ok\. 1 passed; 0 failed; 0 ignored;' "$log_root/$label.log" || true)" == 1 ]] || {
    echo "$label did not execute exactly one passing test" >&2; exit 3;
  }
  if [[ ${#profiler[@]} != 0 ]]; then
    [[ -s "${PEGAINFER_GDN_NSYS_REPORT%.nsys-rep}.nsys-rep" ]] || {
      echo "Shared-SM test passed without the requested Nsight report" >&2; exit 3;
    }
  fi
}

run_exact_gate gate1-candidate-contract qwen35_gdn::tests::candidate_contract_rejects_mutations ordinary \
  -p pegainfer-build --features qwen35-gdn --lib
run_exact_gate gate1-in-place-layout \
  ops::qwen35::tests::sm120_stable_in_place_abi_matches_upstream_layout_reference ignored \
  -p pegainfer-kernels --features qwen35 --lib
run_exact_gate gate2-native-prepare-cpu-oracle \
  recurrent::native_prepare_tests::test_gdn_native_prepare_matches_cpu_reference_on_finite_inputs ignored \
  -p pegainfer-qwen35 --features qwen35 --lib
run_exact_gate gate3-hf-golden \
  executor::hf_golden_gate::pega_logprobs_match_hf_golden_within_qwen35_tolerance ordinary \
  -p pegainfer-qwen35 --features qwen35 --lib
run_exact_gate gate3-hf-long-golden \
  executor::hf_golden_gate::pega_logprobs_match_hf_long_golden_within_qwen35_tolerance ordinary \
  -p pegainfer-qwen35 --features qwen35 --lib
run_exact_gate gate4-model-continuation \
  prefill::tests::flashinfer_gdn_chunk_continuation_and_model_outputs_match ignored \
  -p pegainfer-qwen35 --features qwen35 --lib
run_exact_gate gate4-scheduler-chunks \
  chunked_prefill_matches_unchunked_prefill_for_resumed_paged_kv ordinary \
  -p pegainfer-qwen35 --features qwen35 --test chunked_prefill
run_exact_gate gate5-scheduler test_e2e_qwen35_scheduler ordinary \
  -p pegainfer-qwen35 --features qwen35 --test e2e_scheduler
run_exact_gate gate5-shared-sm test_e2e_qwen35_shared_sm_last_decoder ordinary \
  -p pegainfer-qwen35 --features qwen35 --test e2e_scheduler

[[ "$(git rev-parse HEAD)" == "$commit_sha" && "$(git rev-parse HEAD^{tree})" == "$tree_sha" \
  && -z "$(git status --short --untracked-files=no)" ]] || {
  echo "validated source changed during acceptance" >&2; exit 3;
}
[[ "$(sha256sum "$bundle/kernel.o" | awk '{print $1}')" == "$object_sha" ]] || {
  echo "candidate object changed during acceptance" >&2; exit 3;
}
echo "all five Qwen3.5 GDN functional gate categories passed for $commit_sha tree $tree_sha object $object_sha"
echo "Stage 20 also requires analysis of the Shared-SM Nsight timeline and same-head performance A/B."
