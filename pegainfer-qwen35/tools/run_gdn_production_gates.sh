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
export PEGAINFER_QWEN35_GDN_LAYOUT_REFERENCE="$log_root/layout-reference"
# The committed oracle must be used; per-developer fixture overrides are not acceptance inputs.
unset PEGAINFER_QWEN35_HF_GOLDEN PEGAINFER_QWEN35_HF_LONG_GOLDEN

commit_sha="$(git rev-parse HEAD)"
tree_sha="$(git rev-parse 'HEAD^{tree}')"
object_sha="$(sha256sum "$bundle/kernel.o" | awk '{print $1}')"
export PEGAINFER_TEST_QWEN35_GDN_OBJECT_SHA256="$object_sha"
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
  -p pegainfer-qwen35 -p pegainfer-kernels --all-targets --no-default-features --features qwen35 \
  -- -D warnings 2>&1 | tee "$log_root/candidate-clippy.log"
timeout 60m cargo test --release --locked -p pegainfer-kernels --features qwen35 --lib --no-run \
  2>&1 | tee "$log_root/kernels-tests-build.log"
timeout 60m cargo test --release --locked -p pegainfer-qwen35 --features qwen35 \
  --lib --no-run --message-format=json-render-diagnostics \
  2>&1 | tee "$log_root/qwen35-tests-build.log"

test_executable() {
  "$python" - "$1" "$2" <<'PY'
import json, os, sys
executables = set()
with open(sys.argv[1]) as records:
    for line in records:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (record.get("reason") == "compiler-artifact"
                and record["target"]["name"] == sys.argv[2]
                and record["profile"]["test"] and record.get("executable")):
            executables.add(record["executable"])
assert len(executables) == 1, f"expected one {sys.argv[2]} test executable: {executables}"
executable = executables.pop()
assert os.path.isfile(executable) and os.access(executable, os.X_OK), executable
print(executable)
PY
}
qwen35_test_binary="$(test_executable "$log_root/qwen35-tests-build.log" pegainfer_qwen35)"
sha256sum "$qwen35_test_binary" | tee "$log_root/qwen35-test-binary-sha256.log"

run_exact_gate() {
  local label="$1" exact_name="$2" mode="$3"
  shift 3
  local extra=()
  local profiler=()
  [[ "$mode" != ignored ]] || extra=(--ignored)
  if [[ "$label" == gate5-shared-sm && -n "${PEGAINFER_GDN_NSYS_REPORT:-}" ]]; then
    profiler=(nsys profile "--trace=cuda,nvtx" --cuda-graph-trace=node --sample=none
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
  case "$label" in
    gate3-hf-*|gate4-model-continuation|gate4-scheduler-chunks|gate5-scheduler|gate5-shared-sm)
      rg -F "CANDIDATE_MODEL_IDENTITY_OK object_sha256=$object_sha" "$log_root/$label.log"
      ;;
  esac
}

run_exact_gate gate1-candidate-contract qwen35_gdn::tests::candidate_contract_rejects_mutations ordinary \
  -p pegainfer-build --features qwen35-gdn --lib

# Only this test executable wraps the C identity query; never carry the flag into
# a production binary or invoke cargo test here (it would rebuild without --wrap).
timeout 60m cargo rustc --release --locked -p pegainfer-kernels --features qwen35 \
  --test gdn_identity --message-format=json-render-diagnostics \
  -- -C link-arg=-Wl,--wrap=pegainfer_qwen35_gdn_artifact_sha256 \
  2>&1 | tee "$log_root/gate1-identity-rejections-build.log"
identity_test_binary="$(test_executable "$log_root/gate1-identity-rejections-build.log" gdn_identity)"
identity_test=production_loader_rejects_invalid_artifact_identity
sha256sum "$identity_test_binary" | tee "$log_root/gate1-identity-rejections-binary-sha256.log"
"$identity_test_binary" "$identity_test" --exact --ignored --list \
  >"$log_root/gate1-identity-rejections-list.log" 2>&1
[[ "$(rg -Fxc "$identity_test: test" "$log_root/gate1-identity-rejections-list.log" || true)" == 1 ]]
timeout --kill-after=10s 5m "$identity_test_binary" "$identity_test" \
  --exact --ignored --test-threads=1 --nocapture \
  2>&1 | tee "$log_root/gate1-identity-rejections.log"
[[ "$(rg -c '^test result: ok\. 1 passed; 0 failed; 0 ignored;' "$log_root/gate1-identity-rejections.log" || true)" == 1 ]]
for rejection in null invalid-utf8 short long non-hex; do
  [[ "$(rg -c "(^|[[:space:]])identity_rejection_passed=$rejection$" "$log_root/gate1-identity-rejections.log" || true)" == 1 ]]
done
[[ "$(rg -Fxc 'identity_rejections_passed=5' "$log_root/gate1-identity-rejections.log" || true)" == 1 ]]
echo "gate1 validations passed"
run_exact_gate gate1-in-place-layout \
  ops::qwen35::tests::sm120_stable_in_place_abi_matches_upstream_layout_reference ignored \
  -p pegainfer-kernels --features qwen35 --lib
run_exact_gate gate2-native-prepare-cpu-oracle \
  recurrent::native_prepare_tests::test_gdn_native_prepare_matches_cpu_reference_on_finite_inputs ignored \
  -p pegainfer-qwen35 --features qwen35 --lib

# Reuse the actual candidate HF entry without running its numeric body. All
# admission failures precede inference; the last case loads the real model first.
hf_short=executor::hf_golden_gate::candidate_pega_logprobs_match_hf_golden_within_qwen35_tolerance
"$qwen35_test_binary" "$hf_short" --exact --ignored --list \
  >"$log_root/gate3-acceptance-rejections-list.log" 2>&1
[[ "$(rg -Fxc "$hf_short: test" "$log_root/gate3-acceptance-rejections-list.log" || true)" == 1 ]]
rejections_root="$(mktemp -d "$log_root/acceptance-rejections.XXXXXX")"
expect_acceptance_rejection() {
  local label="$1" message="$2" status=0
  shift 2
  local log="$log_root/gate3-reject-$label.log"
  (ulimit -c 0
    timeout --kill-after=10s 5m env "$@" "$qwen35_test_binary" "$hf_short" \
      --exact --ignored --test-threads=1 --nocapture
  ) >"$log" 2>&1 || status=$?
  "$python" - "$log" "$status" "$hf_short" "$message" <<'PY'
from pathlib import Path
import re, sys
log, status, name, expected = sys.argv[1:]
text = Path(log).read_text()
assert status in ("101", "134"), f"unexpected failure/timeout (exit {status}); retain {log}"
assert re.search(r"(?m)^running 1 test$", text), f"harness did not run one test: {log}"
panic = re.search(r"(?m)^thread '" + re.escape(name)
                  + r"'(?: \(\d+\))? panicked at [^\n]*\n([^\n]*)", text)
assert panic, f"missing the exact test's panic: {log}"
assert expected in panic[1], f"wrong rejection reason; expected {expected!r}; retain {log}"
assert "test result: ok." not in text, f"negative case unexpectedly passed: {log}"
if status == "101":
    assert re.search(r"(?m)^test result: FAILED\. 0 passed; 1 failed; 0 ignored;", text), log
# Some CUDA-linked release test binaries abort while unwinding. Exit 134 alone
# is not evidence: the exact test's panic and its expected reason are mandatory.
print(f"GDN_ACCEPTANCE_REJECTION_OK log={log} exit={status} reason={expected}")
PY
}
expect_acceptance_rejection missing-identity \
  'candidate acceptance requires PEGAINFER_TEST_QWEN35_GDN_OBJECT_SHA256:' \
  -u PEGAINFER_TEST_QWEN35_GDN_OBJECT_SHA256
expect_acceptance_rejection malformed-identity \
  'candidate acceptance requires a 64-character hexadecimal object SHA256' \
  PEGAINFER_TEST_QWEN35_GDN_OBJECT_SHA256=not-a-sha256
expect_acceptance_rejection missing-model \
  'candidate acceptance requires a readable Qwen3.5 model fixture' \
  "PEGAINFER_TEST_MODEL_PATH=$rejections_root/missing-model"
expect_acceptance_rejection missing-fixture \
  "read $rejections_root/missing-short-golden.safetensors: No such file or directory (os error 2)" \
  "PEGAINFER_QWEN35_HF_GOLDEN=$rejections_root/missing-short-golden.safetensors"

# Remove the temporary symlinks even on failure so evidence packers cannot
# follow them and accidentally archive the model weights. Keep the failure log.
model_without_revision="$rejections_root/model-without-revision"
(
trap 'rm -rf -- "$model_without_revision"' EXIT
"$python" - "$model" "$model_without_revision" <<'PY'
from pathlib import Path
import sys
source, view = map(Path, sys.argv[1:])
assert "snapshots" not in view.resolve().parts, "revision-negative view must not imply a snapshot revision"
view.mkdir()
for item in source.iterdir():
    if item.name not in (".cache", ".git") and item.is_file():
        (view / item.name).symlink_to(item.resolve())
assert (view / "config.json").is_file()
PY
expect_acceptance_rejection unknown-revision \
  "cannot verify model_revision=$expected_revision: local model revision is unknown" \
  -u PEGAINFER_TEST_MODEL_REVISION "PEGAINFER_TEST_MODEL_PATH=$model_without_revision"
)
expect_acceptance_rejection wrong-revision \
  'qwen35 hf_golden_gate model revision mismatch;' \
  PEGAINFER_TEST_MODEL_REVISION=0000000000000000000000000000000000000000
wrong_sha="0${object_sha:1}"
[[ "$wrong_sha" != "$object_sha" ]] || wrong_sha="1${object_sha:1}"
expect_acceptance_rejection wrong-identity \
  "candidate acceptance artifact identity mismatch: expected $wrong_sha, loaded $object_sha" \
  "PEGAINFER_TEST_QWEN35_GDN_OBJECT_SHA256=$wrong_sha"
echo "GDN_ACCEPTANCE_REJECTIONS_OK"
run_exact_gate gate3-hf-golden \
  "$hf_short" ignored \
  -p pegainfer-qwen35 --features qwen35 --lib
run_exact_gate gate3-hf-long-golden \
  executor::hf_golden_gate::candidate_pega_logprobs_match_hf_long_golden_within_qwen35_tolerance ignored \
  -p pegainfer-qwen35 --features qwen35 --lib
run_exact_gate gate4-model-continuation \
  prefill::tests::flashinfer_gdn_chunk_continuation_and_model_outputs_match ignored \
  -p pegainfer-qwen35 --features qwen35 --lib
run_exact_gate gate4-scheduler-chunks \
  scheduler::chunked_prefill_tests::candidate_chunked_prefill_matches_unchunked_prefill_for_resumed_paged_kv ignored \
  -p pegainfer-qwen35 --features qwen35 --lib
run_exact_gate gate5-scheduler scheduler::e2e_tests::candidate_e2e_qwen35_scheduler ignored \
  -p pegainfer-qwen35 --features qwen35 --lib
run_exact_gate gate5-shared-sm scheduler::e2e_tests::candidate_e2e_qwen35_shared_sm_last_decoder ignored \
  -p pegainfer-qwen35 --features qwen35 --lib

[[ "$(git rev-parse HEAD)" == "$commit_sha" && "$(git rev-parse 'HEAD^{tree}')" == "$tree_sha" \
  && -z "$(git status --short --untracked-files=no)" ]] || {
  echo "validated source changed during acceptance" >&2; exit 3;
}
[[ "$(sha256sum "$bundle/kernel.o" | awk '{print $1}')" == "$object_sha" ]] || {
  echo "candidate object changed during acceptance" >&2; exit 3;
}
echo "all five Qwen3.5 GDN functional gate categories passed for $commit_sha tree $tree_sha object $object_sha"
echo "Stage 20 also requires analysis of the Shared-SM Nsight timeline and same-head performance A/B."
