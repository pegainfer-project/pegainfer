#!/usr/bin/env bash
set -euo pipefail

: "${PEGAINFER_STAGE13_MODEL_PATH:?set PEGAINFER_STAGE13_MODEL_PATH}"
: "${PEGAINFER_STAGE13_AOT_BUNDLE:?set PEGAINFER_STAGE13_AOT_BUNDLE to qwen35_4b_candidate directory}"
: "${PEGAINFER_STAGE13_OUTPUT_DIR:?set PEGAINFER_STAGE13_OUTPUT_DIR}"
: "${PEGAINFER_STAGE13_COMMIT:?set PEGAINFER_STAGE13_COMMIT to git rev-parse HEAD}"
: "${PEGAINFER_TRITON_PYTHON:?set PEGAINFER_TRITON_PYTHON to a Python that imports Triton}"

readonly EXPECTED_CONFIG_SHA="ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670"
readonly stage13_manifest="${PEGAINFER_STAGE13_AOT_BUNDLE}/manifest.json"
readonly stage13_object="${PEGAINFER_STAGE13_AOT_BUNDLE}/kernel.o"

mkdir -p "${PEGAINFER_STAGE13_OUTPUT_DIR}"

stage13_cargo="$(command -v cargo || true)"
if [[ -z "${stage13_cargo}" || ! -x "${stage13_cargo}" ]]; then
    echo "cargo is unavailable; source /root/.cargo/env or install Rustup before Stage 13" >&2
    exit 1
fi
if [[ ! -x "${PEGAINFER_TRITON_PYTHON}" ]] \
    || ! "${PEGAINFER_TRITON_PYTHON}" -c 'import triton' >/dev/null 2>&1; then
    echo "PEGAINFER_TRITON_PYTHON cannot import Triton: ${PEGAINFER_TRITON_PYTHON}" >&2
    exit 1
fi

test -f "${PEGAINFER_STAGE13_MODEL_PATH}/config.json"
test -f "${stage13_manifest}"
test -f "${stage13_object}"

readonly actual_commit="$(git rev-parse HEAD)"
if [[ "${PEGAINFER_STAGE13_COMMIT}" != "${actual_commit}" ]]; then
    echo "PEGAINFER_STAGE13_COMMIT mismatch: expected ${actual_commit}, got ${PEGAINFER_STAGE13_COMMIT}" >&2
    exit 1
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Stage 13 refuses a dirty tracked or staged source tree" >&2
    exit 1
fi
if [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
    echo "Stage 13 refuses an untracked source tree" >&2
    exit 1
fi
if git submodule status --recursive | grep -Eq '^[+-U]'; then
    echo "Stage 13 refuses missing or mismatched submodules" >&2
    git submodule status --recursive >&2
    exit 1
fi

python3 pegainfer-kernels/tools/flashinfer_gdn/artifact_contract.py \
    validate-manifest "${stage13_manifest}" \
    --flashinfer-dir pegainfer-kernels/third_party/flashinfer

check_hash() {
    local expected="$1"
    local path="$2"
    local actual
    actual="$(sha256sum "${path}" | awk '{print $1}')"
    if [[ "${actual}" != "${expected}" ]]; then
        echo "SHA-256 mismatch for ${path}: expected ${expected}, got ${actual}" >&2
        exit 1
    fi
}

check_hash "${EXPECTED_CONFIG_SHA}" "${PEGAINFER_STAGE13_MODEL_PATH}/config.json"
readonly stage13_object_sha="$(python3 - "${stage13_manifest}" <<'PY'
import json
import pathlib
import sys

print(json.loads(pathlib.Path(sys.argv[1]).read_text())["artifact"]["object"]["sha256"])
PY
)"
check_hash "${stage13_object_sha}" "${stage13_object}"

export PEGAINFER_QWEN35_GDN_AOT_BUNDLE="${PEGAINFER_STAGE13_AOT_BUNDLE}"
export PEGAINFER_TEST_MODEL_PATH="${PEGAINFER_STAGE13_MODEL_PATH}"

{
    date -u
    git rev-parse HEAD
    git status --short
    git submodule status --recursive
    nvidia-smi
    nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv
    nvcc --version
    sha256sum \
        "${PEGAINFER_STAGE13_MODEL_PATH}/config.json" \
        "${stage13_manifest}" \
        "${stage13_object}"
    stat -c '%n %s bytes' "${stage13_object}"
} | tee "${PEGAINFER_STAGE13_OUTPUT_DIR}/environment.log"

run_gate() {
    local name="$1"
    shift
    echo "=== Stage 13 gate: ${name} ===" | tee "${PEGAINFER_STAGE13_OUTPUT_DIR}/${name}.log"
    "$@" 2>&1 | tee -a "${PEGAINFER_STAGE13_OUTPUT_DIR}/${name}.log"
}

run_gate stable-abi-alias-separate \
    "${stage13_cargo}" test --release \
    -p pegainfer-kernels \
    --features qwen35 \
    --lib \
    ops::qwen35::tests::sm120_stable_abi_alias_and_separate_state_are_bitwise_identical \
    -- --ignored --exact --nocapture

run_gate hv32-operator \
    "${stage13_cargo}" test --release \
    -p pegainfer-qwen35 \
    --features qwen35 \
    --lib \
    gdn_stage13_test::sm120_stable_abi_operator_gate_covers_hv32_dynamic_t_and_first_decode \
    -- --ignored --exact --nocapture

run_gate hf-short \
    "${stage13_cargo}" test --release \
    -p pegainfer-qwen35 \
    --features qwen35 \
    --test hf_golden_gate \
    flashinfer_gdn_and_triton_match_hf_short_golden \
    -- --ignored --exact --nocapture

run_gate hf-long \
    "${stage13_cargo}" test --release \
    -p pegainfer-qwen35 \
    --features qwen35 \
    --test hf_golden_gate \
    flashinfer_gdn_and_triton_match_hf_long_golden \
    -- --ignored --exact --nocapture

run_gate chunked-prefill \
    "${stage13_cargo}" test --release \
    -p pegainfer-qwen35 \
    --features qwen35 \
    --test chunked_prefill \
    flashinfer_gdn_chunked_prefill_matches_unchunked_prefill \
    -- --ignored --exact --nocapture

run_gate scheduler \
    "${stage13_cargo}" test --release \
    -p pegainfer-qwen35 \
    --features qwen35 \
    --test e2e_scheduler \
    test_e2e_qwen35_scheduler_flashinfer_gdn \
    -- --ignored --exact --nocapture

python3 - "${PEGAINFER_STAGE13_OUTPUT_DIR}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
gates = [
    "stable-abi-alias-separate",
    "hv32-operator",
    "hf-short",
    "hf-long",
    "chunked-prefill",
    "scheduler",
]
summary = {}
for gate in gates:
    text = (root / f"{gate}.log").read_text()
    passed = "test result: ok." in text
    summary[gate] = {"passed": passed}
    if not passed:
        raise SystemExit(f"Stage 13 gate did not report success: {gate}")
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "Stage 13 correctness results: ${PEGAINFER_STAGE13_OUTPUT_DIR}"
