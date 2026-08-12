#!/usr/bin/env bash
set -euo pipefail

: "${PEGAINFER_STAGE9_MODEL_PATH:?set PEGAINFER_STAGE9_MODEL_PATH}"
: "${PEGAINFER_STAGE9_AOT_BUNDLE:?set PEGAINFER_STAGE9_AOT_BUNDLE to qwen35_4b_candidate directory}"
: "${PEGAINFER_STAGE9_OUTPUT_DIR:?set PEGAINFER_STAGE9_OUTPUT_DIR}"
: "${PEGAINFER_STAGE9_COMMIT:?set PEGAINFER_STAGE9_COMMIT to the exact code/archive provenance}"
: "${PEGAINFER_TRITON_PYTHON:?set PEGAINFER_TRITON_PYTHON to a Python that imports Triton}"

readonly EXPECTED_CONFIG_SHA="ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670"
readonly STAGE9_TARGET_DIR="${CARGO_TARGET_DIR:-target}"
readonly STAGE9_BIN="${STAGE9_TARGET_DIR}/release/gdn_stage9_bench"
readonly stage9_manifest="${PEGAINFER_STAGE9_AOT_BUNDLE}/manifest.json"

mkdir -p "${PEGAINFER_STAGE9_OUTPUT_DIR}"

stage9_cargo_path="$(command -v cargo || true)"
if [[ -z "${stage9_cargo_path}" || ! -x "${stage9_cargo_path}" ]]; then
    echo "cargo is unavailable; source /root/.cargo/env or install Rustup before Stage 9" >&2
    exit 1
fi
if [[ ! -x "${PEGAINFER_TRITON_PYTHON}" ]] \
    || ! "${PEGAINFER_TRITON_PYTHON}" -c 'import triton' >/dev/null 2>&1; then
    echo "PEGAINFER_TRITON_PYTHON cannot import Triton: ${PEGAINFER_TRITON_PYTHON}" >&2
    exit 1
fi

test -f "${PEGAINFER_STAGE9_MODEL_PATH}/config.json"
test -f "${stage9_manifest}"

readonly actual_commit="$(git rev-parse HEAD)"
if [[ "${PEGAINFER_STAGE9_COMMIT}" != "${actual_commit}" ]]; then
    echo "PEGAINFER_STAGE9_COMMIT mismatch: expected ${actual_commit}, got ${PEGAINFER_STAGE9_COMMIT}" >&2
    exit 1
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Stage 9 refuses a dirty tracked or staged source tree" >&2
    exit 1
fi
if [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
    echo "Stage 9 refuses an untracked source tree" >&2
    exit 1
fi
if git submodule status --recursive | grep -Eq '^[+-U]'; then
    echo "Stage 9 refuses missing or mismatched submodules" >&2
    git submodule status --recursive >&2
    exit 1
fi

python3 pegainfer-kernels/tools/flashinfer_gdn/artifact_contract.py \
    validate-manifest "${stage9_manifest}" \
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

check_hash "${EXPECTED_CONFIG_SHA}" "${PEGAINFER_STAGE9_MODEL_PATH}/config.json"
export PEGAINFER_STAGE9_MANIFEST_SHA256
PEGAINFER_STAGE9_MANIFEST_SHA256="$(sha256sum "${stage9_manifest}" | awk '{print $1}')"
export PEGAINFER_QWEN35_GDN_AOT_BUNDLE="${PEGAINFER_STAGE9_AOT_BUNDLE}"
readonly stage9_object_path="${PEGAINFER_STAGE9_AOT_BUNDLE}/kernel.o"
test -f "${stage9_object_path}"
readonly stage9_object_sha="$(python3 - "${stage9_manifest}" <<'PY'
import json
import pathlib
import sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["artifact"]["object"]["sha256"])
PY
)"
check_hash "${stage9_object_sha}" "${stage9_object_path}"

{
    date -u
    git rev-parse HEAD
    git status --short
    git submodule status --recursive
    nvidia-smi
    nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv
    nvcc --version
    sha256sum \
        "${PEGAINFER_STAGE9_MODEL_PATH}/config.json" \
        "${stage9_manifest}" \
        "${stage9_object_path}"
    stat -c '%n %s bytes' "${stage9_object_path}"
    printf 'PEGAINFER_STAGE9_COMMIT=%s\n' "${PEGAINFER_STAGE9_COMMIT}"
    printf 'PEGAINFER_STAGE9_ARCHIVE_SHA=%s\n' "${PEGAINFER_STAGE9_ARCHIVE_SHA:-not-set}"
} | tee "${PEGAINFER_STAGE9_OUTPUT_DIR}/environment.log"

export PEGAINFER_STAGE9_GPU
PEGAINFER_STAGE9_GPU="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -n 1)"
export PEGAINFER_STAGE9_CUDA
PEGAINFER_STAGE9_CUDA="$(nvcc --version | tail -n 1)"

stage9_build_command=(
    "${stage9_cargo_path}" build --release
    -p pegainfer-qwen35
    --features qwen35
    --bin gdn_stage9_bench
)
if [[ -x /usr/bin/time ]]; then
    /usr/bin/time -v \
        -o "${PEGAINFER_STAGE9_OUTPUT_DIR}/build-time.txt" \
        "${stage9_build_command[@]}" \
        2>&1 | tee "${PEGAINFER_STAGE9_OUTPUT_DIR}/build.log"
else
    echo "warning: /usr/bin/time is unavailable; recording wall-clock build time only" \
        | tee "${PEGAINFER_STAGE9_OUTPUT_DIR}/build-time.txt"
    stage9_build_started_ns="$(date +%s%N)"
    "${stage9_build_command[@]}" \
        2>&1 | tee "${PEGAINFER_STAGE9_OUTPUT_DIR}/build.log"
    stage9_build_finished_ns="$(date +%s%N)"
    python3 - "${stage9_build_started_ns}" "${stage9_build_finished_ns}" <<'PY' \
        | tee -a "${PEGAINFER_STAGE9_OUTPUT_DIR}/build-time.txt"
import sys

started = int(sys.argv[1])
finished = int(sys.argv[2])
print(f"wall_seconds={(finished - started) / 1_000_000_000:.6f}")
PY
fi

"${STAGE9_BIN}" --help | tee "${PEGAINFER_STAGE9_OUTPUT_DIR}/help.log"

readonly stage9_cases="${PEGAINFER_STAGE9_CASES:-63:1 64:1 65:1 128:1 128:4 128:8 2048:1 2048:4}"
readonly stage9_warmup="${PEGAINFER_STAGE9_WARMUP:-2}"
readonly stage9_iterations="${PEGAINFER_STAGE9_ITERATIONS:-10}"
readonly stage9_max_new_tokens="${PEGAINFER_STAGE9_MAX_NEW_TOKENS:-8}"
read -r -a stage9_case_array <<<"${stage9_cases}"
readonly -a stage9_backend_order=(triton flashinfer flashinfer triton)

for stage9_case in "${stage9_case_array[@]}"; do
    IFS=: read -r stage9_prompt_len stage9_concurrency <<<"${stage9_case}"
    if [[ -z "${stage9_prompt_len}" || -z "${stage9_concurrency}" ]]; then
        echo "invalid Stage 9 case '${stage9_case}', expected prompt_len:concurrency" >&2
        exit 1
    fi

    stage9_order=0
    for stage9_backend in "${stage9_backend_order[@]}"; do
        stage9_order=$((stage9_order + 1))
        stage9_stem="t${stage9_prompt_len}-c${stage9_concurrency}-o${stage9_order}-${stage9_backend}"
        stage9_args=(
            --backend "${stage9_backend}"
            --model-path "${PEGAINFER_STAGE9_MODEL_PATH}"
            --prompt-len "${stage9_prompt_len}"
            --concurrency "${stage9_concurrency}"
            --warmup "${stage9_warmup}"
            --iterations "${stage9_iterations}"
            --max-new-tokens "${stage9_max_new_tokens}"
            --run-label "${stage9_stem}"
            --output "${PEGAINFER_STAGE9_OUTPUT_DIR}/${stage9_stem}.json"
        )
        "${STAGE9_BIN}" "${stage9_args[@]}" \
            2>&1 | tee "${PEGAINFER_STAGE9_OUTPUT_DIR}/${stage9_stem}.log"
    done
done

python3 - "${PEGAINFER_STAGE9_OUTPUT_DIR}" "${stage9_object_sha}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
expected_object_sha = sys.argv[2]
rows = []
for path in sorted(root.glob("t*-c*-o*-*.json")):
    report = json.loads(path.read_text())
    evidence = report.get("flashinfer_evidence")
    if evidence is not None and evidence["artifact_sha256"] != expected_object_sha:
        raise SystemExit(
            f"{path.name}: linked object hash {evidence['artifact_sha256']} "
            f"does not match validated manifest {expected_object_sha}"
        )
    rows.append(
        {
            "file": path.name,
            "backend": report["backend"],
            "prompt_len": report["prompt_len"],
            "concurrency": report["concurrency"],
            "startup_ms": report["engine_startup_ms"],
            "ttft_p50_ms": report["ttft"]["p50_ms"],
            "ttft_p99_ms": report["ttft"]["p99_ms"],
            "tpot_p50_ms": report["tpot"]["p50_ms"],
            "tpot_p99_ms": report["tpot"]["p99_ms"],
            "throughput_mean": report["batch_throughput_tokens_per_second"]["mean"],
            "successful_launches": None if evidence is None else evidence["successful_launches"],
        }
    )
(root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows, indent=2))
PY

echo "Stage 9 unprofiled ABBA results: ${PEGAINFER_STAGE9_OUTPUT_DIR}"
