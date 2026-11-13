#!/usr/bin/env bash
# Optional independent CPU Torch oracle for the production native sources.
set -euo pipefail
sdk_root=${ROCM_PATH:-/opt/rocm}
core=${PEGAINFER_ROCM_CORE:-$sdk_root}
libraries=${PEGAINFER_ROCM_LIBRARIES:-$sdk_root}
artifact_root="${CARGO_TARGET_DIR:-$PWD/target}/native-probe/${PEGAINFER_ROCM_ARCH}"
mkdir -p "$artifact_root"
hip_flags=(-x hip -std=c++17 -O3 -fPIC -c "--hip-path=$core"
  "-I$core/include" "-I$libraries/include" "-I${PEGAINFER_ROCM_DEV_INCLUDE:-$sdk_root/include}"
  -Ipegainfer-device/csrc)
if [[ -d "$core/lib/llvm/amdgcn/bitcode" ]]; then
  hip_flags+=("--rocm-device-lib-path=$core/lib/llvm/amdgcn/bitcode")
fi
IFS=',' read -r -a targets <<< "$PEGAINFER_ROCM_ARCH"
for target in "${targets[@]}"; do hip_flags+=("--offload-arch=$target"); done
hipcc=${PEGAINFER_HIPCC:-$core/bin/hipcc}
[[ -x "$hipcc" ]] || hipcc=hipcc
"$hipcc" "${hip_flags[@]}" pegainfer-device/csrc/runtime.cpp -o "$artifact_root/runtime.o"
"$hipcc" "${hip_flags[@]}" pegainfer-kernels/csrc/hip/qwen3.cpp -o "$artifact_root/kernels.o"
resolve_library() {
  local candidate
  for candidate in "$1"/lib/lib"$2".so* "$1"/lib64/lib"$2".so*; do
    if [[ -f "$candidate" ]]; then readlink -f "$candidate"; return; fi
  done
  echo "Missing $2 runtime library under $1" >&2
  return 1
}
g++ -shared "$artifact_root/runtime.o" "$artifact_root/kernels.o" \
  "$(resolve_library "$core" amdhip64)" "$(resolve_library "$libraries" rocblas)" \
  -o "$artifact_root/libpegainfer_hip.so"
python3 scripts/amd/check_native.py "$artifact_root/libpegainfer_hip.so"
