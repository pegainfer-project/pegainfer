#!/usr/bin/env bash
# Run inside run_rocm10.sh. This links the exact production native sources.
set -euo pipefail
artifact_root="$PWD/../artifacts/amd/native-probe/${PEGAINFER_ROCM_ARCH}"
mkdir -p "$artifact_root"
hip_flags=(-x hip -std=c++17 -O3 -fPIC -c
  "--hip-path=$PEGAINFER_ROCM_CORE"
  "--rocm-device-lib-path=$PEGAINFER_ROCM_CORE/lib/llvm/amdgcn/bitcode"
  "-I$PEGAINFER_ROCM_CORE/include" "-I$PEGAINFER_ROCM_LIBRARIES/include"
  "-I${PEGAINFER_ROCM_DEV_INCLUDE:-/opt/rocm/core-10.1/include}"
  -Ipegainfer-device/csrc)
IFS=',' read -r -a targets <<< "$PEGAINFER_ROCM_ARCH"
for target in "${targets[@]}"; do hip_flags+=("--offload-arch=$target"); done
"${PEGAINFER_HIPCC:-hipcc}" "${hip_flags[@]}" pegainfer-device/csrc/runtime.cpp -o "$artifact_root/runtime.o"
"${PEGAINFER_HIPCC:-hipcc}" "${hip_flags[@]}" pegainfer-kernels/csrc/hip/qwen3.cpp -o "$artifact_root/kernels.o"
g++ -shared "$artifact_root/runtime.o" "$artifact_root/kernels.o" \
  "$PEGAINFER_ROCM_CORE/lib/libamdhip64.so.7" \
  "$PEGAINFER_ROCM_LIBRARIES/lib/librocblas.so.5" -o "$artifact_root/libpegainfer_hip.so"
python scripts/amd/check_native.py "$artifact_root/libpegainfer_hip.so"
