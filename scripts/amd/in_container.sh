#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
artifact_root="$project_root/../artifacts/amd"
export RUSTUP_HOME="$artifact_root/toolchain/rustup"
export CARGO_HOME="$artifact_root/toolchain/cargo"
export PATH="$CARGO_HOME/bin:$PATH"
export CARGO_TARGET_DIR="$artifact_root/target/rocm10.1-${PEGAINFER_ROCM_ARCH:-gfx1201}"
export TRITON_CACHE_DIR="$artifact_root/cache/triton"
export TORCH_EXTENSIONS_DIR="$artifact_root/cache/torch"
export PEGAINFER_ROCM_ARCH="${PEGAINFER_ROCM_ARCH:-gfx1201}"
export PEGAINFER_ROCM_CORE="${PEGAINFER_ROCM_CORE:-/opt/venv/lib/python3.13/site-packages/_rocm_sdk_core}"
export PEGAINFER_ROCM_LIBRARIES="${PEGAINFER_ROCM_LIBRARIES:-/opt/venv/lib/python3.13/site-packages/_rocm_sdk_libraries}"
export LD_LIBRARY_PATH="$PEGAINFER_ROCM_LIBRARIES/lib:$PEGAINFER_ROCM_CORE/lib:${LD_LIBRARY_PATH:-}"
if [[ -d "$artifact_root/toolchain/openssl/root/usr/include/openssl" ]]; then
    export OPENSSL_INCLUDE_DIR="$artifact_root/toolchain/openssl/root/usr/include"
    export OPENSSL_LIB_DIR="$artifact_root/toolchain/openssl/root/usr/lib/x86_64-linux-gnu"
    export LD_LIBRARY_PATH="$OPENSSL_LIB_DIR:$LD_LIBRARY_PATH"
    export CFLAGS="-I$OPENSSL_INCLUDE_DIR -I$OPENSSL_INCLUDE_DIR/x86_64-linux-gnu ${CFLAGS:-}"
fi
cd "$project_root"
python scripts/amd/check_environment.py
exec "$@"
