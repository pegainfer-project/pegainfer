#!/usr/bin/env bash
# Run in an installed ROCm SDK. No container, GPU selection, or host paths implied.
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
# An optional user-owned profile may select a container or set SDK paths.
# It is never checked into the repository or loaded implicitly.
if [[ -n "${PEGAINFER_ROCM_ENV:-}" ]]; then
  source "$PEGAINFER_ROCM_ENV"
fi
export PEGAINFER_ROCM_ARCH="${PEGAINFER_ROCM_ARCH:?Set PEGAINFER_ROCM_ARCH to an explicit gfx target}"
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$project_root/target/rocm-$PEGAINFER_ROCM_ARCH}"
cd "$project_root"
exec "$@"
