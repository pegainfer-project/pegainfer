#!/usr/bin/env bash
# Build the CUPTI kernel-capture injection library.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cuda="${CUDA_HOME:-/usr/local/cuda}"
target_dir="$(echo "$cuda"/targets/*-linux)"
cupti_inc="$target_dir/include"
cupti_lib="$target_dir/lib"

cc -O2 -fPIC -shared \
  -I"$cupti_inc" -I"$cuda/include" \
  "$here/capture.c" \
  -o "$here/libkernelcapture.so" \
  -L"$cupti_lib" -lcupti -lcuda

echo "built $here/libkernelcapture.so"
