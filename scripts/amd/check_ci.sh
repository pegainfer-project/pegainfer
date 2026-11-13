#!/usr/bin/env bash
# CPU-only backend checks. Requires an installed SDK; never creates a HIP device.
set -euo pipefail
packages=(-p pegainfer-device -p pegainfer-kernels -p pegainfer-core -p pegainfer-qwen3 -p pegainfer-frontend -p pegainfer-build)
backend=(--no-default-features --features pegainfer-qwen3/rocm,pegainfer-core/rocm,pegainfer-kernels/rocm,pegainfer-device/rocm)
cargo build --locked --release -p pegainfer-server --no-default-features --features rocm,qwen3
cargo run --locked --release -p pegainfer-server --no-default-features --features rocm,qwen3 -- --help > /dev/null
cargo test --locked --release "${packages[@]}" "${backend[@]}" --all-targets --no-run
cargo test --locked --release "${packages[@]}" "${backend[@]}" --lib
cargo test --locked --release -p pegainfer-server --no-default-features --features rocm,qwen3 --bin pegainfer
cargo clippy --locked --release "${packages[@]}" -p pegainfer-server "${backend[@]}" --features pegainfer-server/rocm,pegainfer-server/qwen3 --all-targets
