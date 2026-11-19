#!/usr/bin/env bash
# CPU-only backend checks. Requires an installed SDK; never creates a HIP device.
set -euo pipefail
packages=(-p pegainfer-rocm -p pegainfer-frontend -p pegainfer-build)
backend=(--no-default-features --features pegainfer-rocm/rocm)
cargo build --locked --release -p pegainfer-server --no-default-features --features rocm
cargo run --locked --release -p pegainfer-server --no-default-features --features rocm -- --help > /dev/null
cargo test --locked --release "${packages[@]}" "${backend[@]}" --all-targets --no-run
cargo test --locked --release "${packages[@]}" "${backend[@]}" --lib
cargo test --locked --release -p pegainfer-server --no-default-features --features rocm --bin pegainfer
cargo clippy --locked --release "${packages[@]}" -p pegainfer-server "${backend[@]}" --features pegainfer-server/rocm --all-targets
