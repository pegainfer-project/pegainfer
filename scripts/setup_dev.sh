#!/usr/bin/env bash
# setup_dev.sh — one-shot dev-environment bootstrap for a fresh Ubuntu GPU box.
#
# Goal: from a fresh CUDA GPU box to a green `cargo build --release` for the
# default Qwen3-4B build (pure Rust + CUDA, no Python). Installs apt build deps,
# uv, and the rustup nightly toolchain pinned by rust-toolchain.toml.
# Idempotent — safe to re-run.
#
# The CUDA toolkit (nvcc + cuBLAS) is a PREREQUISITE, not something this script
# installs: use a CUDA image (e.g. Prime Intellect's `ubuntu_22_cuda_12`) or an
# already-provisioned toolkit. The script only detects it and fails loudly if
# absent — it will not apt-install a toolkit onto someone's machine.
#
# Usage:  bash scripts/setup_dev.sh
# Also invoked by scripts/prime_devbox.sh after it provisions a fresh box.
set -euo pipefail

log()  { printf '\033[1;32m[setup_dev]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[setup_dev] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SUDO=""
if [ "$(id -u)" -ne 0 ]; then command -v sudo >/dev/null && SUDO="sudo"; fi

# --- 0. sanity: this must be a GPU box (build.rs reads the live arch) ----------
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found — this script expects an NVIDIA GPU host."
log "GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"

# --- 1. apt build deps ---------------------------------------------------------
if command -v apt-get >/dev/null 2>&1; then
  log "Installing apt build deps…"
  export DEBIAN_FRONTEND=noninteractive
  # A freshly-booted cloud box runs cloud-init / unattended-upgrades, which hold
  # the apt/dpkg lock for the first minute or two. DPkg::Lock::Timeout makes apt
  # wait for the lock instead of racing it and dying with "Could not get lock".
  APT="$SUDO apt-get -o DPkg::Lock::Timeout=300"
  $APT update -qq
  # protobuf-compiler: pegaflow-proto's build.rs invokes `protoc`.
  $APT install -y -qq build-essential git curl ca-certificates pkg-config libssl-dev protobuf-compiler
fi

# --- 2. CUDA toolkit (nvcc + cuBLAS) — PREREQUISITE, detected not installed ----
detect_cuda() {
  local c
  for c in "${CUDA_HOME:-}" /usr/local/cuda /usr/local/cuda-13* /usr/local/cuda-12*; do
    [ -n "$c" ] && [ -x "$c/bin/nvcc" ] && { echo "$c"; return 0; }
  done
  if command -v nvcc >/dev/null 2>&1; then dirname "$(dirname "$(command -v nvcc)")"; return 0; fi
  return 1
}

CUDA_HOME="$(detect_cuda)" || die "No CUDA toolkit (nvcc) found.
  This script does not install CUDA — boot a CUDA image (e.g. Prime Intellect's
  'ubuntu_22_cuda_12') or install the CUDA Toolkit (>=12.2) yourself, then re-run.
  If it is installed at a non-standard path, export CUDA_HOME first."
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
log "CUDA toolkit: $CUDA_HOME ($("$CUDA_HOME/bin/nvcc" --version | grep -oE 'release [0-9.]+' | head -1))"

# --- 3. uv ---------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# --- 4. Rust (nightly channel is pinned by rust-toolchain.toml) ----------------
if ! command -v rustup >/dev/null 2>&1; then
  log "Installing rustup…"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
fi
# shellcheck disable=SC1091
. "$HOME/.cargo/env"
log "Provisioning toolchain pinned by rust-toolchain.toml…"
rustup show active-toolchain >/dev/null   # triggers install of the pinned nightly + components

# --- 5. vendored submodules (flashinfer headers + its pinned cccl) -------------
# A fresh `git clone` checks out no submodules. build.rs needs the flashinfer
# headers, and `-I`s flashinfer's pinned cccl ahead of the toolkit's own CCCL so
# the kernels compile against a new-enough libcudacxx (<cuda/cmath>, cuda::maximum)
# on any CUDA >=12.2. (A CUDA 13.x host ships a new-enough system CCCL and masks a
# missing cccl; CUDA 12.x does not — so require it explicitly.) Only flashinfer +
# cccl are inited; the heavy unused nested submodules (cutlass/nccl/nixl/spdlog)
# and the DeepEP submodule are deliberately left untouched.
FI=pegainfer-kernels/third_party/flashinfer
is_git() { git -C "$1" rev-parse --git-dir >/dev/null 2>&1; }

if [ ! -e "$FI/include" ]; then
  is_git . || die "flashinfer submodule not checked out and this is not a git checkout.
  Boot from a git clone (prime_devbox.sh does this) or run: git submodule update --init $FI"
  log "Initializing flashinfer submodule…"
  git submodule update --init "$FI"
fi

if [ ! -e "$FI/3rdparty/cccl/libcudacxx/include/cuda/cmath" ]; then
  is_git "$FI" || die "vendored CCCL missing and flashinfer is not a git checkout.
  Without it the build fails on CUDA <13 with 'cuda/cmath: No such file'."
  log "Initializing vendored CCCL submodule…"
  git -C "$FI" submodule update --init 3rdparty/cccl
fi

# --- 6. build ------------------------------------------------------------------
if [ -n "${PEGAINFER_CUDA_SM:-}" ]; then
  log "PEGAINFER_CUDA_SM=$PEGAINFER_CUDA_SM — compiling kernels for this arch instead of the live GPU."
else
  log "PEGAINFER_CUDA_SM unset — build.rs auto-detects the arch from nvidia-smi."
fi
log "Building (release, default Qwen3-4B feature)… first build compiles CUDA kernels, give it a few minutes."
cargo build --release

cat <<EOF

$(log "✅ Dev environment ready.")
  CUDA_HOME : $CUDA_HOME
  rust      : $(rustc --version)
  next      : huggingface-cli download Qwen/Qwen3-4B --local-dir models/Qwen3-4B
              cargo run --release -- --model-path models/Qwen3-4B
EOF
