# FlashKDA (vendored)

Upstream: https://github.com/MoonshotAI/FlashKDA
Commit: `1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b` (2026-07-29)
License: MIT (Copyright (c) 2026 MoonshotAI) — see `LICENSE` in this directory.
Authors (upstream citation): Yutian Chen, Zhiyuan Li, Yucheng Wang, Ming Wei.

FlashKDA is MoonshotAI's chunkwise KDA (Kimi Delta Attention) prefill forward
kernel, built on CUTLASS/CuTe (SM90 TMA + cluster barriers). PegaInfer uses it
as the K3 model line's chunked-prefill KDA time-axis kernel.

## What is vendored

Only the CUDA sources and the launch layer:

- `csrc/fwd.h` — the `launch_fwd` template declaration (unmodified)
- `csrc/smxx/utils.cuh`, `fwd_kernel1.cuh`, `fwd_kernel2.cuh` (unmodified)
- `csrc/smxx/fwd_launch.cu` — **modified**: the explicit-instantiation list at
  the bottom is trimmed from upstream's 14 variants to the ones PegaInfer
  launches (see the marked block at the end of the file). Everything above the
  marker is upstream verbatim.

NOT vendored: the PyTorch binding (`csrc/flash_kda.cpp`), Python package,
tests, benchmarks. PegaInfer's own C ABI shim lives at
`pegainfer-kernels/csrc/k3/k3_flash_kda.cu` and the workspace-size arithmetic
is reproduced there from the upstream binding.

CUTLASS: upstream pins `5c149f52a436782210263fb2f19b354443a61c6a` as a
submodule; this build compiles against the CUTLASS already vendored in this
repo (see `pegainfer-kernels/build.rs` for which copy).

## Updating

Re-copy the four files from upstream, then re-apply the instantiation trim in
`fwd_launch.cu`.
