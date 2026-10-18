# PegaInfer development container

The development image contains the native toolchain required to build
PegaInfer: CUDA, the Rust nightly pinned by `rust-toolchain.toml`, Python 3,
uv, Triton, TileLang, clang, OpenSSL, protoc, and NCCL 2.30.4 or newer.

Build it once:

```bash
docker/dev.sh build
```

Open a shell with the repository at its original absolute path and persistent
compiler caches mounted:

```bash
docker/dev.sh shell
```

Linked Git worktrees are supported: the wrapper detects an external Git
common directory and mounts it at the same path so build scripts can inspect
and initialize submodules.

Or run a one-off build:

```bash
docker/dev.sh run cargo build --release
```

Qwen3.5's build-time AOT compiler uses the image's pinned Triton environment:

```bash
docker/dev.sh run cargo build --release --features qwen35
```

GLM5.2 builds targeting Hopper use the same environment's pinned TileLang:

```bash
PEGAINFER_CUDA_SM=90 docker/dev.sh run \
  cargo build --release --features glm52
```

Mount model weights read-only at their existing absolute path:

```bash
PEGAINFER_MODEL_DIR=/models/Qwen3-4B docker/dev.sh shell
```

The default base is the pinned CUDA 13.2 development image, which is the
newest version supported by the current GB300 tray driver. Override it without
changing the Dockerfile after upgrading the host driver:

```bash
CUDA_IMAGE=nvidia/cuda:<version>-devel-ubuntu24.04 docker/dev.sh build
```

The persistent native target cache is automatically namespaced by the CUDA
base recorded in the image. Set `PEGAINFER_DEV_CACHE_KEY` only when a custom
image needs a more specific namespace.

On a tray without a GIN-capable NIC, pass `EP_DISABLE_GIN=1` when starting
the container. It is intentionally not baked into the image because networked
deployments require GIN.

On GB300 NVL72 hosts, the wrapper automatically forwards the IMEX channel
device required by cross-tray LSA when that device exists.
