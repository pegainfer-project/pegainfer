#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${PEGAINFER_DEV_IMAGE:-pegainfer-dev:cu132}"
container="${PEGAINFER_DEV_CONTAINER:-pegainfer-dev}"
cache_root="${PEGAINFER_DEV_CACHE:-$HOME/.cache/pegainfer-dev}"

usage() {
  cat <<'EOF'
Usage:
  docker/dev.sh build             Build the development image.
  docker/dev.sh shell [COMMAND]   Start an interactive development container.
  docker/dev.sh run COMMAND...    Run a command in a disposable container.

Environment:
  CUDA_IMAGE              CUDA devel base (default: CUDA 13.2 / Ubuntu 24.04).
  PEGAINFER_DEV_IMAGE     Image tag (default: pegainfer-dev:cu132).
  PEGAINFER_DEV_CONTAINER Interactive container name (default: pegainfer-dev).
  PEGAINFER_DEV_CACHE     Persistent build-cache directory.
  PEGAINFER_DEV_CACHE_KEY Override the native build-cache namespace.
  PEGAINFER_MODEL_DIR     Read-only model directory mounted at the same path.
  PEGAINFER_CUDA_SM       Forwarded CUDA SM target override.
  EP_DISABLE_GIN          Forwarded when set; useful on trays without a GIN NIC.
EOF
}

build_image() {
  docker build \
    --file "$repo_root/docker/Dockerfile.dev" \
    --build-arg "CUDA_IMAGE=${CUDA_IMAGE:-nvidia/cuda:13.2.0-devel-ubuntu24.04}" \
    --build-arg "DEV_USER=$(id -un)" \
    --build-arg "DEV_UID=$(id -u)" \
    --build-arg "DEV_GID=$(id -g)" \
    --tag "$image" \
    "$repo_root"
}

toolkit_id="$(
  docker image inspect \
    --format '{{ index .Config.Labels "org.pegainfer.cuda-image" }}' \
    "$image" 2>/dev/null || true
)"
toolkit_id="${toolkit_id:-$image}"
cache_key="${PEGAINFER_DEV_CACHE_KEY:-$(printf '%s' "$toolkit_id" | sed 's/[^A-Za-z0-9_.-]/_/g')}"
target_cache="$cache_root/target/$cache_key"

docker_args=(
  --gpus all
  --ipc host
  --network host
  --ulimit memlock=-1
  --ulimit stack=67108864
  --volume "$repo_root:$repo_root"
  --volume "$cache_root/cargo-registry:/opt/cargo/registry"
  --volume "$cache_root/cargo-git:/opt/cargo/git"
  --volume "$target_cache:$repo_root/target"
  --workdir "$repo_root"
)

# GB300 NVL72 cross-tray LSA requires the IMEX channel in addition to the
# devices exposed by --gpus. Single-node and non-IMEX hosts have no such path.
imex_channel=/dev/nvidia-caps-imex-channels/channel0
if [[ -e "$imex_channel" ]]; then
  docker_args+=(--device "$imex_channel:$imex_channel")
fi

# A linked worktree stores a .git file that points at the main checkout's
# common git directory. Mount that directory at the same absolute path so
# build.rs can inspect and initialize submodules without making the whole
# parent tree visible.
git_common_dir="$(git -C "$repo_root" rev-parse --path-format=absolute --git-common-dir)"
case "$git_common_dir/" in
  "$repo_root/"*) ;;
  *) docker_args+=(--volume "$git_common_dir:$git_common_dir") ;;
esac

if [[ -n "${EP_DISABLE_GIN:-}" ]]; then
  docker_args+=(--env "EP_DISABLE_GIN=$EP_DISABLE_GIN")
fi

if [[ -n "${PEGAINFER_CUDA_SM:-}" ]]; then
  docker_args+=(--env "PEGAINFER_CUDA_SM=$PEGAINFER_CUDA_SM")
fi

if [[ -n "${PEGAINFER_MODEL_DIR:-}" ]]; then
  [[ "$PEGAINFER_MODEL_DIR" = /* ]] || {
    echo "PEGAINFER_MODEL_DIR must be an absolute path" >&2
    exit 2
  }
  [[ -d "$PEGAINFER_MODEL_DIR" ]] || {
    echo "PEGAINFER_MODEL_DIR does not exist: $PEGAINFER_MODEL_DIR" >&2
    exit 2
  }
  docker_args+=(--volume "$PEGAINFER_MODEL_DIR:$PEGAINFER_MODEL_DIR:ro")
fi

case "${1:-}" in
  build)
    build_image
    ;;
  shell)
    shift
    mkdir -p "$cache_root"/{cargo-registry,cargo-git} "$target_cache"
    if (( $# == 0 )); then
      set -- /bin/bash
    fi
    docker run --rm -it --name "$container" "${docker_args[@]}" "$image" "$@"
    ;;
  run)
    shift
    (( $# > 0 )) || { usage >&2; exit 2; }
    mkdir -p "$cache_root"/{cargo-registry,cargo-git} "$target_cache"
    docker run --rm "${docker_args[@]}" "$image" "$@"
    ;;
  *)
    usage
    exit 2
    ;;
esac
