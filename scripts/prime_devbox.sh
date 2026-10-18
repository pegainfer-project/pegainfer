#!/usr/bin/env bash
# prime_devbox.sh — one command: provision a fresh Prime Intellect CUDA GPU box,
# have it git-clone this repo over HTTPS, and run setup_dev.sh. No rsync — the
# box pulls from GitHub itself (datacenter network, not your uplink). Only the
# main repo is private (needs a token); its submodules are public.
#
# Uses the `prime` CLI directly — SkyPilot 0.12.2's Prime provisioner orphans
# billing pods on a hardcoded 30s pod-create timeout.
#
# One-time prereqs:
#   uv tool install prime              # plus `jq` and `gh` on PATH
#   mkdir -p ~/.prime && printf '{"api_key":"%s"}\n' "$PRIME_API_KEY" > ~/.prime/config.json
#   # The key at `prime config view` → "SSH Key Path" MUST be set as PRIMARY at
#   # https://app.primeintellect.ai/dashboard (SSH Keys): pods inject the primary
#   # key at boot, and it cannot be added to an already-running pod.
#   gh auth login                      # supplies the read-only token for the clone
#
# Usage (env-configurable):
#   scripts/prime_devbox.sh                                          # spot V100, compile-only (sm_90), main
#   BRANCH=feat/x GPU=L40S_48GB SPOT=0 PEGAINFER_CUDA_SM= scripts/prime_devbox.sh
#
# Tear down when done (id is printed at the end):  prime pods terminate <id>
set -euo pipefail

GPU="${GPU:-V100_16GB}"                  # prime availability gpu-type, e.g. L40S_48GB, A100_80GB
SPOT="${SPOT:-1}"                        # 1 = spot (cheapest), 0 = on-demand
IMAGE="${IMAGE:-ubuntu_22_cuda_12}"      # MUST ship nvcc; setup_dev.sh never installs CUDA
DISK="${DISK:-100}"
NAME="${NAME:-oi-dev}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa}"  # the key set PRIMARY on Prime (to ssh into the box)
REMOTE_DIR="${REMOTE_DIR:-/root/pegainfer}"
REPO_SLUG="${REPO_SLUG:-pegainfer-project/pegainfer}"
BRANCH="${BRANCH:-main}"
# Compile target. V100 is sm_70, which the FlashInfer/CUDA kernels do not build
# for, so default to sm_90 (a pure compile box). Set PEGAINFER_CUDA_SM= (empty)
# to let build.rs auto-detect the live GPU's arch (use that on L40S/A100/H100).
CUDA_SM="${PEGAINFER_CUDA_SM-90}"

log(){ printf '\033[1;36m[prime_devbox]\033[0m %s\n' "$*"; }
die(){ printf '\033[1;31m[prime_devbox] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

command -v prime >/dev/null 2>&1 || die "prime CLI not found — run: uv tool install prime"
command -v jq    >/dev/null 2>&1 || die "jq not found — install jq"
[ -f "$SSH_KEY" ] || die "SSH key $SSH_KEY not found (set SSH_KEY=/path/to/key)."

# Read-only token for the private clone: explicit GITHUB_TOKEN, else the local gh login.
TOKEN="${GITHUB_TOKEN:-$(gh auth token 2>/dev/null || true)}"
[ -n "$TOKEN" ] || die "no GitHub token — export GITHUB_TOKEN=... or run 'gh auth login'."

# 1. cheapest matching offer
log "Finding $GPU (spot=$SPOT)…"
want_spot=$([ "$SPOT" = 1 ] && echo true || echo false)
SEL=$(prime availability list --gpu-type "$GPU" --output json 2>/dev/null \
  | jq -r --argjson spot "$want_spot" \
      '[.gpu_resources[] | select(.is_spot==$spot)] | sort_by(.price_value)[0].id // empty')
[ -n "$SEL" ] || die "no $GPU (spot=$SPOT) in stock now — try SPOT=$([ "$SPOT" = 1 ] && echo 0 || echo 1) or a different GPU."

# 2. create
log "Creating pod ($GPU, $IMAGE, ${DISK}GB)…"
POD=$(prime pods create --id "$SEL" --image "$IMAGE" --disk-size "$DISK" --name "$NAME" --yes --plain 2>&1 \
  | grep -oE 'Successfully created pod [0-9a-f]+' | awk '{print $NF}')
[ -n "$POD" ] || die "pod create failed — check 'prime pods list'."
log "Pod $POD created.  (terminate with: prime pods terminate $POD)"

# 3. wait for ACTIVE + SSH (tolerant of intermittent API hiccups)
log "Waiting for ACTIVE + SSH…"
HOSTPORT=""
for _ in $(seq 1 90); do
  st=$(prime pods status "$POD" --plain 2>/dev/null) || { sleep 8; continue; }
  s=$(printf '%s\n' "$st" | awk '/^Status /{print $2}')
  c=$(printf '%s\n' "$st" | sed -nE 's/^SSH +//p')
  case "$s" in
    ACTIVE|RUNNING) [ -n "$c" ] && [ "$c" != "N/A" ] && { HOSTPORT="$c"; break; } ;;
    FAILED|TERMINATED|ERROR) die "pod entered $s. Terminate: prime pods terminate $POD" ;;
  esac
  sleep 8
done
[ -n "$HOSTPORT" ] || die "timed out waiting for SSH. Check: prime pods status $POD ; terminate: prime pods terminate $POD"
HOST="${HOSTPORT%% -p *}"; PORT="${HOSTPORT##*-p }"; PORT="${PORT// /}"
log "SSH up: $HOST -p $PORT"

ssh_box(){ ssh -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30 -p "$PORT" "$HOST" "$@"; }

# 4. box clones the repo (token only on this short call; scrubbed right after)
log "Cloning $REPO_SLUG@$BRANCH on the box…"
ssh_box "set -e
  rm -rf '$REMOTE_DIR'
  git clone --depth 1 --branch '$BRANCH' 'https://x-access-token:${TOKEN}@github.com/${REPO_SLUG}.git' '$REMOTE_DIR' 2>&1 | tail -1
  git -C '$REMOTE_DIR' remote set-url origin 'https://github.com/${REPO_SLUG}.git'"

# 5. bootstrap + build (no token in this long-running call)
log "Running setup_dev.sh (PEGAINFER_CUDA_SM=${CUDA_SM:-auto})…"
ssh_box "cd '$REMOTE_DIR' && PEGAINFER_CUDA_SM='$CUDA_SM' bash scripts/setup_dev.sh"

log "✅ Done."
log "   SSH in:    ssh -i $SSH_KEY -p $PORT $HOST"
log "   Terminate: prime pods terminate $POD"
