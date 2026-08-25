#!/usr/bin/env bash
# Maintainer runner for the Kimi-K3 checkpoint-backed gates.
#
# The K3 gates need a real checkpoint and one-to-four otherwise-free GB300s,
# so CI only compiles them. This script owns their execution on a tray: it
# refuses a busy tray (shared machines — someone else's job landing mid-suite
# reads as OOM, not as contention), runs each gate with its documented
# invocation (golden_decode is serial-only: its parallel default oversubscribes
# one device with 13 model loads), and archives every log.
#
# The host has no cargo: gates run as prebuilt test binaries out of
# target/release/deps (build in the kernel-lab container first). The newest
# binary per target is selected; a binary older than HEAD gets a warning, not
# a refusal, because the tree may be dirty in ways that do not touch K3.
#
#   PEGAINFER_K3_TEST_224=<224-expert checkpoint> scripts/k3_gates.sh [filter]
#
# Optional environment:
#   PEGAINFER_K3_WEIGHT_STAGING  the gates load weights through the pinned
#                            double-buffer uploader by default (#964) — same
#                            bytes, ~6x faster full-depth loads on a warm page
#                            cache. Set 0 for the serial pageable-mmap path,
#                            e.g. on a cold network-filesystem first run.
#   PEGAINFER_K3_NCCL_LIB    directory prepended to LD_LIBRARY_PATH (the
#                            bare-host NCCL 2.30.7; must contain the
#                            unversioned libnccl.so symlink — see #810)
#   PEGAINFER_K3_CP_PROMPT   cp_prefill gate prompt length (default 16384;
#                            65536 exercises the 64k single superstep)
#   PEGAINFER_K3_TEST_DSPARK RadixArk drafter dir for the spec_verify draft
#                            lane gates
#   K3_GATES_ALLOW_BUSY=1    skip the idle-tray refusal (you own the overlap)
#   K3_GATES_LOG_DIR         log directory (default /tmp/k3-gates-<timestamp>)
#
# A filter runs the subset of manifest entries whose "<target> <gate>" line
# contains it, e.g. `scripts/k3_gates.sh cp_prefill` or `... oracle`.
set -uo pipefail

die() { echo "k3 gates: $*" >&2; exit 1; }

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root" || die "cannot enter the repository root"

# --- the suite ------------------------------------------------------------
# "<gpus> <target> <gate-or-ALL> [env NAME=VALUE ...]"
# ALL runs the target's whole --ignored set in one process. Everything runs
# --test-threads=1: each K3 gate loads a multi-GiB model, and the harness
# default of one thread per CPU oversubscribes a device (the golden_decode
# suite is 13 loads).
MANIFEST=(
  "1 golden_decode ALL"
  "1 paged_kv ALL"
  "1 spec_verify ALL"
  "4 ep_mega_oracle ep4_mega_matches_ep1_mega env PEGAINFER_K3_LAYERS=4 PEGAINFER_K3_MAX_BATCH=16"
  "4 ep_mega_oracle ep4_mega_is_invariant_to_peer_traffic env PEGAINFER_K3_LAYERS=4 PEGAINFER_K3_MAX_BATCH=16"
  "4 cp_prefill cp4_prefill_matches_cp1"
)

# --- prerequisites --------------------------------------------------------
[ -n "${PEGAINFER_K3_TEST_224:-}" ] || die "PEGAINFER_K3_TEST_224 is unset"
[ -d "$PEGAINFER_K3_TEST_224" ] || die "checkpoint $PEGAINFER_K3_TEST_224 does not exist"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable — run this on the tray"

if [ -n "${PEGAINFER_K3_NCCL_LIB:-}" ]; then
  [ -e "$PEGAINFER_K3_NCCL_LIB/libnccl.so" ] || die \
    "$PEGAINFER_K3_NCCL_LIB has no unversioned libnccl.so symlink — cudarc \
dlopens that name, and falling back to the system NCCL mixes two libraries (#810)"
  export LD_LIBRARY_PATH="$PEGAINFER_K3_NCCL_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Shared trays: Slurm's ledger and the GPUs' actual occupancy are separate
# books, so the only trustworthy check is nvidia-smi on the spot. Residual
# memory means a parked job that can wake mid-suite.
busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
  awk -F', *' '$2 > 1024 { printf "  GPU %s: %s MiB\n", $1, $2 }')
if [ -n "$busy" ] && [ "${K3_GATES_ALLOW_BUSY:-0}" != 1 ]; then
  die "the tray is not idle (K3_GATES_ALLOW_BUSY=1 to override):"$'\n'"$busy"
fi

# --- binary discovery -----------------------------------------------------
# Newest executable per target. The hash suffix changes with every rebuild
# and stale siblings accumulate, so "the" binary is a moving name.
find_binary() {
  local target=$1 found
  found=$(find target/release/deps -maxdepth 1 -name "$target-*" ! -name '*.d' \
    -type f -perm -u+x -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2)
  [ -n "$found" ] || die \
    "no $target binary under target/release/deps — build it in kernel-lab: \
PEGAINFER_CUDA_SM=103 cargo test --release -p pegainfer-k3 --no-run"
  echo "$found"
}

head_epoch=$(git log -1 --format=%ct 2>/dev/null || echo 0)

# --- selection ------------------------------------------------------------
filter=${1:-}
selected=()
for entry in "${MANIFEST[@]}"; do
  [ -z "$filter" ] || [[ $entry == *"$filter"* ]] || continue
  selected+=("$entry")
done
[ ${#selected[@]} -gt 0 ] || die "filter ${filter:-<none>} selected no gate"

log_dir=${K3_GATES_LOG_DIR:-/tmp/k3-gates-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$log_dir" || die "cannot create log directory $log_dir"

echo "k3 gates: source $(git rev-parse HEAD)$([ -n "$(git status --porcelain)" ] && echo ' (dirty)')"
echo "k3 gates: checkpoint $PEGAINFER_K3_TEST_224"
echo "k3 gates: logs in $log_dir"
echo "k3 gates: ${#selected[@]} selected of ${#MANIFEST[@]} in the manifest"

# --- execution: one gate per process, serialized --------------------------
completed=0
failed=()
for entry in "${selected[@]}"; do
  read -r gpus target gate rest <<<"$entry"
  binary=$(find_binary "$target") || exit 1
  if [ "$(stat -c %Y "$binary")" -lt "$head_epoch" ]; then
    echo "k3 gates: WARNING $binary is older than HEAD — rebuild if K3 changed"
  fi
  gate_env=()
  [ "${rest:-}" = "" ] || { read -r _env_kw envs <<<"$rest"; read -ra gate_env <<<"$envs"; }
  args=(--ignored --test-threads=1 --nocapture)
  [ "$gate" = ALL ] || args=("$gate" --exact "${args[@]}")
  label="$target/${gate/ALL/all}"
  log="$log_dir/${label//\//-}.log"
  echo "--- $label (${gpus} GPU)"
  if env "${gate_env[@]}" "$binary" "${args[@]}" >"$log" 2>&1; then
    tail -3 "$log" | sed 's/^/    /'
    completed=$((completed + 1))
  else
    echo "k3 gates: FAILED $label — full log: $log"
    tail -15 "$log" | sed 's/^/    /'
    failed+=("$label")
  fi
done

echo "k3 gates: selected ${#selected[@]}, completed $completed, failed ${#failed[@]}"
if [ ${#failed[@]} -gt 0 ]; then
  printf 'k3 gates: FAILED %s\n' "${failed[@]}"
  exit 1
fi
echo "k3 gates: all selected gates completed"
