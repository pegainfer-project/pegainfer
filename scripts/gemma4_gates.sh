#!/usr/bin/env bash
# Maintainer runner for the Gemma 4 checkpoint-backed gates and the Gemma
# contracts that live in the kernels and frontend crates.
#
# The checkpoint-backed gates need real weights, fixtures and a device; the
# kernels contracts need only a device. This script owns their execution:
# it refuses to start unless every prerequisite is present, it holds each
# discovered gate set against the manifest here so a gate cannot quietly leave
# the suite. It claims one physical device for the suite's lifetime and runs one
# gate per process — repeated checkpoint loads in one test binary exhaust a
# 48 GiB card.
#
#   PEGAINFER_TEST_MODEL_PATH=<dense-checkpoint> \
#     PEGAINFER_NVFP4_MODEL=<routed-checkpoint> \
#     [PEGAINFER_GATE_GPU=<index-or-UUID-or-pair>] scripts/gemma4_gates.sh [filter]
#
# A filter runs the subset of manifest gates whose names contain it; the
# membership check still covers the whole manifest.
set -uo pipefail

CRATE=pegainfer-gemma4
KERNELS_CRATE=pegainfer-kernels
FEATURE=gemma4
GPU_LOCK_ROOT=/tmp

# The maintained suite, grouped by the invariant each gate owns. Adding a
# gate to the crate without adding it here fails the membership check, which
# covers every target an ignored test can live in — the library and each
# integration binary — so a gate cannot hide in one the runner never looks at.
#
# Each entry is "<needs> <gate>". `needs` is a comma-separated subset of
#   gpu         a CUDA device
#   ckpt        PEGAINFER_TEST_MODEL_PATH and the config it holds
#   moeckpt     PEGAINFER_NVFP4_MODEL and the routed config it holds
#   prompts     the generate fixture, read for its prompts only
#   fixtures    all four tensor fixtures, held against the checkpoint's digests
# and a run demands only the union over the gates it selects, so a filter can
# run a gate without producing the whole suite's prerequisites.
GATES_NUMERIC_PARITY=(
  "gpu,ckpt,fixtures serve::oracle::fp8_argmax_agreement_meets_the_bf16_floor"
)
GATES_ADMISSION=(
  "gpu,ckpt,prompts serve::oracle::mixed_step_matches_serial"
  "gpu,ckpt,prompts serve::oracle::fp8_mixed_walk_holds_its_structure"
  "gpu,ckpt,prompts engine::lane_gates_walk::the_gathered_walk_does_not_depend_on_its_batching"
  "gpu,ckpt,prompts engine::lane_gates_walk::the_gathered_transient_leaves_headroom"
  "gpu,ckpt,prompts engine::lane_gates_walk::the_served_bounds_provision_a_split_walk"
)
# These production contracts apply to both checkpoint geometries. They stay
# unique in the ignored-test manifest and expand into two execution profiles.
GATES_DENSE_AND_ROUTED=(
  "gpu,ckpt,fixtures serve::oracle::context_waypoints_match_hf"
  "gpu,ckpt,fixtures serve::oracle::greedy_matches_hf_generate"
  "gpu,ckpt engine::lane_gates_lifecycle::the_shared_lane_lifecycle_completes"
  "gpu,ckpt engine::lane_gates_lifecycle::the_green_lane_lifecycle_completes"
  "gpu,ckpt serve::oracle::overlapped_prefill_matches_the_sync_step"
  "gpu,ckpt serve::oracle::a_ragged_batch_does_not_depend_on_row_order"
)
# The idle-refill gate borrows the generate fixture's prompts; the roster-edge
# gates build their own.
GATES_SERVING_CONTRACT=(
  "gpu,ckpt engine::lane_gates_lifecycle::the_gathered_lifecycle_completes"
  "gpu,ckpt engine::lane_gates_roster::the_coalesce_door_releases_one_admission_burst"
  "gpu,ckpt engine::lane_gates_roster::the_raised_ceiling_and_slots_hold_at_the_roster_edge"
  "gpu,ckpt engine::lane_gates_roster::the_full_roster_keeps_its_pipeline_under_a_queue"
  "gpu,ckpt,prompts engine::lane_gates_roster::an_idle_refill_matches_a_fresh_engine"
  "gpu,ckpt engine::lane_gates_lifecycle::the_raise_reaches_the_frontend"
  "gpu,ckpt,fixtures engine::lane_gates_logprobs::prompt_scores_bypass_the_prefix_cache_and_match_teacher_forced_decode"
  "gpu,ckpt engine::lane_gates_logprobs::a_scored_prompt_beside_a_live_batch_is_prefilled_whole"
  "gpu,ckpt engine::lane_gates_logprobs::a_low_slot_chunked_pool_scores_up_to_what_it_holds"
  "gpu,ckpt engine::lane_gates_logprobs::a_failed_prefill_costs_that_request_not_the_engine"
)
GATES_KV_AND_LANES=(
  "gpu,ckpt,fixtures serve::oracle::incremental_serving_matches_recompute"
  "gpu,ckpt serve::oracle::prefix_restore_matches_cold_path"
)
# The replacement global-attention kernel only exists in a build that had
# TileLang or a pre-generated directory, and the gate says so rather than
# passing quietly; it is listed apart because that is a build property, not a
# prerequisite this script can arrange.
GATES_TILELANG_GLOBAL=(
  "gpu,ckpt,prompts,tlgeom serve::oracle::the_replacement_global_kernel_matches_the_incumbent"
  "gpu,ckpt,prompts,tlgeom serve::oracle::the_replacement_global_decode_matches_the_incumbent"
  "gpu,ckpt,prompts,tlgeom serve::oracle::the_folded_pool_matches_the_split_one"
  "gpu,ckpt,tlgeom engine::lane_gates_roster::the_full_roster_serves_through_the_generated_kernels"
)
# The disagreeing-config gate deliberately fails before any device is opened.
GATES_LOADER=(
  "ckpt weights::load::tests::a_disagreeing_config_names_every_faulty_tensor"
)
GATES_DEVICE=(
  "gpu kv::tests::admission_is_atomic_across_pools"
)
GATES_ROUTED=(
  "gpu,moeckpt moe::tests::the_routed_block_matches_the_reference_formulas"
)
# These live in pegainfer-kernels under the gemma4 feature and need no checkpoint.
GATES_KERNELS=(
  "gpu ops::elementwise::tests::the_suppression_mask_writes_only_the_ids_it_is_given"
  "gpu ops::gemma4::tests::router_topk_matches_the_exact_128_expert_contract"
  "gpu ops::norm::parity::the_dual_norm_matches_two_standalone_norms"
  "gpu ops::norm::parity::the_layer_tail_matches_its_parts"
  "gpu ops::norm::parity::the_epilogue_norm_pair_matches_its_parts"
  "gpu ops::norm::parity::the_moe_combine_tail_matches_its_parts"
)
GATES_KERNELS_HD256_FP8_POOL=(
  "gpu fp8_prep_stores_exact_bytes_at_layout_offsets"
  "gpu fp8_decode_prep_stores_exact_bytes_at_layout_offsets"
  "gpu fp8_window_read_matches_bf16_for_exact_values"
  "gpu fp8_finite_window_read_matches_bf16_and_changes_the_result"
  "gpu varied_fp8_window_read_is_geometry_invariant_for_the_probed_row"
  "gpu decode_wrapper_without_fp8_twin_refuses_e4m3"
  "gpu the_generated_windowed_prefill_refuses_e4m3"
)
GATES_TENSOR_PARALLEL=(
  "tp2 engine::lane_gates_tp::an_aborted_communicator_refuses_the_next_reduction"
  "tp2,ckpt engine::lane_gates_tp::the_two_rank_engine_matches_one_rank"
  "tp2,ckpt engine::lane_gates_tp::a_failed_rank_zero_segment_stops_the_engine_at_two_ranks"
  "tp2,ckpt engine::lane_gates_tp::the_two_rank_engine_scores_prompt_logprobs"
  "tp2,ckpt engine::lane_gates_tp::the_two_rank_engine_is_prefix_consistent"
  "tp2,ckpt engine::lane_gates_tp::the_two_rank_engine_is_prefix_consistent_within_a_single_rank_control"
  "tp2,ckpt,hf31 engine::lane_gates_tp::the_two_rank_engine_matches_the_hf_reference"
)
MANIFEST_LIB=(
  "${GATES_NUMERIC_PARITY[@]}"
  "${GATES_ADMISSION[@]}"
  "${GATES_DENSE_AND_ROUTED[@]}"
  "${GATES_SERVING_CONTRACT[@]}"
  "${GATES_KV_AND_LANES[@]}"
  "${GATES_LOADER[@]}"
  "${GATES_DEVICE[@]}"
  "${GATES_ROUTED[@]}"
  "${GATES_TILELANG_GLOBAL[@]}"
  "${GATES_TENSOR_PARALLEL[@]}"
)
GATES_FP8_PROFILE=(
  "serve::oracle::context_waypoints_match_hf"
  "serve::oracle::greedy_matches_hf_generate"
  "serve::oracle::a_ragged_batch_does_not_depend_on_row_order"
  "serve::oracle::incremental_serving_matches_recompute"
  "serve::oracle::overlapped_prefill_matches_the_sync_step"
)

# Integration gates live in their own binaries, which `--lib` cannot see. One
# array per binary, named GATES_<TARGET>; the target list itself is held
# against `tests/*.rs` below, so a new binary fails the check rather than
# going unowned.
INTEGRATION_TARGETS=()
# Frontend-owned integration binaries: the chat-render parity gate rides the
# frontend crate (which owns the render path) but stays under this runner's
# manifest and execution ownership.
FRONTEND_CRATE=pegainfer-frontend
FRONTEND_INTEGRATION_TARGETS=(gemma4_tokenizer_parity)
# shellcheck disable=SC2034
GATES_GEMMA4_TOKENIZER_PARITY=(
  "ckpt,chatgolden string_form_chat_renders_match_hf_reference"
)

# The fixture set is named by the checkpoint it was dumped from; the
# committed set is 12b. Another tag selects fixtures dumped for another
# checkpoint under the same names, and the gates are told where they are.
FIXTURE_TAG=${PEGAINFER_GEMMA4_FIXTURE_TAG:-12b}
[[ $FIXTURE_TAG =~ ^[a-z0-9]+$ ]] || { echo "gemma4 gates: PEGAINFER_GEMMA4_FIXTURE_TAG must be alphanumeric" >&2; exit 1; }
# The committed chat reference predates the tag, so 12b keeps its own name.
chat_golden_for() {
  if [ "$1" = 12b ]; then
    echo "test_data/gemma4-tokenizer-golden.json"
  else
    echo "test_data/gemma4-$1-tokenizer-golden.json"
  fi
}
fixtures_for() {
  printf '%s\n' \
    "test_data/gemma4-$1-hf-golden.safetensors" \
    "test_data/gemma4-$1-hf-window-golden.safetensors" \
    "test_data/gemma4-$1-hf-longctx-golden.safetensors" \
    "test_data/gemma4-$1-generate.safetensors"
}
CHAT_GOLDEN=$(chat_golden_for "$FIXTURE_TAG")
mapfile -t FIXTURES < <(fixtures_for "$FIXTURE_TAG")
PROMPT_FIXTURE=${FIXTURES[3]}

# The routed profile compares a different checkpoint against its own
# references, so it needs a set of its own; the tag above names the dense
# one. Unset means the routed profile runs only the gates that need no
# fixture, which is what it did before this existed.
ROUTED_FIXTURE_TAG=${PEGAINFER_GEMMA4_ROUTED_FIXTURE_TAG:-}

die() { echo "gemma4 gates: $*" >&2; exit 1; }

gate_is_in() {
  local wanted=$1 candidate
  shift
  for candidate in "$@"; do
    [ "$candidate" = "$wanted" ] && return 0
  done
  return 1
}

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root" || die "cannot enter the repository root"
export PEGAINFER_GEMMA4_GOLDEN=$root/${FIXTURES[0]}
export PEGAINFER_GEMMA4_WINDOW_GOLDEN=$root/${FIXTURES[1]}
export PEGAINFER_GEMMA4_LONGCTX_GOLDEN=$root/${FIXTURES[2]}
export PEGAINFER_GEMMA4_GENERATE=$root/${FIXTURES[3]}
export PEGAINFER_GEMMA4_CHAT_GOLDEN=$root/$CHAT_GOLDEN

[ -z "${PEGAINFER_KV_FP8+x}" ] || die \
  "PEGAINFER_KV_FP8 is ambient; PEGAINFER_GATE_STORAGE is the only storage switch"
gate_storage=${PEGAINFER_GATE_STORAGE-bf16}
case "$gate_storage" in
  bf16) ;;
  fp8) export PEGAINFER_KV_FP8=local ;;
  *) die "PEGAINFER_GATE_STORAGE must be unset, bf16, or fp8" ;;
esac

# --- prerequisites, one refusal per tier ----------------------------------
# Each is demanded only when a selected gate declares it, so a focused run
# carries the cost of what it runs: the device-only gates need no checkpoint,
# and the chat-render gate needs no card.
ckpt=
moe_ckpt=
gpu_uuid=
gpu_lock_fd=

# Claim `count` devices for this run and export them as one
# `CUDA_VISIBLE_DEVICES` list. The single-device gates and the two-rank gate
# draw from the same reservation, so a full-suite run never tries to lock a
# device it already holds, and a pair may be named in `PEGAINFER_GATE_GPU`.
# `PEGAINFER_GATE_GPU` names the devices ("a" or "a,b"); otherwise they come
# from `CUDA_VISIBLE_DEVICES`, else the first `count` visible devices.
require_devices() {
  local count=$1
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable, so no devices can be claimed"
  command -v flock >/dev/null 2>&1 || die "flock is unavailable, so device ownership cannot be enforced"

  local selector=${PEGAINFER_GATE_GPU:-}
  local explicit=0
  if [ -z "$selector" ] && [ "${CUDA_VISIBLE_DEVICES+x}" = x ]; then
    [ -n "$CUDA_VISIBLE_DEVICES" ] || die \
      "CUDA_VISIBLE_DEVICES is empty; set PEGAINFER_GATE_GPU to claim a device"
    selector=$CUDA_VISIBLE_DEVICES
  fi
  [ -n "$selector" ] && explicit=1
  local picks=()
  [ -z "$selector" ] || IFS=',' read -r -a picks <<<"$selector"
  if [ ${#picks[@]} -lt "$count" ]; then
    # An explicit selector is never substituted: naming one device (as a SLURM
    # job does) means "use this one", so a two-rank run drops its two-rank gates
    # rather than take a different device; naming none at all is a mistake.
    if [ "$explicit" -eq 1 ]; then
      [ ${#picks[@]} -gt 0 ] || die "the device selector names no device"
      return 1
    fi
    picks=()
    mapfile -t picks < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
  fi
  # Not enough devices and nobody named them: `return 1` lets a run that can drop
  # its two-rank gates carry on rather than fail.
  [ ${#picks[@]} -ge "$count" ] || return 1

  local uuids=() rows=() row compute_mode sel uuid seen
  for sel in "${picks[@]:0:$count}"; do
    mapfile -t rows < <(
      nvidia-smi -i "$sel" --query-gpu=uuid,compute_mode --format=csv,noheader 2>/dev/null
    )
    [ ${#rows[@]} -eq 1 ] || die "device selector $sel does not resolve to one GPU"
    row=${rows[0]}
    uuid=${row%%,*}
    uuid=${uuid//[[:space:]]/}
    compute_mode=${row#*,}
    compute_mode=${compute_mode#"${compute_mode%%[![:space:]]*}"}
    compute_mode=${compute_mode%"${compute_mode##*[![:space:]]}"}
    [ "$compute_mode" != Prohibited ] || die "GPU $uuid prohibits compute contexts"
    [[ $uuid =~ ^[A-Za-z0-9._:/-]+$ ]] || die "nvidia-smi returned an unsafe GPU identity"
    for seen in "${uuids[@]:-}"; do
      [ "$seen" != "$uuid" ] || die "device selector names $uuid twice"
    done
    uuids+=("$uuid")
  done

  gpu_uuid=$(IFS=,; printf '%s' "${uuids[*]}")
  export CUDA_VISIBLE_DEVICES=$gpu_uuid
  local lock_fds=() fd uuid2 lock_key lock_path
  for uuid2 in "${uuids[@]}"; do
    lock_key=${uuid2//\//_}
    lock_key=${lock_key//:/_}
    lock_path=$GPU_LOCK_ROOT/pegainfer-gemma4-gates-$lock_key.lock
    if (umask 022; set -o noclobber; : >"$lock_path") 2>/dev/null; then
      :
    elif [ ! -e "$lock_path" ]; then
      die "cannot create device lock $lock_path"
    fi
    # A read-only descriptor lets separate Unix accounts lock the same inode.
    exec {fd}<"$lock_path" || die "cannot open device lock $lock_path"
    flock -n "$fd" || die "GPU $uuid2 is already owned by another Gemma 4 gate runner"
    lock_fds+=("$fd")
  done
  gpu_lock_fd=${lock_fds[0]}
  echo "gemma4 gates: claimed $count GPU(s) $gpu_uuid"
  echo "gemma4 gates: storage profile $gate_storage"
  if [ -n "${PEGAINFER_KV_FP8:-}" ]; then
    echo "gemma4 gates: PEGAINFER_KV_FP8=$PEGAINFER_KV_FP8"
  fi
}

# A config alone is not a checkpoint. This mirrors the loader's discovery
# (`load_shard_info`): a single-file model.safetensors, else every shard the
# safetensors index names, so an incomplete download stops the suite here
# rather than inside its first 12B load.
require_weights() {
  local dir=$1 label=$2
  [ -d "$dir" ] || die "$label directory $dir does not exist"
  [ -f "$dir/config.json" ] || die "$dir has no config.json"
  python3 - "$dir" "$label" <<'PY' || die "$label weight preflight failed"
import json, os, sys

ckpt, label = sys.argv[1], sys.argv[2]
if os.path.isfile(os.path.join(ckpt, "model.safetensors")):
    raise SystemExit(0)
index = os.path.join(ckpt, "model.safetensors.index.json")
if not os.path.isfile(index):
    raise SystemExit(f"{label} {ckpt}: neither model.safetensors nor model.safetensors.index.json")
with open(index) as fh:
    shards = sorted(set(json.load(fh)["weight_map"].values()))
missing = [s for s in shards if not os.path.isfile(os.path.join(ckpt, s))]
if missing:
    raise SystemExit(f"{label} {ckpt}: the index names {len(shards)} shards; missing {missing}")
PY
}

require_ckpt() {
  [ -n "${PEGAINFER_TEST_MODEL_PATH:-}" ] || die "PEGAINFER_TEST_MODEL_PATH is unset"
  ckpt=$PEGAINFER_TEST_MODEL_PATH
  require_weights "$ckpt" checkpoint
}

require_moeckpt() {
  [ -n "${PEGAINFER_NVFP4_MODEL:-}" ] || die "PEGAINFER_NVFP4_MODEL is unset"
  moe_ckpt=$PEGAINFER_NVFP4_MODEL
  require_weights "$moe_ckpt" "routed checkpoint"
}

require_prompts() {
  [ -f "$PROMPT_FIXTURE" ] || die "fixture $PROMPT_FIXTURE is missing (dump it on the test box first)"
}


# A checkpoint and the fixture set dumped from it, held together. The
# fixtures pin the checkpoint they came from; the gates assert it per-run, but
# a mismatch should stop the suite before the first load.
check_fixture_set() {
  local what=$1 against=$2
  shift 2
  local fixture
  for fixture in "$@"; do
    [ -f "$fixture" ] || die "$what fixture $fixture is missing (dump it on the test box first)"
  done
  python3 - "$against" "$@" <<'PY' || die "$what fixture metadata preflight failed"
import hashlib, json, os, struct, sys

ckpt, fixtures = sys.argv[1], sys.argv[2:]

def manifest(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        meta = json.loads(fh.read(n)).get("__metadata__") or {}
    if len(meta) != 1:
        raise SystemExit(f"{path}: expected exactly one metadata key, found {sorted(meta)}")
    return json.loads(next(iter(meta.values())))

base = manifest(fixtures[0])
revision = base.get("revision")
if not revision:
    raise SystemExit(f"{fixtures[0]}: manifest carries no revision")
for path in fixtures[1:]:
    other = manifest(path).get("revision")
    if other != revision:
        raise SystemExit(f"{path}: revision {other} does not match {revision}")

digests = base.get("file_sha256") or {}
if not digests:
    raise SystemExit(f"{fixtures[0]}: manifest carries no file_sha256 block")
# The dumper's convention, mirrored from the crate's own checker: a
# "<file>#header" key hashes the safetensors header alone — the tensor
# layout without the 22 GiB payload — and a plain key hashes the whole file.
for name, want in digests.items():
    header_only = name.endswith("#header")
    filename = name[: -len("#header")] if header_only else name
    target = os.path.join(ckpt, filename)
    if not os.path.exists(target):
        raise SystemExit(f"checkpoint is missing {filename}")
    with open(target, "rb") as fh:
        if header_only:
            length = struct.unpack("<Q", fh.read(8))[0]
            payload = fh.read(length)
        else:
            payload = fh.read()
    if hashlib.sha256(payload).hexdigest() != want:
        raise SystemExit(f"{name}: checkpoint digest does not match the fixture's")
print(f"preflight: {len(fixtures)} fixtures agree on revision {revision[:12]}")
PY
}

require_fixtures() {
  require_ckpt
  check_fixture_set dense "$ckpt" "${FIXTURES[@]}"
}

# The routed profile's own set, named by its own tag.
routed_fixtures=()
require_routedfixtures() {
  require_moeckpt
  [ -n "$ROUTED_FIXTURE_TAG" ] || die \
    "a routed fixture gate was selected but PEGAINFER_GEMMA4_ROUTED_FIXTURE_TAG is unset"
  mapfile -t routed_fixtures < <(fixtures_for "$ROUTED_FIXTURE_TAG")
  check_fixture_set routed "$moe_ckpt" "${routed_fixtures[@]}"
}

# The generated kernels are compiled for one attention geometry; a gate that
# declares `tlgeom` is dropped by name where the checkpoint has another.
tlgeom_mismatch=""
require_tlgeom() {
  require_ckpt
  local generator=pegainfer-gemma4/kernels/generate.py
  tlgeom_mismatch=$(python3 - "$generator" "$ckpt" <<'PY'
import json, os, re, sys

generator, ckpt = sys.argv[1], sys.argv[2]
source = open(generator).read()
def const(name):
    match = re.search(rf"^{name} = (\d+)$", source, re.M)
    if not match:
        raise SystemExit(f"{generator} no longer states {name}")
    return int(match.group(1))

heads, groups, head_dim = const("HEADS"), const("GROUPS"), const("HEAD_DIM")
with open(os.path.join(ckpt, "config.json")) as fh:
    config = json.load(fh)
text = config.get("text_config", config)
theirs = (
    text["num_attention_heads"],
    text["num_global_key_value_heads"],
    text["global_head_dim"],
)
if theirs != (heads, groups and heads // groups, head_dim):
    print(
        f"the kernels are compiled for {heads} query heads over {heads // groups} "
        f"at head dim {head_dim}; this checkpoint's global family is "
        f"{theirs[0]} over {theirs[1]} at {theirs[2]}"
    )
PY
  ) || die "the TileLang geometry preflight failed"
}

require_chatgolden() {
  [ -f "$CHAT_GOLDEN" ] || die "reference $CHAT_GOLDEN is missing (dump it on the test box first)"
}


# --- membership: the crate's ignored set must be exactly the manifest ------
ignored_in() {
  local crate=$1
  shift
  local feat=(--features "$FEATURE")
  [ "$crate" = "$FRONTEND_CRATE" ] && feat=()
  cargo test --release -p "$crate" "${feat[@]}" "$@" -- \
    --ignored --list 2>/dev/null | sed -n 's/^\(.*\): test$/\1/p' | sort
}

listed_in() {
  local crate=$1
  shift
  cargo test --release -p "$crate" --features "$FEATURE" "$@" -- \
    --list 2>/dev/null | sed -n 's/^\(.*\): test$/\1/p' | sort
}

check_membership() {
  local what=$1 listing=$2 expected=$3 missing extra
  missing=$(comm -13 <(printf '%s\n' "$listing") <(printf '%s\n' "$expected"))
  extra=$(comm -23 <(printf '%s\n' "$listing") <(printf '%s\n' "$expected"))
  [ -z "$missing" ] || die "the manifest names $what gates that do not exist:"$'\n'"$missing"
  [ -z "$extra" ] || die "$what has ignored gates the manifest does not name:"$'\n'"$extra"
}

lib_listing=$(ignored_in "$CRATE" --lib)
[ -n "$lib_listing" ] || die "could not list the library's ignored gates"
lib_names=()
for entry in "${MANIFEST_LIB[@]}"; do lib_names+=("${entry##* }"); done
check_membership "library" "$lib_listing" "$(printf '%s\n' "${lib_names[@]}" | sort)"
for gate in "${GATES_FP8_PROFILE[@]}"; do
  gate_is_in "$gate" "${lib_names[@]}" || die \
    "the fp8 storage profile names a gate outside the library manifest: $gate"
done

kernels_listing=$(ignored_in "$KERNELS_CRATE" --lib)
[ -n "$kernels_listing" ] || die "could not list the kernels library's ignored gates"
kernels_names=()
for entry in "${GATES_KERNELS[@]}"; do kernels_names+=("${entry##* }"); done
check_membership "kernels library" "$kernels_listing" \
  "$(printf '%s\n' "${kernels_names[@]}" | sort)"

kernels_pool_listing=$(listed_in "$KERNELS_CRATE" --test hd256_fp8_pool)
[ -n "$kernels_pool_listing" ] || die "could not list the kernels hd256_fp8_pool integration gates"
kernels_pool_names=()
for entry in "${GATES_KERNELS_HD256_FP8_POOL[@]}"; do kernels_pool_names+=("${entry##* }"); done
check_membership "kernels integration binary hd256_fp8_pool" "$kernels_pool_listing" \
  "$(printf '%s\n' "${kernels_pool_names[@]}" | sort)"

# The integration binaries the crate actually has, so adding one without a
# manifest entry fails here instead of leaving its gates unowned.
discovered=$(find "$CRATE/tests" -maxdepth 1 -name '*.rs' -exec basename {} .rs \; 2>/dev/null | sort)
declared=$(printf '%s\n' "${INTEGRATION_TARGETS[@]}" | sort)
[ "$discovered" = "$declared" ] || die \
  "integration binaries disagree with INTEGRATION_TARGETS:"$'\n'"on disk: $discovered"$'\n'"declared: $declared"

discovered=$(find "$FRONTEND_CRATE/tests" -maxdepth 1 -name '*.rs' -exec basename {} .rs \; 2>/dev/null | sort)
declared=$(printf '%s\n' "${FRONTEND_INTEGRATION_TARGETS[@]}" | sort)
[ "$discovered" = "$declared" ] || die \
  "frontend integration binaries disagree with FRONTEND_INTEGRATION_TARGETS:"$'\n'"on disk: $discovered"$'\n'"declared: $declared"

all_gates=()
append_gate() {
  local needs=$1 target=$2 gate=$3 profile=${4:-}
  if [ -z "$profile" ]; then
    case ",$needs," in
      *,moeckpt,*) profile=routed ;;
      *,ckpt,*) profile=dense ;;
      *) profile=device ;;
    esac
  fi
  all_gates+=("$needs|$target|$profile|$gate")
}

for entry in "${MANIFEST_LIB[@]}"; do
  append_gate "${entry%% *}" lib "${entry##* }"
done
for target in "${INTEGRATION_TARGETS[@]}"; do
  group="GATES_$(printf '%s' "$target" | tr '[:lower:]' '[:upper:]')[@]"
  target_names=()
  for entry in "${!group}"; do target_names+=("${entry##* }"); done
  check_membership "integration binary $target" \
    "$(ignored_in "$CRATE" --test "$target")" "$(printf '%s\n' "${target_names[@]}" | sort)"
  for entry in "${!group}"; do
    append_gate "${entry%% *}" "$target" "${entry##* }"
  done
done
for target in "${FRONTEND_INTEGRATION_TARGETS[@]}"; do
  group="GATES_$(printf '%s' "$target" | tr '[:lower:]' '[:upper:]')[@]"
  target_names=()
  for entry in "${!group}"; do target_names+=("${entry##* }"); done
  check_membership "frontend integration binary $target" \
    "$(ignored_in "$FRONTEND_CRATE" --test "$target")" "$(printf '%s\n' "${target_names[@]}" | sort)"
  for entry in "${!group}"; do
    append_gate "${entry%% *}" "frontend:$target" "${entry##* }"
  done
done
for entry in "${GATES_KERNELS[@]}"; do
  append_gate "${entry%% *}" kernels "${entry##* }"
done
for entry in "${GATES_KERNELS_HD256_FP8_POOL[@]}"; do
  append_gate "${entry%% *}" kernels:hd256_fp8_pool "${entry##* }"
done
manifest_gate_count=${#all_gates[@]}
for entry in "${GATES_DENSE_AND_ROUTED[@]}"; do
  routed_needs=${entry%% *}
  routed_needs=${routed_needs/ckpt/moeckpt}
  routed_needs=${routed_needs/fixtures/routedfixtures}
  append_gate "$routed_needs" lib "${entry##* }" routed
done

filter=${1:-}
selected=()
for entry in "${all_gates[@]}"; do
  if [ "$gate_storage" = fp8 ]; then
    gate=${entry##*|}
    gate_is_in "$gate" "${GATES_FP8_PROFILE[@]}" || continue
  fi
  [ -z "$filter" ] || [[ ${entry##*|} == *"$filter"* ]] || continue
  selected+=("$entry")
done
[ ${#selected[@]} -gt 0 ] || die "filter ${filter:-<none>} selected no gate"

# Without a routed set those gates leave the run by name, the way a geometry
# mismatch does, so the rest of the suite still runs whole.
if [ -z "$ROUTED_FIXTURE_TAG" ]; then
  kept=()
  for entry in "${selected[@]}"; do
    case ",${entry%%|*}," in
      *,routedfixtures,*)
        echo "gemma4 gates: not selected, PEGAINFER_GEMMA4_ROUTED_FIXTURE_TAG is unset: ${entry##*|}" ;;
      *) kept+=("$entry") ;;
    esac
  done
  selected=("${kept[@]}")
  [ ${#selected[@]} -gt 0 ] || die "every selected gate needed the routed fixture set"
fi

# The 31B Hugging Face fixture is a separate artifact from its checkpoint, so a
# gate that compares against it leaves the run by name when it is not selected.
# Only that the variable is set is checked here: the fixture's own manifest is
# held against the checkpoint under test by `golden_bytes`, so checking the file
# name on top of that could not catch a mismatched dump — only refuse a valid one
# that happens to be named otherwise.
hf31_mismatch=
require_hf31() {
  if [ -z "${PEGAINFER_GEMMA4_GOLDEN:-}" ]; then
    hf31_mismatch="PEGAINFER_GEMMA4_GOLDEN is unset (point it at the 31B HF fixture)"
  fi
}

# --- prerequisites: the union over what this run selected, and no more -----
needs=" "
for entry in "${selected[@]}"; do needs="$needs${entry%%|*} "; done
needs=" ${needs//,/ } "
demanded=""
# One reservation covers both the single-device gates and the two-rank gate:
# claim two devices when a tp2 gate is selected, one otherwise. Claiming per
# gate type would try to lock a device this same process already holds.
device_count=0
case "$needs" in
  *" tp2 "*) device_count=2 ;;
  *" gpu "*) device_count=1 ;;
esac
if [ "$device_count" -gt 0 ] && ! require_devices "$device_count"; then
  # Two devices were wanted and only one is visible; a one-GPU host should still
  # run the rest, so drop the two-rank gates by name and claim the single one.
  [ "$device_count" -eq 2 ] || die "a device is required but none could be claimed"
  kept=(); dropped=0
  for entry in "${selected[@]}"; do
    case ",${entry%%|*}," in
      *,tp2,*) dropped=$((dropped + 1)) ;;
      *) kept+=("$entry") ;;
    esac
  done
  selected=("${kept[@]}")
  [ ${#selected[@]} -gt 0 ] || die \
    "only two-rank gates were selected and a second device is unavailable"
  echo "gemma4 gates: not selected, a second device is unavailable: $dropped two-rank gate(s)"
  needs=" "
  for entry in "${selected[@]}"; do needs="$needs${entry%%|*} "; done
  needs=" ${needs//,/ } "
  require_devices 1 || die "a device is required but none could be claimed"
fi
for want in gpu tp2 ckpt moeckpt prompts fixtures routedfixtures chatgolden tlgeom hf31; do
  case "$needs" in
    *" $want "*)
      case "$want" in
        gpu|tp2) : ;;
        *) "require_$want" ;;
      esac
      demanded="$demanded $want"
      ;;
  esac
done
echo "gemma4 gates: prerequisites$demanded"

# A gate that declares `tlgeom` has nothing to compare on a checkpoint the
# kernels were not compiled for, so it leaves the run by name.
if [ -n "$tlgeom_mismatch" ]; then
  kept=()
  for entry in "${selected[@]}"; do
    case ",${entry%%|*}," in
      *,tlgeom,*) echo "gemma4 gates: not selected, $tlgeom_mismatch: ${entry##*|}" ;;
      *) kept+=("$entry") ;;
    esac
  done
  selected=("${kept[@]}")
  [ ${#selected[@]} -gt 0 ] || die "every selected gate needed the kernels' own geometry"
fi

# And the same for a gate that needs the 31B Hugging Face fixture.
if [ -n "$hf31_mismatch" ]; then
  kept=()
  for entry in "${selected[@]}"; do
    case ",${entry%%|*}," in
      *,hf31,*) echo "gemma4 gates: not selected, $hf31_mismatch: ${entry##*|}" ;;
      *) kept+=("$entry") ;;
    esac
  done
  selected=("${kept[@]}")
  [ ${#selected[@]} -gt 0 ] || die "every selected gate needed the 31B HF fixture"
fi

echo "gemma4 gates: source $(git rev-parse HEAD)$([ -n "$(git status --porcelain)" ] && echo ' (dirty)')"
[ -z "$ckpt" ] || echo "gemma4 gates: checkpoint $ckpt"
[ -z "$moe_ckpt" ] || echo "gemma4 gates: routed checkpoint $moe_ckpt"
echo "gemma4 gates: ${#selected[@]} selected executions from $manifest_gate_count manifest gates"
for entry in "${selected[@]}"; do
  IFS='|' read -r _needs _target profile gate <<<"$entry"
  printf '  [%s] %s\n' "$profile" "$gate"
done

# --- execution: one gate per process, serialized --------------------------
completed=0
failed=()
for entry in "${selected[@]}"; do
  IFS='|' read -r _needs target profile gate <<<"$entry"
  test_crate=$CRATE
  require_gpu_env=0
  ignored_args=(--ignored)
  feature_args=(--features "$FEATURE")
  if [[ $target == frontend:* ]]; then
    test_crate=$FRONTEND_CRATE
    target_args=(--test "${target#frontend:}")
    feature_args=()
  elif [ "$target" = kernels ]; then
    test_crate=$KERNELS_CRATE
    target_args=(--lib)
  elif [[ $target == kernels:* ]]; then
    test_crate=$KERNELS_CRATE
    target_args=(--test "${target#kernels:}")
    require_gpu_env=1
    ignored_args=()
  elif [ "$target" = lib ]; then
    target_args=(--lib)
  else
    target_args=(--test "$target")
  fi
  model_env=()
  case "$profile" in
    dense) model_env=(env "PEGAINFER_TEST_MODEL_PATH=$ckpt") ;;
    routed)
      model_env=(env "PEGAINFER_TEST_MODEL_PATH=$moe_ckpt")
      # The exported paths name the dense set.
      if [ -n "$ROUTED_FIXTURE_TAG" ]; then
        mapfile -t routed_fixtures < <(fixtures_for "$ROUTED_FIXTURE_TAG")
        model_env+=(
          "PEGAINFER_GEMMA4_GOLDEN=$root/${routed_fixtures[0]}"
          "PEGAINFER_GEMMA4_WINDOW_GOLDEN=$root/${routed_fixtures[1]}"
          "PEGAINFER_GEMMA4_LONGCTX_GOLDEN=$root/${routed_fixtures[2]}"
          "PEGAINFER_GEMMA4_GENERATE=$root/${routed_fixtures[3]}"
          "PEGAINFER_GEMMA4_CHAT_GOLDEN=$root/$(chat_golden_for "$ROUTED_FIXTURE_TAG")"
        )
      fi
      ;;
    device) ;;
    *) die "unknown execution profile $profile" ;;
  esac
  if [ "$require_gpu_env" -eq 1 ]; then
    model_env=(env PEGAINFER_REQUIRE_GPU=1)
  fi
  # A gate that sweeps prints a row per cell, and the build chatter ahead of
  # it is longer than it looks; at twenty lines the first cells of a nine-cell
  # sweep fell off and the table reaching the log was missing its worst rows.
  echo "--- [$profile] $gate"
  if "${model_env[@]}" cargo test --release -p "$test_crate" "${feature_args[@]}" \
      "${target_args[@]}" -- \
      "${ignored_args[@]}" --exact "$gate" --test-threads=1 --nocapture 2>&1 | tail -40; then
    completed=$((completed + 1))
  else
    failed+=("[$profile] $gate")
  fi
done

echo "gemma4 gates: selected ${#selected[@]}, completed $completed, failed ${#failed[@]}"
if [ ${#failed[@]} -gt 0 ]; then
  printf 'gemma4 gates: FAILED %s\n' "${failed[@]}"
  exit 1
fi
[ "$completed" -eq "${#selected[@]}" ] || die "a selected gate neither completed nor failed"
echo "gemma4 gates: all selected gates completed"
