#!/usr/bin/env bash
set -euo pipefail

# Fixed-tray GLM5.2 native-MTP P/D development stack:
#   tray03: PegaFlow MetaServer + TP4 prefill + vLLM Router (optional)
#   tray13+tray14: EP8 decode fleet
#
# Every value can be overridden from the environment so the same script can
# be reused after the fixed trays are reassigned.
#
# Multi-process decode fleet: set D_TOPO to a topology wider than one tray
# (e.g. ep8) and D_HOSTS to the space-separated decode hosts; ranks are
# split evenly in order (ep8 over 2 hosts → host0 0..4, host1 4..8), the
# first host serves the bootstrap rendezvous on D_RENDEZVOUS_PORT, and every
# decode process joins the same KV P2P mesh (advertise port is per-host).
#
# `decode-only` starts the decode fleet WITHOUT the P/D side (no metaserver,
# prefill, router, or KV P2P mesh): requests are served by local prefill on
# the decode ranks. Use a separate GLM52_PD_CONFIG env file for that fleet
# (P_HOST is only a placeholder for the shared stop/status paths there).

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
CONFIG_FILE=${GLM52_PD_CONFIG:-${XDG_CONFIG_HOME:-$HOME/.config}/pegainfer/glm52-pd.env}

# The jump host drops connections intermittently; bound each connect/banner
# wait and retry it a few times so a transient blip does not abort the run.
ssh() { command ssh -o ConnectTimeout=10 -o ConnectionAttempts=4 "$@"; }
if [[ -f $CONFIG_FILE ]]; then
    # shellcheck disable=SC1090
    source "$CONFIG_FILE"
fi

required_vars=(
    P_HOST
    P_IMAGE
    P_MODEL_PATH
    D_MODEL_PATH
    PEGAINFER_NCCL_ROOT
)
for var in "${required_vars[@]}"; do
    if [[ -z ${!var:-} ]]; then
        printf '%s must be set in the environment or %s\n' "$var" "$CONFIG_FILE" >&2
        exit 2
    fi
done

P_CONTAINER=${P_CONTAINER:-pegainfer-pd-prefill}
D_CONTAINER=${D_CONTAINER:-pegainfer-pd-decode}
D_IMAGE=${D_IMAGE:-$P_IMAGE}
D_TOPO=${D_TOPO:-ep4}
D_HOSTS=${D_HOSTS:-${D_HOST:-}}
D_RENDEZVOUS_PORT=${D_RENDEZVOUS_PORT:-19200}
P_HTTP_PORT=${P_HTTP_PORT:-8000}
D_HTTP_PORT=${D_HTTP_PORT:-8000}
ROUTER_PORT=${ROUTER_PORT:-10001}
METASERVER_PORT=${METASERVER_PORT:-50056}
METASERVER_HTTP_PORT=${METASERVER_HTTP_PORT:-19092}
P_TRANSFER_PORT=${P_TRANSFER_PORT:-50103}
D_TRANSFER_PORT=${D_TRANSFER_PORT:-50104}
RDMA_NIC=${RDMA_NIC:-mlx5_bond_0}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
KV_OFFLOAD_HOST_GIB=${KV_OFFLOAD_HOST_GIB:-8}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-glm-5.2-fp8}

case "$D_TOPO" in
    ep4)  d_fleet_ranks=4 ;;
    ep8)  d_fleet_ranks=8 ;;
    ep16) d_fleet_ranks=16 ;;
    ep32) d_fleet_ranks=32 ;;
    ep64) d_fleet_ranks=64 ;;
    *) printf 'unsupported D_TOPO: %s\n' "$D_TOPO" >&2; exit 2 ;;
esac
read -ra d_hosts <<< "$D_HOSTS"
d_host_count=${#d_hosts[@]}
if (( d_host_count < 1 )) || (( d_fleet_ranks % d_host_count != 0 )); then
    printf 'D_HOSTS ("%s") must hold 1+ hosts splitting %s (%d ranks) evenly\n' \
        "$D_HOSTS" "$D_TOPO" "$d_fleet_ranks" >&2
    exit 2
fi
d_ranks_per_host=$(( d_fleet_ranks / d_host_count ))

HOST_REPO=${HOST_REPO:-$REPO_ROOT}
P_REPO=${P_REPO:-/workspace/pegainfer}
D_REPO=${D_REPO:-/workspace/pegainfer}

discover_ip() {
    # The RDMA device (mlx5_bond_0) is a bond of two links; the tray's
    # routable address sits on a VLAN subinterface of the bond master
    # (bond0.225), not on the RDMA alias itself. Walk slave → master → the
    # master's subinterface holding the global IPv4.
    ssh "$1" "slave=\$(ls /sys/class/infiniband/$RDMA_NIC/device/net | head -1) && \
        master=\$(basename \"\$(readlink /sys/class/net/\$slave/master)\") && \
        ip -o -4 addr show scope global | \
        awk -v m=\"\$master.\" 'index(\$2, m) == 1 { split(\$4, a, \"/\"); print a[1]; exit }'"
}

SUBCOMMAND=${1:-}
if [[ $SUBCOMMAND != decode-only ]]; then
    P_IP=${P_IP:-$(discover_ip "$P_HOST")}
else
    # decode-only never touches a prefill peer; a placeholder P_HOST in a
    # decode-only env file must not cost an SSH probe.
    P_IP=${P_IP:-}
fi
d_ips=()
for d_host_spec in "${d_hosts[@]}"; do
    d_ips+=("$(discover_ip "$d_host_spec")")
done
if [[ $SUBCOMMAND != decode-only && -z $P_IP ]]; then
    printf 'could not discover an IPv4 address on %s for P; set P_IP explicitly\n' \
        "$RDMA_NIC" >&2
    exit 2
fi
for d_ip_entry in "${d_ips[@]}"; do
    if [[ -z $d_ip_entry ]]; then
        printf 'could not discover an IPv4 address on %s for every D host; set d_ips via the script\n' \
            "$RDMA_NIC" >&2
        exit 2
    fi
done

role_pid_file() {
    printf '/tmp/pegainfer-glm52-pd-%s.pid' "$1"
}

role_log_file() {
    printf '/tmp/pegainfer-glm52-pd-%s.log' "$1"
}

shell_join() {
    local joined
    printf -v joined '%q ' "$@"
    printf '%s' "$joined"
}

container_start() {
    local host=$1 container=$2 role=$3 command=$4
    local pid_file log_file wrapped quoted
    pid_file=$(role_pid_file "$role")
    log_file=$(role_log_file "$role")
    container_stop "$host" "$container" "$role"
    printf -v wrapped 'echo $$ > %q; %s > %q 2>&1' \
        "$pid_file" "$command" "$log_file"
    printf -v quoted '%q' "$wrapped"
    ssh "$host" "docker exec -d $container bash -lc $quoted"
}

container_stop() {
    local host=$1 container=$2 role=$3
    local pid_file pid
    pid_file=$(role_pid_file "$role")
    pid=$(ssh "$host" "docker exec $container bash -lc 'test -f $pid_file && cat $pid_file'" \
        2>/dev/null || true)
    if [[ $pid =~ ^[0-9]+$ ]]; then
        ssh "$host" "docker exec $container kill -TERM $pid" 2>/dev/null || true
        for _ in $(seq 1 20); do
            if ! ssh "$host" "docker exec $container kill -0 $pid" 2>/dev/null; then
                break
            fi
            sleep 0.5
        done
    fi
    ssh "$host" "docker exec $container rm -f $pid_file" 2>/dev/null || true
}

host_start() {
    local host=$1 role=$2 command=$3
    local pid_file log_file wrapped quoted
    pid_file=$(role_pid_file "$role")
    log_file=$(role_log_file "$role")
    host_stop "$host" "$role"
    printf -v wrapped 'echo $$ > %q; %s > %q 2>&1' \
        "$pid_file" "$command" "$log_file"
    printf -v quoted '%q' "$wrapped"
    ssh "$host" "nohup bash -lc $quoted >/dev/null 2>&1 &"
}

host_stop() {
    local host=$1 role=$2
    local pid_file pid
    pid_file=$(role_pid_file "$role")
    pid=$(ssh "$host" "test -f $pid_file && cat $pid_file" 2>/dev/null || true)
    if [[ $pid =~ ^[0-9]+$ ]]; then
        ssh "$host" "kill -TERM $pid" 2>/dev/null || true
        for _ in $(seq 1 20); do
            if ! ssh "$host" "kill -0 $pid" 2>/dev/null; then
                break
            fi
            sleep 0.5
        done
    fi
    ssh "$host" "rm -f $pid_file" 2>/dev/null || true
}

wait_http() {
    # The trays are private-only: the agent box reaches them over ssh, not
    # HTTP, so the probe curl runs on a host inside the rack.
    local name=$1 via_host=$2 url=$3 timeout=${4:-600}
    local start=$SECONDS
    until ssh -o ConnectTimeout=5 "$via_host" "curl -fsS --max-time 2 $url >/dev/null 2>&1"; do
        if (( SECONDS - start >= timeout )); then
            printf '%s did not become ready within %ss: %s\n' "$name" "$timeout" "$url" >&2
            return 1
        fi
        sleep 2
    done
    printf '%s ready: %s\n' "$name" "$url"
}

ensure_containers() {
    if ! ssh "$P_HOST" "docker inspect $P_CONTAINER >/dev/null 2>&1"; then
        ssh "$P_HOST" "docker run -d \
            --name $P_CONTAINER \
            --gpus all \
            --network host \
            --ipc host \
            --security-opt label=disable \
            --ulimit memlock=-1:-1 \
            --device /dev/infiniband/uverbs4 \
            --device /dev/infiniband/rdma_cm \
            -v /dev/infiniband:/dev/infiniband \
            -v $HOST_REPO:$P_REPO \
            -v $P_MODEL_PATH:$P_MODEL_PATH:ro \
            --entrypoint bash \
            $P_IMAGE -lc 'sleep infinity' >/dev/null"
    else
        ssh "$P_HOST" "docker start $P_CONTAINER >/dev/null"
    fi
    for d_host_spec in "${d_hosts[@]}"; do
        if ! ssh "$d_host_spec" "docker inspect $D_CONTAINER >/dev/null 2>&1"; then
            ssh "$d_host_spec" "docker run -d \
                --name $D_CONTAINER \
                --gpus all \
                --network host \
                --ipc host \
                --security-opt label=disable \
                --ulimit memlock=-1:-1 \
                --device /dev/infiniband/uverbs4 \
                --device /dev/infiniband/rdma_cm \
                -v /dev/infiniband:/dev/infiniband \
                -v $HOST_REPO:$D_REPO \
                -v $D_MODEL_PATH:$D_MODEL_PATH:ro \
                --entrypoint bash \
                $D_IMAGE -lc 'sleep infinity' >/dev/null"
        else
            ssh "$d_host_spec" "docker start $D_CONTAINER >/dev/null"
        fi
    done

    ensure_nccl "$P_HOST" "$P_CONTAINER"
    ssh "$P_HOST" "docker exec $P_CONTAINER bash -lc \
        'nvidia-smi -L >/dev/null && test -c /dev/infiniband/uverbs4'"
    for d_host_spec in "${d_hosts[@]}"; do
        ensure_nccl "$d_host_spec" "$D_CONTAINER"
        ssh "$d_host_spec" "docker exec $D_CONTAINER bash -lc \
            'nvidia-smi -L >/dev/null && test -c /dev/infiniband/uverbs4'"
    done
}

ensure_nccl() {
    local host=$1 container=$2
    if ssh "$host" "docker exec $container bash -lc \
        'nm -D /usr/lib/aarch64-linux-gnu/libnccl.so.2 | grep -q ncclCommQueryProperties'"; then
        return
    fi
    printf 'Upgrading NCCL in %s to 2.30.7...\n' "$container"
    ssh "$host" "docker exec $container bash -lc \
        'apt-get update -qq && \
         apt-get install -y -qq \
            libnccl2=2.30.7-1+cuda13.3 \
            libnccl-dev=2.30.7-1+cuda13.3'"
}

prepare() {
    local pegaflow_manifest
    ensure_containers
    printf 'Building PegaInfer GLM5.2 release binary on %s...\n' "$P_HOST"
    ssh "$P_HOST" "docker exec \
        -e PEGAINFER_NCCL_ROOT=$PEGAINFER_NCCL_ROOT \
        -e PEGAINFER_CUDA_SM=103 \
        $P_CONTAINER bash -lc \
        'cd $P_REPO && cargo build --release --no-default-features --features glm52'"

    pegaflow_manifest=$(ssh "$P_HOST" "docker exec $P_CONTAINER bash -lc \
        'find /root/.cargo/git/checkouts/pegaflow-* -path \"*/1473c53/pegaflow-metaserver/Cargo.toml\" -print -quit'")
    if [[ -z $pegaflow_manifest ]]; then
        printf 'pegaflow 1473c53 checkout is missing in %s\n' "$P_CONTAINER" >&2
        return 1
    fi
    printf 'Building PegaFlow MetaServer...\n'
    ssh "$P_HOST" "docker exec \
        -e CARGO_TARGET_DIR=$P_REPO/target/pegaflow \
        $P_CONTAINER bash -lc \
        'cargo build --release --manifest-path $pegaflow_manifest'"

    for d_host_spec in "${d_hosts[@]}"; do
        ssh "$d_host_spec" "docker exec $D_CONTAINER bash -lc \
            'nm -D /usr/lib/aarch64-linux-gnu/libnccl.so.2 | grep -q ncclCommQueryProperties'"
        ssh "$d_host_spec" "docker exec $D_CONTAINER test -x $D_REPO/target/release/pegainfer"
    done
    if [[ -n ${ROUTER_BIN:-} ]]; then
        ssh "$P_HOST" "test -x $ROUTER_BIN"
    fi
    printf 'prepare complete\n'
}

wait_log() {
    local name=$1 host=$2 container=$3 file=$4 pattern=$5 timeout=${6:-600}
    local start=$SECONDS
    until ssh "$host" "docker exec $container grep -q '$pattern' $file 2>/dev/null"; do
        if (( SECONDS - start >= timeout )); then
            printf '%s: pattern %q not seen within %ss in %s\n' \
                "$name" "$pattern" "$timeout" "$file" >&2
            return 1
        fi
        sleep 3
    done
    printf '%s: saw %q\n' "$name" "$pattern"
}

start_decode_fleet() {
    # Decode fleet: one process per host. With more than one host the first
    # serves the bootstrap rendezvous and must log its serving line before
    # the others start fetching (their connect is not retried forever).
    # $1: KV P2P mesh flags for P/D (empty for standalone decode-only).
    local p2p_base=$1 idx start_rank end_rank fleet_flags p2p_flags d_cmd
    for idx in "${!d_hosts[@]}"; do
        start_rank=$(( idx * d_ranks_per_host ))
        end_rank=$(( start_rank + d_ranks_per_host ))
        fleet_flags=""
        if (( d_host_count > 1 )); then
            fleet_flags="--glm52-ranks $start_rank..$end_rank \
--glm52-rendezvous ${d_ips[0]}:$D_RENDEZVOUS_PORT"
        fi
        p2p_flags=""
        if [[ -n $p2p_base ]]; then
            p2p_flags="$p2p_base --kv-p2p-advertise-addr ${d_ips[$idx]}:$D_TRANSFER_PORT"
        fi
        d_cmd="cd $(printf %q "$D_REPO") && exec env RUST_LOG=info \
EP_DISABLE_GIN=1 \
$(printf %q "$D_REPO/target/release/pegainfer") \
--model-path $(printf %q "$D_MODEL_PATH") \
--served-model-name $(printf %q "$SERVED_MODEL_NAME") \
--port $D_HTTP_PORT --moe-topo $D_TOPO \
--glm52-native-mtp --glm52-weight-staging \
--max-model-len $MAX_MODEL_LEN \
$p2p_flags $fleet_flags"
        container_start "${d_hosts[$idx]}" "$D_CONTAINER" "decode$idx" "$d_cmd"
        if (( idx == 0 && d_host_count > 1 )); then
            wait_log "decode0 rendezvous" "${d_hosts[0]}" "$D_CONTAINER" \
                "$(role_log_file decode0)" "serving DeepEP id" 900
        fi
    done
    for idx in "${!d_hosts[@]}"; do
        wait_http "decode$idx" "${d_hosts[$idx]}" \
            "http://${d_ips[$idx]}:$D_HTTP_PORT/health" 600
    done
}

start_decode_only() {
    # Standalone decode fleet: no metaserver, prefill, router, or KV P2P
    # mesh — requests are served by local prefill on the decode ranks.
    start_decode_fleet ""
}

start() {
    local meta_cmd p_cmd d_cmd router_cmd
    local common_p2p
    common_p2p="--kv-offload --kv-offload-host-gib $KV_OFFLOAD_HOST_GIB \
--kv-p2p-metaserver-addr http://$P_IP:$METASERVER_PORT \
--kv-p2p-nics $RDMA_NIC"

    meta_cmd="exec $(shell_join \
        "$P_REPO/target/pegaflow/release/pegaflow-metaserver" \
        --addr "0.0.0.0:$METASERVER_PORT" \
        --http-addr "0.0.0.0:$METASERVER_HTTP_PORT" \
        --log-level info)"
    container_start "$P_HOST" "$P_CONTAINER" meta "$meta_cmd"
    wait_http metaserver "$P_HOST" "http://$P_IP:$METASERVER_HTTP_PORT/health" 30

    p_cmd="cd $(printf %q "$P_REPO") && exec env RUST_LOG=info \
EP_DISABLE_GIN=1 NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=32 \
$(printf %q "$P_REPO/target/release/pegainfer") \
--model-path $(printf %q "$P_MODEL_PATH") \
--served-model-name $(printf %q "$SERVED_MODEL_NAME") \
--port $P_HTTP_PORT --tp-size 4 --moe-topo tp4 \
--glm52-prefill-only --glm52-native-mtp \
--glm52-weight-staging --max-model-len $MAX_MODEL_LEN \
$common_p2p --kv-p2p-advertise-addr $P_IP:$P_TRANSFER_PORT"

    container_start "$P_HOST" "$P_CONTAINER" prefill "$p_cmd"

    start_decode_fleet "$common_p2p"

    wait_http prefill "$P_HOST" "http://$P_IP:$P_HTTP_PORT/health" 600
    for idx in "${!d_hosts[@]}"; do
        wait_http "decode$idx" "${d_hosts[$idx]}" \
            "http://${d_ips[$idx]}:$D_HTTP_PORT/health" 600
    done

    if [[ -n ${ROUTER_BIN:-} ]]; then
        # A multi-host decode fleet exposes one HTTP endpoint per process;
        # register all of them so --decode-policy spreads requests across
        # the whole fleet.
        decode_flags=()
        for d_ip in "${d_ips[@]}"; do
            decode_flags+=(--decode "http://$d_ip:$D_HTTP_PORT")
        done
        router_cmd="exec $(shell_join \
            "$ROUTER_BIN" \
            --host 0.0.0.0 \
            --port "$ROUTER_PORT" \
            --vllm-pd-disaggregation \
            --prefill "http://$P_IP:$P_HTTP_PORT" none \
            "${decode_flags[@]}" \
            --prefill-policy round_robin \
            --decode-policy round_robin \
            --intra-node-data-parallel-size 1 \
            --kv-connector nixl \
            --disable-retries \
            --prometheus-port 29001)"
        host_start "$P_HOST" router "$router_cmd"
        wait_http router "$P_HOST" "http://$P_IP:$ROUTER_PORT/health" 30
    fi
}

stop() {
    host_stop "$P_HOST" router
    for idx in "${!d_hosts[@]}"; do
        container_stop "${d_hosts[$idx]}" "$D_CONTAINER" "decode$idx"
    done
    container_stop "$P_HOST" "$P_CONTAINER" prefill
    container_stop "$P_HOST" "$P_CONTAINER" meta
}

status_one() {
    local host=$1 where=$2 role=$3 pid_file pid
    pid_file=$(role_pid_file "$role")
    if [[ $where == host ]]; then
        pid=$(ssh "$host" "test -f $pid_file && cat $pid_file" 2>/dev/null || true)
        if [[ $pid =~ ^[0-9]+$ ]] && ssh "$host" "kill -0 $pid" 2>/dev/null; then
            printf '%-10s running pid=%s host=%s\n' "$role" "$pid" "$host"
        else
            printf '%-10s stopped host=%s\n' "$role" "$host"
        fi
    else
        pid=$(ssh "$host" "docker exec $where bash -lc 'test -f $pid_file && cat $pid_file'" \
            2>/dev/null || true)
        if [[ $pid =~ ^[0-9]+$ ]] \
            && ssh "$host" "docker exec $where kill -0 $pid" 2>/dev/null; then
            printf '%-10s running pid=%s host=%s container=%s\n' "$role" "$pid" "$host" "$where"
        else
            printf '%-10s stopped host=%s container=%s\n' "$role" "$host" "$where"
        fi
    fi
}

status() {
    status_one "$P_HOST" "$P_CONTAINER" meta
    status_one "$P_HOST" "$P_CONTAINER" prefill
    for idx in "${!d_hosts[@]}"; do
        status_one "${d_hosts[$idx]}" "$D_CONTAINER" "decode$idx"
    done
    status_one "$P_HOST" host router
}

logs() {
    local role=${1:-all}
    case "$role" in
        meta|prefill)
            ssh "$P_HOST" "docker exec $P_CONTAINER tail -n 120 $(role_log_file "$role")"
            ;;
        decode*)
            local idx=${role#decode}
            idx=${idx:-0}
            ssh "${d_hosts[$idx]}" \
                "docker exec $D_CONTAINER tail -n 120 $(role_log_file "decode$idx")"
            ;;
        router)
            ssh "$P_HOST" "tail -n 120 $(role_log_file "$role")"
            ;;
        all)
            for item in meta prefill; do
                printf '\n===== %s =====\n' "$item"
                logs "$item" || true
            done
            for idx in "${!d_hosts[@]}"; do
                printf '\n===== decode%s =====\n' "$idx"
                logs "decode$idx" || true
            done
            printf '\n===== router =====\n'
            logs router || true
            ;;
        *)
            printf 'unknown role: %s\n' "$role" >&2
            return 2
            ;;
    esac
}

smoke() {
    # Same ssh-only reachability as wait_http: run the smoke on the P host
    # (pure stdlib, the host python3 is enough).
    ssh "$P_HOST" "python3 - --base-url http://$P_IP:$ROUTER_PORT --model $SERVED_MODEL_NAME" \
        < "$(dirname "$0")/glm52_pd_smoke.py"
}

usage() {
    printf 'usage: %s {prepare|start|decode-only|stop|restart|status|logs [role]|smoke}\n' "$0"
    printf '  role: meta | prefill | decode | decode<N> | router | all\n'
}

case "${1:-}" in
    prepare) prepare ;;
    start) start ;;
    decode-only) start_decode_only ;;
    stop) stop ;;
    restart) stop; start ;;
    status) status ;;
    logs) logs "${2:-all}" ;;
    smoke) smoke ;;
    *) usage; exit 2 ;;
esac
