#!/usr/bin/env bash
# K3 multi-machine EP fleet over ssh: one pegainfer process per GB300 tray,
# RANKS_PER_HOST (default 4) EP ranks each, paired through the NVLink-fabric
# bootstrap (--k3-rendezvous, bound by the first host).
#
#   HOSTS="pod4-gb300-3-tray04-f3 pod4-gb300-3-tray05-f3" \
#   MODEL_PATH=/mnt/shared/weights/kimi-k3 \
#   scripts/k3_ep_fleet.sh start|stop|status|logs|smoke
#
# Environment:
#   HOSTS           space-separated tray hostnames, rank order (required)
#   MODEL_PATH      checkpoint directory (required for start/smoke)
#   RANKS_PER_HOST  EP ranks (= GPUs) per host             [4]
#   EP_SIZE         world size                             [hosts * RANKS_PER_HOST]
#   PORT            HTTP port each process serves          [8300]
#   RDV_PORT        bootstrap port on the first host       [19300]
#   BIN             pegainfer binary (shared FS path)      [<repo>/target/release/pegainfer]
#   SERVED_NAME     --served-model-name                    [kimi-k3]
#   LOG_DIR         per-fleet logs (shared FS)             [~/k3-fleet-logs/<timestamp>]
#   EXTRA_ENV       extra "K=V K=V" exported to every rank process
#                   (e.g. "PEGAINFER_K3_MAX_BATCH=16 PEGAINFER_K3_MAX_CTX=8192")
#   EXTRA_ARGS      extra CLI flags appended to every rank process
#                   (e.g. "--dflash-draft-model-path /mnt/shared/weights/kimi-k3-dspark")
#
# Every process serves its own HTTP endpoint with one scheduler partition per
# local rank; requests land on whichever host you curl (front them with a
# router for real traffic). The fleet is fail-stop: one dead process strands
# the MegaMoE device barriers on every peer, so restarts are whole-fleet
# (`stop` then `start`).
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-$REPO_ROOT/target/release/pegainfer}
RANKS_PER_HOST=${RANKS_PER_HOST:-4}
PORT=${PORT:-8300}
RDV_PORT=${RDV_PORT:-19300}
SERVED_NAME=${SERVED_NAME:-kimi-k3}
STATE_DIR=${STATE_DIR:-$HOME/.k3-ep-fleet}
EXTRA_ENV=${EXTRA_ENV:-}
EXTRA_ARGS=${EXTRA_ARGS:-}

SSH=(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10)

die() { echo "k3_ep_fleet: $*" >&2; exit 1; }

hosts() {
  [[ -n ${HOSTS:-} ]] || die "set HOSTS to the ordered tray hostnames"
  echo "$HOSTS"
}

ep_size() {
  local -a fleet=($(hosts))
  echo "${EP_SIZE:-$(( ${#fleet[@]} * RANKS_PER_HOST ))}"
}

start() {
  [[ -n ${MODEL_PATH:-} ]] || die "set MODEL_PATH to the K3 checkpoint directory"
  [[ -x $BIN ]] || die "binary $BIN is missing or not executable"
  local -a fleet=($(hosts))
  local world; world=$(ep_size)
  (( world == ${#fleet[@]} * RANKS_PER_HOST )) \
    || die "EP_SIZE=$world does not equal hosts(${#fleet[@]}) * RANKS_PER_HOST($RANKS_PER_HOST)"
  local log_dir=${LOG_DIR:-$HOME/k3-fleet-logs/$(date +%Y%m%d-%H%M%S)}
  mkdir -p "$log_dir" "$STATE_DIR"
  local rdv="${fleet[0]}:$RDV_PORT"
  echo "$log_dir" > "$STATE_DIR/log_dir"
  printf '%s\n' "${fleet[@]}" > "$STATE_DIR/hosts"

  local i
  for i in "${!fleet[@]}"; do
    local host=${fleet[$i]}
    local start_rank=$(( i * RANKS_PER_HOST ))
    local end_rank=$(( start_rank + RANKS_PER_HOST ))
    local log="$log_dir/$host.log"
    local pid_file="$STATE_DIR/$host.pid"
    echo "[$host] ranks $start_rank..$end_rank -> :$PORT (log $log)"
    # shellcheck disable=SC2029
    "${SSH[@]}" "$host" \
      "nohup env RUST_LOG=info $EXTRA_ENV '$BIN' \
         --model-path '$MODEL_PATH' \
         --served-model-name '$SERVED_NAME' \
         --port $PORT \
         --k3-ep-size $world \
         --k3-ranks $start_rank..$end_rank \
         --k3-rendezvous '$rdv' \
         $EXTRA_ARGS \
         > '$log' 2>&1 & echo \$! > '$pid_file'" </dev/null
  done
  echo "fleet launched: ep_size=$world over ${#fleet[@]} hosts, bootstrap $rdv"
  echo "watch: $0 logs   |   probe: $0 status   |   first tokens: $0 smoke"
}

stop() {
  local -a fleet=($(hosts))
  local host
  for host in "${fleet[@]}"; do
    local pid_file="$STATE_DIR/$host.pid"
    # shellcheck disable=SC2029
    "${SSH[@]}" "$host" \
      "if [[ -f '$pid_file' ]]; then kill \$(cat '$pid_file') 2>/dev/null && echo '[$host] stopped' || echo '[$host] already gone'; rm -f '$pid_file'; else echo '[$host] no pid file'; fi" \
      </dev/null
  done
}

status() {
  local -a fleet=($(hosts))
  local host
  for host in "${fleet[@]}"; do
    local pid_file="$STATE_DIR/$host.pid"
    # shellcheck disable=SC2029
    "${SSH[@]}" "$host" \
      "pid=\$(cat '$pid_file' 2>/dev/null); \
       if [[ -n \$pid ]] && kill -0 \$pid 2>/dev/null; then \
         ready=\$(curl -sf -o /dev/null -w '%{http_code}' http://localhost:$PORT/v1/models || true); \
         echo \"[$host] pid \$pid running, /v1/models -> \${ready:-unreachable}\"; \
       else echo '[$host] not running'; fi" </dev/null
  done
}

logs() {
  local log_dir
  log_dir=$(cat "$STATE_DIR/log_dir" 2>/dev/null) || die "no fleet state; start one first"
  tail -n "${1:-30}" "$log_dir"/*.log
}

smoke() {
  local -a fleet=($(hosts))
  local host
  for host in "${fleet[@]}"; do
    echo "== $host =="
    curl -sf "http://$host:$PORT/v1/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\": \"$SERVED_NAME\", \"prompt\": \"The capital of France is\", \"max_tokens\": 16, \"temperature\": 0}" \
      | python3 -m json.tool || echo "[$host] request failed"
  done
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  logs) logs "${2:-30}" ;;
  smoke) smoke ;;
  *) die "usage: HOSTS=... MODEL_PATH=... $0 start|stop|status|logs [n]|smoke" ;;
esac
