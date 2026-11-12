#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
launcher=${PEGAINFER_ROCM10_LAUNCHER:-/home/zihaomu/zihaomu_workspace/rocm10-launch.sh}
[[ -x "$launcher" ]] || { echo "ROCm 10 foundation launcher is missing: $launcher" >&2; exit 1; }
cd "$project_root"
readarray -t profile < <(python3 -c 'import json; p=json.load(open("scripts/amd/environment.lock.json")); print(p["image_id"]); print(p["device_selection"]["default_hip_visible_devices"]); print(p["device_selection"]["pci_bus_id"])')
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-${profile[1]}}"
export PEGAINFER_EXPECTED_PCI_BUS_ID="${PEGAINFER_EXPECTED_PCI_BUS_ID:-${profile[2]}}"
# HIP enumeration differs from amd-smi physical indices on this host. Inspect
# metadata in the locked image before the common launcher runs its GPU verifier.
docker run --rm --network none --device /dev/kfd --device /dev/dri --group-add 992 --group-add 44 \
  --env "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES" \
  --env "PEGAINFER_EXPECTED_PCI_BUS_ID=$PEGAINFER_EXPECTED_PCI_BUS_ID" \
  --mount "type=bind,src=$project_root,dst=/project,readonly" \
  "${profile[0]}" python /project/scripts/amd/check_device_selection.py
exec "$launcher" env PEGAINFER_ROCM_ARCH="${PEGAINFER_ROCM_ARCH:-gfx1201}" \
  PEGAINFER_EXPECTED_PCI_BUS_ID="$PEGAINFER_EXPECTED_PCI_BUS_ID" bash scripts/amd/in_container.sh "$@"
