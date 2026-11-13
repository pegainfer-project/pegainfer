#!/usr/bin/env bash
# Public, CPU-only SDK setup used by CI. Install into a dedicated Python venv.
set -euo pipefail
sdk_env=${1:?Usage: install_sdk.sh VENV_DIRECTORY [RUNTIME_GFX_TARGET]}
extras=libraries,devel
if [[ -n "${2:-}" ]]; then
  case "$2" in
    gfx1100|gfx1101|gfx1102|gfx1200|gfx1201) extras="$extras,device-$2" ;;
    *) echo "Unsupported runtime target: $2" >&2; exit 2 ;;
  esac
fi
python3 -m venv "$sdk_env"
"$sdk_env/bin/python" -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "rocm[$extras]==10.1.0"
"$sdk_env/bin/rocm-sdk" init
"$sdk_env/bin/rocm-sdk" path --root
