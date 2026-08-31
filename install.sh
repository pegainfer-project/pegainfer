#!/usr/bin/env bash
set -euo pipefail

repository=pegainfer-project/pegainfer
asset_name=pegainfer-qwen3-linux-x86_64-cu130
requested_version=${PEGAINFER_VERSION:-latest}

die() {
  echo "pegainfer installer: $*" >&2
  exit 1
}

command -v curl >/dev/null || die "curl is required"
command -v sha256sum >/dev/null || die "sha256sum is required"
command -v tar >/dev/null || die "tar is required"
[[ $(uname -s) == Linux ]] || die "the prebuilt release supports Linux only"
[[ $(uname -m) == x86_64 ]] || die "the prebuilt release supports x86_64 only"
command -v nvidia-smi >/dev/null || die "nvidia-smi is required"

driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n '1p')
driver_major=${driver_version%%.*}
[[ $driver_major =~ ^[0-9]+$ ]] || die "could not parse NVIDIA driver version: $driver_version"
((driver_major >= 580)) || die "CUDA 13 requires NVIDIA driver 580 or newer; found $driver_version"

supported_gpu=false
while IFS= read -r compute_capability; do
  compute_major=${compute_capability%%.*}
  if [[ $compute_major =~ ^(8|9|10|11|12)$ ]]; then
    supported_gpu=true
    break
  fi
done < <(nvidia-smi --query-gpu=compute_cap --format=csv,noheader)
[[ $supported_gpu == true ]] \
  || die "the Qwen3 binary requires an NVIDIA GPU with compute capability 8.x through 12.x"

if [[ $requested_version == latest ]]; then
  release_url="https://github.com/$repository/releases/latest/download"
else
  [[ $requested_version == v* ]] || requested_version="v$requested_version"
  release_url="https://github.com/$repository/releases/download/$requested_version"
fi

data_home=${XDG_DATA_HOME:-$HOME/.local/share}
install_root=${PEGAINFER_INSTALL_ROOT:-$data_home/pegainfer}
bin_dir=${PEGAINFER_BIN_DIR:-$HOME/.local/bin}
[[ $install_root == /* ]] || die "PEGAINFER_INSTALL_ROOT must be an absolute path"
[[ $bin_dir == /* ]] || die "PEGAINFER_BIN_DIR must be an absolute path"
temporary_dir=$(mktemp -d)
trap 'rm -rf -- "$temporary_dir"' EXIT

archive="$temporary_dir/$asset_name.tar.gz"
checksums="$temporary_dir/SHA256SUMS"
curl --fail --location --silent --show-error \
  "$release_url/$asset_name.tar.gz" --output "$archive"
curl --fail --location --silent --show-error \
  "$release_url/SHA256SUMS" --output "$checksums"

(
  cd "$temporary_dir"
  grep "  $asset_name.tar.gz$" SHA256SUMS >SHA256SUMS.selected \
    || die "release checksum does not list $asset_name.tar.gz"
  sha256sum --check SHA256SUMS.selected
)

tar -xzf "$archive" -C "$temporary_dir"
extracted="$temporary_dir/$asset_name"
[[ -x $extracted/bin/pegainfer ]] || die "release archive does not contain pegainfer"
installed_version=$("$extracted/bin/pegainfer" --version | awk '{print $2}')
[[ $installed_version =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-].*)?$ ]] \
  || die "release binary returned an invalid version: $installed_version"
if [[ $requested_version != latest && v$installed_version != "$requested_version" ]]; then
  die "downloaded version $installed_version does not match $requested_version"
fi

version_dir="$install_root/versions/v$installed_version"
mkdir -p "$install_root/versions" "$bin_dir"
if [[ -e $version_dir ]]; then
  [[ -x $version_dir/bin/pegainfer ]] \
    || die "existing installation is incomplete: $version_dir"
  existing_version=$("$version_dir/bin/pegainfer" --version | awk '{print $2}')
  [[ $existing_version == "$installed_version" ]] \
    || die "existing installation at $version_dir has version $existing_version"
else
  mv "$extracted" "$version_dir"
fi
ln -sfn "versions/v$installed_version" "$install_root/current.new"
mv -Tf "$install_root/current.new" "$install_root/current"
ln -sfn "$install_root/current/bin/pegainfer" "$bin_dir/pegainfer"

echo "installed PegaInfer v$installed_version to $version_dir"
echo "binary: $bin_dir/pegainfer"
if [[ :$PATH: != *":$bin_dir:"* ]]; then
  echo "pegainfer is not on PATH in this shell; run:"
  printf '  export PATH="%s:%s"\n' "$bin_dir" "\$PATH"
  echo "add the same line to your shell profile to keep it available"
fi
