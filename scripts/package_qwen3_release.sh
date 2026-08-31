#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 VERSION OUTPUT_DIR" >&2
  exit 2
}

[[ $# -eq 2 ]] || usage

version=${1#v}
output_dir=$2
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
binary="$repo_root/target/release/pegainfer"
cuda_root=${CUDA_PATH:-${CUDA_HOME:-/usr/local/cuda}}
asset_name=pegainfer-qwen3-linux-x86_64-cu130

[[ $(uname -s) == Linux ]] || {
  echo "release packaging requires Linux" >&2
  exit 1
}
[[ $(uname -m) == x86_64 ]] || {
  echo "release packaging currently supports x86_64 only" >&2
  exit 1
}
[[ -x $binary ]] || {
  echo "release binary is missing: $binary" >&2
  exit 1
}
command -v patchelf >/dev/null || {
  echo "patchelf is required" >&2
  exit 1
}

cuda_lib_dir=
for candidate in \
  "$cuda_root/targets/x86_64-linux/lib" \
  "$cuda_root/lib64"; do
  if [[ -e $candidate/libcudart.so.13 ]]; then
    cuda_lib_dir=$candidate
    break
  fi
done
[[ -n $cuda_lib_dir ]] || {
  echo "CUDA 13 runtime libraries were not found under $cuda_root" >&2
  exit 1
}

staging_dir=$(mktemp -d)
trap 'rm -rf -- "$staging_dir"' EXIT
package_root="$staging_dir/$asset_name"
mkdir -p "$package_root/bin" "$package_root/lib" "$output_dir"

install -m 0755 "$binary" "$package_root/bin/pegainfer"
for library in libcudart.so.13 libcublas.so.13 libcublasLt.so.13; do
  [[ -e $cuda_lib_dir/$library ]] || {
    echo "required CUDA library is missing: $cuda_lib_dir/$library" >&2
    exit 1
  }
  install -m 0644 "$(readlink -f "$cuda_lib_dir/$library")" "$package_root/lib/$library"
done

install -m 0644 "$repo_root/LICENSE" "$package_root/LICENSE"
install -m 0644 "$repo_root/NOTICE" "$package_root/NOTICE"
install -m 0644 "$repo_root/NOTICE_DYNAMO" "$package_root/NOTICE_DYNAMO"
[[ -f $cuda_root/EULA.txt ]] || {
  echo "CUDA EULA is missing: $cuda_root/EULA.txt" >&2
  exit 1
}
install -m 0644 "$cuda_root/EULA.txt" "$package_root/NVIDIA-CUDA-EULA.txt"

patchelf --set-rpath "\$ORIGIN/../lib" "$package_root/bin/pegainfer"

binary_version=$(env -u LD_LIBRARY_PATH \
  "$package_root/bin/pegainfer" --version | awk '{print $2}')
[[ $binary_version == "$version" ]] || {
  echo "binary version $binary_version does not match release version $version" >&2
  exit 1
}

packaged_dependencies=$(env -u LD_LIBRARY_PATH ldd "$package_root/bin/pegainfer")
if grep -Fq 'not found' <<<"$packaged_dependencies"; then
  echo "packaged binary has unresolved dynamic libraries" >&2
  printf '%s\n' "$packaged_dependencies" >&2
  exit 1
fi
for library in libcudart.so.13 libcublas.so.13 libcublasLt.so.13; do
  resolved_library=$(awk -v library="$library" \
    '$1 == library {print $3}' <<<"$packaged_dependencies")
  if [[ -z $resolved_library \
    || $(readlink -f "$resolved_library") != $(readlink -f "$package_root/lib/$library") ]]; then
    echo "packaged binary does not resolve bundled $library" >&2
    printf '%s\n' "$packaged_dependencies" >&2
    exit 1
  fi
done

archive="$output_dir/$asset_name.tar.gz"
tar -C "$staging_dir" -czf "$archive" "$asset_name"
(
  cd "$output_dir"
  sha256sum "$asset_name.tar.gz" >SHA256SUMS
)
echo "$archive"
