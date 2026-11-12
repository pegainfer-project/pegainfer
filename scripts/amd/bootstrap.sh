#!/usr/bin/env bash
# Project-local build dependencies; never mutate the shared ROCm foundation.
set -euo pipefail
artifact_root="$PWD/../artifacts/amd"
mkdir -p "$artifact_root/bootstrap" "$artifact_root/toolchain/openssl/packages"
if [[ ! -x "$CARGO_HOME/bin/rustup" ]]; then
  installer="$artifact_root/bootstrap/rustup-init.sh"
  curl --fail --location --retry 3 https://sh.rustup.rs -o "$installer"
  echo "d890e2fef6fa991b19cde2e608199bc1dca4e37f41f884edf8849ff693048de6  $installer" | sha256sum -c -
  sh "$installer" -y --no-modify-path --profile minimal --default-toolchain none
fi
rustup toolchain install nightly-2026-07-10 --profile minimal --component rustfmt,clippy
if [[ ! -f "$artifact_root/toolchain/openssl/root/usr/include/openssl/ssl.h" ]]; then
  apt_root="$artifact_root/toolchain/apt"
  mkdir -p "$apt_root/lists/partial" "$apt_root/cache/archives/partial"
  apt-get -o "Dir::State::lists=$apt_root/lists" -o "Dir::Cache=$apt_root/cache" update
  (
    cd "$artifact_root/toolchain/openssl/packages"
    apt-get -o "Dir::State::lists=$apt_root/lists" download \
      libssl-dev=3.0.13-0ubuntu3.16 libssl3t64=3.0.13-0ubuntu3.16
    echo '2b4734df0475e305ac03abb0a2cc5659651ed37ef02d967a1c4033a154cf714b  libssl-dev_3.0.13-0ubuntu3.16_amd64.deb' | sha256sum -c -
    echo '219f43b1cd836a4da550938db5fda93160d269d39a4edcf1f1ce698a470db797  libssl3t64_3.0.13-0ubuntu3.16_amd64.deb' | sha256sum -c -
    for package in *.deb; do dpkg-deb -x "$package" ../root; done
  )
fi
printf '%s\n' 'Build dependencies ready; re-enter through run_rocm10.sh to load the OpenSSL environment.'
