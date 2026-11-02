#!/usr/bin/env bash
# Re-derives every number in README.md. Builds the kernel crate twice -- once with the extraction
# and once from the parent commit -- and drives the real Q4_K batched-matmul FFI entry point.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
crate="$here/../../backend-native/src/main/rust/model-kernels"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cc -O2 -o "$work/ab" "$here/ab.c" -ldl

build() { # $1 = git revision for lib.rs, $2 = output name
  git -C "$crate" show "$1:backend-native/src/main/rust/model-kernels/src/lib.rs" \
    > "$crate/src/lib.rs" 2>/dev/null || true
  ( cd "$crate" && touch src/lib.rs && cargo build --release >/dev/null 2>&1 )
  cp "$crate/target/release/libjmodels_kernels.dylib" "$work/$2.dylib"
}
orig="$(git -C "$crate" rev-parse HEAD)"
build "$orig" with_extraction
build "$orig^" parent
git -C "$crate" checkout "$orig" -- backend-native/src/main/rust/model-kernels/src/lib.rs

for shape in "7168 2304 32 1" "2304 7168 16 1" "7168 2304 32 8"; do
  set -- $shape
  echo "## rows=$1 cols=$2 batch=$3 threads=$4"
  for pass in $(seq 1 12); do
    echo "extracted $("$work/ab" "$work/with_extraction.dylib" "$1" "$2" "$3" 20 "$4")"
    echo "parent    $("$work/ab" "$work/parent.dylib" "$1" "$2" "$3" 20 "$4")"
  done
done
