#!/usr/bin/env bash
# External reference only: never copied into a Models distribution or linked by a backend.
set -euo pipefail
[[ $(uname -s) == Linux ]] || { echo 'Build the real-model comparator on remote Linux' >&2; exit 2; }
[[ $# == 1 && ! -e $1 ]] || { echo 'Usage: build-decoder-reference.sh NEW_OUTPUT_DIRECTORY' >&2; exit 2; }
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
mkdir -p "$1"
output=$(realpath "$1")
source=$output/source
revision=2cbca389f515df455412dafa07e2b1359cc72a6d
git init "$source"
git -C "$source" remote add origin https://github.com/ggml-org/llama.cpp.git
git -C "$source" fetch --depth 1 origin "$revision"
git -C "$source" checkout --detach FETCH_HEAD
cmake -S "$source" -B "$source/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DGGML_NATIVE=ON -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF
cmake --build "$source/build" --target llama -j "${VISION_BUILD_THREADS:-8}"
c++ -O3 -std=c++17 -I "$source/include" -I "$source/ggml/include" "$script_dir/decoder-oracle.cpp" \
    -L "$source/build/bin" -Wl,-rpath,"$source/build/bin" -lllama -o "$output/decoder-oracle"
sha256sum "$output/decoder-oracle" "$source/build/bin/"*.so* > "$output/binary-sha256.txt"
