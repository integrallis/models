#!/usr/bin/env bash
# Build the publisher's patched llama.cpp as an external oracle for EdgeRock.
#
# Nothing from this build is copied into our runtime. It is used only to pin our output and as the
# comparator arm for qualification, which is what rule 6 permits: the patch is Apache-2.0
# (LICENSE-CODE in the GGUF repository) and llama.cpp is MIT (backend/llama-cpp-LICENSE).
#
# The prebuilt edgerock-native-linux-sm90.tar.gz in that repository is Linux x86-64 + CUDA SM90
# only, so every other platform has to rebuild. The publisher's own receipts are CUDA runs with
# --n-gpu-layers 99: their throughput numbers are GPU numbers and are not ours and not comparable
# to a CPU arm.
#
# Verified on 2026-10-30: all 9 hunks apply cleanly at the pinned commit and the CPU build
# succeeds on darwin.
set -euo pipefail

# Pinned by the publisher's README, which states that other platforms must rebuild against it.
LLAMA_COMMIT=62318cd57dc63900ecb153272c0b10f78550855d
REPO=hkust-aaron/EdgeRock-4B-GGUF
BASE="https://huggingface.co/${REPO}/resolve/main"

WORK=${1:?usage: build_reference.sh <work-dir> [llama.cpp-clone-or-url]}
UPSTREAM=${2:-https://github.com/ggml-org/llama.cpp}

mkdir -p "$WORK/backend"
for f in llama-cpp-runtime.patch edgerock-rma.h edgerock-rma-graph.inc llama-cpp-LICENSE; do
  [ -s "$WORK/backend/$f" ] || curl -fsSL -o "$WORK/backend/$f" "$BASE/backend/$f"
done
[ -s "$WORK/LICENSE-CODE" ] || curl -fsSL -o "$WORK/LICENSE-CODE" "$BASE/LICENSE-CODE"

if [ ! -d "$WORK/llama-er/.git" ]; then
  # --local against an existing clone when one is given, so this does not refetch a large repo.
  git clone --quiet --no-checkout "$UPSTREAM" "$WORK/llama-er"
fi
git -C "$WORK/llama-er" fetch --quiet origin || true
git -C "$WORK/llama-er" checkout --quiet "$LLAMA_COMMIT"

# The two sources the patch includes, placed where each include resolves from:
#   src/llama-memory-recurrent.h includes "edgerock-rma.h"      -> src/
#   src/models/qwen35.cpp        includes "edgerock-rma-graph.inc" -> src/models/
cp "$WORK/backend/edgerock-rma.h" "$WORK/llama-er/src/edgerock-rma.h"
cp "$WORK/backend/edgerock-rma-graph.inc" "$WORK/llama-er/src/models/edgerock-rma-graph.inc"

# Applied as a diff rather than by running the repository's patch_llama.py: the two make the same
# edits, and a declarative patch does not execute third-party code to do it.
git -C "$WORK/llama-er" apply --verbose "$WORK/backend/llama-cpp-runtime.patch"

cmake -S "$WORK/llama-er" -B "$WORK/llama-er/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=OFF -DGGML_CUDA=OFF \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build "$WORK/llama-er/build" --target llama-server llama-cli -j "$(getconf _NPROCESSORS_ONLN)"

echo "built: $WORK/llama-er/build/bin/llama-server"
echo "built: $WORK/llama-er/build/bin/llama-cli"
