#!/usr/bin/env bash

set -euo pipefail

OLLAMA_VERSION=${OLLAMA_VERSION:-v0.32.0}
OLLAMA_SHA256=${OLLAMA_SHA256:-56362d7609dfa9e35aaebb7c9cab25605d8f0528ec3d5d585dc83d6642002bab}
LLAMA_BUILD=${LLAMA_BUILD:-b10012}
LLAMA_SHA256=${LLAMA_SHA256:-f5d5e201aab21b889cbd731b2fd85aa14b112ee9ee55a68fceeac7ee9905aba4}

if [[ $(id -u) -ne 0 ]]; then
  echo "run this bootstrap as root" >&2
  exit 1
fi
if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
  echo "this pinned bootstrap supports Linux x86_64 only" >&2
  exit 1
fi
for command in curl sha256sum tar zstd; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "required command not found: $command" >&2
    exit 1
  fi
done

TEMP_DIR=$(mktemp -d)
cleanup() {
  rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

OLLAMA_ARCHIVE="$TEMP_DIR/ollama-linux-amd64.tar.zst"
LLAMA_ARCHIVE="$TEMP_DIR/llama-${LLAMA_BUILD}-bin-ubuntu-x64.tar.gz"
OLLAMA_URL="https://github.com/ollama/ollama/releases/download/$OLLAMA_VERSION/ollama-linux-amd64.tar.zst"
LLAMA_URL="https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_BUILD/llama-${LLAMA_BUILD}-bin-ubuntu-x64.tar.gz"

curl -fL "$OLLAMA_URL" -o "$OLLAMA_ARCHIVE"
printf '%s  %s\n' "$OLLAMA_SHA256" "$OLLAMA_ARCHIVE" | sha256sum --check --strict
tar --use-compress-program=unzstd -xf "$OLLAMA_ARCHIVE" -C /usr/local

LLAMA_HOME="/opt/llama.cpp/$LLAMA_BUILD"
install -d -m 0755 "$LLAMA_HOME"
curl -fL "$LLAMA_URL" -o "$LLAMA_ARCHIVE"
printf '%s  %s\n' "$LLAMA_SHA256" "$LLAMA_ARCHIVE" | sha256sum --check --strict
tar -xzf "$LLAMA_ARCHIVE" -C "$LLAMA_HOME"

for binary in llama-server llama-cli llama-bench; do
  binary_path=$(find "$LLAMA_HOME" -type f -name "$binary" -perm -u+x -print -quit)
  if [[ -z "$binary_path" ]]; then
    echo "$binary was not present in the llama.cpp archive" >&2
    exit 1
  fi
  printf '#!/usr/bin/env bash\nexec %q "$@"\n' "$binary_path" >"/usr/local/bin/$binary"
  chmod 0755 "/usr/local/bin/$binary"
done

ollama --version
llama-server --version
