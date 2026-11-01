#!/usr/bin/env bash
set -euo pipefail

revision="b5c6e9da73abc3711f593f705371cdbe9e0fe422"
repository="mixedbread-ai/mxbai-rerank-xsmall-v1"
destination="${1:-${HOME}/.jvllm/models/mxbai-rerank-xsmall-v1}"

mkdir -p "$destination"

download() {
  local path="$1"
  local expected_size="$2"
  local expected_sha="$3"
  local target="$destination/$path"

  if [[ -f "$target" ]] && [[ "$(wc -c < "$target" | tr -d ' ')" == "$expected_size" ]] \
      && [[ "$(checksum "$target")" == "$expected_sha" ]]; then
    return
  fi

  local temporary="$target.part"
  curl --fail --location --retry 3 --retry-all-errors \
    "https://huggingface.co/${repository}/resolve/${revision}/${path}" \
    --output "$temporary"
  if [[ "$(wc -c < "$temporary" | tr -d ' ')" != "$expected_size" ]]; then
    echo "Size mismatch for $path" >&2
    exit 1
  fi
  if [[ "$(checksum "$temporary")" != "$expected_sha" ]]; then
    echo "SHA-256 mismatch for $path" >&2
    exit 1
  fi
  mv "$temporary" "$target"
}

checksum() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

download config.json 968 470a53befc79da411cc04e466770d9f219f3c14adb276bfa0a58df28774ceade
download tokenizer.json 8649139 305674b4d785287feecfb5f73f24aa75e9b57c87c579cfe24fbd207987d4b4c4
download model.safetensors 141685186 a29bc212faf59c136ad0fd5712ecd2346e7b32c44a25b690625bc9ecebb14b8f

printf '%s\n' "$destination"
