#!/usr/bin/env bash
# S1: convert Qwen2.5-Math-1.5B-Instruct from the official safetensors at a pinned revision with a
# pinned llama.cpp, because no official GGUF of it exists. Mirrors the September study's
# Amendment 1 section 6. Produces Q8_0 and Q4_K_M and prints their sha256 for models.json.
set -euo pipefail
WORK="${1:?work dir}"
REVISION=aafeb0fc6f
LLAMA_TAG=v1.0.0            # ggml-org/llama.cpp release of 2026-11-13, commit 2d8a3250
mkdir -p "$WORK" && cd "$WORK"
if [ ! -d llama.cpp ]; then git clone -q --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp.git; fi
( cd llama.cpp && git rev-parse HEAD > ../llama.cpp.commit && cmake -B build -DGGML_NATIVE=ON >/dev/null && cmake --build build --target llama-quantize -j >/dev/null )
python3 -m pip install -q -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt huggingface_hub
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-Math-1.5B-Instruct", revision="aafeb0fc6f", local_dir="hf-math", allow_patterns=["*.json","*.safetensors","*.txt"])
PY
python3 llama.cpp/convert_hf_to_gguf.py hf-math --outtype f16 --outfile qwen2.5-math-1.5b-instruct-f16.gguf
./llama.cpp/build/bin/llama-quantize qwen2.5-math-1.5b-instruct-f16.gguf qwen2.5-math-1.5b-instruct-q8_0.gguf Q8_0 >/dev/null
./llama.cpp/build/bin/llama-quantize qwen2.5-math-1.5b-instruct-f16.gguf qwen2.5-math-1.5b-instruct-q4_k_m.gguf Q4_K_M >/dev/null
echo "llama.cpp commit: $(cat llama.cpp.commit)"
sha256sum qwen2.5-math-1.5b-instruct-q8_0.gguf qwen2.5-math-1.5b-instruct-q4_k_m.gguf
