#!/usr/bin/env bash
# S1: convert Qwen2.5-Math-1.5B-Instruct from the official safetensors at a pinned revision with a
# pinned llama.cpp, because no official GGUF of it exists. Mirrors the September study's
# Amendment 1 section 6. Produces Q8_0 and Q4_K_M and prints their sha256 for models.json.
set -euo pipefail
WORK="${1:?work dir}"
REVISION=aafeb0fc6f
LLAMA_TAG=v1.0.0            # ggml-org/llama.cpp release of 2026-11-13; the tag resolves to b1b770d9 (recorded in llama.cpp.commit)
mkdir -p "$WORK" && cd "$WORK"
if [ ! -d llama.cpp ]; then git clone -q --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp.git; fi
( cd llama.cpp && git rev-parse HEAD > ../llama.cpp.commit && cmake -B build -DGGML_NATIVE=ON >/dev/null && cmake --build build --target llama-quantize -j >/dev/null )
# A fresh box has no system pip; prefer a uv-managed interpreter when uv is present.
if command -v uv >/dev/null 2>&1; then
  [ -x "$WORK/venv/bin/python" ] || uv venv --quiet --python 3.12 "$WORK/venv"
  uv pip install --quiet --python "$WORK/venv/bin/python" -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt huggingface_hub
  PY="$WORK/venv/bin/python"
else
  python3 -m pip install -q -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt huggingface_hub
  PY=python3
fi
"$PY" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-Math-1.5B-Instruct", revision="aafeb0fc6f", local_dir="hf-math", allow_patterns=["*.json","*.safetensors","*.txt"])
PY
"$PY" llama.cpp/convert_hf_to_gguf.py hf-math --outtype f16 --outfile qwen2.5-math-1.5b-instruct-f16.gguf
./llama.cpp/build/bin/llama-quantize qwen2.5-math-1.5b-instruct-f16.gguf qwen2.5-math-1.5b-instruct-q8_0.gguf Q8_0 >/dev/null
./llama.cpp/build/bin/llama-quantize qwen2.5-math-1.5b-instruct-f16.gguf qwen2.5-math-1.5b-instruct-q4_k_m.gguf Q4_K_M >/dev/null
# Gate G0 compares tokenizer metadata field by field. The official Qwen2.5 GGUFs carry no
# tokenizer.ggml.add_eos_token key; this converter writes it as false, the same meaning, so G0 read
# a mismatch on the first S1 attempt. Remove the key so the converted files carry exactly the
# official files' tokenizer fields; vocabulary, merges, tokens, types and special tokens are untouched.
for q in q8_0 q4_k_m; do
  PYTHONPATH=llama.cpp/gguf-py "$PY" llama.cpp/gguf-py/gguf/scripts/gguf_new_metadata.py --force \
    --remove-metadata tokenizer.ggml.add_eos_token \
    "qwen2.5-math-1.5b-instruct-$q.gguf" "qwen2.5-math-1.5b-instruct-$q.fixed.gguf" >/dev/null
  mv -f "qwen2.5-math-1.5b-instruct-$q.fixed.gguf" "qwen2.5-math-1.5b-instruct-$q.gguf"
done
echo "llama.cpp commit: $(cat llama.cpp.commit)"
sha256sum qwen2.5-math-1.5b-instruct-q8_0.gguf qwen2.5-math-1.5b-instruct-q4_k_m.gguf
