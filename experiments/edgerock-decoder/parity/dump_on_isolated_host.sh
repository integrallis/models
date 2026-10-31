#!/bin/bash
# Produce the memory-layer parity fixture on a Linux host, because this workstation cannot.
#
# modeling_edgerock.py imports transformers.models.qwen3_5 and
# transformers.cache_utils.{Cache, DynamicSlidingWindowLayer, LinearAttentionLayer}, which exist
# only in transformers 5.x; that requires torch >= 2.6, and the last PyTorch wheel for x86-64
# macOS is 2.2.2. On this Mac transformers disables PyTorch outright:
#   "[transformers] Disabling PyTorch because PyTorch >= 2.6 is required but found 2.2.2"
#
# Those imports are also the end of the backbone question: EdgeRock subclasses Qwen3_5GatedDeltaNet,
# Qwen3_5Attention, Qwen3_5DecoderLayer, Qwen3_5MLP and Qwen3_5ForCausalLM by name.
#
# Only modeling_edgerock.py and its config are fetched. An earlier version pulled the whole
# 8.6 GB checkpoint and dumped one real layer, which produced an 848 MB fixture whose expected
# output was all zeros -- 3 tokens never reach commit_chunk 256, so the bank stayed empty and the
# test would have passed whatever the read did.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
TOOLS=/Users/briansam-bodden/Code/java-ai/articles/models-deep-dive/tools
OUT=${OUT:-$HERE/fixtures}
LAYER=${LAYER:-3}
TOKENS=${TOKENS:-3}
mkdir -p "$OUT"

: "${HETZNER_TOKEN:?export HETZNER_TOKEN}"
# The fixture is small; the smallest dedicated shape is enough.
export TYPE=${TYPE:-ccx13}
# shellcheck disable=SC1091
. "$TOOLS/isolated_host.sh"

isolated_host_up edgerock-parity
remote 'apt-get -qq update >/dev/null 2>&1; apt-get -qq install -y python3-venv python3-pip curl >/dev/null 2>&1; python3 --version'

push "$HERE/dump_memory_layer.py" /root/dump_memory_layer.py

remote 'set -euo pipefail
  mkdir -p /root/edgerock && cd /root/edgerock
  base=https://huggingface.co/rockmatrix-ai/EdgeRock/resolve/main
  # Only the reference module and its config. The fixture is built from the class at small
  # dimensions, so the 8.6 GB checkpoint is not needed: it would conflate our arithmetic against
  # theirs with Q4_K quantisation against float32.
  for f in config.json configuration_edgerock.py modeling_edgerock.py; do
    curl -sfL -o "$f" "$base/$f"
  done
  ls -la'

remote 'set -euo pipefail
  python3 -m venv /root/venv
  /root/venv/bin/pip -q install --upgrade pip
  /root/venv/bin/pip -q install torch --index-url https://download.pytorch.org/whl/cpu
  /root/venv/bin/pip -q install transformers safetensors accelerate numpy
  /root/venv/bin/python -c "import torch, transformers; print(\"  torch\", torch.__version__, \"transformers\", transformers.__version__)"'

remote "set -euo pipefail
  /root/venv/bin/python /root/dump_memory_layer.py \
    --reference /root/edgerock/modeling_edgerock.py \
    --tokens $TOKENS \
    --out /root/memory-layer-parity.json"

pull "/root/memory-layer-parity.json" "$OUT/"
echo "=== fixture pulled:"
ls -la "$OUT"
