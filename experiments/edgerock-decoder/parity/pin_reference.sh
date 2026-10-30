#!/usr/bin/env bash
# Pin our EdgeRock output against the patched llama.cpp built by build_reference.sh.
#
# This is the correctness gate for the Java decoder. Until it has run, "the model produces coherent
# text" is all we know, and coherent text does not distinguish a correct implementation from one
# whose memory contributes nothing -- a plain Qwen3.5 hybrid is also fluent.
#
# Both arms must be greedy and must see the same token ids, so the prompt is sent raw: no chat
# template on either side. The publisher's own template (ChatML with a forced <think> opener,
# temperature 0) is in Modelfile.Q4_K_M and belongs in the qualification run, not here -- a
# template difference would show up as a decode difference and tell us nothing about the decoder.
set -euo pipefail

REFERENCE=${1:?usage: pin_reference.sh <llama-server-binary> <model.gguf> [port]}
MODEL=${2:?usage: pin_reference.sh <llama-server-binary> <model.gguf> [port]}
PORT=${3:-18080}
PROMPT=${PROMPT:-"The capital of France is"}
PREDICT=${PREDICT:-16}
THREADS=${THREADS:-$(getconf _NPROCESSORS_ONLN)}

# Flags follow the publisher's own receipt command, minus the CUDA-only ones.
"$REFERENCE" --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --ctx-size 16384 --batch-size 256 --ubatch-size 256 --parallel 1 \
  --cache-ram 0 --no-webui --no-context-shift --threads "$THREADS" &
server=$!
trap 'kill "$server" 2>/dev/null || true' EXIT

for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then break; fi
  sleep 2
done

# n_probs with post_sampling_probs gives the chosen token id per step, which is what we compare.
curl -fsS "http://127.0.0.1:${PORT}/completion" \
  -H 'Content-Type: application/json' \
  -d "$(printf '{"prompt":%s,"n_predict":%d,"temperature":0,"top_k":1,"cache_prompt":false}' \
        "$(printf '%s' "$PROMPT" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')" \
        "$PREDICT")" \
  | python3 -c '
import json, sys
d = json.load(sys.stdin)
print("reference text:", repr(d.get("content")))
t = d.get("timings") or {}
if t:
    print("reference prompt/s:", t.get("prompt_per_second"), "predicted/s:", t.get("predicted_per_second"))
print()
print("Compare against the Java arm:")
print("  ./gradlew :backend-java:edgeRockCompatibilityTest -Dmodels.fixtures.edgeRock4bGguf=<model>")
print("Our measured text on this prompt, 2026-10-30:")
print("  \" Paris. Paris is known for its historical landmarks and attractions. One of the most\"")
print("  tokens [11751, 13, 11751, 369, 3750, 364, 1141, 13263, 57902, 321, 37214, 13, 3648, 314, 279, 1379]")
print()
print("A mismatch is a defect in our decoder, not a tolerance to widen. The first places to look,")
print("in order of prior probability: mRoPE (we apply plain NeoX over rope.dimension_count while")
print("the file publishes rope.dimension_sections [11,11,10,0]), then the sliding-window span, then")
print("the memory read gate ordering.")
'
