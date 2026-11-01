#!/bin/bash
# Two-arm qualification on the pod. Strictly sequential, identical workload parameters to both
# arms -- sameWorkload() compares workload, corpus, cases, template, topK, maxOutputTokens,
# contextLength, THREADS, groundingPolicy and the generation controls, and a comparator that
# defaults its thread count is excluded with "benchmark workload differs".
set -e
ART="$1"; MODEL_ID="$2"; TPL="$3"; WORKLOAD="$4"; TOPK="$5"; OUT="$6"
THREADS=16; CONTEXT=2048; MAXTOK=256; PORT=18080
JAVA=/root/jdk/bin/java
CP=$(ls /root/models-rag-bench/lib/*.jar | tr '\n' ':')
KERNEL=/root/model-kernels/target/release/libjmodels_kernels.so
SERVER=/root/llama.cpp/build/bin/llama-server
mkdir -p "$OUT"

echo "### host: $(nproc) vCPU, $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2), avx512=$(grep -c avx512f /proc/cpuinfo | head -1)"
echo "### load before: $(cat /proc/loadavg)"
echo "### ARM 1 candidate (rust-ffm) model=$MODEL_ID wl=$WORKLOAD tpl=$TPL topk=$TOPK threads=$THREADS"
$JAVA --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
  -Dmodels.native.kernels.library=$KERNEL -cp "$CP" \
  com.integrallis.models.rag.RagBenchmarkCli \
  --framework plain-java --backend rust-ffm --backend-version "$BACKEND_VERSION" \
  --model "$ART" --model-id "$MODEL_ID" --workload "$WORKLOAD" --prompt-template "$TPL" \
  --top-k "$TOPK" --context $CONTEXT --threads $THREADS --max-tokens $MAXTOK \
  --warmups 0 --iterations 1 --output "$OUT/candidate.json" > "$OUT/candidate.log" 2>&1
echo "### candidate rc=$? -> $(ls -la $OUT/candidate.json 2>/dev/null | awk '{print $5}') bytes"

echo "### starting llama-server at $(cd /root/llama.cpp && git log -1 --format=%h)"
$SERVER -m "$ART" -c $CONTEXT -t $THREADS --port $PORT --host 127.0.0.1 > "$OUT/server.log" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null' EXIT
READY=0
for i in $(seq 1 300); do
  CODE=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null || echo 000)
  [ "$CODE" = "200" ] && { READY=1; echo "### server ready after ${i}s (HTTP 200)"; break; }
  sleep 1
done
[ "$READY" = "1" ] || { echo "### SERVER NEVER REACHED HTTP 200 (last $CODE) -- aborting rather than recording an error arm"; tail -25 "$OUT/server.log"; exit 1; }

echo "### ARM 2 comparator (llama.cpp)"
$JAVA --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -cp "$CP" \
  com.integrallis.models.rag.RagBenchmarkCli \
  --framework plain-java --backend llama.cpp \
  --backend-version "ggml-org/llama.cpp@3f03b17724ad8b69040825f45c43c59b092a0244" \
  --endpoint "http://127.0.0.1:$PORT" \
  --model "$ART" --model-id "$MODEL_ID" --workload "$WORKLOAD" --prompt-template "$TPL" \
  --top-k "$TOPK" --context $CONTEXT --threads $THREADS --max-tokens $MAXTOK \
  --warmups 0 --iterations 1 --output "$OUT/comparator.json" > "$OUT/comparator.log" 2>&1
echo "### comparator rc=$? -> $(ls -la $OUT/comparator.json 2>/dev/null | awk '{print $5}') bytes"
echo "### load after: $(cat /proc/loadavg)"
echo DONE > "$OUT/arms.done"
