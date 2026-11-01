#!/bin/bash
# Two-arm qualification on a single box, strictly sequential, for a pod the fleet worker cannot
# drive (qual-worker-two-arm.sh bootstraps from S3 and reports to an EC2 serial console).
#
# Supersedes benchmark-results/2026-10-31-maple-qualification/run-two-arm.sh, which is left in place
# as the record of the runs it produced. See two-arm-protocol.sh for what that script got wrong.
set -e
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$SCRIPT_DIR/two-arm-protocol.sh"

ART="$1"; MODEL_ID="$2"; TPL="$3"; WORKLOAD="$4"; TOPK="$5"; OUT="$6"
[ -n "$OUT" ] || { echo "usage: $0 <artifact> <model-id> <template> <workload> <top-k> <out-dir>"; exit 2; }
[ -f "$ART" ] || { echo "artifact does not exist: $ART"; exit 2; }
THREADS=${THREADS:-16}; CONTEXT=${CONTEXT:-2048}; MAXTOK=${MAXTOK:-256}; PORT=${PORT:-18080}
JAVA=${JAVA_BIN:-/root/jdk/bin/java}
CP=$(ls /root/models-rag-bench/lib/*.jar | tr '\n' ':')
KERNEL=${KERNEL:-/root/model-kernels/target/release/libjmodels_kernels.so}
SERVER=${SERVER:-/root/llama.cpp/build/bin/llama-server}
: "${BACKEND_VERSION:?BACKEND_VERSION is required: it is stamped into the candidate report}"
mkdir -p "$OUT"
ARGS=$(two_arm_common_args "$WORKLOAD" "$TPL" "$TOPK" "$CONTEXT" "$THREADS" "$MAXTOK")

# /proc/loadavg inside this container reports the HOST, not our cpuset: cpu.max carries no quota and
# /proc/cpuinfo lists every core on the machine while nproc reports the 16 we are pinned to. A
# reading of 30 on a 256-core host is a neighbour's work, so it is recorded for the file and not
# used as a gate. Pairing the arms in one session on one box is what controls for the neighbour.
echo "### host: $(nproc) vCPU of $(grep -c processor /proc/cpuinfo) on host, cpu.max=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null)"
echo "### $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2), avx512f=$(grep -c avx512f /proc/cpuinfo)"
echo "### host loadavg before: $(cat /proc/loadavg)   [host-wide, not ours]"
echo "### protocol: warmups=$TWO_ARM_WARMUPS iterations=$TWO_ARM_ITERATIONS threads=$THREADS"

echo "### ARM 1 candidate (rust-ffm) model=$MODEL_ID wl=$WORKLOAD tpl=$TPL topk=$TOPK"
$JAVA --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
  -Dmodels.native.kernels.library=$KERNEL -cp "$CP" \
  com.integrallis.models.rag.RagBenchmarkCli \
  --framework plain-java --backend rust-ffm --backend-version "$BACKEND_VERSION" \
  --model "$ART" --model-id "$MODEL_ID" $ARGS \
  --output "$OUT/candidate.json" > "$OUT/candidate.log" 2>&1
echo "### candidate rc=$?"

# COMPARATOR picks the engine. ollama is the default because it is the only one that can produce a
# QUALIFIED verdict here: RagProductionQualificationPolicy's AUTHORITATIVE_ENGINES is
# {ollama, transformers}, and while a llama.cpp arm is admitted at 0.45/2.0 it can never be the
# backend that qualifies a model -- it reaches FAILED_RELATIVE_GATE at best unless a reviewed
# --publisher-reference is supplied, and the two pinned references exist only because those
# publishers ship their own llama.cpp (a fork, and a patch). Upstream serves mellum, lfm2moe and
# maple without either, so the publisher-reference route does not apply to them.
COMPARATOR=${COMPARATOR:-ollama}

if [ "$COMPARATOR" = "ollama" ]; then
  export HOME=${HOME:-/root}
  export OLLAMA_MODELS=${OLLAMA_MODELS:-/root/ollama}
  # HOME must be set before any ollama invocation: envconfig resolves the model directory while
  # building the CLI, so `serve` and `create` alike die on an unset HOME before parsing an argument.
  command -v ollama >/dev/null || { echo "### ollama is not installed"; exit 1; }
  CODE=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:11434/api/tags 2>/dev/null || echo 000)
  [ "$CODE" = "200" ] || (OLLAMA_HOST=127.0.0.1:11434 OLLAMA_MAX_LOADED_MODELS=1 nohup ollama serve >> /root/ollama-serve.log 2>&1 &)
  for i in $(seq 1 120); do
    CODE=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:11434/api/tags 2>/dev/null || echo 000)
    [ "$CODE" = "200" ] && { echo "### ollama ready after ${i}s"; break; }
    sleep 1
  done
  [ "$CODE" = "200" ] || { echo "### OLLAMA NEVER ANSWERED 200 -- aborting"; exit 1; }
  # `ollama create` from this exact file, never a registry copy of the same name, so the layer
  # digest equals the candidate's artifactSha256 and the comparison is not excluded.
  printf 'FROM %s\n' "$ART" > "$OUT/Modelfile"
  ollama create "qual-$MODEL_ID" -f "$OUT/Modelfile" > "$OUT/create.log" 2>&1 || {
    echo "### ollama create failed"; tail -5 "$OUT/create.log"; exit 1; }
  # Warm the weights off disk so the first measured case is not paying a 5-20 GB cold read. The
  # protocol's --warmups covers JIT and caches on our side; it does not cover the comparator's load.
  curl -s http://127.0.0.1:11434/api/generate --max-time 900 \
    -d "{\"model\":\"qual-$MODEL_ID\",\"prompt\":\"warm\",\"stream\":false,\"options\":{\"num_predict\":1,\"num_thread\":$THREADS}}" \
    > "$OUT/warm.json" 2>&1 || true
  COMPARATOR_ARGS="--backend ollama --backend-version ollama@$(ollama --version 2>&1 | head -1 | tr -d '\n' | tr ' ' '-')"
  COMPARATOR_ARGS="$COMPARATOR_ARGS --endpoint http://127.0.0.1:11434 --model qual-$MODEL_ID"
else
  echo "### starting llama-server"
  $SERVER -m "$ART" -c $CONTEXT -t $THREADS --port $PORT --host 127.0.0.1 > "$OUT/server.log" 2>&1 &
  SPID=$!
  trap 'kill $SPID 2>/dev/null' EXIT
  READY=0
  for i in $(seq 1 300); do
    # Readiness is the HTTP status, never the body: a loading server answers 503 with prose that
    # contains the word "ok".
    CODE=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null || echo 000)
    [ "$CODE" = "200" ] && { READY=1; echo "### server ready after ${i}s"; break; }
    sleep 1
  done
  [ "$READY" = "1" ] || { echo "### SERVER NEVER REACHED HTTP 200 (last $CODE) -- aborting rather than recording an error arm"; tail -25 "$OUT/server.log"; exit 1; }
  : "${COMPARATOR_VERSION:?COMPARATOR_VERSION is required for a llama.cpp arm}"
  COMPARATOR_ARGS="--backend llama.cpp --backend-version $COMPARATOR_VERSION"
  COMPARATOR_ARGS="$COMPARATOR_ARGS --endpoint http://127.0.0.1:$PORT --model $ART"
fi

# --artifact is what makes this report admissible: without it artifactSha256 is null and the policy
# excludes the comparison as "artifact SHA-256 differs". RagBenchmarkCli now refuses the run
# outright, so this cannot be forgotten again, but it is passed explicitly here for the reader.
echo "### ARM 2 comparator ($COMPARATOR)"
$JAVA --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -cp "$CP" \
  com.integrallis.models.rag.RagBenchmarkCli \
  --framework plain-java $COMPARATOR_ARGS \
  --artifact "$ART" --model-id "$MODEL_ID" $ARGS \
  --output "$OUT/comparator.json" > "$OUT/comparator.log" 2>&1
echo "### comparator rc=$?"
echo "### host loadavg after: $(cat /proc/loadavg)   [host-wide, not ours]"

# A verdict comes from the policy, never from dividing the two reports by hand.
$JAVA -cp "$CP" com.integrallis.models.rag.RagQualificationCli \
  --candidate "$OUT/candidate.json" --comparator "$OUT/comparator.json" \
  --output "$OUT/verdict.json" > "$OUT/verdict.log" 2>&1 || true
python3 "$SCRIPT_DIR/print-verdict.py" "$OUT/verdict.json" || true
echo DONE > "$OUT/arms.done"
