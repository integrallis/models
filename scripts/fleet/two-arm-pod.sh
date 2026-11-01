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
: "${COMPARATOR_VERSION:?COMPARATOR_VERSION is required: it identifies the comparator build}"
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

# --artifact is what makes this report admissible: without it artifactSha256 is null and the policy
# excludes the comparison as "artifact SHA-256 differs". RagBenchmarkCli now refuses the run
# outright, so this cannot be forgotten again, but it is passed explicitly here for the reader.
echo "### ARM 2 comparator (llama.cpp)"
$JAVA --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -cp "$CP" \
  com.integrallis.models.rag.RagBenchmarkCli \
  --framework plain-java --backend llama.cpp --backend-version "$COMPARATOR_VERSION" \
  --endpoint "http://127.0.0.1:$PORT" \
  --model "$ART" --artifact "$ART" --model-id "$MODEL_ID" $ARGS \
  --output "$OUT/comparator.json" > "$OUT/comparator.log" 2>&1
echo "### comparator rc=$?"
echo "### host loadavg after: $(cat /proc/loadavg)   [host-wide, not ours]"

# A verdict comes from the policy, never from dividing the two reports by hand. A llama.cpp
# comparator is admitted at 0.45/2.0 but is not an AUTHORITATIVE_ENGINE, so the verdict it can
# reach alone is FAILED_RELATIVE_GATE -- it takes ollama, transformers, or a reviewed
# --publisher-reference to reach QUALIFIED. Run it anyway: the comparison it prints is the evidence.
$JAVA -cp "$CP" com.integrallis.models.rag.RagQualificationCli \
  --candidate "$OUT/candidate.json" --comparator "$OUT/comparator.json" \
  --output "$OUT/verdict.json" > "$OUT/verdict.log" 2>&1 || true
python3 - "$OUT/verdict.json" <<'PY' || true
import json,sys
q=json.load(open(sys.argv[1]))['qualification']
print('### verdict:',q['verdict'],'qualified=',q['qualified'])
print('### contribution: answerRate=%.4f correctRate=%.4f'%(q['modelAnswerRate'],q['modelAnswerCorrectRate']))
for c in q.get('comparisons',[]):
    print('### vs %s: decodeRatio=%.4f e2eRatio=%.4f qualified=%s'
          %(c['comparatorBackend'],c['decodeThroughputRatio'],c['endToEndLatencyRatio'],c['qualified']))
for k,v in (q.get('exclusions') or {}).items():
    print('### EXCLUDED %s: %s'%(k,v))
PY
echo DONE > "$OUT/arms.done"
