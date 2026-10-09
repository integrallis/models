#!/bin/bash
# Catalogue-wide default-configuration smoke on the SHIPPED library, Models 0.3.54.
#
# CAMPAIGN-RUNBOOK.md: "Run the smoke across the whole catalogue whenever the released library
# changes the grounding policy, the decoder, or the kernel -- not only across the entries that are
# new." 0.3.54 changed kernel dispatch defaults (ten of eleven pure-Java plan optimizations now
# enabled when unset; five native settings removed), so every published entry needs a current proof
# that it answers its whole workload correctly under library defaults.
#
# ONE ARM BY DESIGN, and that is not the two-arm rule being broken. The two-arm rule governs a
# qualification *verdict*, which this does not produce: a smoke is a correctness claim about the
# shipped defaults, not a comparative performance claim. No Ollama here, which also removes the
# $HOME, systemd and OLLAMA_MODELS traps the two-arm worker has to handle.
#
# Reporting goes to the SERIAL CONSOLE first, as in qual-worker-two-arm.sh, because
# get-console-output needs no key, no agent and no bucket and survives termination.
SHARD=${SHARD:-500}
DEADLINE_SECONDS=18000
BUCKET=models-qual-077051030817
# Overridable so the identical protocol can be pointed at an earlier library: attributing a smoke
# failure to a release needs the same worker, same shard and same gate on both sides, with the
# library as the only variable.
PAYLOAD=${PAYLOAD:-models-rag-bench-0.3.54-v23.tar}
# No separate kernel jar. The dist's backend-native-0.3.54.jar is the artifact published to Maven
# Central and already carries every platform's kernel, linux-x86_64 included. Adding the same jar
# again under another name put two copies of
# META-INF/models/native/linux-x86_64/native.properties on the classpath and
# BundledNativeKernelLibrary.uniqueResource refused the ambiguity -- correctly -- which failed all
# eight pilot jobs with rc=1. The older two-arm worker needs that extra jar because its dist was
# built without Linux natives; this one is not.
# Names the shipped library exactly: version + grounding policy + the released commit (tag v0.3.54).
BACKEND_VERSION=${BACKEND_VERSION:-"models@0.3.54+v23-85f74b1d8345"}
LOG=/var/log/smoke-worker.log
say() { printf '[smoke %s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$LOG" >/dev/console 2>/dev/null; }
exec 2>>"$LOG"
set -x
STATUS=STARTED
say "boot shard=$SHARD payload=$PAYLOAD"

finish() {
  rc=$?
  say "FINAL_STATUS=$STATUS exit=$rc"
  if command -v aws >/dev/null 2>&1; then
    echo "$STATUS" > /work/out/STATUS 2>/dev/null
    aws s3 cp /work/out/STATUS "s3://$BUCKET/results/shard-$SHARD/" --only-show-errors 2>/dev/null
    aws s3 cp "$LOG" "s3://$BUCKET/results/shard-$SHARD/smoke-worker-$SHARD.log" --only-show-errors 2>/dev/null
    [ -f /work/out/progress.tsv ] && aws s3 cp /work/out/progress.tsv "s3://$BUCKET/results/shard-$SHARD/" --only-show-errors 2>/dev/null
    say "uploaded to s3://$BUCKET/results/shard-$SHARD/"
  else
    say "no aws cli: console output is the only record"
  fi
  if [ "$STATUS" != "SHARD_COMPLETE" ]; then
    say "holding 45m for inspection"
    sleep 2700
  fi
  say "shutting down"
  shutdown -h now
}
trap finish EXIT
mkdir -p /work/models /work/out /opt/jdk || { STATUS=NO_WORKDIR; exit 1; }

# Tested by whether a response ARRIVES: s3 root answers an unauthenticated HEAD with 307, so
# `curl -f` would report NO_EGRESS on a healthy box. 000 means no response.
EGRESS_CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 25 https://s3.amazonaws.com 2>/dev/null)
if [ "$EGRESS_CODE" = "000" ] || [ -z "$EGRESS_CODE" ]; then
  STATUS=NO_EGRESS; say "no outbound HTTPS (curl code=$EGRESS_CODE)"; exit 1
fi
say "egress ok (s3 root http=$EGRESS_CODE)"

# The bundled installer is the PRIMARY path: `apt-cache policy awscli` reports no candidate on this
# AMI, so apt-get install awscli can never succeed.
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq || say "apt-get update failed (continuing)"
apt-get install -y -qq unzip python3 || say "apt-get install failed (continuing)"
if ! command -v aws >/dev/null 2>&1; then
  say "installing aws cli from the bundled installer"
  if curl -sfL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscli.zip; then
    (cd /tmp && unzip -qo awscli.zip && ./aws/install --update >/dev/null 2>&1); hash -r
  fi
fi
command -v aws >/dev/null 2>&1 || { STATUS=NO_AWS_CLI; exit 1; }
command -v python3 >/dev/null 2>&1 || { STATUS=NO_PYTHON; exit 1; }
curl -sfL "https://api.adoptium.net/v3/binary/latest/25/ga/linux/x64/jdk/hotspot/normal/eclipse" -o /tmp/jdk.tgz \
  || { STATUS=NO_JDK_DOWNLOAD; exit 1; }
tar xzf /tmp/jdk.tgz -C /opt/jdk --strip-components=1 || { STATUS=NO_JDK_UNPACK; exit 1; }
export JAVA_HOME=/opt/jdk; export PATH=$JAVA_HOME/bin:$PATH
[ -x /opt/jdk/bin/java ] || { STATUS=NO_JDK; exit 1; }
say "jdk $(/opt/jdk/bin/java -version 2>&1 | head -1)"

cd /work
aws s3 cp "s3://$BUCKET/payload/$PAYLOAD" . --only-show-errors
aws s3 cp "s3://$BUCKET/payload/smoke-shard-$SHARD.json" /work/shard.json --only-show-errors
[ -s "$PAYLOAD" ]      || { STATUS=NO_DIST; exit 1; }
[ -s /work/shard.json ] || { STATUS=NO_SHARD; exit 1; }
tar xf "$PAYLOAD" || { STATUS=NO_DIST_UNPACK; exit 1; }
DIST=$(ls -d /work/models-rag-bench-*/ 2>/dev/null | head -1)
[ -n "$DIST" ] || DIST=/work/models-rag-bench
CP="$DIST/lib/*"
NATIVE_JAR=$(ls "$DIST"/lib/backend-native-*.jar 2>/dev/null | head -1)
if ! unzip -l "$NATIVE_JAR" 2>/dev/null | grep -q "linux-x86_64/libjmodels_kernels.so"; then
  STATUS=NO_LINUX_KERNEL_IN_DIST; say "dist backend-native carries no linux-x86_64 kernel"; exit 1
fi
say "dist carries the linux-x86_64 kernel"
JOBS=$(python3 -c "import json;print(len(json.load(open('/work/shard.json'))))")
THREADS=$(nproc 2>/dev/null || echo 16)
# A RagBenchmarkCli report has no tuning-properties field, so the artifact cannot prove "nothing
# was configured". It is asserted here, where it is knowable: any models.* property reaching the JVM
# through an options variable would silently make this a tuned run masquerading as a default one.
for var in JAVA_TOOL_OPTIONS JAVA_OPTS JDK_JAVA_OPTIONS _JAVA_OPTIONS; do
  val=$(printenv "$var" 2>/dev/null || true)
  case "$val" in
    *models.*) STATUS=TUNED_ENVIRONMENT; say "$var carries a models.* property: $val"; exit 1 ;;
  esac
done
say "environment clean: no models.* property in any JVM options variable"
say "payload ok, $JOBS jobs, threads=$THREADS, backend-version=$BACKEND_VERSION"
STATUS=RUNNING
T_START=$(date +%s)

# The gate, exactly as CAMPAIGN-RUNBOOK.md states it for a default-configuration smoke.
cat > /work/gate.py <<'GATE'
import json, sys
r = json.load(open(sys.argv[1]))
s = r.get('summary') or {}
env = (r.get('backendDiagnostics') or {}).get('environment') or {}
st = r.get('settings') or {}
reasons = []
if s.get('correctAnswerRate') != 1:   reasons.append(f"correctAnswerRate={s.get('correctAnswerRate')}")
if s.get('abstentionAccuracy') != 1:  reasons.append(f"abstentionAccuracy={s.get('abstentionAccuracy')}")
if r.get('failures'):                 reasons.append(f"failures={len(r['failures'])}")
if st.get('warmups') != 0:            reasons.append(f"warmups={st.get('warmups')}")
if st.get('iterations') != 1:         reasons.append(f"iterations={st.get('iterations')}")
gc = st.get('generationControls') or {}
if gc.get('promptCache') != 'longest-common-prefix':
    reasons.append(f"promptCache={gc.get('promptCache')}")
if r.get('backend') == 'rust-ffm' and env.get('native-quantized-decode') != 'true':
    # 0.3.54 removed the setting: the shim always serves decode, so "false" means it fell back.
    reasons.append(f"native-quantized-decode={env.get('native-quantized-decode')}")
print('PASS' if not reasons else 'FAIL:' + ';'.join(reasons))
GATE

done_n=0
while [ "$done_n" -lt "$JOBS" ]; do
  if [ $(( $(date +%s) - T_START )) -gt "$DEADLINE_SECONDS" ]; then
    STATUS=DEADLINE; say "deadline reached after $done_n/$JOBS"; break
  fi
  job=$(python3 -c "import json;print(json.dumps(json.load(open('/work/shard.json'))[$done_n]))")
  id=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['id'])" "$job")
  uri=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['uri'])" "$job")
  tpl=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['tpl'])" "$job")
  wl=$(python3 -c "import json,sys;print(json.loads(sys.argv[1]).get('wl') or 'general')" "$job")
  bk=$(python3 -c "import json,sys;print(json.loads(sys.argv[1]).get('backend') or 'rust-ffm')" "$job")
  gguf="/work/models/$id.gguf"
  T0=$(date +%s)
  say "start $id (backend=$bk template=$tpl workload=$wl)"
  if curl -sfL "$uri" -o "$gguf"; then
    # NO -Dmodels.* of any kind: the gate requires tuningSystemProperties to be empty, and the
    # whole point is to measure what a user gets with nothing configured.
    java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
      -cp "$CP" \
      com.integrallis.models.rag.RagBenchmarkCli \
      --framework plain-java --backend "$bk" --backend-version "$BACKEND_VERSION" \
      --model "$gguf" --model-id "$id" --workload "$wl" --prompt-template "$tpl" \
      --context 2048 --threads "$THREADS" --max-tokens 256 --warmups 0 --iterations 1 \
      --output "/work/out/$id.json" > "/work/out/$id.log" 2>&1
    rc=$?
    verdict="RUN_FAILED_rc$rc"
    if [ -s "/work/out/$id.json" ]; then
      verdict=$(python3 /work/gate.py "/work/out/$id.json" 2>/dev/null || echo "GATE_ERROR")
    fi
    printf '%s\t%s\t%s\n' "$id" "$verdict" "$(( $(date +%s)-T0 ))" >> /work/out/progress.tsv
    say "done $id -> $verdict in $(( $(date +%s)-T0 ))s"
    aws s3 cp "/work/out/$id.json" "s3://$BUCKET/results/shard-$SHARD/" --only-show-errors 2>/dev/null
    aws s3 cp "/work/out/$id.log"  "s3://$BUCKET/results/shard-$SHARD/" --only-show-errors 2>/dev/null
    aws s3 cp /work/out/progress.tsv "s3://$BUCKET/results/shard-$SHARD/" --only-show-errors 2>/dev/null
  else
    printf '%s\tDOWNLOAD_FAIL\t0\n' "$id" >> /work/out/progress.tsv
    say "download failed $id"
  fi
  rm -f "$gguf"
  done_n=$((done_n+1))
done
[ "$STATUS" = "DEADLINE" ] || STATUS=SHARD_COMPLETE
say "shard finished: $done_n/$JOBS"
