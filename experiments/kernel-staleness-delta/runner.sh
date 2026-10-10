#!/bin/bash
# EC2 user-data payload for the kernel-staleness delta. See README.md for the registered protocol.
#
# One variable: which kernels JAR is on the classpath. Everything else -- payload, model, workload,
# template, context, max tokens, threads, decode threads, warmups, iterations, seed -- is identical
# between the two arms, and both orders are run on the same box because a first-run arm pays
# page-cache and JIT costs the second does not.
#
# Console-first: a failure before the tooling exists is only visible through get-console-output.
set -u
say() { printf '[kdelta %s] %s\n' "$(date -u +%H:%M:%S)" "$*" >/dev/console 2>&1; }
STATUS=BOOT
finish() {
  printf '%s\n' "$STATUS" > /work/STATUS 2>/dev/null
  aws s3 cp /work/STATUS "s3://$BUCKET/results/kdelta-$RUNID/STATUS" --only-show-errors 2>/dev/null
  for f in /work/out/*; do
    [ -e "$f" ] && aws s3 cp "$f" "s3://$BUCKET/results/kdelta-$RUNID/" --only-show-errors 2>/dev/null
  done
  say "finish status=$STATUS"
  shutdown -h now
  exit 0
}
trap finish EXIT

BUCKET=${BUCKET:-models-qual-077051030817}
RUNID=${RUNID:?RUNID required}
PAYLOAD=${PAYLOAD:?PAYLOAD required}
KERNEL_A=${KERNEL_A:?KERNEL_A required}
KERNEL_B=${KERNEL_B:?KERNEL_B required}
MODELS_SPEC=${MODELS_SPEC:?MODELS_SPEC required}
THREADS=${THREADS:-16}
DECODE_THREADS=${DECODE_THREADS:-8}

mkdir -p /work/out && cd /work
say "start runid=$RUNID"

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq unzip curl python3 >/dev/null 2>&1

curl -sfL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscli.zip \
  || { STATUS=NO_AWSCLI_DOWNLOAD; exit 1; }
unzip -q /tmp/awscli.zip -d /tmp && /tmp/aws/install >/dev/null 2>&1
command -v aws >/dev/null || { STATUS=NO_AWSCLI; exit 1; }

mkdir -p /opt/jdk
curl -sfL "https://api.adoptium.net/v3/binary/latest/25/ga/linux/x64/jdk/hotspot/normal/eclipse" -o /tmp/jdk.tgz \
  || { STATUS=NO_JDK_DOWNLOAD; exit 1; }
tar xzf /tmp/jdk.tgz -C /opt/jdk --strip-components=1 || { STATUS=NO_JDK_UNPACK; exit 1; }
export JAVA_HOME=/opt/jdk PATH=/opt/jdk/bin:$PATH
[ -x /opt/jdk/bin/java ] || { STATUS=NO_JDK; exit 1; }
say "jdk ok"

for obj in "$PAYLOAD" "$KERNEL_A" "$KERNEL_B"; do
  aws s3 cp "s3://$BUCKET/payload/$obj" . --only-show-errors || { STATUS=NO_FETCH_$obj; exit 1; }
  [ -s "$obj" ] || { STATUS=EMPTY_$obj; exit 1; }
done
tar xf "$PAYLOAD" || { STATUS=NO_DIST_UNPACK; exit 1; }
DIST=$(ls -d /work/models-rag-bench-*/ | head -1)

# The kill criterion: prove the two arms load different kernels, or report nothing. The .so digest
# from each JAR's own native.properties is the only thing distinguishing two ABI-6 kernels.
sha_of() { unzip -p "$1" "META-INF/models/native/linux-x86_64/native.properties" 2>/dev/null \
             | sed -n 's/^sha256=//p'; }
SHA_A=$(sha_of "$KERNEL_A"); SHA_B=$(sha_of "$KERNEL_B")
say "kernel A $KERNEL_A sha=$SHA_A"
say "kernel B $KERNEL_B sha=$SHA_B"
if [ -z "$SHA_A" ] || [ -z "$SHA_B" ] || [ "$SHA_A" = "$SHA_B" ]; then
  say "FATAL cannot prove the arms differ"
  STATUS=KERNELS_NOT_DISTINCT; exit 1
fi
printf '{"runId":"%s","kernelA":{"jar":"%s","sha256":"%s"},"kernelB":{"jar":"%s","sha256":"%s"},"threads":%s,"decodeThreads":%s}\n' \
  "$RUNID" "$KERNEL_A" "$SHA_A" "$KERNEL_B" "$SHA_B" "$THREADS" "$DECODE_THREADS" \
  > /work/out/kernels.json

run_arm() {
  local arm=$1 kjar=$2 id=$3 uri=$4 tpl=$5 wl=$6 pass=$7
  local gguf="/work/$id.gguf"
  if [ ! -s "$gguf" ]; then
    curl -sfL "$uri" -o "$gguf" || { say "download failed $id"; return 1; }
  fi
  local out="/work/out/$id.$arm.pass$pass.json"
  java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
    "-Dmodels.native.kernels.decodeThreads=$DECODE_THREADS" \
    -cp "$DIST/lib/*:/work/$kjar" \
    com.integrallis.models.rag.RagBenchmarkCli \
    --framework plain-java --backend rust-ffm \
    --backend-version "kdelta@$arm-$(printf '%s' "$kjar" | tr -c 'A-Za-z0-9.-' '_')" \
    --model "$gguf" --model-id "$id" --workload "$wl" --prompt-template "$tpl" \
    --context 2048 --threads "$THREADS" --max-tokens 256 --warmups 1 --iterations 3 \
    --output "$out" > "/work/out/$id.$arm.pass$pass.log" 2>&1
  local rc=$?
  say "arm=$arm pass=$pass id=$id rc=$rc"
  return 0
}

# MODELS_SPEC: id|uri|tpl|wl, one per line.
printf '%s\n' "$MODELS_SPEC" | while IFS='|' read -r id uri tpl wl; do
  [ -n "${id:-}" ] || continue
  # Both orders, same box: pass 1 is A,B and pass 2 is B,A.
  run_arm A "$KERNEL_A" "$id" "$uri" "$tpl" "$wl" 1
  run_arm B "$KERNEL_B" "$id" "$uri" "$tpl" "$wl" 1
  run_arm B "$KERNEL_B" "$id" "$uri" "$tpl" "$wl" 2
  run_arm A "$KERNEL_A" "$id" "$uri" "$tpl" "$wl" 2
  rm -f "/work/$id.gguf"
done

STATUS=COMPLETE
exit 0
