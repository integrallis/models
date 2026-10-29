#!/bin/bash
# Template ablation for VibeThinker 1.5B on the `math` workload, on a Hetzner CPU host.
#
# Why a VPS and not this workstation: the native kernel ships per platform and there is no
# macos-x86_64 build, so `--backend rust-ffm` cannot load here -- it fails in
# BundledNativeKernelLibrary.resolve. Falling back to pure-java would measure a route the fleet
# never runs. A linux-x86_64 host takes the same payload tar and the same kernels jar the
# qualification fleet uses, so this ablation runs the production route on the production build.
#
# Why this is not a qualification: one arm, no comparator, a different machine than the fleet. It
# selects a prompt template. The verdict that can reach the catalog is produced afterwards by
# scripts/fleet/run-shards.sh under the two-arm protocol.
#
# Everything the decision rule in README.md reads -- truncatedAnswerRate, modelAnswerRate,
# modelAnswerCorrectRate -- comes from the candidate report alone, which is why one arm is enough
# to choose between templates.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
TOOLS=/Users/briansam-bodden/Code/java-ai/articles/models-deep-dive/tools
OUT=${OUT:-$HERE/evidence}
mkdir -p "$OUT"

BUCKET=models-qual-077051030817
PAYLOAD=models-rag-bench-0.5.7-vibethinker1.tar
KERNELS=models-kernels-linux-x86_64-0.5.7.jar
BACKEND_VERSION=models@0.5.7+vibethinker1-ec3ae4ff2fd3
URI="https://huggingface.co/mradermacher/VibeThinker-1.5B-GGUF/resolve/d0d66139a78030a92a582f966b0f7cbbb3b19406/VibeThinker-1.5B.Q4_K_M.gguf"
# The artifact shard 703 measured. Checked on the host, so a silently different download is a
# failure here rather than an unexplained difference in the numbers.
ARTIFACT_SHA=59fa32aaaf5c4512452a9bff8bb6d08f1e39a75acee226b5e01372f7509e3458
# Crossed cap x template. 768 is shard 703's cap, so the control cell at 768 is the closest thing
# to a like-for-like repeat of the measured cell; 2048 is the ~3x step above it. One cap could not
# separate "the template does not help" from "the cap is still binding".
CAPS=${CAPS:-"768 1792"}
THREADS=${THREADS:-8}
# Greedy and pinned (temperature 0, topK 1, seed 42), so iterations re-generate the same text and
# the rates this experiment reads do not move with iteration count. Iterations buy timing
# stability and no timing is reported here.
WARMUPS=${WARMUPS:-0}
ITERATIONS=${ITERATIONS:-1}

: "${HETZNER_TOKEN:?export HETZNER_TOKEN}"
# shellcheck disable=SC1091
. "$TOOLS/isolated_host.sh"

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
aws s3 cp "s3://$BUCKET/payload/$PAYLOAD" "$STAGE/" --only-show-errors
aws s3 cp "s3://$BUCKET/payload/$KERNELS" "$STAGE/" --only-show-errors

isolated_host_up vibethinker-tpl
isolated_host_install_java

push "$STAGE/$PAYLOAD" /root/
push "$STAGE/$KERNELS" /root/

remote "set -e
  export PATH=/opt/java/bin:\$PATH
  cd /root && tar xf $PAYLOAD
  DIST=\$(ls -d /root/models-rag-bench-*/ | head -1)
  echo \"dist \$DIST\"
  unzip -p /root/$KERNELS META-INF/models/native/linux-x86_64/native.properties \
    | sed -n 's/^sha256=/kernel sha256=/p'
  curl -sfL '$URI' -o /root/model.gguf
  got=\$(sha256sum /root/model.gguf | cut -d' ' -f1)
  if [ \"\$got\" != '$ARTIFACT_SHA' ]; then
    echo \"FATAL artifact sha \$got is not $ARTIFACT_SHA\"; exit 1
  fi
  echo 'artifact sha matches shard 703'
  nproc; free -g | head -2"

for mt in $CAPS; do
for tpl in chatml-direct chatml-answer chatml-no-think; do
  short=$(echo "$tpl" | tr -d '-' | sed 's/chatml/tpl_/')
  id="vibethinker_1_5b_q4_k_m_${short}_mt${mt}"
  echo "=== $tpl cap $mt -> $id"
  remote "set -e
    export PATH=/opt/java/bin:\$PATH
    DIST=\$(ls -d /root/models-rag-bench-*/ | head -1)
    mkdir -p /root/out
    java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
      -cp \"\$DIST/lib/*:/root/$KERNELS\" \
      com.integrallis.models.rag.RagBenchmarkCli \
      --framework plain-java --backend rust-ffm --backend-version '$BACKEND_VERSION' \
      --model /root/model.gguf --model-id '$id' \
      --workload math --prompt-template '$tpl' --context 2048 --threads $THREADS \
      --max-tokens $mt --warmups $WARMUPS --iterations $ITERATIONS \
      --output /root/out/$id.json > /root/out/$id.log 2>&1
    echo \"  exit \$?\"
    tail -3 /root/out/$id.log" || echo "  CELL FAILED: $tpl cap $mt"
  pull "/root/out/$id.json" "$OUT/" 2>/dev/null || true
  pull "/root/out/$id.log"  "$OUT/" 2>/dev/null || true
done
done

echo "=== pulled into $OUT"
ls -la "$OUT"
