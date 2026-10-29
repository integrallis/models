#!/bin/bash
# Collect the library-default correctness smoke for every model in a shard manifest.
#
# Why this exists. A qualification is not landable without it. ModelJars'
# `tools/qualification-smoke-gate.mjs` requires `defaultConfigurationSmoke` on every new or
# changed entry, and `qual-worker-two-arm.sh` does not produce one -- it writes only the tuned
# two-arm performance comparison. So the fleet path measured eleven models on 2026-10-29 and none
# of them could be published: "unsloth_qwen3_5_0_8b_gguf_q5_k_s/rust-ffm.defaultConfigurationSmoke
# is required for a new or changed qualification". The host-based
# `scripts/run-controlled-rag-qualification.sh` writes both halves, which is why the 41 entries
# that carry the field came from it.
#
# This closes that gap for the fleet path. The parameters and every assertion mirror the
# `default-correctness` phase of run-controlled-rag-qualification.sh (:171-218): library-default
# JAVA_OPTS with no tuning properties, `--warmups 0 --iterations 1` over every case in the
# workload, and a hard requirement of perfect correctness. Per RAG_BENCHMARKS.md: "A performance
# profile can improve a qualified deployment, but it cannot make a model/backend pair eligible when
# the public library defaults fail."
#
# A model that fails here is NOT qualified, whatever its tuned comparison said. That is the point
# of the gate and this script does not soften it.
#
#   SHARD=920 HETZNER_TOKEN=... bash scripts/fleet/collect-default-smoke.sh
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
TOOLS=/Users/briansam-bodden/Code/java-ai/articles/models-deep-dive/tools

SHARD=${SHARD:?SHARD is required}
MANIFEST="$HERE/fleet-shard-$SHARD.json"
[ -f "$MANIFEST" ] || { echo "no manifest at $MANIFEST" >&2; exit 1; }
OUT=${OUT:-$ROOT/benchmark-results/default-smoke-$SHARD}
mkdir -p "$OUT"

BUCKET=${BUCKET:-models-qual-077051030817}
PAYLOAD=${PAYLOAD:-models-rag-bench-0.5.7-vibethinker1.tar}
KERNELS=${KERNELS:-models-kernels-linux-x86_64-0.5.7.jar}
BACKEND_VERSION=${BACKEND_VERSION:-models@0.5.7+vibethinker1-ec3ae4ff2fd3}
THREADS=${THREADS:-8}
CONTEXT=${CONTEXT:-2048}
MAX_TOKENS=${MAX_TOKENS:-256}

: "${HETZNER_TOKEN:?export HETZNER_TOKEN}"
# shellcheck disable=SC1091
. "$TOOLS/isolated_host.sh"

STAGE=$(mktemp -d); trap 'rm -rf "$STAGE"' EXIT
aws s3 cp "s3://$BUCKET/payload/$PAYLOAD" "$STAGE/" --only-show-errors
aws s3 cp "s3://$BUCKET/payload/$KERNELS" "$STAGE/" --only-show-errors

isolated_host_up "default-smoke-$SHARD"
isolated_host_install_java
remote 'apt-get -qq install -y jq >/dev/null 2>&1; jq --version'

push "$STAGE/$PAYLOAD" /root/
push "$STAGE/$KERNELS" /root/
push "$MANIFEST" /root/shard.json
remote "cd /root && tar xf $PAYLOAD >/dev/null 2>&1 && ls -d /root/models-rag-bench-*/ | head -1"

JOBS=$(python3 -c "import json;print(len(json.load(open('$MANIFEST'))))")
echo "=== $JOBS models to smoke"

push "$HERE/default-smoke-remote.sh" /root/default-smoke-remote.sh
remote 'chmod +x /root/default-smoke-remote.sh'

for i in $(seq 0 $((JOBS - 1))); do
  job=$(python3 -c "import json;print(json.dumps(json.load(open('$MANIFEST'))[$i]))")
  read -r id uri wl tpl <<<"$(python3 -c "
import json,sys
j = json.loads(sys.argv[1])
print(j['id'], j['uri'], j['wl'], j['tpl'])" "$job")"
  echo "=== $id ($wl / $tpl)"
  if remote "THREADS=$THREADS CONTEXT=$CONTEXT MAX_TOKENS=$MAX_TOKENS \
       /root/default-smoke-remote.sh '$id' '$uri' '$wl' '$tpl' '$KERNELS' '$BACKEND_VERSION'"; then
    :
  else
    echo "  >>> $id DID NOT PASS THE LIBRARY-DEFAULT SMOKE"
  fi
  pull "/root/smoke/$id.default-smoke.json" "$OUT/" 2>/dev/null || true
  pull "/root/smoke/$id.default-smoke.log"  "$OUT/" 2>/dev/null || true
  pull "/root/smoke/$id.smoke.json"         "$OUT/" 2>/dev/null || true
done

echo "=== collected into $OUT"
ls -la "$OUT" | tail -25
