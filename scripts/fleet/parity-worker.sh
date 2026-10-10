#!/bin/bash
# Framework verification pass: run every qualified model through plain Java, LangChain4j and Spring AI
# and compare their grounded decisions case by case.
#
# Separate from qualification on purpose. It needs no comparator, no Ollama and no verdict -- the three
# applications receive the identical generation client and differ only in the wrapper, so what is being
# established is that each surface serves the model, not how fast it is. Running it as its own pass keeps
# candidate runs lean and covers models published before the check existed in the same sweep.
SHARD=${SHARD:-200}
BUCKET=models-qual-077051030817
LOG=/var/log/parity-worker.log
say() { printf '[parity %s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$LOG" >/dev/console 2>/dev/null; }
exec 2>>"$LOG"
set -x
STATUS=STARTED
finish() {
  rc=$?
  say "FINAL_STATUS=$STATUS exit=$rc"
  if command -v aws >/dev/null 2>&1; then
    echo "$STATUS" > /work/out/STATUS 2>/dev/null
    aws s3 cp /work/out/STATUS "s3://$BUCKET/results/parity-$SHARD/" --only-show-errors 2>/dev/null
    aws s3 cp "$LOG" "s3://$BUCKET/results/parity-$SHARD/parity-worker-$SHARD.log" --only-show-errors 2>/dev/null
    [ -f /work/out/progress.tsv ] && aws s3 cp /work/out/progress.tsv "s3://$BUCKET/results/parity-$SHARD/" --only-show-errors 2>/dev/null
  fi
  shutdown -h now
}
trap finish EXIT
mkdir -p /work/models /work/out /opt/jdk || { STATUS=NO_WORKDIR; exit 1; }
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq || true
apt-get install -y -qq unzip python3 || true
if ! command -v aws >/dev/null 2>&1; then
  curl -sfL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscli.zip
  (cd /tmp && unzip -qo awscli.zip && ./aws/install --update >/dev/null 2>&1); hash -r
fi
command -v aws >/dev/null 2>&1 || { STATUS=NO_AWS_CLI; exit 1; }
curl -sfL "https://api.adoptium.net/v3/binary/latest/25/ga/linux/x64/jdk/hotspot/normal/eclipse" -o /tmp/jdk.tgz \
  || { STATUS=NO_JDK_DOWNLOAD; exit 1; }
tar xzf /tmp/jdk.tgz -C /opt/jdk --strip-components=1 || { STATUS=NO_JDK_UNPACK; exit 1; }
export JAVA_HOME=/opt/jdk; export PATH=$JAVA_HOME/bin:$PATH
say "jdk $(/opt/jdk/bin/java -version 2>&1 | head -1)"
cd /work
# The kernels JAR is named explicitly, never by a fixed name. backend-native's published JAR
# carries classes only -- zero META-INF/models/native entries -- so this JAR is the ONLY source of
# the .so, and a fixed name is how a 2026-09-28 kernel came to be measured against a 0.3.56
# library with no symptom: both declare abi=6. Same defect, same file family, second instance.
KERNELS=${PARITY_KERNELS:?PARITY_KERNELS required; refusing to pick a kernel by a fixed name}
aws s3 cp "s3://$BUCKET/payload/$PARITY_DIST" . --only-show-errors
aws s3 cp "s3://$BUCKET/payload/$KERNELS" . --only-show-errors
aws s3 cp "s3://$BUCKET/payload/parity-shard-$SHARD.json" /work/shard.json --only-show-errors
[ -s "$PARITY_DIST" ] && [ -s /work/shard.json ] || { STATUS=NO_PAYLOAD; exit 1; }
[ -s "/work/$KERNELS" ] || { STATUS=NO_KERNEL_JAR; exit 1; }
tar xf "$PARITY_DIST" || { STATUS=NO_DIST_UNPACK; exit 1; }
DIST=$(ls -d /work/models-rag-bench-*/ | head -1)
CP="$DIST/lib/*:/work/$KERNELS"
# Record which kernel ran. The .so digest is the only thing distinguishing two ABI-6 builds.
say "kernel $KERNELS sha256=$(unzip -p "/work/$KERNELS" \
  META-INF/models/native/linux-x86_64/native.properties 2>/dev/null | sed -n 's/^sha256=//p')"
THREADS=$(nproc 2>/dev/null || echo 8)
: > /work/out/progress.tsv
python3 -c "
import json
for j in json.load(open('/work/shard.json')): print(json.dumps(j))
" > /work/jobs.ndjson
JOBS=$(wc -l < /work/jobs.ndjson); say "payload ok, $JOBS models to verify"

while IFS= read -r job; do
  id=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['id'])" "$job")
  uri=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['uri'])" "$job")
  tpl=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['tpl'])" "$job")
  wl=$(python3 -c "import json,sys;print(json.loads(sys.argv[1]).get('wl') or 'general')" "$job")
  dt=$(python3 -c "import json,sys;print(json.loads(sys.argv[1]).get('dt') or '')" "$job")
  DECODE_THREAD_OPT=""
  [ -n "$dt" ] && DECODE_THREAD_OPT="-Dmodels.native.kernels.decodeThreads=$dt"
  gguf="/work/models/$id.gguf"
  T0=$(date +%s)
  say "verify $id (tpl=$tpl wl=$wl)"
  if ! curl -sfL "$uri" -o "$gguf"; then
    printf '%s\tDOWNLOAD_FAIL\t0\n' "$id" >> /work/out/progress.tsv
    say "download failed $id"; continue
  fi
  for fw in plain-java langchain4j spring-ai; do
    java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
      $DECODE_THREAD_OPT -cp "$CP" \
      com.integrallis.models.rag.RagBenchmarkCli \
      --framework "$fw" --backend rust-ffm --backend-version "$PARITY_VERSION" \
      --model "$gguf" --model-id "$id" --workload "$wl" --prompt-template "$tpl" \
      --context 2048 --threads "$THREADS" --max-tokens 256 --warmups 0 --iterations 1 \
      --output "/work/out/$id.$fw.json" >> "/work/out/$id.parity.log" 2>&1
    say "  $fw rc=$?"
  done
  python3 - "$id" >> "/work/out/$id.parity.log" 2>&1 <<'COMPARE'
import json, os, sys
model = sys.argv[1]
out = "/work/out"


def decisions(path):
    runs = json.load(open(path))["runs"]
    return {(r.get("caseId") or r.get("case") or str(i)):
            (r.get("grounding") or {}).get("decision") for i, r in enumerate(runs)}


result = {"model": model, "verified": ["plain-java"], "servable": True,
          "mismatches": [], "missing": []}
base = os.path.join(out, f"{model}.plain-java.json")
if not os.path.isfile(base):
    result.update(servable=False, missing=["plain-java"])
else:
    reference = decisions(base)
    for framework in ("langchain4j", "spring-ai"):
        path = os.path.join(out, f"{model}.{framework}.json")
        if not os.path.isfile(path):
            result["servable"] = False
            result["missing"].append(framework)
            continue
        other = decisions(path)
        result["verified"].append(framework)
        # A case that reaches the extractive fallback under one adapter and a model answer under another
        # is a serving defect in that adapter, not a difference of opinion.
        for case, verdict in reference.items():
            if other.get(case) != verdict:
                result["servable"] = False
                result["mismatches"].append(
                    {"case": case, "plain-java": verdict, framework: other.get(case)})
json.dump(result, open(os.path.join(out, f"{model}.serving.json"), "w"), indent=1)
print(f"servable={result['servable']} verified={result['verified']} "
      f"mismatches={len(result['mismatches'])} missing={result['missing']}")
COMPARE
  servable=$(python3 -c "import json;print(json.load(open('/work/out/$id.serving.json')).get('servable'))" 2>/dev/null || echo False)
  printf '%s\tservable\t%s\t%s\n' "$id" "$servable" "$(( $(date +%s)-T0 ))" >> /work/out/progress.tsv
  say "done $id servable=$servable in $(( $(date +%s)-T0 ))s"
  for f in serving.json parity.log plain-java.json langchain4j.json spring-ai.json; do
    [ -s "/work/out/$id.$f" ] && aws s3 cp "/work/out/$id.$f" "s3://$BUCKET/results/parity-$SHARD/" --only-show-errors 2>/dev/null
  done
  rm -f "$gguf"
done < /work/jobs.ndjson
STATUS=PARITY_COMPLETE
say "shard complete"
