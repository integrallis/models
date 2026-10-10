#!/bin/bash
# Default-correctness smoke against a published Models payload. One arm, no performance verdict.
# Requires explicit payload, backend label and checksum-bearing jobs. See CAMPAIGN-RUNBOOK.md.
# The versioned 0354 workers are retained for historical reproduction.
: "${SHARD:?SHARD is required}"
: "${SMOKE_PAYLOAD:?SMOKE_PAYLOAD is required}"
: "${SMOKE_BACKEND_VERSION:?SMOKE_BACKEND_VERSION is required}"
: "${SMOKE_PAYLOAD_SHA256:?SMOKE_PAYLOAD_SHA256 is required}"
DEADLINE_SECONDS=${DEADLINE_SECONDS:-18000}
[[ "$SHARD" =~ ^[0-9]+$ ]] || { echo "SHARD must be numeric" >&2; exit 1; }
[[ "$SMOKE_PAYLOAD" =~ ^models-rag-bench-[A-Za-z0-9_.-]+\.tar$ ]] \
  || { echo "invalid payload name" >&2; exit 1; }
[[ "$SMOKE_PAYLOAD_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || { echo "invalid payload SHA-256" >&2; exit 1; }
[[ "$DEADLINE_SECONDS" =~ ^[1-9][0-9]*$ ]] \
  || { echo "DEADLINE_SECONDS must be positive" >&2; exit 1; }
BUCKET=models-qual-077051030817
# Overridable so the identical protocol can be pointed at an earlier library: attributing a smoke
# failure to a release needs the same worker, same shard and same gate on both sides, with the
# library as the only variable.
PAYLOAD=$SMOKE_PAYLOAD
# No separate kernel jar. The dist's backend-native JAR is the artifact published to Maven
# Central and already carries every platform's kernel, linux-x86_64 included. Adding the same jar
# again under another name put two copies of
# META-INF/models/native/linux-x86_64/native.properties on the classpath and
# BundledNativeKernelLibrary.uniqueResource refused the ambiguity -- correctly -- which failed all
# eight pilot jobs with rc=1. The older two-arm worker needs that extra jar because its dist was
# built without Linux natives; this one is not.
# The caller supplies the released version and source revision used to assemble the payload.
BACKEND_VERSION=$SMOKE_BACKEND_VERSION
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
    say "failure recorded; terminating without an idle hold"
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
printf '%s  %s\n' "$SMOKE_PAYLOAD_SHA256" "$PAYLOAD" | sha256sum --check --status \
  || { STATUS=PAYLOAD_CHECKSUM_MISMATCH; exit 1; }
tar xf "$PAYLOAD" || { STATUS=NO_DIST_UNPACK; exit 1; }
DIST=$(ls -d /work/models-rag-bench-*/ 2>/dev/null | head -1)
[ -n "$DIST" ] || DIST=/work/models-rag-bench
CP="$DIST/lib/*"
NATIVE_JAR=$(ls "$DIST"/lib/backend-native-*.jar 2>/dev/null | head -1)
python3 - "$BACKEND_VERSION" "$PAYLOAD" "$NATIVE_JAR" <<'VERSION_CHECK'
import pathlib, re, sys
label, payload, native = sys.argv[1:]
match = re.fullmatch(r"models@(\d+\.\d+\.\d+)\+[A-Za-z0-9_.-]+", label)
if not match:
    sys.exit("backend label must name the released version and source revision")
version = match.group(1)
if (not payload.startswith(f"models-rag-bench-{version}-")
        or pathlib.Path(native).name != f"backend-native-{version}.jar"):
    sys.exit("payload, published native JAR and backend label disagree")
VERSION_CHECK
[ "$?" -eq 0 ] || { STATUS=VERSION_PAYLOAD_MISMATCH; exit 1; }
if ! unzip -l "$NATIVE_JAR" 2>/dev/null | grep -q "linux-x86_64/libjmodels_kernels.so"; then
  STATUS=NO_LINUX_KERNEL_IN_DIST; say "dist backend-native carries no linux-x86_64 kernel"; exit 1
fi
say "dist carries the linux-x86_64 kernel"
JOBS=$(python3 -c "import json;print(len(json.load(open('/work/shard.json'))))")
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
say "payload ok, $JOBS jobs, backend-version=$BACKEND_VERSION"
STATUS=RUNNING
T_START=$(date +%s)

# This worker is deployed as one file. Tests extract this exact embedded helper.
cat > /work/fetch-smoke-artifact.py <<'ARTIFACT_HELPER'
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import urllib.request
from urllib.parse import urlsplit


def validate_job(job):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", job["id"]):
        raise ValueError("invalid model id")
    if job["format"] not in ("gguf", "safetensors"):
        raise ValueError("unsupported model format")
    if job["backend"] not in ("pure-java", "rust-ffm"):
        raise ValueError("unsupported backend")
    for field in ("context", "threads", "mt", "topK"):
        if type(job[field]) is not int or job[field] <= 0:
            raise ValueError(f"{field} must be a positive integer")
    for field in ("wl", "tpl"):
        if not isinstance(job[field], str) or not job[field]:
            raise ValueError(f"{field} is required")
    paths = set()
    for entry in job["files"]:
        name = entry["path"]
        path = PurePosixPath(name)
        if (not name or name == "." or path.is_absolute() or str(path) != name
                or ".." in path.parts or "\\" in name or name in paths):
            raise ValueError(f"invalid or duplicate artifact path: {name}")
        paths.add(name)
        if not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ValueError(f"invalid SHA-256: {name}")
        if type(entry["sizeBytes"]) is not int or entry["sizeBytes"] <= 0:
            raise ValueError(f"invalid size: {name}")
        uri = urlsplit(entry["uri"])
        if uri.scheme != "https" or not uri.hostname or uri.username or uri.password:
            raise ValueError(f"artifact URI must be HTTPS without credentials: {name}")
    if job["format"] == "gguf":
        if len(paths) != 1 or not next(iter(paths)).endswith(".gguf"):
            raise ValueError("GGUF requires one .gguf file")
    elif not {"config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"} <= paths:
        # Sharded snapshots need index validation before this worker can support them.
        raise ValueError("Safetensors requires weights, config and both tokenizer files")


def fetch(job, root, receipt, opener=urllib.request.urlopen):
    validate_job(job)
    directory = Path(root) / job["id"]
    # Never reuse a partial snapshot or silently consume files left by another run.
    directory.mkdir(parents=True, exist_ok=False)
    verified = []
    for entry in job["files"]:
        target = directory / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".partial")
        digest = hashlib.sha256()
        size = 0
        try:
            with opener(entry["uri"], timeout=60) as response, partial.open("xb") as out:
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > entry["sizeBytes"]:
                        raise ValueError(f"size exceeds manifest: {entry['path']}")
                    digest.update(chunk)
                    out.write(chunk)
            if size != entry["sizeBytes"] or digest.hexdigest() != entry["sha256"]:
                raise ValueError(f"checksum or size mismatch: {entry['path']}")
            partial.rename(target)
        finally:
            partial.unlink(missing_ok=True)
        verified.append({**entry, "verifiedSha256": digest.hexdigest(), "verifiedSizeBytes": size})
    artifact = directory if job["format"] == "safetensors" else directory / job["files"][0]["path"]
    Path(receipt).write_text(json.dumps({"modelId": job["id"], "format": job["format"],
        "modelPath": str(artifact), "job": job, "files": verified}, indent=2) + "\n")
    return artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-shard")
    parser.add_argument("--job-json")
    parser.add_argument("--root")
    parser.add_argument("--receipt")
    args = parser.parse_args()
    if args.validate_shard:
        jobs = json.loads(Path(args.validate_shard).read_text())
        if not isinstance(jobs, list) or not jobs:
            raise ValueError("shard must be a non-empty list")
        ids = set()
        for job in jobs:
            validate_job(job)
            if job["id"] in ids:
                raise ValueError("duplicate model id")
            ids.add(job["id"])
    else:
        print(fetch(json.loads(args.job_json), args.root, args.receipt))


if __name__ == "__main__":
    main()
ARTIFACT_HELPER
python3 /work/fetch-smoke-artifact.py --validate-shard /work/shard.json \
  || { STATUS=INVALID_SHARD; exit 1; }

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
if r.get('failures') != []:           reasons.append(f"failures={r.get('failures')}")
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
  if [ $(( $(date +%s) - T_START )) -ge "$DEADLINE_SECONDS" ]; then
    STATUS=DEADLINE; say "deadline reached after $done_n/$JOBS"; break
  fi
  job=$(python3 -c "import json;print(json.dumps(json.load(open('/work/shard.json'))[$done_n]))")
  id=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['id'])" "$job")
  tpl=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['tpl'])" "$job")
  wl=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['wl'])" "$job")
  bk=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['backend'])" "$job")
  read -r context threads mt topk < <(python3 -c \
    "import json,sys;j=json.loads(sys.argv[1]);print(j['context'],j['threads'],j['mt'],j['topK'])" "$job")
  artifact_dir="/work/models/$id"
  T0=$(date +%s)
  say "start $id (backend=$bk template=$tpl workload=$wl)"
  remaining=$((DEADLINE_SECONDS - $(date +%s) + T_START))
  if [ "$remaining" -le 0 ]; then STATUS=DEADLINE; break; fi
  if artifact=$(timeout --kill-after=30 "$remaining" python3 /work/fetch-smoke-artifact.py --job-json "$job" \
      --root /work/models --receipt "/work/out/$id.inputs.json"); then
    # NO -Dmodels.* of any kind: the gate requires tuningSystemProperties to be empty, and the
    # whole point is to measure what a user gets with nothing configured.
    remaining=$((DEADLINE_SECONDS - $(date +%s) + T_START))
    if [ "$remaining" -le 0 ]; then STATUS=DEADLINE; break; fi
    timeout --kill-after=30 "$remaining" java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
      -cp "$CP" \
      com.integrallis.models.rag.RagBenchmarkCli \
      --framework plain-java --backend "$bk" --backend-version "$BACKEND_VERSION" \
      --model "$artifact" --model-id "$id" --workload "$wl" --prompt-template "$tpl" \
      --context "$context" --threads "$threads" --max-tokens "$mt" --top-k "$topk" \
      --warmups 0 --iterations 1 \
      --output "/work/out/$id.json" > "/work/out/$id.log" 2>&1
    rc=$?
    verdict="RUN_FAILED_rc$rc"
    if [ "$rc" -eq 0 ] && [ -s "/work/out/$id.json" ]; then
      verdict=$(python3 /work/gate.py "/work/out/$id.json" 2>/dev/null || echo "GATE_ERROR")
    fi
    printf '%s\t%s\t%s\n' "$id" "$verdict" "$(( $(date +%s)-T0 ))" >> /work/out/progress.tsv
    say "done $id -> $verdict in $(( $(date +%s)-T0 ))s"
    aws s3 cp "/work/out/$id.inputs.json" "s3://$BUCKET/results/shard-$SHARD/" --only-show-errors
    aws s3 cp "/work/out/$id.json" "s3://$BUCKET/results/shard-$SHARD/" --only-show-errors 2>/dev/null
    aws s3 cp "/work/out/$id.log"  "s3://$BUCKET/results/shard-$SHARD/" --only-show-errors 2>/dev/null
    aws s3 cp /work/out/progress.tsv "s3://$BUCKET/results/shard-$SHARD/" --only-show-errors 2>/dev/null
  else
    printf '%s\tDOWNLOAD_FAIL\t0\n' "$id" >> /work/out/progress.tsv
    say "download failed $id"
  fi
  rm -rf "$artifact_dir"
  done_n=$((done_n+1))
done
[ "$STATUS" = "DEADLINE" ] || STATUS=SHARD_COMPLETE
say "shard finished: $done_n/$JOBS"
