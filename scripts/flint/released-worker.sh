#!/usr/bin/env bash
# Copyright 2025-2026 Integrallis Software, LLC
# SPDX-License-Identifier: Apache-2.0
# Disposable remote Linux host; bootstrap must enforce an independent termination bound.
set -euo pipefail
INPUT=$(realpath "${1:?MANIFEST.json}")
WORK=/work/flint-released
OUT=$WORK/results
mkdir -p "$OUT/default-correctness" "$OUT/visible" "$WORK/jdk" "$WORK/probe"
cp "$INPUT" "$OUT/inputs.json"
DEST=$(jq -er '.resultsUri' "$INPUT")
export AWS_DEFAULT_REGION=us-east-1
exec > >(tee -a "$OUT/worker.log" /dev/console) 2>&1
sync_results() { aws s3 cp "$OUT/" "$DEST/" --recursive --only-show-errors; }
cleanup() {
  local rc=$?
  trap - EXIT
  printf '%s\n' "$rc" >"$OUT/worker-exit-code.txt"
  date -u +%FT%TZ >"$OUT/finished-at.txt"
  sync_results || true
  shutdown -h now
  exit "$rc"
}
trap cleanup EXIT
(while sleep 30; do sync_results || true; done) &
date -u +%FT%TZ >"$OUT/started-at.txt"
uname -a >"$OUT/host.txt"
lscpu >>"$OUT/host.txt"
free -b >>"$OUT/host.txt"
# Default smoke must not inherit hidden deployment tuning.
for key in JAVA_OPTS JAVA_TOOL_OPTIONS JDK_JAVA_OPTIONS _JAVA_OPTIONS; do
  if [[ "${!key:-}" == *models.* ]]; then echo "$key contains Models overrides" >&2; exit 1; fi
done
stage() {
  local name=$1 seconds=$2
  shift 2
  date -u +%FT%TZ >"$OUT/$name.started"
  local rc=0
  timeout --signal=TERM --kill-after=20 "$seconds" "$@" >"$OUT/$name.log" 2>&1 || rc=$?
  printf '%s\n' "$rc" >"$OUT/$name.exit-code"
  sync_results
  return "$rc"
}
verify() { printf '%s  %s\n' "$1" "$2" | sha256sum -c -; }
download() { curl -fL --retry 3 --connect-timeout 30 --max-time 900 "$1" -o "$2"; }
aws s3 cp "$(jq -er '.payloadUri' "$INPUT")" "$WORK/payload.tar.gz" --only-show-errors
verify "$(jq -er '.payloadSha256' "$INPUT")" "$WORK/payload.tar.gz"
tar -xzf "$WORK/payload.tar.gz" -C "$WORK"
python3 - "$WORK/released" "$INPUT" <<'PY'
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]);manifest=json.loads(Path(sys.argv[2]).read_text())
provenance=json.loads((root/'provenance.json').read_text())
assert provenance['modelsVersion']==manifest['modelsVersion']
assert provenance['vectorsVersion']==manifest['vectorsVersion']
for record in json.loads((root/'classpath.json').read_text()):
    name=record['file'];assert Path(name).name==name
    assert hashlib.sha256((root/'lib'/name).read_bytes()).hexdigest()==record['sha256']
    harness = name in (f"models-rag-bench-{manifest['modelsVersion']}.jar", f"models-bench-{manifest['modelsVersion']}.jar")
    if harness:
        assert record['origin']=='source-built measurement harness',name
    elif name.startswith(('models-','backend-','vectors-')):
        assert record['origin']=='Maven Central',name
print('All released product JAR identities verified')
PY
cp "$WORK/released/provenance.json" "$WORK/released/classpath.json" "$OUT/"
SOURCES=(FlintKernelParity.java FlintReleasedProfileProbe.java reference.properties)
if jq -e '.toolDiagnostic == true' "$INPUT" >/dev/null; then
  SOURCES+=(FlintToolTaskProbe.java tool-task-held-out-v1.json)
fi
for source in "${SOURCES[@]}"; do
  uri=$(jq -er --arg name "$source" '.probeSources[$name].uri' "$INPUT")
  sha=$(jq -er --arg name "$source" '.probeSources[$name].sha256' "$INPUT")
  aws s3 cp "$uri" "$WORK/$source" --only-show-errors
  verify "$sha" "$WORK/$source"
done
download 'https://github.com/adoptium/temurin25-binaries/releases/download/jdk-25.0.3%2B9/OpenJDK25U-jdk_x64_linux_hotspot_25.0.3_9.tar.gz' "$WORK/jdk.tar.gz"
verify 69264a7a211bf5029830d07bc3370f879769d62ebc5b5488e90c9343a2da0e1f "$WORK/jdk.tar.gz"
tar -xzf "$WORK/jdk.tar.gz" -C "$WORK/jdk" --strip-components=1
export JAVA_HOME=$WORK/jdk
export PATH=$JAVA_HOME/bin:$PATH
java -version >"$OUT/java-version.txt" 2>&1
MODEL=$WORK/Flint-Q4_K_M.gguf
MODEL_SHA=c3c4d93f9db92627a2fc47ea8fe8c64a094cabd3549db544bcdd2c00e9a9a347
download 'https://huggingface.co/ggml-org/Flint-GGUF/resolve/468307a1b38259e9e4c9d4fe53d4e3b76386572c/Flint-Q4_K_M.gguf' "$MODEL"
test "$(stat -c %s "$MODEL")" = 23387911136
verify "$MODEL_SHA" "$MODEL"
printf '%s  %s\n' "$MODEL_SHA" Flint-Q4_K_M.gguf >"$OUT/model.sha256"
CP="$WORK/released/lib/*"
JAVA=(java -Xmx8g --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED -XX:ActiveProcessorCount=16)
javac --release 25 -cp "$CP" -d "$WORK/probe" "$WORK/FlintKernelParity.java" "$WORK/FlintReleasedProfileProbe.java"
stage parity 900 "${JAVA[@]}" -cp "$WORK/probe:$CP" com.integrallis.models.bench.FlintKernelParity "$MODEL" "$WORK/reference.properties" "$OUT/parity.json"
VERSION="models@$(jq -er '.modelsVersion' "$INPUT")+released vectors@$(jq -er '.vectorsVersion' "$INPUT")"
stage native-default 1800 "${JAVA[@]}" -cp "$CP" com.integrallis.models.rag.RagBenchmarkCli \
  --framework plain-java --backend rust-ffm --backend-version "$VERSION" --model "$MODEL" \
  --model-id ggml_org_flint_gguf_q4_k_m --workload general --prompt-template flint \
  --context 2048 --threads 16 --max-tokens 256 --top-k 1 --warmups 0 --iterations 1 \
  --output "$OUT/native-default.json"
jq -e --arg sha "$MODEL_SHA" '.artifactSha256 == $sha and .backend == "rust-ffm"
  and .summary.totalAttempts == 9 and .summary.successfulAttempts == 9
  and .summary.correctAnswerRate == 1 and .summary.abstentionAccuracy == 1
  and .settings.generationControls.promptCache == "longest-common-prefix"
  and (.failures | length) == 0' "$OUT/native-default.json"
cp "$OUT/native-default.json" "$OUT/default-correctness/models-rust-ffm.json"
jq -n --arg sha "$MODEL_SHA" --arg version "$VERSION" \
  --arg reportSha "$(sha256sum "$OUT/native-default.json" | awk '{print $1}')" \
  '{schemaVersion:1,kind:"model-backend-default-correctness-smoke",configuration:"library-defaults",
    backend:"rust-ffm",backendVersion:$version,modelId:"ggml_org_flint_gguf_q4_k_m",
    artifactSha256:$sha,report:"default-correctness/models-rust-ffm.json",reportSha256:$reportSha,
    totalAttempts:9,successfulAttempts:9,tuningSystemProperties:[]}' >"$OUT/default-correctness/smoke.json"
stage native-default-visible 120 "${JAVA[@]}" -cp "$CP" \
  com.integrallis.models.rag.RagVisibleAnswerAudit "$OUT/native-default.json" "$OUT/visible/native-default.json"
stage public-profile 1200 "${JAVA[@]}" -cp "$WORK/probe:$CP" FlintReleasedProfileProbe "$MODEL" \
  "$(jq -er '.expectedRawSha256' "$INPUT")" "$OUT/public-profile.json"
printf '%s\n' 'Released numerical parity, default smoke, and public profile round trip passed. Full catalog qualification is separately hash-bound.' >"$OUT/completed.txt"
sync_results
# This optional research stage cannot turn a failed tool result into a catalog qualification.
# It runs after all required release checks and performs simulated tool actions only.
if jq -e '.toolDiagnostic == true' "$INPUT" >/dev/null; then
  javac --release 25 -cp "$CP" -d "$WORK/probe" "$WORK/FlintToolTaskProbe.java"
  stage tools-diagnostic 2700 "${JAVA[@]}" -cp "$WORK/probe:$CP" \
    com.integrallis.models.bench.FlintToolTaskProbe "$MODEL" \
    "$WORK/tool-task-held-out-v1.json" "$OUT/tools-diagnostic.json" || \
    printf '%s\n' 'Tool diagnostic incomplete; inspect its exit code and raw trace. No tool qualification claimed.' >"$OUT/tools-diagnostic-incomplete.txt"
fi
