#!/bin/bash
# Runs ON an isolated host. One library-default correctness smoke for one model.
#
#   default-smoke-remote.sh <id> <uri> <workload> <template> <kernels-jar> <backend-version>
#
# Every parameter and every assertion mirrors the `default-correctness` phase of
# scripts/run-controlled-rag-qualification.sh. It is a separate file rather than a heredoc inside
# collect-default-smoke.sh because the first version inlined it over ssh and the nested quoting
# turned `pgrep -f llama-server` into a check that matched the script's own command line -- every
# model "failed" with a competing inference process that was itself.
set -euo pipefail

id=$1; uri=$2; wl=$3; tpl=$4; kernels=$5; backend_version=$6
threads=${THREADS:-8}
context=${CONTEXT:-2048}
max_tokens=${MAX_TOKENS:-256}

export PATH=/opt/java/bin:$PATH
DIST=$(ls -d /root/models-rag-bench-*/ | head -1)
mkdir -p /root/smoke
report=/root/smoke/$id.default-smoke.json

curl -sfL "$uri" -o /root/model.gguf
sha=$(sha256sum /root/model.gguf | awk '{print $1}')

# Verbatim from run-controlled-rag-qualification.sh: matches comm exactly and excludes this very
# process, which is what a naive `pgrep -f` cannot do.
assert_no_competing_inference_processes() {
  local competitors
  competitors=$(
    ps -eo pid=,stat=,comm=,args= | awk -v self_pid="$$" '
      {
        is_live = $2 !~ /^Z/
        is_llama = $3 ~ /^(llama-cli|llama-server|llama-bench)$/
        is_ollama_runner = $3 == "ollama" && $0 ~ /(^|[[:space:]])runner([[:space:]]|$)/
        is_models_java = $3 == "java" && $0 ~ /com\.integrallis\.models\.(bench|rag)/
        if ($1 != self_pid && is_live && (is_llama || is_ollama_runner || is_models_java)) {
          print
        }
      }
    '
  )
  if [[ -n "$competitors" ]]; then
    echo "competing inference processes make qualification evidence invalid:" >&2
    printf '%s\n' "$competitors" >&2
    exit 1
  fi
}

assert_no_competing_inference_processes
sync; echo 3 > /proc/sys/vm/drop_caches || true

# Library defaults. No tuning system properties: that is the whole point of this phase, and the
# empty tuningSystemProperties in the sidecar is the claim this run has to earn.
JAVA_OPTS="--enable-native-access=ALL-UNNAMED -XX:ActiveProcessorCount=$threads"
# shellcheck disable=SC2086
java $JAVA_OPTS --add-modules jdk.incubator.vector \
  -cp "$DIST/lib/*:/root/$kernels" \
  com.integrallis.models.rag.RagBenchmarkCli \
  --framework plain-java --backend rust-ffm --backend-version "$backend_version" \
  --model /root/model.gguf --model-id "$id" \
  --workload "$wl" --prompt-template "$tpl" \
  --context "$context" --threads "$threads" --top-k 1 --max-tokens "$max_tokens" \
  --warmups 0 --iterations 1 \
  --output "$report" > /root/smoke/"$id".default-smoke.log 2>&1

if ! jq -e --arg backend rust-ffm --arg model_id "$id" --arg artifact_sha "$sha" \
  '.backend == $backend
   and .modelId == $model_id
   and .artifactSha256 == $artifact_sha
   and .settings.warmups == 0
   and .settings.iterations == 1
   and .settings.generationControls.promptCache == "longest-common-prefix"
   and .summary.totalAttempts > 0
   and .summary.successfulAttempts == .summary.totalAttempts
   and .summary.correctAnswerRate == 1
   and .summary.abstentionAccuracy == 1
   and (.failures | length) == 0' "$report" >/dev/null; then
  echo "SMOKE_FAILED"
  jq -c '{correct: .summary.correctAnswerRate, abstention: .summary.abstentionAccuracy,
          ok: .summary.successfulAttempts, n: .summary.totalAttempts,
          failures: (.failures | length)}' "$report" 2>/dev/null || true
  rm -f /root/model.gguf
  exit 2
fi

if ! jq -e '.backendDiagnostics.environment["native-quantized-decode"] == "true"' \
     "$report" >/dev/null; then
  echo "SMOKE_FAILED decode did not route through the Rust shim"
  rm -f /root/model.gguf
  exit 3
fi

# The sidecar in the shape ModelJars' qualification-smoke-gate.mjs requires.
report_sha=$(sha256sum "$report" | awk '{print $1}')
jq -n \
  --arg backend rust-ffm \
  --arg model_id "$id" \
  --arg artifact_sha "$sha" \
  --arg report "default-correctness/models-rust-ffm.json" \
  --arg report_sha "$report_sha" \
  --argjson total_attempts "$(jq '.summary.totalAttempts' "$report")" \
  '{schemaVersion: 1,
    kind: "model-backend-default-correctness-smoke",
    configuration: "library-defaults",
    backend: $backend,
    modelId: $model_id,
    artifactSha256: $artifact_sha,
    report: $report,
    reportSha256: $report_sha,
    totalAttempts: $total_attempts,
    successfulAttempts: $total_attempts,
    tuningSystemProperties: []}' > /root/smoke/"$id".smoke.json

echo "SMOKE_OK"
rm -f /root/model.gguf
