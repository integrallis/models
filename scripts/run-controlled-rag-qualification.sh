#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 MODEL_GGUF MODEL_ID WORKLOAD PROMPT_TEMPLATE [OUTPUT_DIR]" >&2
  exit 2
fi

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_PATH=$(realpath "$1")
MODEL_ID=$2
WORKLOAD=$3
PROMPT_TEMPLATE=$4
OUTPUT_DIR=${5:-"$ROOT_DIR/build/reports/rag/qualification/$MODEL_ID"}
THREADS=${RAG_THREADS:-$(nproc)}
NATIVE_THREADS=${RAG_NATIVE_THREADS:-$THREADS}
MODELS_BACKEND=${RAG_MODELS_BACKEND:-rust-ffm}
CONTEXT=${RAG_CONTEXT:-2048}
MAX_TOKENS=${RAG_MAX_TOKENS:-64}
STOP_SEQUENCE=${RAG_STOP_SEQUENCE:-}
WARMUPS=${RAG_WARMUPS:-1}
ITERATIONS=${RAG_ITERATIONS:-3}
DROP_CACHES=${RAG_DROP_CACHES:-0}
LLAMA_PORT=${LLAMA_PORT:-18081}
LLAMA_SERVER=${LLAMA_SERVER:-llama-server}
OLLAMA_ENDPOINT=${OLLAMA_ENDPOINT:-http://127.0.0.1:11434}
NATIVE_LIBRARY=${MODELS_NATIVE_LIBRARY:-"$ROOT_DIR/backend-native/build/rust-target/release/libjmodels_kernels.so"}

if [[ "$MODELS_BACKEND" != "pure-java" && "$MODELS_BACKEND" != "rust-ffm" ]]; then
  echo "RAG_MODELS_BACKEND must be pure-java or rust-ffm" >&2
  exit 2
fi
for option_source in JAVA_OPTS JAVA_TOOL_OPTIONS JDK_JAVA_OPTIONS _JAVA_OPTIONS; do
  if [[ "${!option_source:-}" == *-Dmodels.* ]]; then
    echo "$option_source must not contain Models tuning properties during qualification" >&2
    echo "use RAG_TUNED_JAVA_OPTS for the performance phase" >&2
    exit 2
  fi
done

if [[ $(uname -s) != Linux ]]; then
  echo "controlled qualification currently requires Linux" >&2
  exit 1
fi
for command in awk curl git java jq nproc ollama pgrep ps realpath sha256sum sync "$LLAMA_SERVER"; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "required command not found: $command" >&2
    exit 1
  fi
done
JAVA_FEATURE=$(
  java -XshowSettings:properties -version 2>&1 \
    | awk -F= '/^[[:space:]]*java.specification.version[[:space:]]*=/{gsub(/[[:space:]]/, "", $2); print $2}'
)
if [[ "$JAVA_FEATURE" != 25 ]]; then
  echo "qualification requires Java 25; default java is $JAVA_FEATURE" >&2
  exit 1
fi
if [[ "$DROP_CACHES" != 0 && "$DROP_CACHES" != 1 ]]; then
  echo "RAG_DROP_CACHES must be 0 or 1" >&2
  exit 1
fi
if [[ "$DROP_CACHES" == 1 && ! -w /proc/sys/vm/drop_caches ]]; then
  echo "RAG_DROP_CACHES=1 requires permission to write /proc/sys/vm/drop_caches" >&2
  exit 1
fi
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "model not found: $MODEL_PATH" >&2
  exit 1
fi
if ! git -C "$ROOT_DIR" diff --quiet || ! git -C "$ROOT_DIR" diff --cached --quiet; then
  echo "models checkout must be clean" >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR"
TEMP_DIR=$(mktemp -d)
LLAMA_PID=""
OLLAMA_PID=""
OLLAMA_STARTED=0
OLLAMA_MODEL=""

cleanup() {
  if [[ -n "$LLAMA_PID" ]]; then
    kill "$LLAMA_PID" 2>/dev/null || true
    wait "$LLAMA_PID" 2>/dev/null || true
  fi
  if [[ -n "$OLLAMA_MODEL" ]]; then
    ollama stop "$OLLAMA_MODEL" >/dev/null 2>&1 || true
  fi
  if [[ $OLLAMA_STARTED -eq 1 && -n "$OLLAMA_PID" ]]; then
    kill "$OLLAMA_PID" 2>/dev/null || true
    wait "$OLLAMA_PID" 2>/dev/null || true
  fi
  rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

drop_file_cache() {
  if [[ "$DROP_CACHES" == 1 ]]; then
    sync
    printf '3\n' >/proc/sys/vm/drop_caches
  fi
}

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

GRADLE_TASKS=(
  :models-rag-bench:installDist
)
if [[ "$MODELS_BACKEND" == "rust-ffm" ]]; then
  GRADLE_TASKS=(
    :backend-native:cargoBuildRelease
    "${GRADLE_TASKS[@]}"
  )
fi
"$ROOT_DIR/gradlew" --no-daemon -p "$ROOT_DIR" "${GRADLE_TASKS[@]}"
if [[ "$MODELS_BACKEND" == "rust-ffm" && ! -f "$NATIVE_LIBRARY" ]]; then
  echo "Models native library not found: $NATIVE_LIBRARY" >&2
  exit 1
fi

RAG_CLI="$ROOT_DIR/models-rag-bench/build/install/models-rag-bench/bin/models-rag-bench"
MODEL_SHA=$(sha256sum "$MODEL_PATH" | awk '{print $1}')
SAFE_MODEL_ID=$(printf '%s' "$MODEL_ID" | tr -cs '[:alnum:]._-' '-')
OLLAMA_MODEL="modeljars-rag-${SAFE_MODEL_ID}:${MODEL_SHA:0:12}"
MODELS_COMMIT=$(git -C "$ROOT_DIR" rev-parse HEAD)
VECTORS_VERSION=$(sed -n -E 's/^vectorsVersion[[:space:]]*=[[:space:]]*//p' "$ROOT_DIR/gradle.properties")
MODELS_VERSION="models@$MODELS_COMMIT com.integrallis:vectors-core@$VECTORS_VERSION"

DEFAULT_JAVA_OPTS="${JAVA_OPTS:-} --enable-native-access=ALL-UNNAMED"
DEFAULT_JAVA_OPTS="$DEFAULT_JAVA_OPTS -XX:ActiveProcessorCount=$THREADS"
if [[ "$MODELS_BACKEND" == "rust-ffm" ]]; then
  DEFAULT_JAVA_OPTS="$DEFAULT_JAVA_OPTS -Dmodels.native.kernels.library=$NATIVE_LIBRARY"
fi

COMMON_ARGS=(
  --framework plain-java
  --model-id "$MODEL_ID"
  --workload "$WORKLOAD"
  --prompt-template "$PROMPT_TEMPLATE"
  --context "$CONTEXT"
  --threads "$THREADS"
  --top-k 1
  --max-tokens "$MAX_TOKENS"
)
if [[ -n "$STOP_SEQUENCE" ]]; then
  COMMON_ARGS+=(--stop-sequence "$STOP_SEQUENCE")
fi

# Correctness is qualified against the public library defaults before any
# model- or host-specific performance properties are applied. One measured
# iteration covers every case in the workload, including prefix-cache reuse.
DEFAULT_CORRECTNESS_DIR="$OUTPUT_DIR/default-correctness"
DEFAULT_REPORT="$DEFAULT_CORRECTNESS_DIR/models-$MODELS_BACKEND.json"
mkdir -p "$DEFAULT_CORRECTNESS_DIR"
export JAVA_OPTS="$DEFAULT_JAVA_OPTS"
assert_no_competing_inference_processes
drop_file_cache
"$RAG_CLI" \
  --backend "$MODELS_BACKEND" \
  --backend-version "$MODELS_VERSION" \
  --model "$MODEL_PATH" \
  --output "$DEFAULT_REPORT" \
  "${COMMON_ARGS[@]}" \
  --warmups 0 \
  --iterations 1

if ! jq -e \
  --arg backend "$MODELS_BACKEND" \
  --arg model_id "$MODEL_ID" \
  --arg artifact_sha "$MODEL_SHA" \
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
   and (.failures | length) == 0' \
  "$DEFAULT_REPORT" >/dev/null; then
  echo "library-default correctness smoke failed: $DEFAULT_REPORT" >&2
  exit 1
fi
if [[ "$MODELS_BACKEND" == "rust-ffm" ]] &&
   ! jq -e '.backendDiagnostics.environment["native-quantized-decode"] == "false"' \
     "$DEFAULT_REPORT" >/dev/null; then
  echo "default smoke unexpectedly enabled models.native.quantizedDecode" >&2
  exit 1
fi
DEFAULT_REPORT_SHA=$(sha256sum "$DEFAULT_REPORT" | awk '{print $1}')
jq -n \
  --arg backend "$MODELS_BACKEND" \
  --arg model_id "$MODEL_ID" \
  --arg artifact_sha "$MODEL_SHA" \
  --arg report "default-correctness/models-$MODELS_BACKEND.json" \
  --arg report_sha "$DEFAULT_REPORT_SHA" \
  --argjson total_attempts "$(jq '.summary.totalAttempts' "$DEFAULT_REPORT")" \
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
    tuningSystemProperties: []}' \
  >"$DEFAULT_CORRECTNESS_DIR/smoke.json"

# Performance qualification is deliberately separate and may use explicitly
# recorded tuning. It cannot compensate for a failed default-correctness smoke.
export JAVA_OPTS="$DEFAULT_JAVA_OPTS"
if [[ "$MODELS_BACKEND" == "rust-ffm" ]]; then
  export JAVA_OPTS="$JAVA_OPTS -Dmodels.native.quantizedDecode=true"
  export JAVA_OPTS="$JAVA_OPTS -Dmodels.native.kernels.threads=$NATIVE_THREADS"
fi
if [[ -n "${RAG_TUNED_JAVA_OPTS:-}" ]]; then
  export JAVA_OPTS="$JAVA_OPTS $RAG_TUNED_JAVA_OPTS"
fi

assert_no_competing_inference_processes
drop_file_cache
"$RAG_CLI" \
  --backend "$MODELS_BACKEND" \
  --backend-version "$MODELS_VERSION" \
  --model "$MODEL_PATH" \
  --output "$OUTPUT_DIR/models-$MODELS_BACKEND.json" \
  "${COMMON_ARGS[@]}" \
  --warmups "$WARMUPS" \
  --iterations "$ITERATIONS"

if ! curl -fsS "$OLLAMA_ENDPOINT/api/version" >/dev/null 2>&1; then
  ollama serve >"$OUTPUT_DIR/ollama.log" 2>&1 &
  OLLAMA_PID=$!
  OLLAMA_STARTED=1
  for _ in $(seq 1 300); do
    if curl -fsS "$OLLAMA_ENDPOINT/api/version" >/dev/null 2>&1; then
      break
    fi
    if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
      echo "ollama exited before becoming ready" >&2
      exit 1
    fi
    sleep 0.1
  done
else
  OLLAMA_PID=$(pgrep -xo ollama || true)
fi
if ! curl -fsS "$OLLAMA_ENDPOINT/api/version" >/dev/null 2>&1; then
  echo "ollama did not become ready before the timeout" >&2
  exit 1
fi
if [[ -z "$OLLAMA_PID" ]]; then
  echo "could not identify the Ollama daemon PID" >&2
  exit 1
fi
printf 'FROM %s\n' "$MODEL_PATH" >"$TEMP_DIR/Modelfile"
ollama create "$OLLAMA_MODEL" -f "$TEMP_DIR/Modelfile"
OLLAMA_VERSION=$(ollama --version 2>&1 | head -n 1)

drop_file_cache
"$RAG_CLI" \
  --backend ollama \
  --backend-version "$OLLAMA_VERSION" \
  --model "$OLLAMA_MODEL" \
  --artifact "$MODEL_PATH" \
  --endpoint "$OLLAMA_ENDPOINT" \
  --pid "$OLLAMA_PID" \
  --output "$OUTPUT_DIR/ollama.json" \
  "${COMMON_ARGS[@]}" \
  --warmups "$WARMUPS" \
  --iterations "$ITERATIONS"
ollama stop "$OLLAMA_MODEL" >/dev/null

assert_no_competing_inference_processes
drop_file_cache
"$LLAMA_SERVER" \
  --model "$MODEL_PATH" \
  --ctx-size "$CONTEXT" \
  --threads "$THREADS" \
  --host 127.0.0.1 \
  --port "$LLAMA_PORT" \
  --no-warmup \
  >"$OUTPUT_DIR/llama.cpp.log" 2>&1 &
LLAMA_PID=$!
for _ in $(seq 1 600); do
  if curl -fsS "http://127.0.0.1:$LLAMA_PORT/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
    echo "llama-server exited before becoming ready" >&2
    exit 1
  fi
  sleep 0.1
done
if ! curl -fsS "http://127.0.0.1:$LLAMA_PORT/health" >/dev/null 2>&1; then
  echo "llama-server did not become ready before the timeout" >&2
  exit 1
fi
LLAMA_VERSION=$("$LLAMA_SERVER" --version 2>&1 | head -n 1)

"$RAG_CLI" \
  --backend llama.cpp \
  --backend-version "$LLAMA_VERSION" \
  --model "$MODEL_PATH" \
  --artifact "$MODEL_PATH" \
  --endpoint "http://127.0.0.1:$LLAMA_PORT" \
  --pid "$LLAMA_PID" \
  --output "$OUTPUT_DIR/llama.cpp.json" \
  "${COMMON_ARGS[@]}" \
  --warmups "$WARMUPS" \
  --iterations "$ITERATIONS"

"$RAG_CLI" qualify \
  --candidate "$OUTPUT_DIR/models-$MODELS_BACKEND.json" \
  --comparator "$OUTPUT_DIR/llama.cpp.json" \
  --comparator "$OUTPUT_DIR/ollama.json" \
  --output "$OUTPUT_DIR/qualification.json" \
  --require-qualified

for report in "models-$MODELS_BACKEND.json" llama.cpp.json ollama.json; do
  report_sha=$(jq -r '.artifactSha256' "$OUTPUT_DIR/$report")
  if [[ "$report_sha" != "$MODEL_SHA" ]]; then
    echo "artifact mismatch in $report" >&2
    exit 1
  fi
done

echo "library-default correctness evidence: $DEFAULT_CORRECTNESS_DIR"
echo "qualified tuned RAG evidence: $OUTPUT_DIR"
