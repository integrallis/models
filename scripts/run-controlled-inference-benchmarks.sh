#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 MODEL_GGUF MODEL_ID [OUTPUT_DIR]" >&2
  exit 2
fi

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_PATH=$(realpath "$1")
MODEL_ID=$2
OUTPUT_DIR=${3:-"$ROOT_DIR/build/reports/inference/$MODEL_ID"}
THREADS=${BENCH_THREADS:-$(nproc)}
CONTEXT=${BENCH_CONTEXT:-2048}
MAX_TOKENS=${BENCH_MAX_TOKENS:-64}
WARMUPS=${BENCH_WARMUPS:-2}
ITERATIONS=${BENCH_ITERATIONS:-10}
DROP_CACHES=${BENCH_DROP_CACHES:-0}
LLAMA_PORT=${LLAMA_PORT:-18080}
LLAMA_SERVER=${LLAMA_SERVER:-llama-server}
PROMPT_FILE=${BENCH_PROMPT_FILE:-"$ROOT_DIR/models-bench/prompts/completion.txt"}
BENCH_MODELJAR_ALIAS=${BENCH_MODELJAR_ALIAS:-}
MODELS_BACKEND=${BENCH_MODELS_BACKEND:-rust-ffm}

for command in awk curl git java jq nproc ollama ps realpath sha256sum sync "$LLAMA_SERVER"; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "required command not found: $command" >&2
    exit 1
  fi
done
JAVA_FEATURE=$(java -XshowSettings:properties -version 2>&1 \
  | awk -F= '/^[[:space:]]*java.specification.version[[:space:]]*=/{gsub(/[[:space:]]/, "", $2); print $2}')
if [[ "$JAVA_FEATURE" != 25 ]]; then
  echo "the benchmark CLI requires Java 25; default java is $JAVA_FEATURE" >&2
  exit 1
fi
if [[ "$DROP_CACHES" != 0 && "$DROP_CACHES" != 1 ]]; then
  echo "BENCH_DROP_CACHES must be 0 or 1" >&2
  exit 1
fi
case "$MODELS_BACKEND" in
  pure-java|rust-ffm) ;;
  *)
    echo "BENCH_MODELS_BACKEND must be pure-java or rust-ffm" >&2
    exit 1
    ;;
esac
if [[ "$MODELS_BACKEND" == rust-ffm && -n "$BENCH_MODELJAR_ALIAS" ]]; then
  echo "BENCH_MODELJAR_ALIAS is currently supported only with BENCH_MODELS_BACKEND=pure-java" >&2
  exit 1
fi
if [[ "$DROP_CACHES" == 1 && ! -w /proc/sys/vm/drop_caches ]]; then
  echo "BENCH_DROP_CACHES=1 requires permission to write /proc/sys/vm/drop_caches" >&2
  exit 1
fi
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "model not found: $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "prompt not found: $PROMPT_FILE" >&2
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

cleanup() {
  if [[ -n "$LLAMA_PID" ]]; then
    kill "$LLAMA_PID" 2>/dev/null || true
    wait "$LLAMA_PID" 2>/dev/null || true
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
    ps -eo pid=,comm=,args= | awk -v self_pid="$$" '
      {
        is_llama = $2 ~ /^(llama-cli|llama-server|llama-bench)$/
        is_ollama_runner = $2 == "ollama" && $0 ~ /(^|[[:space:]])runner([[:space:]]|$)/
        is_models_java = $2 == "java" && $0 ~ /com\.integrallis\.models\.(bench|rag)/
        if ($1 != self_pid && (is_llama || is_ollama_runner || is_models_java)) {
          print
        }
      }
    '
  )
  if [[ -n "$competitors" ]]; then
    echo "competing inference processes make benchmark evidence invalid:" >&2
    printf '%s\n' "$competitors" >&2
    exit 1
  fi
}

"$ROOT_DIR/gradlew" --no-daemon -p "$ROOT_DIR" :models-bench:installDist
BENCHMARK_CLI="$ROOT_DIR/models-bench/build/install/models-bench/bin/models-bench"
assert_no_competing_inference_processes

MODEL_SHA=$(sha256sum "$MODEL_PATH" | awk '{print $1}')
SAFE_MODEL_ID=$(printf '%s' "$MODEL_ID" | tr -cs '[:alnum:]._-' '-')
OLLAMA_MODEL="modeljars-bench-${SAFE_MODEL_ID}:${MODEL_SHA:0:12}"
MODELS_COMMIT=$(git -C "$ROOT_DIR" rev-parse HEAD)
VECTORS_VERSION=$(sed -n -E 's/^vectorsVersion[[:space:]]*=[[:space:]]*//p' "$ROOT_DIR/gradle.properties")
MODELS_VERSION="models@$MODELS_COMMIT com.integrallis:vectors-core@$VECTORS_VERSION"

BENCHMARK_JVM_OPTIONS="-XX:ActiveProcessorCount=$THREADS"
if [[ "$MODELS_BACKEND" == rust-ffm ]]; then
  BENCHMARK_JVM_OPTIONS="$BENCHMARK_JVM_OPTIONS -Dmodels.native.quantizedDecode=true"
  BENCHMARK_JVM_OPTIONS="$BENCHMARK_JVM_OPTIONS -Dmodels.native.kernels.threads=$THREADS"
fi
export MODELS_BENCH_OPTS="${MODELS_BENCH_OPTS:-} $BENCHMARK_JVM_OPTIONS"

COMMON_ARGS=(
  --model-id "$MODEL_ID"
  --prompt-file "$PROMPT_FILE"
  --max-tokens "$MAX_TOKENS"
  --warmups "$WARMUPS"
  --iterations "$ITERATIONS"
  --context "$CONTEXT"
  --threads "$THREADS"
)

MODELS_MODEL_ARGS=(--model "$MODEL_PATH")
if [[ -n "$BENCH_MODELJAR_ALIAS" ]]; then
  MODELS_MODEL_ARGS=(--modeljar "$BENCH_MODELJAR_ALIAS")
fi

drop_file_cache
"$BENCHMARK_CLI" \
  --backend "$MODELS_BACKEND" \
  "${MODELS_MODEL_ARGS[@]}" \
  --backend-version "$MODELS_VERSION" \
  --output "$OUTPUT_DIR/models.json" \
  "${COMMON_ARGS[@]}"
MODELS_ARTIFACT_SHA=$(jq -r '.artifactSha256' "$OUTPUT_DIR/models.json")
if [[ "$MODELS_ARTIFACT_SHA" != "$MODEL_SHA" ]]; then
  echo "Models and comparator benchmark artifacts differ" >&2
  exit 1
fi

if ! curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
  ollama serve >"$OUTPUT_DIR/ollama.log" 2>&1 &
  OLLAMA_PID=$!
  OLLAMA_STARTED=1
  for _ in $(seq 1 300); do
    if curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
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
if ! curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
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
"$BENCHMARK_CLI" \
  --backend ollama \
  --model "$OLLAMA_MODEL" \
  --artifact "$MODEL_PATH" \
  --endpoint http://127.0.0.1:11434 \
  --backend-version "$OLLAMA_VERSION" \
  --pid "$OLLAMA_PID" \
  --output "$OUTPUT_DIR/ollama.json" \
  "${COMMON_ARGS[@]}"
ollama stop "$OLLAMA_MODEL" >/dev/null

drop_file_cache
LLAMA_START_NS=$(date +%s%N)
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
LLAMA_LOAD_MS=$(( ($(date +%s%N) - LLAMA_START_NS) / 1000000 ))
LLAMA_VERSION=$("$LLAMA_SERVER" --version 2>&1 | head -n 1)

"$BENCHMARK_CLI" \
  --backend llama.cpp \
  --model "$MODEL_PATH" \
  --artifact "$MODEL_PATH" \
  --endpoint "http://127.0.0.1:$LLAMA_PORT" \
  --backend-version "$LLAMA_VERSION" \
  --pid "$LLAMA_PID" \
  --load-ms "$LLAMA_LOAD_MS" \
  --output "$OUTPUT_DIR/llama.cpp.json" \
  "${COMMON_ARGS[@]}"

"$BENCHMARK_CLI" compare \
  --models "$OUTPUT_DIR/models.json" \
  --llama-cpp "$OUTPUT_DIR/llama.cpp.json" \
  --ollama "$OUTPUT_DIR/ollama.json" \
  --json "$OUTPUT_DIR/comparison.json" \
  --markdown "$OUTPUT_DIR/comparison.md"

echo "benchmark evidence: $OUTPUT_DIR"
