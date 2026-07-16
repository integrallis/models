#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 MODEL_GGUF MODEL_ID [OUTPUT_DIR]" >&2
  exit 2
fi

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_INPUT=$1
MODEL_PATH=$(cd "$(dirname "$MODEL_INPUT")" && pwd -P)/$(basename "$MODEL_INPUT")
MODEL_ID=$2
OUTPUT_DIR=${3:-"$ROOT_DIR/build/reports/determinism/$MODEL_ID"}
PROMPT=${AUDIT_PROMPT:-}
PROMPT_FILE=${AUDIT_PROMPT_FILE:-"$ROOT_DIR/models-bench/prompts/completion.txt"}
TOKENS=${AUDIT_TOKENS:-4}
ITERATIONS=${AUDIT_ITERATIONS:-3}
CONTEXT=${AUDIT_CONTEXT:-128}
PREFILL=${AUDIT_PREFILL:-sequential}

for command in awk java; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "required command not found: $command" >&2
    exit 1
  fi
done
JAVA_FEATURE=$(java -XshowSettings:properties -version 2>&1 \
  | awk -F= '/^[[:space:]]*java.specification.version[[:space:]]*=/{gsub(/[[:space:]]/, "", $2); print $2}')
if [[ "$JAVA_FEATURE" != 25 ]]; then
  echo "the determinism audit requires Java 25; default java is $JAVA_FEATURE" >&2
  exit 1
fi
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "model not found: $MODEL_PATH" >&2
  exit 1
fi
if [[ -z "$PROMPT" && ! -f "$PROMPT_FILE" ]]; then
  echo "prompt not found: $PROMPT_FILE" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
"$ROOT_DIR/gradlew" --no-daemon -p "$ROOT_DIR" :models-bench:installDist
AUDIT_CLI="$ROOT_DIR/models-bench/build/install/models-bench/bin/models-bench"
if [[ -n "$PROMPT" ]]; then
  PROMPT_ARGS=(--prompt "$PROMPT")
else
  PROMPT_ARGS=(--prompt-file "$PROMPT_FILE")
fi

run_audit() {
  local name=$1
  local java_options=$2
  echo "auditing $name with JAVA_OPTS='$java_options'"
  env JAVA_OPTS="$java_options" "$AUDIT_CLI" determinism \
    --model "$MODEL_PATH" \
    --model-id "$MODEL_ID" \
    "${PROMPT_ARGS[@]}" \
    --tokens "$TOKENS" \
    --iterations "$ITERATIONS" \
    --context "$CONTEXT" \
    --prefill "$PREFILL" \
    --output "$OUTPUT_DIR/$name.json"
}

run_audit panama-default ""
run_audit panama-serial "-Dvectors.gguf.parallel=false"
run_audit panama-no-fma \
  "-Dvectors.gguf.parallel=false -Dvectors.useVectorFMA=false -Dvectors.useScalarFMA=false"
run_audit panama-128 \
  "-Dvectors.gguf.parallel=false -Dvectors.maxBits=128"
run_audit scalar \
  "-Dvectors.forceScalar=true -Dvectors.gguf.parallel=false"

echo "determinism evidence: $OUTPUT_DIR"
