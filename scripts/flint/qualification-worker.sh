#!/usr/bin/env bash
# Copyright 2025-2026 Integrallis Software, LLC
# SPDX-License-Identifier: Apache-2.0
# Dedicated, disposable Linux host only. Bootstrap must enforce shutdown independently.
set -euo pipefail
INPUT=$(realpath "${1:?usage: qualification-worker.sh MANIFEST.json}")
WORK=/work/flint
OUT=$WORK/results
mkdir -p "$OUT" "$WORK/ollama" "$WORK/jdk"
cp "$INPUT" "$OUT/inputs.json"
DEST=$(jq -er '.resultsUri' "$INPUT")
REV=$(jq -er '.sourceRevision | select(test("^[0-9a-f]{40}$"))' "$INPUT")
PAYLOAD=$(jq -er '.payloadUri' "$INPUT")
PAYLOAD_SHA=$(jq -er '.payloadSha256' "$INPUT")
THREADS=16
MODEL=$WORK/Flint-Q4_K_M.gguf
MODEL_ID=ggml_org_flint_gguf_q4_k_m
MODEL_SHA=c3c4d93f9db92627a2fc47ea8fe8c64a094cabd3549db544bcdd2c00e9a9a347
LLAMA_REV=0893ec98f5eafad36a43a72cd1c3723d72b0d03a
export AWS_DEFAULT_REGION=us-east-1
export OLLAMA_MODELS=$WORK/ollama
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export JAVA_OPTS="-Xmx8g --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED -XX:ActiveProcessorCount=$THREADS"
exec > >(tee -a "$OUT/worker.log" /dev/console) 2>&1
sync_results() { aws s3 sync "$OUT/" "$DEST/" --only-show-errors; }
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
date -u +%FT%TZ >"$OUT/started-at.txt"
uname -a >"$OUT/host.txt"
lscpu >>"$OUT/host.txt"
free -b >>"$OUT/host.txt"
sync_results
# Independent periodic uploads preserve partial reports and diagnostics on timeout.
(while sleep 30; do sync_results || true; done) &

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
download() { curl -fL --retry 3 --connect-timeout 30 --max-time 900 "$1" -o "$2"; }
verify() { printf '%s  %s\n' "$1" "$2" | sha256sum -c -; }
idle() {
  # A comparator daemon may be idle; its runner must be gone before the next arm.
  if pgrep -f 'ollama runner|llama-server|com.integrallis.models.(bench|rag)' >"$OUT/competing-pids.txt"; then
    echo 'Competing inference processes; refusing incomparable measurements'
    exit 1
  fi
  sync
  printf '3\n' >/proc/sys/vm/drop_caches
}

aws s3 cp "$PAYLOAD" "$WORK/payload.tar.gz" --only-show-errors
verify "$PAYLOAD_SHA" "$WORK/payload.tar.gz"
tar -xzf "$WORK/payload.tar.gz" -C "$WORK"
test "$(cat "$WORK/source-revision.txt")" = "$REV"
RAG=$WORK/models-rag-bench/bin/models-rag-bench
BENCH=$WORK/models-bench/bin/models-bench
download 'https://github.com/adoptium/temurin25-binaries/releases/download/jdk-25.0.3%2B9/OpenJDK25U-jdk_x64_linux_hotspot_25.0.3_9.tar.gz' "$WORK/jdk.tar.gz"
verify 69264a7a211bf5029830d07bc3370f879769d62ebc5b5488e90c9343a2da0e1f "$WORK/jdk.tar.gz"
tar -xzf "$WORK/jdk.tar.gz" -C "$WORK/jdk" --strip-components=1
export JAVA_HOME=$WORK/jdk
export PATH=$JAVA_HOME/bin:$PATH
java -version >"$OUT/java-version.txt" 2>&1
download 'https://ollama.com/download/ollama-linux-amd64.tar.zst?version=0.40.9' "$WORK/ollama.tar.zst"
sha256sum "$WORK/ollama.tar.zst" >"$OUT/ollama-download.sha256"
tar --zstd -xf "$WORK/ollama.tar.zst" -C /usr/local
git clone --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp.git "$WORK/llama.cpp"
git -C "$WORK/llama.cpp" checkout --detach "$LLAMA_REV"
cmake -S "$WORK/llama.cpp" -B "$WORK/llama.cpp/build" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_BLAS=OFF -DLLAMA_CURL=ON
cmake --build "$WORK/llama.cpp/build" -j "$THREADS" --target llama-server
LLAMA=$WORK/llama.cpp/build/bin/llama-server
"$LLAMA" --version >"$OUT/llama-version.txt" 2>&1
download 'https://huggingface.co/ggml-org/Flint-GGUF/resolve/468307a1b38259e9e4c9d4fe53d4e3b76386572c/Flint-Q4_K_M.gguf' "$MODEL"
test "$(stat -c %s "$MODEL")" = 23387911136
verify "$MODEL_SHA" "$MODEL"
printf '%s  %s\n' "$MODEL_SHA" Flint-Q4_K_M.gguf >"$OUT/model.sha256"
sync_results

COMMON=(--framework plain-java --model-id "$MODEL_ID" --workload general --prompt-template flint --context 2048 --threads "$THREADS" --top-k 1 --max-tokens 256)
VERSION="models@0.5.7+flint-$REV com.integrallis:vectors-core@0.1.32"
idle
stage tool-qualification 3600 "$BENCH" tool-qualification --candidate flint-40b-a2.5b --model "$MODEL" --models-revision "$REV" --backend pure-java --max-tokens 256 --report "$OUT/tool-qualification.json" || true
idle
mkdir -p "$OUT/default-correctness"
DEFAULT=$OUT/default-correctness/models-pure-java.json
stage default-correctness 1800 "$RAG" --backend pure-java --backend-version "$VERSION" --model "$MODEL" --output "$DEFAULT" "${COMMON[@]}" --warmups 0 --iterations 1 || true
if [[ -f "$DEFAULT" ]] && jq -e --arg sha "$MODEL_SHA" --arg id "$MODEL_ID" '
  .backend == "pure-java" and .modelId == $id and .artifactSha256 == $sha
  and .settings.warmups == 0 and .settings.iterations == 1
  and .settings.generationControls.promptCache == "longest-common-prefix"
  and .summary.totalAttempts > 0 and .summary.successfulAttempts == .summary.totalAttempts
  and .summary.correctAnswerRate == 1 and .summary.abstentionAccuracy == 1
  and (.failures | length) == 0' "$DEFAULT" >/dev/null; then
  jq -n --arg id "$MODEL_ID" --arg sha "$MODEL_SHA" --arg reportSha "$(sha256sum "$DEFAULT" | awk '{print $1}')" --argjson attempts "$(jq '.summary.totalAttempts' "$DEFAULT")" \
    '{schemaVersion:1,kind:"model-backend-default-correctness-smoke",configuration:"library-defaults",backend:"pure-java",modelId:$id,artifactSha256:$sha,report:"default-correctness/models-pure-java.json",reportSha256:$reportSha,totalAttempts:$attempts,successfulAttempts:$attempts,tuningSystemProperties:[]}' >"$OUT/default-correctness/smoke.json"
else
  printf '%s\n' 'FAILED_OR_INCOMPLETE: comparative arms retained for diagnosis only' >"$OUT/default-correctness/failed.txt"
fi
idle
stage candidate 4200 "$RAG" --backend pure-java --backend-version "$VERSION" --model "$MODEL" --output "$OUT/candidate.json" "${COMMON[@]}" --warmups 1 --iterations 3 || true
idle
ollama serve >"$OUT/ollama-server.log" 2>&1 &
OLLAMA_PID=$!
for ((i=0; i<60; i++)); do
  curl -fsS http://127.0.0.1:11434/api/version >"$OUT/ollama-version.json" && break
  sleep 1
done
jq -e '.version == "0.40.9"' "$OUT/ollama-version.json"
printf 'FROM %s\n' "$MODEL" >"$WORK/Modelfile"
stage ollama-import 600 ollama create flint-qualification -f "$WORK/Modelfile"
stage ollama 1800 "$RAG" --backend ollama --backend-version ollama@0.40.9 --model flint-qualification --artifact "$MODEL" --endpoint http://127.0.0.1:11434 --pid "$OLLAMA_PID" --output "$OUT/ollama.json" "${COMMON[@]}" --warmups 1 --iterations 3 || true
ollama stop flint-qualification
kill "$OLLAMA_PID"
wait "$OLLAMA_PID" || true
for ((i=0; i<60; i++)); do
  pgrep -f 'ollama runner' >/dev/null || break
  sleep 1
done
idle
"$LLAMA" --model "$MODEL" --ctx-size 2048 --threads "$THREADS" --host 127.0.0.1 --port 18081 --no-warmup >"$OUT/llama-server.log" 2>&1 &
LLAMA_PID=$!
for ((i=0; i<180; i++)); do
  curl -fsS http://127.0.0.1:18081/health >"$OUT/llama-health.json" && break
  kill -0 "$LLAMA_PID"
  sleep 1
done
curl -fsS http://127.0.0.1:18081/health >/dev/null
stage llama 1800 "$RAG" --backend llama.cpp --backend-version "llama.cpp@$LLAMA_REV" --model "$MODEL" --artifact "$MODEL" --endpoint http://127.0.0.1:18081 --pid "$LLAMA_PID" --output "$OUT/llama.json" "${COMMON[@]}" --warmups 1 --iterations 3 || true
kill "$LLAMA_PID"
wait "$LLAMA_PID" || true
if [[ -f "$OUT/candidate.json" && -f "$OUT/ollama.json" && -f "$OUT/llama.json" ]]; then
  stage qualification 120 "$RAG" qualify --candidate "$OUT/candidate.json" --comparator "$OUT/ollama.json" --comparator "$OUT/llama.json" --output "$OUT/qualification.json" || true
fi
printf '%s\n' 'All planned stages attempted; consult reports and exit codes for verdicts.' >"$OUT/completed.txt"
