#!/usr/bin/env bash
# Copyright 2025-2026 Integrallis Software, LLC
# SPDX-License-Identifier: Apache-2.0
# Flint improvement campaign: see docs/flint.md for the frozen experiment protocol.
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
export AWS_DEFAULT_REGION=us-east-1
export OLLAMA_MODELS=$WORK/ollama
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export JAVA_OPTS="-Xmx8g --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED -XX:ActiveProcessorCount=$THREADS"
exec > >(tee -a "$OUT/worker.log" /dev/console) 2>&1
# The existing worker role has GetObject/PutObject, but deliberately no ListBucket.
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
download 'https://static.rust-lang.org/dist/rust-1.96.0-x86_64-unknown-linux-gnu.tar.xz' "$WORK/rust.tar.xz"
verify c295047583a56238ea06b43f849f4b877fa12bfd4c7103f8d9a74c94c9c4e108 "$WORK/rust.tar.xz"
mkdir -p "$WORK/rust-install" "$WORK/rust"
tar -xJf "$WORK/rust.tar.xz" -C "$WORK/rust-install" --strip-components=1
"$WORK/rust-install/install.sh" --prefix="$WORK/rust" --components=rustc,cargo,rust-std-x86_64-unknown-linux-gnu --disable-ldconfig
export PATH=$WORK/rust/bin:$PATH
rustc --version >"$OUT/rust-version.txt"
export CARGO_TARGET_DIR=$WORK/rust-target
stage native-build 900 cargo build --offline --locked --release --manifest-path "$WORK/model-kernels/Cargo.toml"
NATIVE=$WORK/rust-target/release/libjmodels_kernels.so
sha256sum "$NATIVE" >"$OUT/native-library.sha256"
export JAVA_OPTS="$JAVA_OPTS -Dmodels.native.kernels.library=$NATIVE"
download 'https://huggingface.co/ggml-org/Flint-GGUF/resolve/468307a1b38259e9e4c9d4fe53d4e3b76386572c/Flint-Q4_K_M.gguf' "$MODEL"
test "$(stat -c %s "$MODEL")" = 23387911136
verify "$MODEL_SHA" "$MODEL"
printf '%s  %s\n' "$MODEL_SHA" Flint-Q4_K_M.gguf >"$OUT/model.sha256"
sync_results


CP="$WORK/models-bench/lib/*:$WORK/models-rag-bench/lib/*"
JAVA=(java -Xmx8g --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED -XX:ActiveProcessorCount=16 "-Dmodels.native.kernels.library=$NATIVE")
BENCH=("${JAVA[@]}" -cp "$CP" com.integrallis.models.bench.InferenceBenchmarkCli)
AUDIT=("${JAVA[@]}" -cp "$CP" com.integrallis.models.rag.RagVisibleAnswerAudit)
COMMON=(--framework plain-java --model-id "$MODEL_ID" --prompt-template flint --context 2048 --threads "$THREADS" --top-k 1)
VERSION="models@0.5.7+flint-$REV com.integrallis:vectors-core@0.1.32"
mkdir -p "$WORK/probe" "$OUT/visible" "$OUT/default-correctness"
javac --release 25 -cp "$CP" -d "$WORK/probe" "$WORK/FlintKernelParity.java"
idle
stage parity 900 "${JAVA[@]}" -cp "$WORK/probe:$CP" com.integrallis.models.bench.FlintKernelParity "$MODEL" "$WORK/reference.properties" "$OUT/parity.json"

benchmark() {
  local label=$1 seconds=$2 backend=$3 cap=$4 warmups=$5 iterations=$6 workload=$7
  local output=$OUT/$label.json
  local model=$MODEL version=$VERSION
  local extra=()
  if [[ "$backend" == ollama ]]; then
    model=flint-qualification
    version=ollama@0.40.9
    extra=(--artifact "$MODEL" --endpoint http://127.0.0.1:11434 --pid "$RUNNER_PID")
  fi
  stage "$label" "$seconds" "$RAG" --backend "$backend" --backend-version "$version" --model "$model" --output "$output" "${COMMON[@]}" --workload "$workload" --max-tokens "$cap" --warmups "$warmups" --iterations "$iterations" "${extra[@]}" || true
  if [[ -f "$output" ]]; then
    stage "$label-visible" 120 "${AUDIT[@]}" "$output" "$OUT/visible/$label.json"
  fi
}

start_ollama() {
  ollama serve >"$OUT/ollama-server.log" 2>&1 &
  OLLAMA_PID=$!
  for ((i=0; i<60; i++)); do
    curl -fsS http://127.0.0.1:11434/api/version >"$OUT/ollama-version.json" && break
    sleep 1
  done
  jq -e '.version == "0.40.9"' "$OUT/ollama-version.json"
  printf 'FROM %s\n' "$MODEL" >"$WORK/Modelfile"
  stage ollama-import 900 ollama create flint-qualification -f "$WORK/Modelfile"
  # Prime a single runner, then measure its PID rather than the small HTTP daemon.
  stage ollama-prime 600 curl -fsS --max-time 590 http://127.0.0.1:11434/api/generate -H 'Content-Type: application/json' -d '{"model":"flint-qualification","prompt":"Warmup","raw":true,"stream":false,"keep_alive":-1,"options":{"num_predict":1,"num_ctx":2048,"num_thread":16,"temperature":0}}'
  # Ollama 0.40.9 launches llama-server; older versions used an `ollama runner` command.
  # Bind discovery to this daemon's children so unrelated inference can never be sampled.
  mapfile -t runners < <(ps -eo pid=,ppid=,args= | awk -v parent="$OLLAMA_PID" \
    '$2 == parent && ($3 == "/usr/local/lib/ollama/llama-server" || $3 == "/usr/local/bin/ollama" || $3 == "ollama") { if ($3 ~ /llama-server$/ || $4 == "runner") print $1 }')
  if ((${#runners[@]} != 1)); then
    ps -eo pid,ppid,args >"$OUT/runner-discovery-failure.txt"
    echo 'Expected exactly one Ollama runner' >&2
    exit 1
  fi
  RUNNER_PID=${runners[0]}
  printf '%s\n' "$RUNNER_PID" >"$OUT/ollama-runner-pid.txt"
  ps -p "$OLLAMA_PID,$RUNNER_PID" -o pid,ppid,rss,args >"$OUT/ollama-processes.txt"
}
stop_ollama() {
  ollama stop flint-qualification
  kill "$OLLAMA_PID"
  wait "$OLLAMA_PID" || true
  for ((i=0; i<60; i++)); do
    if ! kill -0 "$RUNNER_PID" 2>/dev/null; then return; fi
    sleep 1
  done
  echo 'Ollama runner did not stop' >&2
  exit 1
}

# Short development comparison only. It cannot satisfy any catalog qualification gate.
if jq -e '.diagnosticOnly == true' "$INPUT" >/dev/null; then
  CAP=512
  COMMON+=(--case auto-glass-deadline)
  idle
  benchmark native-speed-pilot 1200 rust-ffm "$CAP" 1 3 general
  if jq -e '.diagnosticProfiles == "workers"' "$INPUT" >/dev/null; then
    ORIGINAL_JAVA_OPTS=$JAVA_OPTS
    for decode_threads in 8 4; do
      export JAVA_OPTS="$ORIGINAL_JAVA_OPTS -Dmodels.native.kernels.decodeThreads=$decode_threads"
      idle
      benchmark "native-decode$decode_threads-speed-pilot" 1200 rust-ffm "$CAP" 1 3 general
    done
    export JAVA_OPTS=$ORIGINAL_JAVA_OPTS
  fi
  idle
  start_ollama
  benchmark ollama-speed-pilot 1200 ollama "$CAP" 1 3 general
  stop_ollama
  printf '%s\n' 'Single-case performance diagnosis only; no catalog qualification attempted.' >"$OUT/completed.txt"
  exit 0
fi

idle
benchmark java-default 1800 pure-java 256 0 1 general
idle
benchmark native-default 1800 rust-ffm 256 0 1 general
DEFAULT=$OUT/native-default.json
if [[ -f "$DEFAULT" ]] && jq -e --arg sha "$MODEL_SHA" --arg id "$MODEL_ID" '
  .backend == "rust-ffm" and .modelId == $id and .artifactSha256 == $sha
  and .settings.warmups == 0 and .settings.iterations == 1
  and .settings.generationControls.promptCache == "longest-common-prefix"
  and .summary.totalAttempts == 9 and .summary.successfulAttempts == 9
  and .summary.correctAnswerRate == 1 and .summary.abstentionAccuracy == 1
  and (.failures | length) == 0' "$DEFAULT" >/dev/null; then
  cp "$DEFAULT" "$OUT/default-correctness/models-rust-ffm.json"
  jq -n --arg id "$MODEL_ID" --arg sha "$MODEL_SHA" --arg reportSha "$(sha256sum "$DEFAULT" | awk '{print $1}')" \
    '{schemaVersion:1,kind:"model-backend-default-correctness-smoke",configuration:"library-defaults",backend:"rust-ffm",modelId:$id,artifactSha256:$sha,report:"default-correctness/models-rust-ffm.json",reportSha256:$reportSha,totalAttempts:9,successfulAttempts:9,tuningSystemProperties:[]}' >"$OUT/default-correctness/smoke.json"
else
  printf '%s\n' 'FAILED_OR_INCOMPLETE' >"$OUT/default-correctness/failed.txt"
fi

# Choose the smallest budget whose nine pilot responses all satisfy the visible-answer audit.
# The held-out workload is never consulted when choosing the budget.
CAP=1024
PROFILE_PASSED=false
for cap in 256 512 1024; do
  label=native-default
  if ((cap != 256)); then
    label=native-budget-$cap
    idle
    benchmark "$label" 2400 rust-ffm "$cap" 0 1 general
  fi
  if [[ -f "$OUT/visible/$label.json" ]] && jq -e '.attempts == 9 and .requiredFactsAndCitationsPasses == 9' "$OUT/visible/$label.json" >/dev/null; then
    CAP=$cap
    PROFILE_PASSED=true
    break
  fi
done
jq -n --argjson cap "$CAP" --argjson passed "$PROFILE_PASSED" \
  '{maxOutputTokens:$cap,pilotVisibleGatePassed:$passed,selectionRule:"smallest of 256/512/1024 with 9/9 complete visible fact/citation/polarity passes; 1024 diagnostic if none",heldOutConsulted:false}' >"$OUT/selected-profile.json"
sync_results
idle
benchmark native-candidate 4200 rust-ffm "$CAP" 1 3 general
idle
benchmark native-reliability 2400 rust-ffm "$CAP" 0 1 answer-reliability
idle
stage native-tools 2400 "${BENCH[@]}" tool-qualification --candidate flint-40b-a2.5b --model "$MODEL" --models-revision "$REV" --backend rust-ffm --max-tokens 512 --follow-up-max-tokens 512 --report "$OUT/native-tools.json" || true
idle
stage native-profile 900 env "JAVA_OPTS=$JAVA_OPTS -XX:StartFlightRecording=filename=$OUT/native-profile.jfr,settings=profile,dumponexit=true" "$RAG" --backend rust-ffm --backend-version "$VERSION" --model "$MODEL" --output "$OUT/native-profile.json" "${COMMON[@]}" --workload general --case auto-glass-deadline --max-tokens 64 --warmups 1 --iterations 1 || true
if [[ -f "$OUT/native-profile.jfr" ]]; then
  jfr summary "$OUT/native-profile.jfr" >"$OUT/native-profile-summary.txt"
  jfr print --json --events jdk.ExecutionSample,jdk.ObjectAllocationSample "$OUT/native-profile.jfr" >"$OUT/native-profile-samples.json"
fi
idle
start_ollama
benchmark ollama-candidate 2400 ollama "$CAP" 1 3 general
benchmark ollama-reliability 1800 ollama "$CAP" 0 1 answer-reliability
stop_ollama
if [[ -f "$OUT/native-candidate.json" && -f "$OUT/ollama-candidate.json" ]]; then
  stage qualification 120 "$RAG" qualify --candidate "$OUT/native-candidate.json" --comparator "$OUT/ollama-candidate.json" --output "$OUT/qualification.json"
fi
printf '%s\n' 'All planned stages attempted; consult reports, default smoke, selected profile, and held-out accuracy before admission.' >"$OUT/completed.txt"
