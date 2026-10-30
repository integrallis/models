#!/usr/bin/env bash
# Copyright 2025-2026 Integrallis Software, LLC
# SPDX-License-Identifier: Apache-2.0
# Flint tool grammar comparison: default and schema profile on the same disposable host.
# Dedicated, disposable Linux host only. Bootstrap must enforce shutdown independently.
set -euo pipefail
INPUT=$(realpath "${1:?usage: tool-grammar-worker.sh MANIFEST.json}")
WORK=/work/flint
OUT=$WORK/results
mkdir -p "$OUT" "$WORK/jdk"
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
download 'https://github.com/adoptium/temurin25-binaries/releases/download/jdk-25.0.3%2B9/OpenJDK25U-jdk_x64_linux_hotspot_25.0.3_9.tar.gz' "$WORK/jdk.tar.gz"
verify 69264a7a211bf5029830d07bc3370f879769d62ebc5b5488e90c9343a2da0e1f "$WORK/jdk.tar.gz"
tar -xzf "$WORK/jdk.tar.gz" -C "$WORK/jdk" --strip-components=1
export JAVA_HOME=$WORK/jdk
export PATH=$JAVA_HOME/bin:$PATH
java -version >"$OUT/java-version.txt" 2>&1
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

CP="$WORK/models-bench/lib/*"
JAVA=(java -Xmx8g --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED -XX:ActiveProcessorCount=16 "-Dmodels.native.kernels.library=$NATIVE")
mkdir -p "$WORK/probe"
javac --release 25 -cp "$CP" -d "$WORK/probe" "$WORK/FlintKernelParity.java"
idle
stage parity 900 "${JAVA[@]}" -cp "$WORK/probe:$CP" com.integrallis.models.bench.FlintKernelParity "$MODEL" "$WORK/reference.properties" "$OUT/parity.json"

# No expected argument values are passed to call-generation constraints. Both use the same
# declared schemas, greedy sampling, and 512-token call budget. The existing follow-up fixture
# supplies a scripted prior weather call/result and allows 512 unconstrained answer tokens.
for profile in default flint-schema; do
  label=tools-$profile
  expected=default
  if [[ "$profile" == flint-schema ]]; then expected=flint-schema-v1; fi
  idle
  stage "$label" 2100 "${JAVA[@]}" -cp "$CP" com.integrallis.models.bench.InferenceBenchmarkCli tool-qualification \
    --candidate flint-40b-a2.5b --model "$MODEL" --models-revision "$REV" --backend rust-ffm \
    --max-tokens 512 --follow-up-max-tokens 512 --tool-constraints "$profile" --report "$OUT/$label.json" || true
  if [[ ! -f "$OUT/$label.json" ]] || ! jq -e --arg sha "$MODEL_SHA" --arg profile "$expected" '
    .complete == true and .plannedAttempts == 14 and (.cases | length) == 14
    and .artifactSha256 == $sha and .generation.maxTokens == 512
    and .generation.toolConstraints == $profile
    and .policyVersion == "small-model-tool-conformance-v2"
    and .generation.followUpMaxTokens == 512' "$OUT/$label.json" >/dev/null; then
    printf '%s\n' 'INCOMPLETE_NOT_A_PASS' >"$OUT/$label.incomplete"
  fi
done
printf '%s\n' 'Both tool profiles attempted; inspect complete reports and visible follow-up grading. No catalog admission is implied.' >"$OUT/completed.txt"

