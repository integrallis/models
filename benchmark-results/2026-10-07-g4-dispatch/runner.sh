#!/bin/bash
export SRC_URL='<presigned, redacted: carried an access key id in its query string>'
export PUT_DEC='<presigned, redacted: carried an access key id in its query string>'
export PUT_CAP='<presigned, redacted: carried an access key id in its query string>'
export PUT_LOG='<presigned, redacted: carried an access key id in its query string>'
export READER_URL='<presigned, redacted: carried an access key id in its query string>'
export REV='1955b55ba16f9a2cd90655be3ac7323f3efa452a'
export MODEL_URL='https://huggingface.co/ibm-granite/granite-4.1-3b-GGUF/resolve/ab4701481089b58a082ef63cc1cee738887293ff/granite-4.1-3b-Q4_K_M.gguf'
export MODEL_SHA='662b0626cd58f443baea23559b469df6576a81d349649c59413b36a9fb32eb29'
# G4 dispatch measurement. Follows benchmark-results/2026-09-18-gpu-large-model/REPRODUCE-RUST-PTX.md
# in its documented order: ptxas, capability, Q6_K device parity, then decode. Each is a gate on the
# next, because the recipe records a GPU host sitting idle at $1.09/hr while undocumented commands
# were discovered by hand.
set -u
LOG=/work/worker.log
mkdir -p /work
exec > >(tee -a "$LOG") 2>&1
say(){ printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
STATUS=BOOT
finish(){
  say "FINAL_STATUS=$STATUS"
  curl -s -X PUT --upload-file "$LOG" "$PUT_LOG" >/dev/null 2>&1 && say "log uploaded"
}
trap finish EXIT

say "nvidia-smi:"; nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader || { STATUS=NO_GPU; exit 1; }
command -v ptxas >/dev/null || { say "ptxas missing from image"; STATUS=NO_PTXAS; exit 1; }

say "installing jdk 25"
curl -sfL "https://api.adoptium.net/v3/binary/latest/25/ga/linux/x64/jdk/hotspot/normal/eclipse" -o /tmp/jdk.tgz || { STATUS=NO_JDK; exit 1; }
mkdir -p /opt/jdk && tar xzf /tmp/jdk.tgz -C /opt/jdk --strip-components=1
export JAVA_HOME=/opt/jdk PATH=/opt/jdk/bin:$PATH
say "jdk: $(java -version 2>&1 | head -1)"

say "installing pinned rust nightly-2026-09-17"
curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none --profile minimal >/dev/null 2>&1 || { STATUS=NO_RUSTUP; exit 1; }
export PATH=$HOME/.cargo/bin:$PATH
rustup toolchain install nightly-2026-09-17 --profile minimal --no-self-update >/dev/null 2>&1 || { STATUS=NO_TOOLCHAIN; exit 1; }
rustup component add --toolchain nightly-2026-09-17 clippy rustfmt rust-src llvm-bitcode-linker llvm-tools >/dev/null 2>&1
rustup target add --toolchain nightly-2026-09-17 nvptx64-nvidia-cuda >/dev/null 2>&1
say "rust: $(rustup run nightly-2026-09-17 rustc --version)"

say "fetching source"
curl -sfL "$SRC_URL" -o /work/src.tar.gz || { STATUS=NO_SRC; exit 1; }
tar xzf /work/src.tar.gz -C /work || { STATUS=NO_SRC_UNPACK; exit 1; }
cd /work/models || { STATUS=NO_SRC_DIR; exit 1; }
say "source at $(pwd), revision $REV"

say "=== STEP 1: ptxas assembly (cheap gate) ==="
# preparePtxResources, not compilePtx: compilePtx emits to build/rust-target/... and only
# preparePtxResources stages the module at the packaged resources path the recipe names. Asking for
# compilePtx alone cost one pod run that died on "no ptx emitted". Both gradle targets are in one
# invocation because a second JVM start is paid for in GPU time.
./gradlew :backend-cuda:preparePtxResources :models-bench:installDist --console=plain -q 2>&1 | tail -8
PTX=backend-cuda/build/generated/cuda-resources/META-INF/models/cuda/models-cuda-kernels.ptx
if [ ! -s "$PTX" ]; then
  say "not at the staged path, searching"
  PTX=$(find backend-cuda/build -name '*.ptx' -size +1k 2>/dev/null | head -1)
fi
[ -s "${PTX:-}" ] || { say "no ptx found anywhere under backend-cuda/build"; STATUS=NO_PTX; exit 1; }
say "ptx: $PTX ($(wc -c < "$PTX") bytes)"
ptxas -arch=sm_80 -O3 "$PTX" -o /dev/null || { STATUS=PTXAS_FAILED; exit 1; }
say "ptxas OK"

say "=== STEP 2: capability gate ==="
BIN=models-bench/build/install/models-bench/bin/models-bench
[ -x "$BIN" ] || { say "models-bench dist missing at $BIN"; STATUS=NO_DIST; exit 1; }
$BIN cuda-kernel-gate --mode capability --require-device true \
  --report /work/capability.json --models-revision "$REV" > /work/capability.out 2>&1
rc=$?
tail -20 /work/capability.out
curl -s -X PUT --upload-file /work/capability.json "$PUT_CAP" >/dev/null 2>&1 && say "capability report uploaded"
[ "$rc" = "0" ] || { say "capability gate rc=$rc"; STATUS=CAPABILITY_FAILED; exit 1; }

say "=== STEP 2a: Q6_K device parity (gate before any model) ==="
./gradlew :backend-cuda:test --tests '*CudaQ6KDeviceParityTest' --console=plain 2>&1 | tail -15
[ "${PIPESTATUS[0]}" = "0" ] || { STATUS=Q6K_DEVICE_PARITY_FAILED; exit 1; }
say "Q6_K device parity OK"

say "=== fetching model (granite 4.1 3b q4_k_m, 1.96GB) ==="
curl -sfL "$MODEL_URL" -o /work/model.gguf || { STATUS=NO_MODEL; exit 1; }
GOT=$(sha256sum /work/model.gguf | cut -d' ' -f1)
[ "$GOT" = "$MODEL_SHA" ] || { say "sha mismatch: $GOT != $MODEL_SHA"; STATUS=MODEL_SHA_MISMATCH; exit 1; }
say "model sha verified"

say "=== STEP 4: decode gate G4, --arm both ==="
$BIN cuda-kernel-gate --mode decode \
  --model /work/model.gguf \
  --prompts benchmark-results/2026-09-18-gpu-large-model/prompts.txt \
  --max-tokens 64 --warmup-tokens 16 --context 4096 \
  --arm both --require-device true \
  --report /work/decode.json --models-revision "$REV" > /work/decode.out 2>&1
rc=$?
tail -40 /work/decode.out
curl -s -X PUT --upload-file /work/decode.json "$PUT_DEC" >/dev/null 2>&1 && say "decode report uploaded"
say "decode gate rc=$rc"
[ -s /work/decode.json ] || { say "no decode report -- the gate rejected the run"; STATUS=DECODE_NO_REPORT; exit 1; }
say "=== the numbers this run exists for ==="
curl -sfL "$READER_URL" -o /work/read_report.py && python3 /work/read_report.py 2>&1 || say "reader unavailable; the report is in S3 regardless"
STATUS=COMPLETE
