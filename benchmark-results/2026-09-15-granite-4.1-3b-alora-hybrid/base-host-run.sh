#!/usr/bin/env bash
# Public base qualification of Granite 4.1 3B Q4_K_M through the controlled RAG harness on a
# fresh Ubuntu 24.04 x86-64 host. Java owns inference; llama.cpp and Ollama are native controls.
#
# Usage: MODELS_COMMIT=<sha> bash base-host-run.sh
set -euo pipefail
MODELS_COMMIT="${MODELS_COMMIT:?MODELS_COMMIT is required}"
RUN_ROOT=/opt/modeljars-runs/granite41-base-20260916
EVIDENCE="$RUN_ROOT/evidence"
MODELS="$RUN_ROOT/models"
STORE="$RUN_ROOT/store"
GGUF_URL='https://huggingface.co/ibm-granite/granite-4.1-3b-GGUF/resolve/ab4701481089b58a082ef63cc1cee738887293ff/granite-4.1-3b-Q4_K_M.gguf'
GGUF_SHA='662b0626cd58f443baea23559b469df6576a81d349649c59413b36a9fb32eb29'

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$EVIDENCE/host-run.log"; }

mkdir -p "$EVIDENCE" "$STORE"
{ hostname; uname -a; lscpu; free -h; df -h /; uptime; } > "$EVIDENCE/host-baseline.txt" 2>&1
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git curl jq zstd build-essential pkg-config python3 unzip > "$EVIDENCE/apt.log" 2>&1

# The control bootstrap requires JDK 21 and JDK 25 installations under /usr/lib/jvm.
mkdir -p /usr/lib/jvm
for feature in 21 25; do
  if ! ls -d /usr/lib/jvm/jdk-$feature* >/dev/null 2>&1; then
    curl -fsSL "https://api.adoptium.net/v3/binary/latest/$feature/ga/linux/x64/jdk/hotspot/normal/eclipse" -o "/tmp/jdk$feature.tar.gz"
    tar -xzf "/tmp/jdk$feature.tar.gz" -C /usr/lib/jvm
  fi
done
JDK25="$(ls -d /usr/lib/jvm/jdk-25* | head -1)"
export JAVA_HOME="$JDK25" PATH="$JDK25/bin:$PATH"
java -version 2>&1 | tee "$EVIDENCE/java-version.txt"

if ! command -v cargo >/dev/null; then
  curl -fsSL https://sh.rustup.rs -o /tmp/rustup-init.sh
  sh /tmp/rustup-init.sh -y --profile minimal > "$EVIDENCE/rustup.log" 2>&1
fi
export PATH="$HOME/.cargo/bin:$PATH"

if [ ! -d "$MODELS/.git" ]; then
  git clone --quiet https://github.com/integrallis/models.git "$MODELS"
fi
git -C "$MODELS" fetch --quiet origin "$MODELS_COMMIT"
git -C "$MODELS" switch --detach --quiet "$MODELS_COMMIT"
test -z "$(git -C "$MODELS" status --short)"
log "models checkout $(git -C "$MODELS" rev-parse HEAD)"

# Pinned native controls: Ollama v0.32.0 and llama.cpp b10012, hash-checked by the bootstrap.
bash "$MODELS/scripts/bootstrap-inference-bench-host.sh" > "$EVIDENCE/bootstrap-controls.log" 2>&1
tail -2 "$EVIDENCE/bootstrap-controls.log" | tee -a "$EVIDENCE/host-run.log"

( cd "$MODELS" && ./gradlew :backend-native:cargoBuildRelease --console=plain -q ) > "$EVIDENCE/build-native.log" 2>&1
ls -l "$MODELS/backend-native/build/rust-target/release/libjmodels_kernels.so" | tee -a "$EVIDENCE/host-run.log"

if [ ! -f "$STORE/granite-4.1-3b-Q4_K_M.gguf" ] || [ "$(sha256sum "$STORE/granite-4.1-3b-Q4_K_M.gguf" | cut -d' ' -f1)" != "$GGUF_SHA" ]; then
  for attempt in 1 2 3 4 5; do
    if curl -fsSL -o "$STORE/gguf.part" "$GGUF_URL"; then break; fi
    sleep $((attempt * 30))
  done
  if [ "$(sha256sum "$STORE/gguf.part" | cut -d' ' -f1)" != "$GGUF_SHA" ]; then log "GGUF HASH MISMATCH"; exit 2; fi
  mv "$STORE/gguf.part" "$STORE/granite-4.1-3b-Q4_K_M.gguf"
fi
log "gguf verified $GGUF_SHA"

export RAG_MODELS_BACKEND=rust-ffm
log "start controlled RAG qualification"
( cd "$MODELS" && bash scripts/run-controlled-rag-qualification.sh \
    "$STORE/granite-4.1-3b-Q4_K_M.gguf" ibm_granite_granite_4_1_3b_gguf_q4_k_m general granite "$EVIDENCE/rag" ) \
  > "$EVIDENCE/rag-harness.log" 2>&1 || log "NONZERO exit from the RAG harness"
log "done controlled RAG qualification: $(tail -1 "$EVIDENCE/rag-harness.log")"
ls "$EVIDENCE/rag" 2>/dev/null | tee -a "$EVIDENCE/host-run.log"
sha256sum "$EVIDENCE"/rag/*.json 2>/dev/null | tee "$EVIDENCE/SHA256SUMS"
