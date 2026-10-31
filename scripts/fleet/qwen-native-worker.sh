#!/usr/bin/env bash
# Disposable EC2 worker only. The launch manifest supplies an immutable payload and deadline.
set -euo pipefail
INPUT=${1:?manifest}
WORK=/work/released-pair
OUT=$WORK/results
mkdir -p "$OUT" "$WORK/jdk" "$WORK/models"
DEST=$(jq -er .resultsUri "$INPUT")
export AWS_DEFAULT_REGION=us-east-1
exec > >(tee -a "$OUT/worker.log" /dev/console) 2>&1
upload() { aws s3 cp "$OUT/" "$DEST/" --recursive --only-show-errors; }
cleanup() { rc=$?; trap - EXIT; printf '%s\n' "$rc" >"$OUT/worker.exit-code"; upload || true; shutdown -h now; exit "$rc"; }
trap cleanup EXIT
(while sleep 30; do upload || true; done) &
cp "$INPUT" "$OUT/manifest.json"
{ uname -a; lscpu; free -b; } >"$OUT/host.txt"
aws s3 cp "$(jq -er .payloadUri "$INPUT")" "$WORK/payload.tar.gz" --only-show-errors
printf '%s  %s\n' "$(jq -er .payloadSha256 "$INPUT")" "$WORK/payload.tar.gz" | sha256sum -c -
tar -xzf "$WORK/payload.tar.gz" -C "$WORK"
cp "$WORK/protocol.json" "$OUT/"
curl -fL --retry 3 --max-time 900 'https://github.com/adoptium/temurin25-binaries/releases/download/jdk-25.0.3%2B9/OpenJDK25U-jdk_x64_linux_hotspot_25.0.3_9.tar.gz' -o "$WORK/jdk.tar.gz"
printf '%s  %s\n' 69264a7a211bf5029830d07bc3370f879769d62ebc5b5488e90c9343a2da0e1f "$WORK/jdk.tar.gz" | sha256sum -c -
tar -xzf "$WORK/jdk.tar.gz" -C "$WORK/jdk" --strip-components=1
curl -fL --retry 3 --max-time 600 https://ollama.com/install.sh -o "$WORK/install-ollama.sh"
OLLAMA_VERSION=0.40.13 sh "$WORK/install-ollama.sh" >"$OUT/ollama-install.log" 2>&1
export OLLAMA_MODELS=/work/ollama
mkdir -p "$OLLAMA_MODELS"
runuser -u root -- ollama serve >"$OUT/ollama-server.log" 2>&1 &
for n in $(seq 1 60); do curl -fsS http://127.0.0.1:11434/api/version >"$OUT/ollama-version.json" && break; sleep 2; done
jq -e '.version == "0.40.13"' "$OUT/ollama-version.json"
curl -fL --retry 3 --max-time 1800 "$(jq -er .job.uri "$WORK/protocol.json")" -o "$WORK/model.gguf"
printf '%s  %s\n' "$(jq -er .job.sha256 "$WORK/protocol.json")" "$WORK/model.gguf" | sha256sum -c -
printf 'FROM %s\n' "$WORK/model.gguf" >"$OUT/Modelfile"
runuser -u root -- ollama create qualification -f "$OUT/Modelfile" >"$OUT/import.log" 2>&1
# Build both variants with one pinned compiler, retaining native symbols for diagnostics.
apt-get install -y -qq build-essential
curl -fL --retry 3 --max-time 600 https://static.rust-lang.org/dist/rust-1.96.0-x86_64-unknown-linux-gnu.tar.xz -o "$WORK/rust.tar.xz"
printf '%s  %s\n' "$(jq -er .rustSha256 "$WORK/protocol.json")" "$WORK/rust.tar.xz" | sha256sum -c -
tar -xJf "$WORK/rust.tar.xz" -C "$WORK"
"$WORK/rust-1.96.0-x86_64-unknown-linux-gnu/install.sh" --prefix="$WORK/rust" --without=rust-docs
export PATH="$WORK/rust/bin:$PATH"
rustc --version --verbose >"$OUT/compiler.txt"
export CARGO_PROFILE_RELEASE_DEBUG=1
export CARGO_PROFILE_RELEASE_STRIP=none
for variant in baseline candidate; do
  cargo test --locked --manifest-path "$WORK/$variant/Cargo.toml" --release >"$OUT/$variant-tests.log" 2>&1
  cargo build --locked --manifest-path "$WORK/$variant/Cargo.toml" --release >"$OUT/$variant-build.log" 2>&1
  nm -C "$WORK/$variant/target/release/libjmodels_kernels.so" >"$OUT/$variant-symbols.txt"
done
cp "$WORK/released/classpath.json" "$WORK/released/provenance.json" "$OUT/"
python3 -u "$WORK/run-qwen-native-experiment.py" --work "$WORK" --output "$OUT"
printf '%s\n' complete >"$OUT/shard.complete"
