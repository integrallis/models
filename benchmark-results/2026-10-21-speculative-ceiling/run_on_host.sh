#!/usr/bin/env bash
# Run the speculative verification sweep on a host that is already provisioned and quiet.
#
# The host runs a JDK and nothing else. The benchmark jar is built here and shipped, because
# building it there drags in backend-cuda, whose jar compiles our Rust PTX kernels with a pinned
# nightly and the nvptx64 target -- a large install for a CPU benchmark that never touches CUDA,
# and a compiler running next to the thing being timed.
#
# Takes no opinion about where the host came from: the caller provisions it (see README.md) and
# passes an ssh target, so one copy of the teardown logic exists in the world rather than two.
#
# Usage: ./run_on_host.sh root@HOST [known_hosts_file]
set -uo pipefail

TARGET="${1:?usage: run_on_host.sh root@HOST [known_hosts]}"
KNOWN="${2:-/dev/null}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
SSH=(ssh -o UserKnownHostsFile="$KNOWN" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15
     -o ServerAliveInterval=30 "$TARGET")
SCP=(scp -q -o UserKnownHostsFile="$KNOWN" -o StrictHostKeyChecking=accept-new)

say() { printf '\n== %s\n' "$*"; }

JAR="$(ls -t "$REPO"/models-bench/build/libs/*-jmh.jar 2>/dev/null | head -1)"
[ -n "$JAR" ] || { echo "no jmh jar: run ./gradlew :models-bench:jmhJar first" >&2; exit 1; }
echo "benchmark jar: $JAR"
shasum -a 256 "$JAR" | tee "$HERE/jmh-jar-digest.txt"

say "host baseline"
"${SSH[@]}" 'lscpu | grep -E "^Model name|^CPU\(s\)|^Thread|^Core"; free -h | head -2; uptime;
  /opt/java/bin/java -version 2>&1 | head -1' | tee "$HERE/host-baseline.txt"

say "benchmark jar"
"${SSH[@]}" 'mkdir -p ~/bench'
"${SCP[@]}" "$JAR" "$TARGET:~/bench/benchmarks.jar"

say "models"
"${SSH[@]}" 'set -e; mkdir -p ~/gguf; cd ~/gguf
  [ -f Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf ] || curl -fsSL --retry 3 -o Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
    "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/bf5b95e96dac0462e2a09145ec66cae9a3f12067/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
  [ -f Llama-3.2-1B-Instruct-Q4_K_M.gguf ] || curl -fsSL --retry 3 -o Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
  sha256sum *.gguf' | tee "$HERE/model-digests.txt"

PARAMS="${PARAMS:-1,2,4,8,16,24,32}"
for model in Meta-Llama-3.1-8B-Instruct-Q4_K_M Llama-3.2-1B-Instruct-Q4_K_M; do
  for context in 256 2048; do
    tag="$model-ctx$context"
    say "$tag  (draftLength=$PARAMS)"
    "${SSH[@]}" "cd ~/bench && /opt/java/bin/java \
        --add-modules=jdk.incubator.vector -Xms2g -Xmx8g -XX:+UseG1GC \
        -jar benchmarks.jar SpeculativeVerificationBenchmark \
        -p draftLength=$PARAMS -f 1 -wi 2 -i 5 -r 1 -w 1 \
        -rf json -rff ~/bench/results-$tag.json \
        -jvmArgsAppend '--add-modules=jdk.incubator.vector -Dmodels.bench.model=/root/gguf/$model.gguf -Dmodels.purejava.maxContextLength=$context'" \
      > "$HERE/log-$tag.txt" 2>&1 \
      || { echo "  FAILED: $tag"; tail -12 "$HERE/log-$tag.txt"; }
    "${SCP[@]}" "$TARGET:~/bench/results-$tag.json" "$HERE/results-$tag.json" 2>/dev/null \
      && echo "  pulled results-$tag.json" || echo "  no results"
  done
done
say "done"
