#!/bin/bash
# Self-contained authoritative arm for the repack-tile A/B. Runs entirely as a pod entrypoint:
# no credentials are placed on the pod, nothing is fetched from our buckets, and every input is a
# public artifact. Results are printed to the pod log, which is read back with stream-pod-logs.
set -u
SHA=a5822222909b785f23ddc74ce3c8f85bd0e38562
MODEL_URL=https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf
PROMPTS="${PROMPTS:-1024 4096}"
REPS="${REPS:-3}"
ALTERNATIONS="${ALTERNATIONS:-3}"

echo "===== HOST ====="
lscpu | grep -E "^Model name|^CPU\(s\):|^Thread\(s\)|^Socket|^CPU max MHz" || true
printf 'avx512 flags: '; lscpu | grep -o 'avx512[a-z_0-9]*' | sort -u | tr '\n' ' '; echo
printf 'amx flags   : '; lscpu | grep -o 'amx[a-z_0-9]*' | sort -u | tr '\n' ' '; echo
printf 'avx2        : '; (lscpu | grep -qw avx2 && echo yes || echo no)
NPROC=$(nproc); echo "nproc: $NPROC"
echo

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq build-essential cmake git curl ca-certificates python3 >/dev/null 2>&1 \
  || { echo "FATAL: apt install failed"; sleep infinity; }

mkdir -p /w && cd /w || exit 1
git init -q lc && cd lc || exit 1
git remote add origin https://github.com/ggml-org/llama.cpp.git
git fetch -q --depth 1 origin "$SHA" || { echo "FATAL: fetch $SHA failed"; sleep infinity; }
git checkout -q FETCH_HEAD || { echo "FATAL: checkout failed"; sleep infinity; }
echo "llama.cpp at $(git rev-parse --short HEAD)"

# Static link so each arm is a single binary and the build tree can be deleted; the pod's
# container disk is small.
for arm in on off; do
  [ "$arm" = on ] && R=ON || R=OFF
  cmake -S . -B b -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_BLAS=OFF -DGGML_METAL=OFF \
    -DGGML_CPU_REPACK=$R -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF \
    > /tmp/cfg-$arm.log 2>&1 || { echo "FATAL: configure $arm"; tail -20 /tmp/cfg-$arm.log; sleep infinity; }
  grep -E "GGML_CPU_REPACK:BOOL" b/CMakeCache.txt
  cmake --build b --target llama-bench -j"$NPROC" > /tmp/bld-$arm.log 2>&1 \
    || { echo "FATAL: build $arm"; tail -25 /tmp/bld-$arm.log; sleep infinity; }
  cp b/bin/llama-bench /w/llama-bench-$arm
  rm -rf b
  echo "built $arm"
done

curl -sL -o /w/m.gguf "$MODEL_URL" || { echo "FATAL: model download"; sleep infinity; }
echo "model bytes: $(stat -c%s /w/m.gguf)"
echo "model sha256: $(sha256sum /w/m.gguf | cut -d' ' -f1)"
echo

for P in $PROMPTS; do
  echo "===== p$P, threads=$NPROC, reps=$REPS, interleaved x$ALTERNATIONS ====="
  for i in $(seq 1 "$ALTERNATIONS"); do
    for arm in on off; do
      v=$(/w/llama-bench-$arm -m /w/m.gguf -p "$P" -n 0 -r "$REPS" -t "$NPROC" -o json 2>/dev/null \
          | python3 -c "import json,sys;d=json.load(sys.stdin)[0];print('%.3f %.3f'%(d['avg_ts'],d['stddev_ts']))" 2>/dev/null)
      echo "RESULT p=$P pass=$i repack=$arm tok_s=$v"
    done
  done
done
echo "===== AB COMPLETE ====="
sleep infinity
