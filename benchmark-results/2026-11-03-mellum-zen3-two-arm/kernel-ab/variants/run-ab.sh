#!/bin/bash
# Two builds of our Q4_K row dot -- baseline and mins-vectorized -- each linked into its own
# kernel_ab, run ALTERNATELY in one session on a quiet dedicated box. ggml runs in-process in both
# as the drift control, and the harness asserts output agreement before any timing is believed.
# Variants come from two complete lib.rs files, not a patch, so there is nothing to misapply.
set -e
export DEBIAN_FRONTEND=noninteractive
W=/root/work
cd /root
if [ ! -f /root/.setup_done ]; then
  apt-get update -qq
  apt-get install -y -qq build-essential cmake git clang >/dev/null
  curl -fsSL https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.96.0 --profile minimal >/dev/null
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git >/dev/null 2>&1
  cd /root/llama.cpp
  git fetch --depth 1 origin 2cbca389f515df455412dafa07e2b1359cc72a6d >/dev/null 2>&1
  git checkout -q 2cbca389f515df455412dafa07e2b1359cc72a6d
  cmake -B bkern -S . -DCMAKE_BUILD_TYPE=Release -DGGML_BLAS=OFF -DGGML_NATIVE=ON \
    -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF -DGGML_OPENMP=OFF >/dev/null
  cmake --build bkern -j8 --target ggml-cpu ggml-base >/dev/null 2>&1
  touch /root/.setup_done
fi
. /root/.cargo/env
git -C /root/llama.cpp log -1 --format='llama.cpp %H' > /root/llamacpp.rev
build_variant() {
  rm -rf /root/kern-$1 && cp -r $W/model-kernels /root/kern-$1
  cp $W/lib_$1.rs /root/kern-$1/src/lib.rs
  ( cd /root/kern-$1 && cargo build --release >/dev/null 2>&1 )
  clang -O2 -I/root/llama.cpp/ggml/include -o /root/kab-$1 $W/kernel_ab.c \
    /root/llama.cpp/bkern/ggml/src/libggml-cpu.a /root/llama.cpp/bkern/ggml/src/libggml-base.a \
    -L/root/kern-$1/target/release -ljmodels_kernels -lm -lstdc++ -lpthread -ldl \
    -Wl,-rpath,/root/kern-$1/target/release
  echo "### built $1 sha=$(sha256sum /root/kern-$1/target/release/libjmodels_kernels.so | cut -c1-16)"
}
build_variant base
build_variant mins
build_variant combo
build_variant chain
echo "### HOST:$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-) nproc=$(nproc) load=$(cut -d' ' -f1-3 /proc/loadavg)"
for rows in 16384 4096; do
  echo "#### rows=$rows"
  for pass in 1 2 3 4 5 6 7 8; do
    if [ $((pass % 2)) -eq 1 ]; then
      echo "base $(/root/kab-base $rows 12 q4_K 2>/dev/null | grep BEST)"
      echo "mins $(/root/kab-mins $rows 12 q4_K 2>/dev/null | grep BEST)"
      echo "combo $(/root/kab-combo $rows 12 q4_K 2>/dev/null | grep BEST)"
      echo "chain $(/root/kab-chain $rows 12 q4_K 2>/dev/null | grep BEST)"
    else
      echo "chain $(/root/kab-chain $rows 12 q4_K 2>/dev/null | grep BEST)"
      echo "combo $(/root/kab-combo $rows 12 q4_K 2>/dev/null | grep BEST)"
      echo "mins $(/root/kab-mins $rows 12 q4_K 2>/dev/null | grep BEST)"
      echo "base $(/root/kab-base $rows 12 q4_K 2>/dev/null | grep BEST)"
    fi
  done
  echo "agree-base $(/root/kab-base $rows 12 q4_K 2>/dev/null | grep agreement)"
  echo "agree-mins $(/root/kab-mins $rows 12 q4_K 2>/dev/null | grep agreement)"
  echo "agree-combo $(/root/kab-combo $rows 12 q4_K 2>/dev/null | grep agreement)"
  echo "agree-chain $(/root/kab-chain $rows 12 q4_K 2>/dev/null | grep agreement)"
done
echo "### AB DONE"
