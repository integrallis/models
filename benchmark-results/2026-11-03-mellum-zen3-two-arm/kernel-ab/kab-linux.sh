#!/bin/bash
# Linux port of the committed kernel-ab build: same harness, same assertions, no Accelerate/rpath.
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get install -y -qq cmake git clang libstdc++-13-dev >/dev/null 2>&1 || apt-get install -y -qq cmake git clang >/dev/null
cd /root
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp.git >/dev/null 2>&1
cd llama.cpp && git log -1 --format='llama.cpp %H %cs' | tee /root/llamacpp.rev
[ -d bkern ] || cmake -B bkern -S . -DCMAKE_BUILD_TYPE=Release \
  -DGGML_BLAS=OFF -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF >/dev/null
cmake --build bkern -j16 --target ggml-cpu ggml-base 2>&1 | tail -2
K=/root/work/rust/model-kernels/target/release
cd /root
clang -O2 -I/root/llama.cpp/ggml/include -o /root/kernel_ab /root/kernel_ab.c \
  /root/llama.cpp/bkern/ggml/src/libggml-cpu.a /root/llama.cpp/bkern/ggml/src/libggml-base.a \
  -L"$K" -ljmodels_kernels -lm -lstdc++ -lpthread -ldl
echo "### built"
export LD_LIBRARY_PATH=$K
for t in q4_K q5_K; do
  for rows in 512 4096 16384; do /root/kernel_ab $rows 12 $t | grep -E 'type=|BEST|agreement'; done
done
echo "### KAB DONE"
