#!/bin/bash
# Head-to-head of ggml's per-row Q4_K/Q5_K x Q8_K vec_dot against our Rust kernel, in ONE process,
# on identical weights, SINGLE-THREADED and interleaved. ggml runs in the same process as the
# in-run control, so machine drift cancels. The harness asserts the two kernels agree before any
# timing is believed -- the first version of this experiment timed a no-op because ggml_cpu_init()
# had not been called, leaving the fp16->fp32 table zeroed so every scale read as 0.
set -e
LLAMA=${LLAMA:-$HOME/Code/llama.cpp}
KERNELS=${KERNELS:?point KERNELS at a model-kernels target/release holding libjmodels_kernels}
[ -d "$LLAMA/bkern" ] || cmake -B "$LLAMA/bkern" -S "$LLAMA" -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF
cmake --build "$LLAMA/bkern" -j8 --target ggml-cpu ggml-base
clang -O2 -I"$LLAMA/ggml/include" -o kernel_ab kernel_ab.c \
  "$LLAMA/bkern/ggml/src/libggml-cpu.a" "$LLAMA/bkern/ggml/src/libggml-base.a" \
  -L"$KERNELS" -ljmodels_kernels -lm -lc++ -framework Accelerate
install_name_tool -add_rpath "$KERNELS" kernel_ab 2>/dev/null || true
for t in q4_K q5_K; do
  for rows in 512 4096 16384; do ./kernel_ab $rows 12 $t | grep -E 'type=|BEST|agreement'; done
done
