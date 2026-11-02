export HOME=/root
set -e
quiet() { for i in $(seq 1 120); do n=$(ps -eo args | grep -cE '[R]agBenchmarkCli|[l]lama-server|[l]lama-bench' || true); [ "$n" -eq 0 ] && { sleep 3; return 0; }; sleep 10; done; }
cd /root/llama.cpp
grep -n "GGML_CPU_REPACK" ggml/CMakeLists.txt ggml/src/ggml-cpu/CMakeLists.txt 2>/dev/null | head
cmake -B bnr -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_NATIVE=ON -DGGML_CPU_REPACK=OFF >/dev/null 2>&1
cmake --build bnr -j16 --target llama-bench >/dev/null 2>&1
quiet
echo "=== ARM B: repack OFF ==="
./bnr/bin/llama-bench -m /root/mathstral/mathstral-7B-v0.1-Q4_K_S.gguf -p 0 -n 128 -t 16 -r 2 2>/dev/null | tail -3
quiet
echo "=== ARM A re-confirm: repack ON ==="
./b/bin/llama-bench -m /root/mathstral/mathstral-7B-v0.1-Q4_K_S.gguf -p 0 -n 128 -t 16 -r 2 2>/dev/null | tail -3
