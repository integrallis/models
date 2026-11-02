export HOME=/root
set -e
quiet() { for i in $(seq 1 120); do n=$(ps -eo args | grep -cE '[R]agBenchmarkCli|[l]lama-server|[l]lama-bench' || true); [ "$n" -eq 0 ] && { sleep 3; return 0; }; sleep 10; done; }
apt-get install -y -q cmake build-essential git >/dev/null 2>&1 || true
[ -d /root/llama.cpp ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp /root/llama.cpp
cd /root/llama.cpp
echo "   llama.cpp HEAD: $(git rev-parse --short HEAD)"
# ARM A: stock (repack active)
cmake -B b -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_NATIVE=ON >/dev/null 2>&1
cmake --build b -j16 --target llama-bench >/dev/null 2>&1
quiet
./b/bin/llama-bench -m /root/mathstral/mathstral-7B-v0.1-Q4_K_S.gguf -p 0 -n 128 -t 16 -r 2 2>/dev/null | tail -4 | sed 's/^/   A-repack  /'
