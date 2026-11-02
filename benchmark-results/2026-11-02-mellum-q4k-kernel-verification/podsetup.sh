set -e
export HOME=/root
export DEBIAN_FRONTEND=noninteractive
echo "### host: $(nproc) vCPU of $(grep -c processor /proc/cpuinfo) on host"
grep -m1 'model name' /proc/cpuinfo | cut -d: -f2
echo -n "### isa: "; grep -oE 'avx2|avx512f' /proc/cpuinfo | sort -u | tr '\n' ' '; echo
apt-get update -q >/dev/null 2>&1 && apt-get install -y -q build-essential curl python3 >/dev/null 2>&1
echo "### jdk"
[ -x /root/jdk/bin/java ] || { curl -sL -o /root/jdk.tgz "https://api.adoptium.net/v3/binary/latest/25/ga/linux/x64/jdk/hotspot/normal/eclipse"; mkdir -p /root/jdk; tar xzf /root/jdk.tgz -C /root/jdk --strip-components=1; }
/root/jdk/bin/java -version 2>&1 | head -1
echo "### rust"
[ -x /root/.cargo/bin/cargo ] || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null 2>&1
. /root/.cargo/env && cargo --version
echo "### ollama"
command -v ollama >/dev/null || curl -fsSL https://ollama.com/install.sh | sh >/dev/null 2>&1
ollama --version 2>&1 | head -1
echo "### mellum gguf (7.5 GB)"
mkdir -p /root/mellum
[ -s /root/mellum/m.gguf ] || curl -sL --retry 3 -o /root/mellum/m.gguf \
  "https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Thinking-GGUF-Q4_K_M/resolve/main/Mellum2-12B-A2.5B-Thinking-Q4_K_M.gguf"
ls -la /root/mellum/m.gguf
sha256sum /root/mellum/m.gguf
echo "### SETUP DONE"
