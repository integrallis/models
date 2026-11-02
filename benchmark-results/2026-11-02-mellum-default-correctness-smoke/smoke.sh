set -e
export HOME=/root
export DEBIAN_FRONTEND=noninteractive
grep -m1 'model name' /proc/cpuinfo | cut -d: -f2
echo -n "### isa: "; grep -oE 'avx2|avx512f' /proc/cpuinfo | sort -u | tr '\n' ' '; echo
apt-get update -q >/dev/null 2>&1; apt-get install -y -q build-essential curl jq python3 >/dev/null 2>&1
[ -x /root/jdk/bin/java ] || { curl -sL -o /root/jdk.tgz "https://api.adoptium.net/v3/binary/latest/25/ga/linux/x64/jdk/hotspot/normal/eclipse"; mkdir -p /root/jdk; tar xzf /root/jdk.tgz -C /root/jdk --strip-components=1; }
[ -x /root/.cargo/bin/cargo ] || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null 2>&1
. /root/.cargo/env
mkdir -p /root/bench && tar xzf /root/pay3.tgz -C /root/bench
echo "### building the native library from source (this is what the jar would bundle)"
cd /root/bench/crate && cargo build --release >/dev/null 2>&1
SO=/root/bench/crate/target/release/libjmodels_kernels.so
sha256sum $SO
echo "### mellum gguf"
mkdir -p /root/mellum
[ -s /root/mellum/m.gguf ] || curl -sL --retry 3 -o /root/mellum/m.gguf \
  "https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Thinking-GGUF-Q4_K_M/resolve/71a489e7b95efacf89feaaa6fe3b2995f3542409/Mellum2-12B-A2.5B-Thinking-Q4_K_M.gguf"
MODEL_SHA=$(sha256sum /root/mellum/m.gguf | awk '{print $1}')
echo "artifact sha256 $MODEL_SHA"
echo "### SETUP DONE"
