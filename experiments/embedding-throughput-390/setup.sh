#!/usr/bin/env bash
set -euo pipefail
cd /opt/embedding390
mkdir -p results models references java profiler
uname -a > results/uname.txt
lscpu --json > results/lscpu.json
free -b > results/memory.txt
cat /proc/cpuinfo > results/cpuinfo.txt
if [[ -f /sys/fs/cgroup/cpu.max ]]; then
  cat /sys/fs/cgroup/cpu.max > results/cpu.max.txt
else
  printf "cpu.max absent (host cgroup layout)\n" > results/cpu.max.txt
fi
cat /proc/self/status > results/process-status.txt
curl --retry 3 --retry-max-time 120 --connect-timeout 20 -fsSL 'https://api.adoptium.net/v3/assets/latest/25/hotspot?architecture=x64&image_type=jdk&os=linux&vendor=eclipse' -o results/jdk-release.json
python3 - <<'PY'
import json,subprocess,hashlib,pathlib
asset=json.load(open('results/jdk-release.json'))[0]['binary']['package']
subprocess.run(['curl','--retry','3','--retry-max-time','180','--connect-timeout','20','-fsSL',asset['link'],'-o','java.tar.gz'],check=True)
assert hashlib.sha256(pathlib.Path('java.tar.gz').read_bytes()).hexdigest()==asset['checksum']
PY
tar -xzf java.tar.gz -C java --strip-components=1
java/bin/java -version > results/java-version.txt 2>&1
curl --retry 3 --retry-max-time 180 --connect-timeout 20 -fsSL 'https://github.com/async-profiler/async-profiler/releases/download/v4.6/async-profiler-4.6-linux-x64.tar.gz' -o profiler.tar.gz
tar -xzf profiler.tar.gz -C profiler --strip-components=1
curl --retry 3 --retry-max-time 120 --connect-timeout 20 -fsSL 'https://github.com/async-profiler/async-profiler/releases/download/v4.6/async-profiler.jar' -o profiler/lib/async-profiler.jar
sha256sum java.tar.gz profiler.tar.gz profiler/lib/async-profiler.jar > results/tooling-sha256.txt
python3 - <<'PY'
import hashlib,json,pathlib,subprocess
models=json.load(open('models.json'))
for m in models:
 p=pathlib.Path('models')/(m['name']+'.gguf')
 subprocess.run(['curl','--retry','3','--retry-max-time','180','--connect-timeout','20','-fsSL',m['url'],'-o',str(p)],check=True)
 h=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
 assert h==m['sha256'],(p,h)
 print('VERIFIED',p,flush=True)
PY
sysctl -w kernel.perf_event_paranoid=1 > results/perf-config.txt
python3 run.py
