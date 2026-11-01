#!/usr/bin/env bash
# One disposable CPU worker. GLINER_URI and GLINER_PAYLOAD_SHA are set by the launcher.
set -euo pipefail
export AWS_DEFAULT_REGION=us-east-1 DEBIAN_FRONTEND=noninteractive
mkdir -p /work/out
exec > >(tee -a /work/out/bootstrap.log /dev/console) 2>&1
shutdown -h +110
finish() {
  rc=$?
  trap - EXIT
  if command -v aws >/dev/null; then
    aws s3 sync /work/out "$GLINER_URI/results" --only-show-errors || true
  fi
  shutdown -h now
  exit "$rc"
}
trap finish EXIT
apt-get update -qq
apt-get install -y -qq curl unzip git build-essential python3-venv pkg-config
curl -fL --retry 3 --max-time 300 https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /work/awscli.zip
unzip -q /work/awscli.zip -d /work
/work/aws/install
(while sleep 30; do aws s3 sync /work/out "$GLINER_URI/results" --only-show-errors || true; done) &
aws s3 cp "$GLINER_URI/payload.tar.gz" /work/payload.tar.gz --only-show-errors
printf '%s  %s\n' "$GLINER_PAYLOAD_SHA" /work/payload.tar.gz | sha256sum -c -
mkdir -p /work/payload /work/jdk /work/snapshot
tar -xzf /work/payload.tar.gz -C /work/payload
curl -fL --retry 3 --max-time 900 'https://github.com/adoptium/temurin25-binaries/releases/download/jdk-25.0.3%2B9/OpenJDK25U-jdk_x64_linux_hotspot_25.0.3_9.tar.gz' -o /work/jdk.tar.gz
printf '%s  %s\n' 69264a7a211bf5029830d07bc3370f879769d62ebc5b5488e90c9343a2da0e1f /work/jdk.tar.gz | sha256sum -c -
tar -xzf /work/jdk.tar.gz -C /work/jdk --strip-components=1
export PATH="/work/jdk/bin:/root/.cargo/bin:$PATH"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /work/rustup.sh
sh /work/rustup.sh -y --profile minimal --default-toolchain 1.96.0
cargo build --locked --release --manifest-path /work/payload/rust/Cargo.toml
python3 -m venv /work/python
/work/python/bin/pip install --require-hashes --extra-index-url https://download.pytorch.org/whl/cpu -r /work/payload/experiment/requirements.lock
git clone https://github.com/fastino-ai/GLiNER2.git /work/reference
git -C /work/reference checkout --detach ca4f68235b50fe4c3ecf3876851d748fe4b15c6c
/work/python/bin/python - <<'PY'
import json, urllib.request, pathlib, sys
sys.path.insert(0, '/work/payload/experiment')
from common import verify_snapshot
manifest=json.load(open('/work/payload/experiment/snapshot.json'))
for entry in manifest['files']:
    destination=pathlib.Path('/work/snapshot')/entry['path']
    destination.parent.mkdir(parents=True,exist_ok=True)
    url=manifest['sourceUri']+'/resolve/'+manifest['revision']+'/'+entry['path']
    with urllib.request.urlopen(url,timeout=300) as source, destination.open('wb') as output:
        while block:=source.read(1024*1024): output.write(block)
verify_snapshot('/work/snapshot')
PY
lscpu > /work/out/lscpu.txt
free -b > /work/out/memory.txt
java -version > /work/out/java-version.txt 2>&1
/work/python/bin/pip freeze > /work/out/python-freeze.txt
printf 'ready\n' > /work/out/STATUS
aws s3 sync /work/out "$GLINER_URI/results" --only-show-errors
while [[ ! -f /work/STOP ]]; do
  if aws s3 cp "$GLINER_URI/next-job.json" /work/next-job.json --only-show-errors 2>/dev/null; then
    job=$(python3 -c 'import json; print(json.load(open("/work/next-job.json"))["id"])')
    [[ "$job" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
    if [[ ! -f "/work/out/$job.done" ]]; then
      aws s3 cp "$GLINER_URI/jobs/$job.sh" /work/job.sh --only-show-errors
      set +e
      timeout --signal=TERM --kill-after=10 1800 bash /work/job.sh > "/work/out/$job.log" 2>&1
      rc=$?
      set -e
      printf '%s\n' "$rc" > "/work/out/$job.done"
      aws s3 sync /work/out "$GLINER_URI/results" --only-show-errors
    fi
  fi
  sleep 10
done
