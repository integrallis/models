#!/bin/bash
set -euo pipefail
exec > >(tee -a /var/log/flint-bootstrap.log /dev/console) 2>&1
# Independent spending bound, even if download, worker, or upload hangs.
shutdown -h +240
finish() {
  rc=$?
  trap - EXIT
  if command -v aws >/dev/null; then
    aws s3 cp /var/log/flint-bootstrap.log 's3://models-qual-077051030817/results/flint-improvement-20261030-23e4e083/bootstrap.log' --region us-east-1 --only-show-errors || true
  fi
  shutdown -h now
  exit "$rc"
}
trap finish EXIT
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq curl unzip jq git cmake build-essential libcurl4-openssl-dev zstd
mkdir -p /work
curl -fL --retry 3 --max-time 300 https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /work/awscli.zip
unzip -q /work/awscli.zip -d /work
/work/aws/install
aws --version
aws s3 cp 's3://models-qual-077051030817/payload/flint-improvement-20261030-23e4e083/manifest.json' /work/manifest.json --region us-east-1 --only-show-errors
aws s3 cp 's3://models-qual-077051030817/payload/flint-improvement-20261030-23e4e083/improvement-worker.sh' /work/improvement-worker.sh --region us-east-1 --only-show-errors
printf '%s  %s\n' '4ed8037dc5dc9ee3dc2a898a81a1f6757c6a815b553245ca6f478fcbdcc8a7ce' /work/improvement-worker.sh | sha256sum -c -
aws s3 cp /var/log/flint-bootstrap.log 's3://models-qual-077051030817/results/flint-improvement-20261030-23e4e083/bootstrap.log' --region us-east-1 --only-show-errors
# Login environment supplies the account's normal home directory to Ollama.
timeout --signal=TERM --kill-after=30 14100 sudo -i bash /work/improvement-worker.sh /work/manifest.json
