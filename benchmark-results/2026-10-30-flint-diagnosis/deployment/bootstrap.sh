#!/bin/bash
set -euo pipefail
exec > >(tee -a /var/log/flint-bootstrap.log /dev/console) 2>&1
# Independent spending bound, even if download, worker, or upload hangs.
shutdown -h +45
finish() {
  rc=$?
  trap - EXIT
  if command -v aws >/dev/null; then
    aws s3 cp /var/log/flint-bootstrap.log 's3://models-qual-077051030817/results/flint-diagnosis-20261030-f68555de/bootstrap.log' --region us-east-1 --only-show-errors || true
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
aws s3 cp 's3://models-qual-077051030817/payload/flint-diagnosis-20261030-f68555de/manifest.json' /work/manifest.json --region us-east-1 --only-show-errors
aws s3 cp 's3://models-qual-077051030817/payload/flint-diagnosis-20261030-f68555de/diagnosis-worker.sh' /work/diagnosis-worker.sh --region us-east-1 --only-show-errors
printf '%s  %s\n' 'dcdbc7cfc0f38bfff93cf4f55690f6ab3e0057acefbea1f861ca907e83f58697' /work/diagnosis-worker.sh | sha256sum -c -
aws s3 cp /var/log/flint-bootstrap.log 's3://models-qual-077051030817/results/flint-diagnosis-20261030-f68555de/bootstrap.log' --region us-east-1 --only-show-errors
# Login environment supplies the account's normal home directory to Ollama.
timeout --signal=TERM --kill-after=30 2400 sudo -i bash /work/diagnosis-worker.sh /work/manifest.json
