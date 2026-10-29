#!/bin/bash
set -euo pipefail
exec > >(tee -a /var/log/flint-bootstrap.log /dev/console) 2>&1
# Independent spending bound, even if download, worker, or upload hangs.
shutdown -h +240
finish() {
  rc=$?
  trap - EXIT
  if command -v aws >/dev/null; then
    aws s3 cp /var/log/flint-bootstrap.log 's3://models-qual-077051030817/results/flint-20261029-7541b1a4/bootstrap.log' --region us-east-1 --only-show-errors || true
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
aws s3 cp 's3://models-qual-077051030817/payload/flint-20261029-7541b1a4/manifest.json' /work/manifest.json --region us-east-1 --only-show-errors
aws s3 cp 's3://models-qual-077051030817/payload/flint-20261029-7541b1a4/qualification-worker.sh' /work/qualification-worker.sh --region us-east-1 --only-show-errors
printf '%s  %s\n' '9732a0711445db49d3591c65a91f698e8625719f73c7553ab546431a45689ccd' /work/qualification-worker.sh | sha256sum -c -
aws s3 cp /var/log/flint-bootstrap.log 's3://models-qual-077051030817/results/flint-20261029-7541b1a4/bootstrap.log' --region us-east-1 --only-show-errors
# Login environment supplies the account's normal home directory to Ollama.
timeout --signal=TERM --kill-after=30 14100 sudo -i bash /work/qualification-worker.sh /work/manifest.json
