#!/bin/bash
set -euo pipefail
exec > >(tee -a /var/log/flint-bootstrap.log /dev/console) 2>&1
# Independent bound applies during installation, download, inference and upload.
shutdown -h +90
finish() {
  rc=$?
  trap - EXIT
  if command -v aws >/dev/null; then
    aws s3 cp /var/log/flint-bootstrap.log 's3://models-qual-077051030817/results/flint-released-058-20261030-282b3960/bootstrap.log' --region us-east-1 --only-show-errors || true
  fi
  shutdown -h now
  exit "$rc"
}
trap finish EXIT
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq curl unzip jq python3
mkdir -p /work
curl -fL --retry 3 --max-time 300 https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /work/awscli.zip
unzip -q /work/awscli.zip -d /work
/work/aws/install
aws s3 cp 's3://models-qual-077051030817/payload/flint-released-058-20261030-282b3960/manifest.json' /work/manifest.json --region us-east-1 --only-show-errors
aws s3 cp 's3://models-qual-077051030817/payload/flint-released-058-20261030-282b3960/released-worker.sh' /work/released-worker.sh --region us-east-1 --only-show-errors
printf '%s  %s\n' '6884147349dbfe1d7dbd29487833e7969bd2fa927c57f68da2e8b923afc4d29e' /work/released-worker.sh | sha256sum -c -
timeout --signal=TERM --kill-after=30 5100 bash /work/released-worker.sh /work/manifest.json
