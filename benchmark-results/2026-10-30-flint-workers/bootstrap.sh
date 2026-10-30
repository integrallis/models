#!/bin/bash
set -euo pipefail
exec > >(tee -a /var/log/flint-bootstrap.log /dev/console) 2>&1
# Independent spending bound, even if download, worker, or upload hangs.
shutdown -h +90
finish() {
  rc=$?
  trap - EXIT
  if command -v aws >/dev/null; then
    aws s3 cp /var/log/flint-bootstrap.log 's3://models-qual-077051030817/results/flint-workers-20261030-6cce978f/bootstrap.log' --region us-east-1 --only-show-errors || true
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
aws s3 cp 's3://models-qual-077051030817/payload/flint-workers-20261030-6cce978f/manifest.json' /work/manifest.json --region us-east-1 --only-show-errors
aws s3 cp 's3://models-qual-077051030817/payload/flint-workers-20261030-6cce978f/improvement-worker.sh' /work/improvement-worker.sh --region us-east-1 --only-show-errors
printf '%s  %s\n' 'fd99ace4cea783573dcd79dced52303915c68b4ba7f495bd904ee82aad46af7b' /work/improvement-worker.sh | sha256sum -c -
aws s3 cp /var/log/flint-bootstrap.log 's3://models-qual-077051030817/results/flint-workers-20261030-6cce978f/bootstrap.log' --region us-east-1 --only-show-errors
# The outer timeout leaves time for final uploads before the spending bound.
timeout --signal=TERM --kill-after=30 5100 sudo -i bash /work/improvement-worker.sh /work/manifest.json
