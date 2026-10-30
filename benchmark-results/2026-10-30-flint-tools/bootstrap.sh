#!/bin/bash
set -euo pipefail
exec > >(tee -a /var/log/flint-bootstrap.log /dev/console) 2>&1
# Independent spending bound, even if download, worker, or upload hangs.
shutdown -h +90
finish() {
  rc=$?
  trap - EXIT
  if command -v aws >/dev/null; then
    aws s3 cp /var/log/flint-bootstrap.log 's3://models-qual-077051030817/results/flint-tools-20261030-07fc6853/bootstrap.log' --region us-east-1 --only-show-errors || true
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
aws s3 cp 's3://models-qual-077051030817/payload/flint-tools-20261030-07fc6853/manifest.json' /work/manifest.json --region us-east-1 --only-show-errors
aws s3 cp 's3://models-qual-077051030817/payload/flint-tools-20261030-07fc6853/tool-grammar-worker.sh' /work/tool-grammar-worker.sh --region us-east-1 --only-show-errors
printf '%s  %s\n' '07a2c03155a772481ac2b77400e0e22adfc3b346354fe07fa0c4affb629f5910' /work/tool-grammar-worker.sh | sha256sum -c -
aws s3 cp /var/log/flint-bootstrap.log 's3://models-qual-077051030817/results/flint-tools-20261030-07fc6853/bootstrap.log' --region us-east-1 --only-show-errors
# The outer timeout leaves time for final uploads before the spending bound.
timeout --signal=TERM --kill-after=30 5100 sudo -i bash /work/tool-grammar-worker.sh /work/manifest.json
