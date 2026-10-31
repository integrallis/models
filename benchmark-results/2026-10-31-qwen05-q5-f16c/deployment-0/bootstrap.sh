#!/bin/bash
set -euo pipefail
exec > >(tee -a /var/log/campaign-bootstrap.log /dev/console) 2>&1
shutdown -h +150
finish() { rc=$?; trap - EXIT; if command -v aws >/dev/null; then aws s3 cp /var/log/campaign-bootstrap.log 's3://models-qual-077051030817/results/qwen05-q5-f16c-20261031T190822Z/shard-0/bootstrap.log' --region us-east-1 --only-show-errors || true; fi; shutdown -h now; exit "$rc"; }
trap finish EXIT
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq curl unzip jq python3 zstd
mkdir -p /work
curl -fL --retry 3 --max-time 300 https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /work/awscli.zip
unzip -q /work/awscli.zip -d /work
/work/aws/install
aws s3 cp 's3://models-qual-077051030817/payload/qwen05-q5-f16c-20261031T190822Z/manifest-0.json' /work/manifest.json --region us-east-1 --only-show-errors
aws s3 cp 's3://models-qual-077051030817/payload/qwen05-q5-f16c-20261031T190822Z/worker.sh' /work/worker.sh --region us-east-1 --only-show-errors
printf '%s  %s\n' '9f224bdeec685f147822750a7d136b070995da446b19225eb1c2484938a9eb11' /work/worker.sh | sha256sum -c -
timeout --signal=TERM --kill-after=30 8500 bash /work/worker.sh /work/manifest.json
