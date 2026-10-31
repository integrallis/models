#!/bin/bash
set -euo pipefail
exec > >(tee -a /var/log/campaign-bootstrap.log /dev/console) 2>&1
shutdown -h +150
finish() { rc=$?; trap - EXIT; if command -v aws >/dev/null; then aws s3 cp /var/log/campaign-bootstrap.log 's3://models-qual-077051030817/results/qwen05-q5-attention-20261031T191743Z/shard-0/bootstrap.log' --region us-east-1 --only-show-errors || true; fi; shutdown -h now; exit "$rc"; }
trap finish EXIT
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq curl unzip jq python3 zstd
mkdir -p /work
curl -fL --retry 3 --max-time 300 https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /work/awscli.zip
unzip -q /work/awscli.zip -d /work
/work/aws/install
aws s3 cp 's3://models-qual-077051030817/payload/qwen05-q5-attention-20261031T191743Z/manifest-0.json' /work/manifest.json --region us-east-1 --only-show-errors
aws s3 cp 's3://models-qual-077051030817/payload/qwen05-q5-attention-20261031T191743Z/worker.sh' /work/worker.sh --region us-east-1 --only-show-errors
printf '%s  %s\n' 'd99fde326ceed35a1005c4ff834eb098d8b6437dbb643b213458ba3fc85a4cfb' /work/worker.sh | sha256sum -c -
timeout --signal=TERM --kill-after=30 8500 bash /work/worker.sh /work/manifest.json
