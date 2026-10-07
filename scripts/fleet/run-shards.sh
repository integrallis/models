#!/bin/bash
# Work a list of shards through the fleet, keeping at most MAX_CONCURRENT boxes alive.
#
# The vCPU quota is the real constraint, and it is read, not assumed: it was 16, then 64 was
# pending, and as of 2026-10-06 it is 192. A hard-coded default sent one run hunting for RunPod
# capacity while 176 vCPUs sat free, so VCPU_QUOTA defaults to whatever the account reports and
# falls back to 64 only if the query fails. Counting only `running` under-counts, because an
# instance in `shutting-down` still holds its quota and the next RunInstances then fails with
# VcpuLimitExceeded -- so every state that holds capacity is counted.
#
# The account is shared. Other projects' boxes hold vCPUs too, which is why held_vcpus counts every
# instance rather than filtering by tag -- but never terminate one this script did not launch.
#
# A shard is finished when it uploads STATUS, not when its instance disappears: the instance
# terminates itself, and gating on instance state would race the upload.
set -u
S=$(cd "$(dirname "$0")" && pwd)
BOOTSTRAPS=${BOOTSTRAPS:-$S}
BUCKET=${BUCKET:-models-qual-077051030817}
VCPU_QUOTA=${VCPU_QUOTA:-$(aws service-quotas get-service-quota --service-code ec2 \
  --quota-code L-1216C47A --query "Quota.Value" --output text 2>/dev/null \
  | awk '{printf "%d", $1+0}')}
VCPU_QUOTA=${VCPU_QUOTA:-64}
VCPUS_PER_BOX=${VCPUS_PER_BOX:-16}
AMI=${AMI:-ami-0045d7fc2ad003464}
TYPE=${TYPE:-m6a.4xlarge}

# vCPUs held, not instances. The quota is 64 vCPU, and a mixed fleet makes an instance count
# meaningless: one m6a.8xlarge takes 32 where a 4xlarge takes 16, so counting boxes would either
# overshoot the quota or leave it idle. Every state that still holds capacity is counted, because an
# instance releases its vCPUs only when it is gone.
held_vcpus() {
  aws ec2 describe-instances \
    --filters Name=instance-state-name,Values=running,pending,shutting-down,stopping \
    --query 'Reservations[].Instances[].InstanceType' --output text 2>/dev/null \
    | tr '\t' '\n' \
    | awk '{ if ($0 ~ /8xlarge/) s+=32; else if ($0 ~ /16xlarge/) s+=64; else if ($0 != "") s+=16 } END { print s+0 }'
}

done_already() {
  [ "$(aws s3 ls "s3://$BUCKET/results/shard-$1/STATUS" 2>/dev/null | wc -l | tr -d ' ')" -gt 0 ]
}

volume_for() {
  local manifest="$BOOTSTRAPS/fleet-shard-$1.json"
  [ -f "$manifest" ] || { echo 120; return; }
  python3 - "$manifest" <<'PY'
import json,sys
jobs=json.load(open(sys.argv[1]))
biggest=max((j.get('gb') or 0) for j in jobs) if jobs else 0
# Peak disk is the payload plus the largest model TWICE: `ollama create` copies the GGUF.
print(max(120, int(biggest*2+40)))
PY
}

for shard in "$@"; do
  if done_already "$shard"; then
    echo "shard-$shard already reported STATUS; skipping"
    continue
  fi
  while [ $(( $(held_vcpus) + VCPUS_PER_BOX )) -gt "$VCPU_QUOTA" ]; do sleep 20; done
  boot="$BOOTSTRAPS/ud-boot-$shard.sh"
  [ -f "$boot" ] || { echo "shard-$shard: no bootstrap at $boot"; continue; }
  vol=$(volume_for "$shard")
  id=$(aws ec2 run-instances --image-id "$AMI" --instance-type "$TYPE" \
    --iam-instance-profile Name=models-qual-worker \
    --security-group-ids sg-0e9ff523da4140d55 --subnet-id subnet-08fcc910aba0d5b23 \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$vol,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
    --instance-initiated-shutdown-behavior terminate \
    --user-data "file://$boot" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=qual-shard-$shard}]" \
    --query 'Instances[0].InstanceId' --output text) || { echo "shard-$shard launch failed"; continue; }
  echo "shard-$shard launched $id vol=${vol}GB at $(date -u +%H:%M:%S)Z"
  sleep 15
done

echo "all shards dispatched; waiting for the last of them"
for shard in "$@"; do
  for _ in $(seq 1 400); do
    done_already "$shard" && break
    sleep 30
  done
  echo "shard-$shard done at $(date -u +%H:%M:%S)Z"
done
