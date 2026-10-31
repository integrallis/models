"""Launch one bounded, disposable native experiment from a frozen payload."""
import argparse, base64, datetime, hashlib, json, pathlib, subprocess
parser=argparse.ArgumentParser()
parser.add_argument('--payload-directory',type=pathlib.Path,required=True)
parser.add_argument('--launch-template',type=pathlib.Path,required=True)
a=parser.parse_args()
root=pathlib.Path(__file__).resolve().parents[2]
out=a.payload_directory.resolve()
assert not (out/'deployment.json').exists(), 'refusing to replace deployment ownership receipt'
payload=out/'payload.tar.gz'
worker=root/'scripts/fleet/qwen-native-worker.sh'
assert payload.is_file()
now=datetime.datetime.now(datetime.timezone.utc)
campaign='qwen05-q5-f16c-'+now.strftime('%Y%m%dT%H%M%SZ')
prefix='s3://models-qual-077051030817/payload/'+campaign
results='s3://models-qual-077051030817/results/'+campaign
aws=lambda *args:json.loads(subprocess.check_output(['aws',*args,'--region','us-east-1','--output','json']))
inventory=aws('ec2','describe-instances','--filters','Name=instance-state-name,Values=running,pending,stopping,shutting-down')
instances=[i for r in inventory['Reservations'] for i in r['Instances']]
quota=aws('service-quotas','get-service-quota','--service-code','ec2','--quota-code','L-1216C47A')['Quota']['Value']
types={i['InstanceType'] for i in instances}|{'m6a.4xlarge'}
cpu={t['InstanceType']:t['VCpuInfo']['DefaultVCpus'] for t in aws('ec2','describe-instance-types','--instance-types',*sorted(types))['InstanceTypes']}
assert sum(cpu[i['InstanceType']] for i in instances)+16<=quota
assert len(instances)+1<=6
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
receipt=dict(campaign=campaign,resultUri=results,instances=[],hourlyUSD=.6912,maxHours=2.5,maxSpendUSD=3,
             stopCondition='worker exit or hard shutdown at150minutes',inventory=inventory,quota=quota,
             creationDeadline=(now+datetime.timedelta(minutes=10)).isoformat(),deletionDeadline=(now+datetime.timedelta(minutes=155)).isoformat())
(out/'deployment.json').write_text(json.dumps(receipt,indent=2)+'\n')
for path,name in [(worker,'worker.sh'),(payload,'payload.tar.gz')]:subprocess.run(['aws','s3','cp',str(path),prefix+'/'+name,'--only-show-errors'],check=True)
old=json.loads(a.launch_template.read_text())
assert old['InstanceInitiatedShutdownBehavior']=='terminate'
assert old['InstanceType']=='m6a.4xlarge'
assert old['MinCount']==old['MaxCount']==1
assert all(b['Ebs']['DeleteOnTermination'] for b in old['BlockDeviceMappings'])
for shard in range(1):
 d=out/f'deployment-{shard}';d.mkdir()
 manifest=dict(shard=shard,payloadUri=prefix+'/payload.tar.gz',payloadSha256=sha(payload),resultsUri=results+f'/shard-{shard}')
 (d/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 subprocess.run(['aws','s3','cp',str(d/'manifest.json'),prefix+f'/manifest-{shard}.json','--only-show-errors'],check=True)
 bootstrap=f'''#!/bin/bash
set -euo pipefail
exec > >(tee -a /var/log/campaign-bootstrap.log /dev/console) 2>&1
shutdown -h +150
finish() {{ rc=$?; trap - EXIT; if command -v aws >/dev/null; then aws s3 cp /var/log/campaign-bootstrap.log '{results}/shard-{shard}/bootstrap.log' --region us-east-1 --only-show-errors || true; fi; shutdown -h now; exit "$rc"; }}
trap finish EXIT
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq curl unzip jq python3 zstd
mkdir -p /work
curl -fL --retry 3 --max-time 300 https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /work/awscli.zip
unzip -q /work/awscli.zip -d /work
/work/aws/install
aws s3 cp '{prefix}/manifest-{shard}.json' /work/manifest.json --region us-east-1 --only-show-errors
aws s3 cp '{prefix}/worker.sh' /work/worker.sh --region us-east-1 --only-show-errors
printf '%s  %s\\n' '{sha(worker)}' /work/worker.sh | sha256sum -c -
timeout --signal=TERM --kill-after=30 8500 bash /work/worker.sh /work/manifest.json
'''
 (d/'bootstrap.sh').write_text(bootstrap)
 request=json.loads(json.dumps(old));request.update(ClientToken=campaign+f'-{shard}',UserData=base64.b64encode(bootstrap.encode()).decode())
 for tag in request['TagSpecifications'][0]['Tags']:
  if tag['Key'] in ['Name','campaign']:tag['Value']=campaign+f'-{shard}'
 (d/'launch-request.json').write_text(json.dumps(request,indent=2)+'\n')
 response=aws('ec2','run-instances','--cli-input-json','file://'+str(d/'launch-request.json'))
 (d/'launch-response.json').write_text(json.dumps(response,indent=2)+'\n')
 iid=response['Instances'][0]['InstanceId'];receipt['instances'].append(dict(id=iid,shard=shard));(out/'deployment.json').write_text(json.dumps(receipt,indent=2)+'\n');print(iid,flush=True)
print(campaign)
