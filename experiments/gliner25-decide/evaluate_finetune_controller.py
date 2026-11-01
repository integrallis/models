"""Same-host CPU evaluation with a deadline inside the original aggregate rental cap."""
import argparse
import base64
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

import boto3

HERE=Path(__file__).resolve().parent
REPO=HERE.parent.parent
EVIDENCE=REPO/'benchmark-results/2026-11-01-gliner25-decide'
BUCKET='models-qual-077051030817'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--training-control',type=Path,required=True)
    args=parser.parse_args()
    training=json.loads(args.training_control.read_text())
    if training['state']!='deleted' or training.get('workerResult',{}).get('stage')!='complete':
        raise ValueError('training must be complete, uploaded and its pod deleted')
    previous=json.loads((EVIDENCE/'deployment.json').read_text())
    prior=2602
    for path in EVIDENCE.glob('gliner25-training-*/control.json'):
        record=json.loads(path.read_text())
        if 'podId' in record and record.get('state')!='deleted':raise ValueError('unresolved training rental')
        prior+=record.get('rentedSecondsUpperBound',0)
    for path in EVIDENCE.glob('gliner25-eval-*/control.json'):
        record=json.loads(path.read_text())
        if 'instanceId' in record and record.get('state')!='terminated':raise ValueError('unresolved evaluation rental')
        prior+=record.get('rentedSecondsUpperBound',0)
    seconds=min(2700,7200-prior-120)
    if seconds<1500:raise ValueError('insufficient remaining budget for complete CPU evaluation')
    ec2=boto3.client('ec2',region_name='us-east-1');s3=boto3.client('s3',region_name='us-east-1')
    quota=boto3.client('service-quotas',region_name='us-east-1').get_service_quota(ServiceCode='ec2',QuotaCode='L-1216C47A')['Quota']['Value']
    active=[i for r in ec2.describe_instances(Filters=[dict(Name='instance-state-name',Values=['pending','running','stopping','shutting-down'])])['Reservations'] for i in r['Instances']]
    types={i['InstanceType'] for i in active}|{'m6a.4xlarge'}
    cpus={t['InstanceType']:t['VCpuInfo']['DefaultVCpus'] for t in ec2.describe_instance_types(InstanceTypes=list(types))['InstanceTypes']}
    if sum(cpus[i['InstanceType']] for i in active)+16>quota:raise ValueError('insufficient verified vCPU headroom')
    if any(any(t['Key']=='campaign' and t['Value'].startswith('gliner25-') for t in i.get('Tags',[])) for i in active):
        raise ValueError('another GLiNER instance is not fully terminated')
    now=datetime.now(timezone.utc);run='gliner25-eval-'+now.strftime('%Y%m%dT%H%M%SZ')
    root=EVIDENCE/run;root.mkdir();work=REPO/'build/gliner-runpod'/run;work.mkdir(parents=True)
    prefix='payload/'+run;results='results/'+run
    overlay=work/'evaluation-overlay.tar.gz'
    with tarfile.open(overlay,'w:gz') as archive:
        for name in ['common.py','run_worker.py','reference_worker.py','compare_parity.py','assess_quality.py','assess_finetune.py',
                     'finetune-policy.json','finetune-splits.json','finetune-qualification-cases.jsonl','policy.json','quality-cases.jsonl','parity-cases.jsonl','performance-parity-cases.jsonl']:
            archive.add(HERE/name,arcname=name)
    sha=hashlib.file_digest(overlay.open('rb'),'sha256').hexdigest()
    s3.copy_object(Bucket=BUCKET,Key=prefix+'/payload.tar.gz',CopySource={'Bucket':BUCKET,'Key':previous['payloadUri'].split(BUCKET+'/')[1]+'/payload.tar.gz'})
    s3.upload_file(str(overlay),BUCKET,prefix+'/evaluation-overlay.tar.gz')
    s3.upload_file(str(HERE/'evaluate_finetune_remote.sh'),BUCKET,prefix+'/jobs/evaluate.sh')
    s3.put_object(Bucket=BUCKET,Key=prefix+'/next-job.json',Body=b'{"id":"evaluate"}')
    worker=(HERE/'remote-worker.sh').read_text().replace('shutdown -h +110','shutdown -h +'+str(seconds//60))
    worker=worker.replace('timeout --signal=TERM --kill-after=10 1800 bash','timeout --signal=TERM --kill-after=10 '+str(seconds-180)+' bash')
    s3.put_object(Bucket=BUCKET,Key=prefix+'/worker.sh',Body=worker.encode())
    body=json.loads((EVIDENCE/'restart-1-request.json').read_text())
    body.pop('UserData',None)
    body['ClientToken']=run
    body['TagSpecifications']=[dict(ResourceType='instance',Tags=[dict(Key='Name',Value=run),dict(Key='project',Value='models-qualification'),dict(Key='campaign',Value=run),dict(Key='delete-by',Value=(now+timedelta(seconds=seconds)).strftime('%Y%m%dT%H%M%SZ'))])]
    env=dict(GLINER_URI='s3://'+BUCKET+'/'+prefix,GLINER_RESULTS='s3://'+BUCKET+'/'+results,
             GLINER_PAYLOAD_SHA=previous['payloadSha256'],GLINER_EVAL_OVERLAY_SHA=sha,
             GLINER_TRAINED_RESULTS=training['resultUri'])
    userdata='#!/bin/bash\nset -euo pipefail\n'+'\n'.join('export '+k+'='+shlex.quote(v) for k,v in env.items())+'\n'
    userdata+="mkdir -p /work\napt-get update -qq\napt-get install -y -qq curl unzip\ncurl -fsSL --max-time 120 "+shlex.quote(s3.generate_presigned_url('get_object',Params={'Bucket':BUCKET,'Key':prefix+'/worker.sh'},ExpiresIn=5400))+" -o /work/worker.sh\nbash /work/worker.sh\n"
    # boto3 encodes UserData; unlike the CLI request it takes the plain script.
    body['UserData']=userdata
    record=dict(experiment=run,provider='EC2',region='us-east-1',plan='m6a.4xlarge',
        sourceRevision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        measuredRuntimeSource=previous['payloadManifest']['sourceRevision'],payloadSha256=previous['payloadSha256'],overlaySha256=sha,
        trainedWeightSha256=training['workerResult']['mergedWeightSha256'],priorAggregateRentedSecondsUpperBound=prior,
        maximumRentalSeconds=seconds,maximumAuthorizedSpendUsd=1.0,expectedMinutes=30,
        creationDeadline=(now+timedelta(minutes=5)).isoformat(),deletionDeadline=(now+timedelta(seconds=seconds)).isoformat(),
        resultUri='s3://'+BUCKET+'/'+results,correctnessGate='unchanged numerical parity; exact checkpoint and CPU thread budget',
        qualityGate='all frozen fine-tune and original regression floors',performanceGate='not measured in this quality run',
        stopCondition='evaluation complete, failure, setup timeout or deadline',secretsOnBox='instance-scoped IAM role; no account API credentials',catalogQualified=False)
    def save():(root/'control.json').write_text(json.dumps(record,indent=2)+'\n')
    save()
    print(json.dumps(dict(control=str(root/'control.json'),remainingAggregateSeconds=7200-prior,rentalLimitSeconds=seconds)),flush=True)
    launched=ec2.run_instances(**body)['Instances'][0]
    instance=launched['InstanceId'];record.update(instanceId=instance,launchTime=launched['LaunchTime'].isoformat(),state='running');save()
    print('created '+instance,flush=True)
    try:
        while datetime.now(timezone.utc)<datetime.fromisoformat(record['deletionDeadline']):
            keys=[o['Key'] for o in s3.list_objects_v2(Bucket=BUCKET,Prefix=results+'/').get('Contents',[])]
            for key in keys:
                s3.download_file(BUCKET,key,str(root/key.rsplit('/',1)[1]))
            if (root/'evaluate.done').exists():
                record['evaluationExitCode']=(root/'evaluate.done').read_text().strip();break
            state=ec2.describe_instances(InstanceIds=[instance])['Reservations'][0]['Instances'][0]['State']['Name']
            if state=='terminated':break
            if (datetime.now(timezone.utc)-now).total_seconds()>600 and not (root/'STATUS').exists():
                record['stopReason']='setup exceeded ten minutes';break
            time.sleep(20)
    finally:
        ec2.terminate_instances(InstanceIds=[instance])
        for _ in range(36):
            state=ec2.describe_instances(InstanceIds=[instance])['Reservations'][0]['Instances'][0]['State']['Name']
            if state=='terminated':break
            time.sleep(5)
        if state!='terminated':raise RuntimeError('instance termination unverified')
        # Recover the final uploader's objects after the worker exits.
        for obj in s3.list_objects_v2(Bucket=BUCKET,Prefix=results+'/').get('Contents',[]):
            s3.download_file(BUCKET,obj['Key'],str(root/obj['Key'].rsplit('/',1)[1]))
        ended=datetime.now(timezone.utc)
        record.update(state='terminated',deletionVerifiedAt=ended.isoformat(),rentedSecondsUpperBound=int((ended-now).total_seconds())+1)
        record['aggregateRentedSecondsUpperBound']=prior+record['rentedSecondsUpperBound']
        save();print(json.dumps(dict(state='terminated',instance=instance,aggregateRentedSecondsUpperBound=record['aggregateRentedSecondsUpperBound'])),flush=True)


if __name__=='__main__':main()
