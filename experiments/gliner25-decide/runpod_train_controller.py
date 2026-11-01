"""Local RunPod lifecycle controller; keeps account credentials off the training pod."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import time
import urllib.error
import urllib.request

import boto3
from rental_budget import prior_rented_seconds, aggregate_limit_seconds
from botocore.config import Config
from botocore.exceptions import ClientError

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BUCKET = 'models-qual-077051030817'


def digest(path):
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()


def read_key(path):
    for line in path.read_text().splitlines():
        line=line.strip().removeprefix('export ')
        if line.startswith('RUNPOD_API_KEY='):
            values=shlex.split(line.split('=',1)[1],comments=True)
            if len(values)==1:return values[0]
    raise ValueError('RUNPOD_API_KEY is missing')


def api(key, method, suffix, body=None):
    request=urllib.request.Request('https://api.runpod.io/v2/'+suffix,
        data=json.dumps(body).encode() if body is not None else None, method=method,
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(request,timeout=45) as response:
            data=response.read()
            return json.loads(data) if data else None
    except urllib.error.HTTPError as exc:
        # API errors can echo request fields. Never print the body or signed URLs.
        raise RuntimeError('RunPod '+method+' '+suffix.split('?')[0]+' HTTP '+str(exc.code)) from None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--env-file',type=Path,required=True)
    parser.add_argument('--publisher-bundle',type=Path,required=True)
    parser.add_argument('--gpu',default='NVIDIA A100-SXM4-80GB')
    parser.add_argument('--launch',action='store_true')
    parser.add_argument('--recipe',type=Path,default=HERE)
    parser.add_argument('--data',type=Path,default=REPO/'build/gliner-finetune/data')
    parser.add_argument('--maximum-rental-seconds',type=int,default=2100)
    args=parser.parse_args()
    key=read_key(args.env_file)
    s3=boto3.client('s3',region_name='us-east-1',config=Config(signature_version='s3v4'))
    account=boto3.client('sts').get_caller_identity()['Account']
    if account!='077051030817':raise ValueError('unexpected AWS account')
    existing=api(key,'GET','pods')['pods']
    if any(p['name'].startswith('gliner25-training-') for p in existing):
        raise ValueError('existing GLiNER training pod must be reconciled before a new rental')
    prior_seconds=prior_rented_seconds(REPO/'benchmark-results/2026-11-01-gliner25-decide',reserve_active=True)
    aggregate_limit=aggregate_limit_seconds()
    if not 1800<=args.maximum_rental_seconds<=3300:raise ValueError('rental must be bounded between 1800 and 3300 seconds')
    rental_seconds=min(args.maximum_rental_seconds,aggregate_limit-prior_seconds-120)
    if rental_seconds<1800:raise ValueError('insufficient aggregate budget for another training attempt')
    policy=json.loads((args.recipe/'finetune-policy.json').read_text())
    splits=json.loads((args.recipe/'finetune-splits.json').read_text())
    for name in ['train.jsonl','development.jsonl']:
        if digest(args.data/name)!=splits['files'][name]['sha256']:
            raise ValueError('training data mismatch')
    now=datetime.now(timezone.utc)
    run='gliner25-training-'+now.strftime('%Y%m%dT%H%M%SZ')
    root=REPO/'benchmark-results/2026-11-01-gliner25-decide'/run
    root.mkdir()
    work=REPO/'build/gliner-runpod'/run
    work.mkdir(parents=True)
    prefix='training/'+run
    payload=work/'payload.tar.gz'
    names=['train_general.py','development_metrics.py','common.py','snapshot.json','finetune-policy.json','finetune-splits.json',
           'requirements-training.in','requirements-training.lock']
    with tarfile.open(payload,'w:gz') as archive:
        for name in names:
            source=args.recipe/name if name in ['finetune-policy.json','finetune-splits.json'] else HERE/name
            archive.add(source,arcname='experiment/'+name)
        for name in ['train.jsonl','development.jsonl','development-cases.jsonl']:
            archive.add(args.data/name,arcname='data/'+name)
        archive.add(args.publisher_bundle,arcname='publisher.bundle')
    def signed(operation,name):
        return s3.generate_presigned_url(operation,Params={'Bucket':BUCKET,'Key':prefix+'/'+name},ExpiresIn=5400)
    for local,name in [(payload,'payload.tar.gz'),(HERE/'runpod_train_worker.py','worker.py')]:
        s3.upload_file(str(local),BUCKET,prefix+'/'+name)
    output_names=['status.json','worker.log','snapshot.json','training-record.json','selected-adapter.tar.gz']
    output_names += ['snapshot/'+e['path'] for e in json.loads((HERE/'snapshot.json').read_text())['files']]
    manifest=dict(payloadUrl=signed('get_object','payload.tar.gz'),payloadSha256=digest(payload),
        workerUrl=signed('get_object','worker.py'),workerSha256=digest(HERE/'runpod_train_worker.py'),
        publisherCommit=policy['publisherCommit'],
        puts={name:signed('put_object','results/'+name) for name in output_names})
    s3.put_object(Bucket=BUCKET,Key=prefix+'/manifest.json',Body=json.dumps(manifest).encode())
    stock=api(key,'GET','catalog/gpus?include=AVAILABILITY&product=POD&cloud=SECURE&minCudaVersion=12.8')
    gpu=next(g for g in stock['gpus'] if g['id']==args.gpu)
    if gpu.get('availability') not in ['HIGH','MEDIUM','LOW'] or gpu['price']['secure']>3:
        raise ValueError('selected GPU unavailable or above hourly limit')
    # Readable control record exists before any provisioning mutation.
    record=dict(experiment=run,sourceRevision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        payloadSha256=digest(payload),workerSha256=manifest['workerSha256'],provider='RunPod',cloud='SECURE',gpu=gpu,
        expectedMinutes=25,maximumRentalSeconds=rental_seconds,maximumAuthorizedSpendUsd=2.50,
        creationDeadline=(now+timedelta(minutes=5)).isoformat(),deletionDeadline=(now+timedelta(seconds=rental_seconds)).isoformat(),
        priorAggregateRentedSecondsUpperBound=prior_seconds,maximumAggregateRentedSeconds=aggregate_limit,
        candidate=policy['candidate'],policySha256=digest(args.recipe/'finetune-policy.json'),
        correctnessGate='exact tensor inventory, finite F32 export, unchanged tokenizer/config',
        qualityGate='frozen finetune-policy; this training run alone cannot qualify',
        performanceGate='unchanged same-class CPU policy; GPU training is not a performance comparison',
        stopCondition='completion, failed job, setup timeout, or rental deadline',
        secretsOnPod='only expiring object-scoped S3 URLs; no provider API credentials',
        resultUri='s3://'+BUCKET+'/'+prefix+'/results',localEvidenceDestination=str(root),catalogQualified=False)
    def save(): (root/'control.json').write_text(json.dumps(record,indent=2)+'\n')
    save()
    print(json.dumps(dict(control=str(root/'control.json'),gpu=gpu['id'],hourlyUsd=gpu['price']['secure'],launch=args.launch)),flush=True)
    if not args.launch:return
    if datetime.now(timezone.utc)>datetime.fromisoformat(record['creationDeadline']):raise ValueError('creation deadline passed')
    bootstrap="""import os,json,urllib.request,pathlib,hashlib,subprocess
p=pathlib.Path('/work');p.mkdir(exist_ok=True)
b=urllib.request.urlopen(os.environ['GLINER_MANIFEST_URL'],timeout=60).read();(p/'manifest.json').write_bytes(b)
m=json.loads(b);w=urllib.request.urlopen(m['workerUrl'],timeout=60).read()
assert hashlib.sha256(w).hexdigest()==m['workerSha256']
(p/'worker.py').write_bytes(w)
raise SystemExit(subprocess.call(['python','-u',str(p/'worker.py'),str(p/'manifest.json')]))
"""
    body=dict(name=run,image='python:3.12-slim-bookworm',cloud='SECURE',disk=40,
        gpu=dict(id=args.gpu,count=1,minRamPerGpu=48,minVcpuCountPerGpu=8,minCudaVersion='12.8'),
        entrypoint=['python','-u','-c'],cmd=[bootstrap],ports=[],
        env=dict(GLINER_MANIFEST_URL=signed('get_object','manifest.json'),NVIDIA_VISIBLE_DEVICES='all',
                 NVIDIA_DRIVER_CAPABILITIES='compute,utility',DEBIAN_FRONTEND='noninteractive'))
    # A controller process supervises deletion independently of the tool conversation.
    try:
        pod=api(key,'POST','pods',body)
    except Exception:
        # Reconcile ambiguous create responses by the unique run name before retrying.
        matches=[p for p in api(key,'GET','pods')['pods'] if p['name']==run]
        if len(matches)!=1:raise
        pod=matches[0]
    pod_id=pod['id']
    started=time.time()
    record.update(podId=pod_id,createdAt=pod.get('createdAt'),costPerHour=pod.get('cost'),
                  dataCenterId=pod.get('dataCenterId'),state='provisioned')
    save()
    print('created pod '+pod_id,flush=True)
    try:
        if pod.get('cost',99)>3:raise RuntimeError('placed GPU exceeds hourly cap')
        previous=None
        while datetime.now(timezone.utc)<datetime.fromisoformat(record['deletionDeadline']):
            current=api(key,'GET','pods/'+pod_id)
            result=None
            for name in ['status.json','worker.log','training-record.json','snapshot.json']:
                try:
                    data=s3.get_object(Bucket=BUCKET,Key=prefix+'/results/'+name)['Body'].read()
                    (root/name).write_bytes(data)
                    if name=='status.json':result=json.loads(data)
                except ClientError as exc:
                    if exc.response['Error']['Code']!='NoSuchKey':raise
            state=(current['status'],result['stage'] if result else 'awaiting-worker')
            if state!=previous:print(json.dumps(dict(pod=pod_id,state=state,elapsedSeconds=round(time.time()-started))),flush=True)
            previous=state
            if result and result['stage'] in ['complete','failed']:
                record['workerResult']=result
                break
            if time.time()-started>600 and (not result or result['stage'] in ['installing','downloading-snapshot']):
                record['stopReason']='setup exceeded ten minutes'
                break
            time.sleep(20)
        else:record['stopReason']='rental deadline reached'
    finally:
        record['terminationRequestedAt']=datetime.now(timezone.utc).isoformat();save()
        for attempt in range(5):
            try:
                api(key,'DELETE','pods/'+pod_id)
                break
            except RuntimeError:
                # A deleted pod may already return 404. Verify through list, scoped to our id.
                if not any(p['id']==pod_id for p in api(key,'GET','pods')['pods']):break
                if attempt==4:raise
                time.sleep(5)
        remaining=api(key,'GET','pods')['pods']
        if any(p['id']==pod_id for p in remaining):raise RuntimeError('pod still exists after delete')
        ended=datetime.now(timezone.utc)
        record.update(state='deleted',deletionVerifiedAt=ended.isoformat(),
                      rentedSecondsUpperBound=int((ended-now).total_seconds())+1)
        record['aggregateRentedSecondsUpperBound']=prior_seconds+record['rentedSecondsUpperBound']
        save()
        print(json.dumps(dict(pod=pod_id,state='deleted',aggregateRentedSecondsUpperBound=record['aggregateRentedSecondsUpperBound'])),flush=True)


if __name__=='__main__':main()
