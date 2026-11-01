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
from rental_budget import prior_rented_seconds, aggregate_limit_seconds

HERE=Path(__file__).resolve().parent
REPO=HERE.parent.parent
EVIDENCE=REPO/'benchmark-results/2026-11-01-gliner25-decide'
BUCKET='models-qual-077051030817'


def main():
    parser=argparse.ArgumentParser()
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--training-control',type=Path)
    source.add_argument('--original-base',action='store_true')
    parser.add_argument('--dense-payload','--runtime-payload',dest='dense_payload',type=Path)
    parser.add_argument('--dense-manifest','--runtime-manifest',dest='dense_manifest',type=Path)
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--development-cases',type=Path)
    parser.add_argument('--publisher-audit',type=Path)
    parser.add_argument('--publisher-single-head',action='store_true')
    parser.add_argument('--overlap-control',type=Path)
    parser.add_argument('--performance',action='store_true')
    parser.add_argument('--quality-and-performance',action='store_true')
    parser.add_argument('--replications',type=int,choices=[1,3],default=1)
    parser.add_argument('--recipe',type=Path,default=HERE)
    args=parser.parse_args()
    if args.publisher_single_head and not args.publisher_audit:raise ValueError('single-head requires publisher audit')
    if args.overlap_control and not args.publisher_single_head:raise ValueError('bounded overlap only supports the single-head publisher audit')
    if args.original_base and not (args.performance or args.publisher_audit):raise ValueError('original-base mode requires performance or publisher audit')
    if args.publisher_audit and (not args.original_base or args.performance or args.profile or args.quality_and_performance or args.development_cases):raise ValueError('publisher audit requires original base in a separate mode')
    if args.development_cases and (args.profile or args.performance or args.quality_and_performance):raise ValueError('development is a separate mode')
    if args.replications!=1 and not (args.performance or args.quality_and_performance):raise ValueError('replications require performance mode')
    if bool(args.dense_payload)!=bool(args.dense_manifest):raise ValueError('dense payload and manifest are required together')
    dense=json.loads(args.dense_manifest.read_text()) if args.dense_manifest else None
    if args.quality_and_performance and (args.profile or args.performance):raise ValueError('choose one evaluation mode')
    if args.profile and (dense or args.performance):raise ValueError('profile the original runtime independently')
    if dense and hashlib.file_digest(args.dense_payload.open('rb'),'sha256').hexdigest()!=dense['sha256']:
        raise ValueError('dense payload identity mismatch')
    training=json.loads(args.training_control.read_text()) if args.training_control else None
    if training and (training['state']!='deleted' or training.get('workerResult',{}).get('stage')!='complete'):
        raise ValueError('training must be complete, uploaded and its pod deleted')
    previous=json.loads((EVIDENCE/'deployment.json').read_text())
    prior=prior_rented_seconds(EVIDENCE,reserve_active=True)
    aggregate_limit=aggregate_limit_seconds()
    seconds=min(2700 if args.publisher_single_head else 5400 if args.publisher_audit else 900 if args.profile else 2700,aggregate_limit-prior-120)
    if seconds<(600 if args.profile else 1500):raise ValueError('insufficient remaining budget for this CPU experiment')
    ec2=boto3.client('ec2',region_name='us-east-1');s3=boto3.client('s3',region_name='us-east-1')
    quota=boto3.client('service-quotas',region_name='us-east-1').get_service_quota(ServiceCode='ec2',QuotaCode='L-1216C47A')['Quota']['Value']
    active=[i for r in ec2.describe_instances(Filters=[dict(Name='instance-state-name',Values=['pending','running','stopping','shutting-down'])])['Reservations'] for i in r['Instances']]
    types={i['InstanceType'] for i in active}|{'m6a.4xlarge'}
    cpus={t['InstanceType']:t['VCpuInfo']['DefaultVCpus'] for t in ec2.describe_instance_types(InstanceTypes=list(types))['InstanceTypes']}
    if sum(cpus[i['InstanceType']] for i in active)+16>quota:raise ValueError('insufficient verified vCPU headroom')
    gliner_active=[i for i in active if any(t['Key']=='campaign' and t['Value'].startswith('gliner25-') for t in i.get('Tags',[]))]
    overlap=None
    if args.overlap_control:
        if args.overlap_control.resolve().parent.parent!=EVIDENCE.resolve():raise ValueError('overlap control must be retained campaign evidence')
        overlap=json.loads(args.overlap_control.read_text())
        if (overlap['mode']!='publisher-and-emotion-audit' or overlap['state']!='running'
                or datetime.fromisoformat(overlap['deletionDeadline'])<=datetime.now(timezone.utc)
                or {i['InstanceId'] for i in gliner_active}!={overlap['instanceId']}
                or any(i['State']['Name']!='running' for i in gliner_active)):
            raise ValueError('overlap must match the one active bounded full emotion audit')
    elif gliner_active:
        raise ValueError('another GLiNER instance is not fully terminated')
    now=datetime.now(timezone.utc);run='gliner25-eval-'+now.strftime('%Y%m%dT%H%M%SZ')
    root=EVIDENCE/run;root.mkdir();work=REPO/'build/gliner-runpod'/run;work.mkdir(parents=True)
    prefix='payload/'+run;results='results/'+run
    # Worker IAM permits GET under payload/, not the training archive prefix. Stage server-side.
    if training:
        trained_prefix=training['resultUri'].split(BUCKET+'/')[1]
        trained_manifest=json.loads(s3.get_object(Bucket=BUCKET,Key=trained_prefix+'/snapshot.json')['Body'].read())
        for name in ['snapshot.json']+['snapshot/'+entry['path'] for entry in trained_manifest['files']]:
            s3.copy_object(Bucket=BUCKET,Key=prefix+'/variant/'+name,CopySource={'Bucket':BUCKET,'Key':trained_prefix+'/'+name})
    else:
        trained_manifest=json.loads((HERE/'snapshot.json').read_text())
    overlay=work/'evaluation-overlay.tar.gz'
    with tarfile.open(overlay,'w:gz') as archive:
        if args.publisher_audit:
            manifest=json.loads((args.publisher_audit/'publisher-audit-manifest.json').read_text())
            expected_mode='publisher-one-head-per-call' if args.publisher_single_head else 'joint-document-heads'
            if manifest.get('executionMode','joint-document-heads')!=expected_mode:raise ValueError('publisher execution mode differs')
            for name,expected in manifest['files'].items():
                source=args.publisher_audit/name
                if hashlib.file_digest(source.open('rb'),'sha256').hexdigest()!=expected:raise ValueError('publisher audit input identity differs')
                archive.add(source,arcname=name)
            archive.add(args.publisher_audit/'publisher-audit-manifest.json',arcname='publisher-audit-manifest.json')
            archive.add(HERE/'report_publisher_audit.py',arcname='report_publisher_audit.py')
        if args.development_cases:
            cases=[json.loads(line) for line in args.development_cases.read_text().splitlines()]
            if not cases or any(c['cohort']!='goemotions' or c['sourceSplit']!='train' for c in cases):
                raise ValueError('development diagnostics require training-source emotion cases')
            archive.add(args.development_cases,arcname='development-diagnostic-cases.jsonl')
            archive.add(HERE/'development_diagnostics.py',arcname='development_diagnostics.py')
        for name in ['performance_remote.sh','run_performance_pairs.py','SummarizeJfr.java','common.py','run_worker.py','reference_worker.py','compare_parity.py','assess_quality.py','assess_finetune.py',
                     'finetune-policy.json','finetune-splits.json','finetune-qualification-cases.jsonl','policy.json','quality-cases.jsonl','parity-cases.jsonl','performance-parity-cases.jsonl','performance-cases.jsonl','performance-cohort.json','compare_performance.py','evaluate_finetune_remote.sh','download_reranker_fixture.sh']:
            source=args.recipe/name if name in ['finetune-policy.json','finetune-splits.json','finetune-qualification-cases.jsonl'] else HERE/name
            if args.original_base and name=='finetune-policy.json':
                policy=json.loads(source.read_text());policy['candidate']=trained_manifest['id']
                source=work/'original-base-policy.json';source.write_text(json.dumps(policy)+'\n')
            archive.add(source,arcname=name)
    sha=hashlib.file_digest(overlay.open('rb'),'sha256').hexdigest()
    if dense:
        s3.upload_file(str(args.dense_payload),BUCKET,prefix+'/payload.tar.gz')
    else:
        s3.copy_object(Bucket=BUCKET,Key=prefix+'/payload.tar.gz',CopySource={'Bucket':BUCKET,'Key':previous['payloadUri'].split(BUCKET+'/')[1]+'/payload.tar.gz'})
    s3.upload_file(str(overlay),BUCKET,prefix+'/evaluation-overlay.tar.gz')
    job='publisher_audit_remote.sh' if args.publisher_audit else 'development_remote.sh' if args.development_cases else 'quality_and_performance_remote.sh' if args.quality_and_performance else 'profile_remote.sh' if args.profile else 'performance_remote.sh' if args.performance else 'evaluate_finetune_remote.sh'
    s3.upload_file(str(HERE/job),BUCKET,prefix+'/jobs/evaluate.sh')
    s3.put_object(Bucket=BUCKET,Key=prefix+'/next-job.json',Body=b'{"id":"evaluate"}')
    worker=(HERE/'remote-worker.sh').read_text().replace('shutdown -h +110','shutdown -h +'+str(seconds//60))
    worker=worker.replace('timeout --signal=TERM --kill-after=10 1800 bash','timeout --signal=TERM --kill-after=10 '+str(seconds-180)+' bash')
    s3.put_object(Bucket=BUCKET,Key=prefix+'/worker.sh',Body=worker.encode())
    body=json.loads((EVIDENCE/'restart-1-request.json').read_text())
    body.pop('UserData',None)
    body['ClientToken']=run
    body['TagSpecifications']=[dict(ResourceType='instance',Tags=[dict(Key='Name',Value=run),dict(Key='project',Value='models-qualification'),dict(Key='campaign',Value=run),dict(Key='delete-by',Value=(now+timedelta(seconds=seconds)).strftime('%Y%m%dT%H%M%SZ'))])]
    env=dict(GLINER_URI='s3://'+BUCKET+'/'+prefix,GLINER_RESULTS='s3://'+BUCKET+'/'+results,
             GLINER_PAYLOAD_SHA=dense['sha256'] if dense else previous['payloadSha256'],GLINER_EVAL_OVERLAY_SHA=sha,
             GLINER_SINGLE_HEAD='1' if args.publisher_single_head else '0',GLINER_ORIGINAL_BASE='1' if args.original_base else '0',GLINER_REPLICATIONS=str(args.replications),GLINER_TRAINED_RESULTS='s3://'+BUCKET+'/'+prefix+'/variant')
    userdata='#!/bin/bash\nset -euo pipefail\n'+'\n'.join('export '+k+'='+shlex.quote(v) for k,v in env.items())+'\n'
    userdata+="mkdir -p /work\napt-get update -qq\napt-get install -y -qq curl unzip\ncurl -fsSL --max-time 120 "+shlex.quote(s3.generate_presigned_url('get_object',Params={'Bucket':BUCKET,'Key':prefix+'/worker.sh'},ExpiresIn=5400))+" -o /work/worker.sh\nbash /work/worker.sh\n"
    # boto3 encodes UserData; unlike the CLI request it takes the plain script.
    body['UserData']=userdata
    record=dict(experiment=run,provider='EC2',region='us-east-1',plan='m6a.4xlarge',
        sourceRevision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        runtimeKind=dense.get('runtimeKind','candidate-source') if dense else 'candidate-source',
        releasedVersion=dense.get('releasedVersion') if dense else None,
        performanceReplications=args.replications if args.performance or args.quality_and_performance else 0,
        mode='publisher-single-head-audit' if args.publisher_single_head else 'publisher-and-emotion-audit' if args.publisher_audit else 'development-diagnostics' if args.development_cases else 'quality-and-performance' if args.quality_and_performance else 'cpu-profile' if args.profile else 'paired-performance' if args.performance else 'fine-tune-quality',
        maximumAggregateRentedSeconds=aggregate_limit,
        measuredRuntimeSource=dense['sourceRevision'] if dense else previous['payloadManifest']['sourceRevision'],payloadSha256=env['GLINER_PAYLOAD_SHA'],overlaySha256=sha,
        trainedWeightSha256=next(e['sha256'] for e in trained_manifest['files'] if e['path']=='model.safetensors'),
        trainedSourceUri=training['resultUri'] if training else trained_manifest['sourceUri'],
        stagedTrainedUri=env['GLINER_TRAINED_RESULTS'],priorAggregateRentedSecondsUpperBound=prior,
        maximumRentalSeconds=seconds,maximumAuthorizedSpendUsd=2.0 if args.publisher_audit else 1.0,expectedMinutes=30 if args.publisher_single_head else 60 if args.publisher_audit else 30,
        creationDeadline=(now+timedelta(minutes=5)).isoformat(),deletionDeadline=(now+timedelta(seconds=seconds)).isoformat(),
        resultUri='s3://'+BUCKET+'/'+results,correctnessGate='unchanged numerical parity; exact checkpoint and CPU thread budget',
        qualityGate='publisher example recipe: one head per call, 2900 public development heads; no changed admission gates' if args.publisher_single_head else 'original publisher public development plus complete official GoEmotions test; no changed admission gates' if args.publisher_audit else 'development-only, not qualification' if args.development_cases else 'not measured in profiling or performance mode' if args.profile or args.performance else 'all frozen fine-tune and original regression floors',
        performanceGate='complete 13-shape pairs; every pair must pass, at least three for final qualification' if args.performance or args.quality_and_performance else 'CPU cost attribution only, not qualification' if args.profile else 'not measured in this quality run',
        stopCondition='evaluation complete, failure, setup timeout or deadline',secretsOnBox='instance-scoped IAM role; no account API credentials',catalogQualified=False)
    if overlap:record['overlappingRun']=dict(experiment=overlap['experiment'],instanceId=overlap['instanceId'],reservedMaximumSeconds=overlap['maximumRentalSeconds'],reason='Independent publisher recipe audit while full emotion audit continues; aggregate cap reserves both rentals')
    record['snapshotOrigin']='original-publisher' if args.original_base else 'fine-tuned'
    def save():(root/'control.json').write_text(json.dumps(record,indent=2)+'\n')
    save()
    print(json.dumps(dict(control=str(root/'control.json'),remainingAggregateSeconds=aggregate_limit-prior,rentalLimitSeconds=seconds)),flush=True)
    launched=ec2.run_instances(**body)['Instances'][0]
    instance=launched['InstanceId']
    volumes=[m['Ebs']['VolumeId'] for m in launched.get('BlockDeviceMappings',[]) if 'Ebs' in m]
    (root/'volume-ids.json').write_text(json.dumps(volumes,indent=2)+'\n')
    if args.development_cases:
        record['developmentCasesSha256']=hashlib.file_digest(args.development_cases.open('rb'),'sha256').hexdigest()
    record.update(instanceId=instance,launchTime=launched['LaunchTime'].isoformat(),state='running');save()
    print('created '+instance,flush=True)
    try:
        ec2.get_waiter('instance_exists').wait(InstanceIds=[instance],WaiterConfig={'Delay':5,'MaxAttempts':12})
        record['instanceVisibilityVerifiedAt']=datetime.now(timezone.utc).isoformat();save()
        while datetime.now(timezone.utc)<datetime.fromisoformat(record['deletionDeadline']):
            keys=[o['Key'] for o in s3.list_objects_v2(Bucket=BUCKET,Prefix=results+'/').get('Contents',[])]
            for key in keys:
                s3.download_file(BUCKET,key,str(root/key.rsplit('/',1)[1]))
            if (root/'evaluate.done').exists():
                record['evaluationExitCode']=(root/'evaluate.done').read_text().strip();break
            observed=ec2.describe_instances(InstanceIds=[instance])['Reservations'][0]['Instances'][0]
            state=observed['State']['Name']
            if not volumes:
                volumes=[m['Ebs']['VolumeId'] for m in observed.get('BlockDeviceMappings',[]) if 'Ebs' in m]
                (root/'volume-ids.json').write_text(json.dumps(volumes,indent=2)+'\n')
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
