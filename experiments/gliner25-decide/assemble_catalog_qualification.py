"""Final admission requires complete same-release quality and three reconstructed performance pairs."""
import argparse
import hashlib
import json
from pathlib import Path
from common import HERE, read_cases
from run_performance_pairs import summarize_pairs


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source,'sha256').hexdigest()


def assemble(quality_path, performance):
    quality = json.loads(quality_path.read_text())
    policy_path = HERE/'catalog-policy-v2.json'
    policy = json.loads(policy_path.read_text())
    if quality['policySha256'] != sha(policy_path) or not quality['fidelityPassed']:
        raise ValueError('complete approved quality evidence required')
    if not quality['numericalParityPassed'] or quality['inferenceErrors'] != 0:
        raise ValueError('quality or parity gate failed')
    roots = {Path(item['rawFile']).parent for item in quality['evidence'].values()}
    for evidence in quality['evidence'].values():
        for kind in ['raw','score']:
            if sha(Path(evidence[kind+'File'])) != evidence[kind+'Sha256']:
                raise ValueError('quality evidence changed')
    if len(roots) != 1:
        raise ValueError('final released quality must be one complete controlled repeat')
    quality_root = roots.pop()
    controls = [json.loads((root/'control.json').read_text()) for root in [quality_root,performance]]
    receipts = [json.loads((root/'released-runtime.json').read_text()) for root in [quality_root,performance]]
    if receipts[0] != receipts[1] or receipts[0]['kind'] != 'central-release':
        raise ValueError('quality and performance must use identical released artifacts')
    if controls[0]['payloadSha256'] != controls[1]['payloadSha256']:
        raise ValueError('released payload identity differs')
    if controls[0]['mode'] != 'publisher-complete-release-audit' or controls[1]['mode'] != 'paired-performance':
        raise ValueError('wrong final experiment modes')
    for root,control in zip([quality_root,performance],controls):
        if control['runtimeKind'] != 'central-release' or control['state'] != 'terminated' or control['evaluationExitCode'] != '0':
            raise ValueError('released run did not finish cleanly')
        cleanup = json.loads((root/'cleanup-verification.json').read_text())
        if cleanup['instanceId'] != control['instanceId'] or cleanup['instanceState'] != 'terminated' or not cleanup['volumeAbsent']:
            raise ValueError('final rental cleanup unverified')
        smoke = json.loads((root/'released-default-discovery.json').read_text())
        if not smoke['defaultNativeDiscovery'] or smoke['selected'] != ['positive']:
            raise ValueError('released default native discovery did not pass')
    records = json.loads((performance/'performance-run-manifest.json').read_text())
    measured = summarize_pairs(read_cases(HERE/'performance-cases.jsonl'),performance,records)
    if not measured['replicationGatePassed']:
        raise ValueError('three complete passing performance replications required')
    rows = []
    for index,pair in enumerate(measured['pairResults'],1):
        candidate = pair['candidateRuntime']
        if candidate['nativeSha256'] != receipts[0]['nativeSha256'] or candidate['snapshot'] != quality['snapshot']:
            raise ValueError('performance kernel or checkpoint differs from released quality')
        for shape,bucket in pair['buckets'].items():
            rows.append({'pair':index,'shape':shape,'samples':bucket['candidate']['samples'],
                'throughputRatio':bucket['throughputRatio'],'p95LatencyRatio':bucket['p95LatencyRatio']})
    for item in quality['evidence'].values():
        if '.models' not in Path(item['rawFile']).stem and not Path(item['rawFile']).stem.endswith('-models'):
            continue
        with Path(item['rawFile']).open() as stream: ready=json.loads(stream.readline())
        if ready['nativeSha256'] != receipts[0]['nativeSha256']:
            raise ValueError('quality native artifact differs')
    return {'policy':policy['policy'],'modelId':quality['snapshot']['id'],'backend':'rust-ffm',
        'backendVersion':'models-'+receipts[0]['version'],'quality':quality['quality'],
        'oracleQuality':quality['oracleQuality'],'maximumOracleQualityLoss':quality['maximumOracleQualityLoss'],
        'cohortCases':quality['cohortCases'],'inferenceErrors':0,'numericalParityPassed':True,
        'performance':rows,'releasedRuntimeVerified':True,'qualified':True,
        'snapshot':quality['snapshot'],'runtimeReceipt':receipts[0],
        'evidence':{'quality':{'path':str(quality_path),'sha256':sha(quality_path)},
            'performanceManifest':{'path':str(performance/'performance-run-manifest.json'),
                'sha256':sha(performance/'performance-run-manifest.json')}},
        'limitations':quality['limitations']+["Performance applies to the recorded Linux x86-64 CPU profile with eight threads per arm."]}

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    for name in ['quality','performance','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();result=assemble(args.quality,args.performance)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'qualified':result['qualified'],'backendVersion':result['backendVersion']}))
