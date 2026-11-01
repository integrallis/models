"""Complete counterbalanced CPU pairs; raw outputs remain the source of every metric."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

from common import HERE, read_cases
from compare_performance import compare_pair


def summarize_pairs(cases, outputs, records):
    policy=json.loads((HERE/'policy.json').read_text())['performance']
    pairs=[]
    identity=None
    for index,record in enumerate(records):
        expected=['reference','models'] if index%2==0 else ['models','reference']
        if record['order']!=expected:raise ValueError('arm order was not counterbalanced')
        for arm in expected:
            path=outputs/record[arm]['file']
            with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            if digest!=record[arm]['sha256']:raise ValueError('raw performance outcome digest mismatch')
        pair=compare_pair(cases,outputs/record['models']['file'],outputs/record['reference']['file'])
        current={}
        for arm,key in [('models','candidateRuntime'),('reference','referenceRuntime')]:
            ready=pair[key]
            for field in ['snapshot','backend','threads','dtype','nativeSha256','javaVersion','sourceRevision','torchVersion','attentionBackend']:
                current[arm+'.'+field]=ready.get(field)
        if identity is not None and identity!=current:raise ValueError('runtime or checkpoint identity changed across replications')
        identity=current
        pairs.append(pair)
    if not pairs:raise ValueError('no performance pairs')
    if any(set(pair['buckets'])!=set(pairs[0]['buckets']) for pair in pairs):raise ValueError('shape coverage differs')
    buckets={}
    for name in pairs[0]['buckets']:
        buckets[name]={}
        for metric in ['throughputRatio','p95LatencyRatio']:
            values=[p['buckets'][name][metric] for p in pairs]
            buckets[name][metric]=dict(values=values,minimum=min(values),maximum=max(values),
                mean=statistics.mean(values),sampleStandardDeviation=statistics.stdev(values) if len(values)>1 else None)
    all_passed=all(pair['passed'] for pair in pairs)
    return dict(stage='performance-replications',pairs=len(pairs),passed=all_passed,
        replicationGatePassed=all_passed and len(pairs)>=policy['minPairedReplications'],
        identity=identity,buckets=buckets,pairResults=pairs,catalogQualified=False)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--replications',type=int,choices=[1,3],required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    commands={
        'reference':['/work/python/bin/python','reference_worker.py','--snapshot','/work/variant','--manifest','/work/variant-manifest.json','--source','/work/reference','--threads','8'],
        'models':['java','--add-modules','jdk.incubator.vector','--enable-native-access=ALL-UNNAMED','-Xmx8g','-Dmodels.deberta.threads=8','-Djdk.virtualThreadScheduler.parallelism=8','-cp','/work/payload/classes:/work/payload/lib/*','GlinerWorker','/work/variant','/work/payload/rust/target/release/libjmodels_kernels.so','/work/variant-manifest.json']}
    records=[]
    for index in range(args.replications):
        order=['reference','models'] if index%2==0 else ['models','reference']
        record=dict(pair=index+1,order=order)
        for arm in order:
            path=args.output/f'performance-{index+1}-{arm}.jsonl'
            started=datetime.now(timezone.utc).isoformat()
            subprocess.run([sys.executable,'run_worker.py','--cases','performance-cases.jsonl','--output',str(path),'--']+commands[arm],check=True)
            with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            record[arm]=dict(file=path.name,sha256=digest,startedAt=started,finishedAt=datetime.now(timezone.utc).isoformat())
        records.append(record)
        (args.output/'performance-run-manifest.json').write_text(json.dumps(records,indent=2)+'\n')
        report=summarize_pairs(read_cases(HERE/'performance-cases.jsonl'),args.output,records)
        (args.output/'performance-replications-report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(pair=index+1,passed=report['pairResults'][-1]['passed'])),flush=True)
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':main()
