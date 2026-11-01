"""Capped rare-label repetition from training labels only; no development/holdout fitting."""
import argparse
import collections
import hashlib
import json
from pathlib import Path


def repeat_counts(cases, target, cap):
    if target < 1 or cap < 1:raise ValueError('target and repetition cap must be positive')
    counts=collections.Counter(label for row in cases if row['cohort']=='goemotions'
        for label in row['expected']['emotions'])
    repeats=[]
    for row in cases:
        factor=max((min(cap,target/counts[label]) for label in row['expected']['emotions']),default=1) if row['cohort']=='goemotions' else 1
        factor=max(1,factor)
        whole=int(factor)
        fraction=int.from_bytes(hashlib.sha256(('gliner-balance:'+row['id']).encode()).digest()[:8],'big')/2**64
        repeats.append(whole+int(fraction < factor-whole))
    return repeats


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--training-cases',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--target',type=int,default=1000)
    p.add_argument('--cap',type=int,default=8)
    args=p.parse_args()
    cases=[json.loads(s) for s in args.training_cases.read_text().splitlines()]
    if any(row['sourceSplit']!='train' for row in cases):raise ValueError('only source training rows may be resampled')
    if len({row['id'] for row in cases})!=len(cases):raise ValueError('input training IDs must be unique')
    counts=repeat_counts(cases,args.target,args.cap)
    repeated=[]
    for row,count in zip(cases,counts):
        for index in range(count):
            repeated.append(dict(row,id=f"{row['id']}:repeat-{index}",trainingSourceId=row['id']))
    repeated.sort(key=lambda row:hashlib.sha256(('gliner-balanced-order:'+row['id']).encode()).digest())
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'train-cases.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in repeated))
    records=[]
    for row in repeated:
        tasks=[dict(task=task['name'],labels=[label['name'] for label in task['labels']],
                    true_label=row['expected'][task['name']],multi_label=task['multiLabel']) for task in row['request']['tasks']]
        records.append(dict(input=row['request']['text'],output=dict(classifications=tasks)))
    (args.output/'train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    report=dict(method='capped inverse-frequency repetition',target=args.target,maximumRepeats=args.cap,
        inputCases=len(cases),outputCases=len(repeated),allInputRowsRetained=True,
        inputSha256=hashlib.sha256(args.training_cases.read_bytes()).hexdigest(),
        repeatedEmotionCounts=dict(collections.Counter(label for row in repeated if row['cohort']=='goemotions' for label in row['expected']['emotions'])),
        repetitions=dict(collections.Counter(counts)))
    (args.output/'balance-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
