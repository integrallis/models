"""Held-out task quality. Oracle agreement alone never satisfies the absolute floors."""
import argparse
import collections
import json
import math
from pathlib import Path

from common import HERE, read_cases


def measure(cases, path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if not rows or rows[0].get('type') != 'ready' or rows[0].get('device') != ['cpu']:
        raise ValueError('missing CPU runtime identity')
    outcomes = {row['id']: row for row in rows[1:]}
    if len(outcomes) != len(rows)-1 or set(outcomes) != {row['id'] for row in cases}:
        raise ValueError('missing, extra or duplicate outcomes')
    groups = collections.defaultdict(list)
    for case in cases:
        row = outcomes[case['id']]
        task = case['request']['tasks'][0]
        truth = set(case['expected'][task['name']])
        chosen = row.get('selected', {}).get(task['name'], [])
        error = str(row['error']) or 'worker error' if 'error' in row else None
        names = {l['name'] for l in task['labels']}
        if (not chosen or len(chosen) != len(set(chosen)) or not set(chosen) <= names
                or (not task['multiLabel'] and len(chosen) != 1)
                or set(row.get('selected', {})) != {task['name']}):
            error = 'invalid selected labels'
        predicted = set(chosen) if not error else set()
        confidence = None
        if not task['multiLabel'] and not error:
            raw = row.get('raw')
            if isinstance(raw, dict):
                confidence = raw[task['name']][1]
            elif isinstance(raw, list):
                scores = [label['score'] for result in raw for label in result['labels'] if label['selected']]
                if len(scores) != 1:
                    raise ValueError('invalid selected score count')
                confidence = scores[0]
            if confidence is not None and (not math.isfinite(confidence) or not 0 <= confidence <= 1):
                raise ValueError('invalid single-label model score')
        entry = dict(id=case['id'], expected=sorted(truth), selected=sorted(predicted),
                     correct=not error and predicted == truth, error=error, labels=sorted(names),
                     kind=case['kind'], selectedModelScore=confidence)
        groups[case['cohort']].append(entry)
    summaries = {}
    for cohort, outcomes in groups.items():
        label_counts = collections.defaultdict(lambda: [0,0,0])
        by_type = collections.defaultdict(list)
        for row in outcomes:
            by_type[row['kind']].append(row['correct'])
            truth, prediction = set(row['expected']), set(row['selected'])
            for label in row['labels']:
                counts = label_counts[label]
                counts[0] += label in truth and label in prediction
                counts[1] += label not in truth and label in prediction
                counts[2] += label in truth and label not in prediction
        def f1(counts):
            tp,fp,fn=counts
            return 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
        total = [sum(counts[i] for counts in label_counts.values()) for i in range(3)]
        bins = [[] for _ in range(10)]
        for row in outcomes:
            score = row['selectedModelScore']
            if score is not None:
                bins[min(9, int(score*10))].append((score, row['correct']))
        scored = sum(len(b) for b in bins)
        ece = (sum(abs(sum(s-y for s,y in b)) for b in bins)/scored) if scored else None
        summaries[cohort] = dict(cases=len(outcomes), errors=sum(bool(r['error']) for r in outcomes),
            accuracy=sum(r['correct'] for r in outcomes)/len(outcomes),
            microF1=f1(total), macroF1=sum(f1(c) for c in label_counts.values())/len(label_counts),
            singleLabelECE=ece, calibrationCases=scored, confidenceQualified=False,
            accuracyByType={t:sum(v)/len(v) for t,v in by_type.items()},
            perLabel={label:dict(zip(['tp','fp','fn'],counts),f1=f1(counts),
                precision=counts[0]/(counts[0]+counts[1]) if counts[0]+counts[1] else 0.0,
                recall=counts[0]/(counts[0]+counts[2]) if counts[0]+counts[2] else 0.0)
                for label,counts in label_counts.items()})
    return dict(runtime=rows[0], cohorts=summaries, outcomes=dict(groups))


def assess(cases, path):
    measured = measure(cases, path)
    summaries = measured['cohorts']
    policy = json.loads((HERE/'policy.json').read_text())['quality']
    required = {'jevbench', 'clinc-banking', 'goemotions'}
    if set(summaries) != required:
        raise ValueError('required quality cohorts differ')
    failures=[]
    for name,s in summaries.items():
        if s['errors']: failures.append(name+': incomplete/invalid inference')
    gates=[('jevbench.accuracy',summaries['jevbench']['accuracy'],policy['jevbenchAccuracy']),
           ('jevbench.choice',summaries['jevbench']['accuracyByType']['choice'],policy['jevbenchChoiceAccuracy']),
           ('jevbench.noul',summaries['jevbench']['accuracyByType']['noul'],policy['jevbenchNoulAccuracy']),
           ('clinc-banking.macroF1',summaries['clinc-banking']['macroF1'],policy['singleLabelMacroF1']),
           ('goemotions.microF1',summaries['goemotions']['microF1'],policy['multiLabelMicroF1']),
           ('goemotions.macroF1',summaries['goemotions']['macroF1'],policy['multiLabelMacroF1'])]
    for name,value,floor in gates:
        if value < floor: failures.append(f'{name}: {value:.6f} < {floor}')
    return dict(policy='schema-classification-v1',stage='absolute-quality',passed=not failures,
                **measured,failures=failures,catalogQualified=False)


def compare_quality(cases, candidate, reference):
    left, right = assess(cases, candidate), assess(cases, reference)
    for key in ['snapshot', 'threads']:
        if left['runtime'][key] != right['runtime'][key]:
            raise ValueError(key + ' differs between arms')
    policy = json.loads((HERE/'policy.json').read_text())['quality']
    failures = ['candidate: '+f for f in left['failures']] + ['reference: '+f for f in right['failures']]
    deltas = {}
    for cohort in left['cohorts']:
        deltas[cohort] = {}
        for metric in ['accuracy', 'macroF1', 'microF1']:
            loss = right['cohorts'][cohort][metric] - left['cohorts'][cohort][metric]
            deltas[cohort][metric] = loss
            if not math.isfinite(loss) or loss > policy['maxOracleQualityLoss']:
                failures.append(f'{cohort}.{metric}: oracle loss {loss}')
    disagreements = []
    for cohort in left['outcomes']:
        for a, b in zip(left['outcomes'][cohort], right['outcomes'][cohort]):
            if a['selected'] != b['selected'] or a['error'] or b['error']:
                disagreements.append(dict(id=a['id'], candidate=a, reference=b))
    return dict(policy='schema-classification-v1', stage='paired-quality', passed=not failures,
                candidate=left, reference=right, oracleLoss=deltas, disagreements=disagreements,
                failures=failures, catalogQualified=False)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['cases','results','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reference', type=Path)
    a=p.parse_args(); cases=read_cases(a.cases)
    report=compare_quality(cases,a.results,a.reference) if a.reference else assess(cases,a.results)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['outcomes','cohorts','runtime','candidate','reference','disagreements']},indent=2))
    if 'cohorts' in report:
        print(json.dumps({k:{m:v for m,v in s.items() if m!='perLabel'} for k,s in report['cohorts'].items()},indent=2))
    raise SystemExit(0 if report['passed'] else 1)
