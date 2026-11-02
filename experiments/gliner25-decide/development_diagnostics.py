"""Development-only error analysis; never admits a model or reads qualification inputs."""
import argparse
import json
import math
from pathlib import Path
from assess_quality import measure
from common import read_cases


def analyze(cases, predictions):
    if not cases or any(c['cohort'] != 'goemotions' or c['sourceSplit'] != 'train' for c in cases):
        raise ValueError('requires the reserved training-source emotion development cohort')
    measured = measure(cases, predictions)
    rows = {r['id']: r for r in map(json.loads, Path(predictions).read_text().splitlines()[1:])}
    prepared=[]
    for case in cases:
        row=rows[case['id']];task=case['request']['tasks'][0]
        logits=row['logits'];names=[v['name'] for v in task['labels']]
        if len(logits)!=len(names) or not all(math.isfinite(v) for v in logits):
            raise ValueError('invalid development logits')
        prepared.append((names, logits, set(case['expected'][task['name']])))
    offsets=[]
    for offset in [-2,-1.5,-1,-.5,0,.5,1,1.5,2]:
        tp=fp=fn=0
        for names,logits,truth in prepared:
            chosen={n for n,l in zip(names,logits) if l+offset>=0}
            if not chosen:chosen={names[max(range(len(logits)),key=logits.__getitem__)]}
            tp+=len(chosen&truth);fp+=len(chosen-truth);fn+=len(truth-chosen)
        offsets.append(dict(logitOffset=offset,microF1=2*tp/(2*tp+fp+fn),tp=tp,fp=fp,fn=fn))
    return dict(stage='development-diagnostics',catalogQualified=False,
        purpose='Training error analysis only; offsets are hypothetical and change no inference or gates',
        measured=measured,globalLogitOffsetSweep=offsets)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['cases','predictions','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();report=analyze(read_cases(a.cases),a.predictions)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(cohorts=report['measured']['cohorts'],offsets=report['globalLogitOffsetSweep'])))
