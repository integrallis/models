"""Score every public benchmark head; retain exact-row and exact-head definitions separately."""
import argparse
import collections
import json
from pathlib import Path
from common import read_cases, digest
from assess_quality import measure


def score(cases, path):
    raw=[json.loads(l) for l in Path(path).read_text().splitlines()]
    if not raw or raw[0].get('type')!='ready' or raw[0].get('device')!=['cpu']:raise ValueError('CPU identity missing')
    rows={r['id']:r for r in raw[1:]}
    if len(rows)!=len(raw)-1 or set(rows)!={c['id'] for c in cases}:raise ValueError('incomplete/duplicate outcomes')
    domains=collections.defaultdict(lambda:dict(cases=0,correctRows=0,heads=0,correctHeads=0,errors=0,labels=collections.defaultdict(lambda:[0,0,0])))
    for case in cases:
        row=rows[case['id']];domain=domains[case['cohort']];domain['cases']+=1
        error='error' in row or set(row.get('selected',{}))!=set(case['expected'])
        exact=not error
        for task in case['request']['tasks']:
            name=task['name'];truth=set(case['expected'][name]);chosen=row.get('selected',{}).get(name,[])
            names={l['name'] for l in task['labels']}
            if not chosen or len(set(chosen))!=len(chosen) or not set(chosen)<=names or (not task['multiLabel'] and len(chosen)!=1):error=True
            predicted=set(chosen) if not error else set()
            correct=not error and predicted==truth
            domain['heads']+=1;domain['correctHeads']+=correct;exact=exact and correct
            for label in names:
                counts=domain['labels'][name+'::'+label]
                counts[0]+=label in truth and label in predicted
                counts[1]+=label not in truth and label in predicted
                counts[2]+=label in truth and label not in predicted
        domain['correctRows']+=exact and not error;domain['errors']+=error
    def f1(c):
        tp,fp,fn=c;return 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    for d in domains.values():
        counts=d.pop('labels');d['documentExactMatch']=d['correctRows']/d['cases'];d['headExactMatch']=d['correctHeads']/d['heads']
        d['microF1']=f1([sum(v[i] for v in counts.values()) for i in range(3)])
        d['macroF1']=sum(map(f1,counts.values()))/len(counts)
        d['perTaskLabelCounts']={k:dict(zip(['tp','fp','fn'],v)) for k,v in counts.items()}
    return dict(runtime=raw[0],domains=dict(domains),cases=len(cases),errors=sum(d['errors'] for d in domains.values()),
        meanDomainDocumentExactMatch=sum(d['documentExactMatch'] for d in domains.values())/len(domains),
        meanDomainHeadExactMatch=sum(d['headExactMatch'] for d in domains.values())/len(domains))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--publisher-only',action='store_true')
    a=p.parse_args();cases=read_cases('publisher-development-cases.jsonl');go=[] if a.publisher_only else read_cases('go-full-test-cases.jsonl')
    result=dict(stage='publisher-public-data-and-full-emotion-audit',catalogQualified=False,
        publisherSplit='public development, not the private benchmark test',
        performanceClaim='none; accuracy workers run concurrently on the same host',
        publisher={arm:score(cases,a.directory/f'publisher-{arm}.jsonl') for arm in ['reference','models']},
        goemotions={} if a.publisher_only else {arm:measure(go,a.directory/f'go-full-{arm}.jsonl') for arm in ['reference','models']})
    for suite,prefix,inputs in [('publisher','publisher',cases),('goemotions','go-full',go)]:
        if not inputs:continue
        left={r['id']:r for r in map(json.loads,(a.directory/f'{prefix}-models.jsonl').read_text().splitlines()[1:])}
        right={r['id']:r for r in map(json.loads,(a.directory/f'{prefix}-reference.jsonl').read_text().splitlines()[1:])}
        if result[suite]['models']['runtime']['snapshot']!=result[suite]['reference']['runtime']['snapshot']:raise ValueError('snapshot differs')
        result[suite]['disagreements']=[c['id'] for c in inputs if left[c['id']].get('selected')!=right[c['id']].get('selected') or 'error' in left[c['id']] or 'error' in right[c['id']]]
    result['rawSha256']={p.name:digest(p) for p in a.directory.glob('*.jsonl')}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(publisher={k:{m:v[m] for m in ['cases','errors','meanDomainDocumentExactMatch','meanDomainHeadExactMatch']} for k,v in result['publisher'].items() if k!='disagreements'},
        goemotions={k:v['cohorts'] for k,v in result['goemotions'].items() if k!='disagreements'})))

if __name__=='__main__':main()
