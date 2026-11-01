"""Prepare the public publisher development suite and full official emotion test, without inference."""
import argparse
import collections
import functools
import importlib.util
import json
import subprocess
from pathlib import Path
from common import HERE, digest, read_cases
from prepare_finetune import make_case


def main():
    p=argparse.ArgumentParser()
    for name in ['publisher-data','go-test','emotions','tokenizer','publisher-source','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    inventory=json.loads((HERE/'publisher-dataset-inventory.json').read_text())
    for entry in inventory['files']:
        if digest(a.publisher_data/entry['path'])!=entry['sha256']:raise ValueError('publisher source changed')
    for source in json.loads((HERE/'cohorts.json').read_text())['sources']:
        if source.get('url','').endswith('/test.tsv') and digest(a.go_test)!=source['sha256']:raise ValueError('official emotion test changed')
        if source.get('url','').endswith('/emotions.txt') and digest(a.emotions)!=source['sha256']:raise ValueError('emotion label mapping changed')
    policy=json.loads((HERE/'policy.json').read_text())
    if subprocess.check_output(['git','-C',str(a.publisher_source),'rev-parse','HEAD'],text=True).strip()!=policy['referenceCommit']:
        raise ValueError('publisher source revision differs')
    config=json.loads((HERE/'snapshot.json').read_text())
    entry=next(e for e in config['files'] if e['path']=='tokenizer.json')
    if digest(a.tokenizer)!=entry['sha256']:raise ValueError('tokenizer changed')
    from tokenizers import Tokenizer, __version__
    if __version__!=policy["tokenizersVersion"]:raise ValueError("tokenizers version differs")
    tokenizer=Tokenizer.from_file(str(a.tokenizer));tokenizer.no_padding();tokenizer.no_truncation()
    spec=importlib.util.spec_from_file_location('splitter',a.publisher_source/'gliner2/processing/word_splitter.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    splitter=module.WhitespaceTokenSplitter()
    @functools.lru_cache(maxsize=50000)
    def piece(word):return tokenizer.encode(word,add_special_tokens=False).ids
    def encode(request):
        pieces=[]
        for task in request['tasks']:
            if pieces:pieces.append('[SEP_STRUCT]')
            desc=task['name']
            for l in task['labels']:
                if l['description']:desc+=' [DESCRIPTION] '+l['name']+': '+l['description']
            pieces+=['(','[P]',desc,'(']
            for label in task['labels']:pieces+=['[L]',label['name']]
            pieces+=[')',')']
        text=request['text']
        if not text.endswith(('.','!','?')):text+='.'
        pieces+=['[SEP_TEXT]']+[word for word,_,_ in splitter(text)]
        return [token for word in pieces for token in piece(word)]
    oracle=HERE.parent.parent/'benchmark-results/2026-11-01-gliner25-decide/results-restart-1/parity-1-reference.jsonl'
    results={r['id']:r for r in map(json.loads,oracle.read_text().splitlines()[1:])}
    for case in read_cases(HERE/'parity-cases.jsonl'):
        if encode(case['request'])!=results[case['id']]['tokenIds']:raise ValueError('token preflight differs')
    suites={}
    for entry in inventory['files']:
        if not entry['path'].endswith('.jsonl'):continue
        domain=Path(entry['path']).stem;cases=[]
        for index,line in enumerate((a.publisher_data/entry['path']).read_text().splitlines()):
            row=json.loads(line);tasks=[];gold={}
            for task in row['output']['classifications']:
                tasks.append(dict(name=task['task'],labels=[dict(name=l,description='') for l in task['labels']],
                    multiLabel=task['multi_label'],threshold=.5))
                gold[task['task']]=task['true_label']
                if not set(task['true_label'])<=set(task['labels']):raise ValueError('invalid publisher labels')
            cases.append(dict(id=f'fast-decisions-{domain}-{index}',cohort=domain,sourceSplit='publisher-development',
                kind='multi-head',request=dict(text=row['input'],tasks=tasks),expected=gold))
        suites[domain]=cases
    publisher=[c for rows in suites.values() for c in rows]
    labels=a.emotions.read_text().splitlines();go=[]
    for line in a.go_test.read_text().splitlines():
        text,ids,id=line.split('\t')
        go.append(make_case('goemotions-'+id,text,'emotions',labels,[labels[int(i)] for i in ids.split(',')],
            'goemotions','test',True))
    for case in publisher+go:case['encodedTokens']=len(encode(case['request']))
    for name,rows in [('publisher-development-cases.jsonl',publisher),('go-full-test-cases.jsonl',go)]:
        (a.output/name).write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    manifest=dict(publisherRevision=inventory['revision'],publisherPublicSplit='development',
        publisherPublicCases=len(publisher),publisherOfficialTestAvailable=False,publisherReportedTestExactMatch=.602,
        goTestCases=len(go),goTestSourceSha256=digest(a.go_test),tokenizerSha256=entry.get('sha256'),
        scope='Original weights only; independent accuracy audit, not admission or performance measurement',
        maximumSupportedModelsTokens=512,
        cohorts={domain:dict(cases=len(rows),overlength=sum(c['encodedTokens']>512 for c in rows),
            maximumTokens=max(c['encodedTokens'] for c in rows)) for domain,rows in dict(suites,goemotions=go).items()},
        files={name:digest(a.output/name) for name in ['publisher-development-cases.jsonl','go-full-test-cases.jsonl']})
    manifest['tokenizerSha256']=digest(a.tokenizer)
    (a.output/'publisher-audit-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
