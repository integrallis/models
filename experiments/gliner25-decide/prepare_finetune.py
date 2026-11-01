"""Metadata-only split construction. Never load model weights or run inference here."""
import argparse
import collections
import functools
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import unicodedata

from common import HERE, digest, read_cases


def fingerprint(text):
    return hashlib.sha256(' '.join(unicodedata.normalize('NFKC', text).casefold().split()).encode()).hexdigest()


def ordered(rows):
    return sorted(rows, key=lambda r: hashlib.sha256(('gliner-general-v1:'+r['id']).encode()).hexdigest())


def stratified(rows, count):
    groups = collections.defaultdict(list)
    for row in rows:
        for label in next(iter(row['expected'].values())):
            groups[label].append(row)
    selected = {r['id']: r for group in groups.values() for r in ordered(group)[:count]}
    return ordered(selected.values())


def make_case(id, text, task, labels, expected, cohort, split, multi=False):
    return dict(id=id, request=dict(text=text, tasks=[dict(name=task,
        labels=[dict(name=label, description='') for label in labels], multiLabel=multi, threshold=.5)]),
        expected={task: expected}, cohort=cohort, sourceSplit=split,
        kind='multi' if multi else 'noul' if cohort=='boolq' else 'choice')


def main():
    parser = argparse.ArgumentParser()
    for name in ['sources', 'publisher', 'tokenizer', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--reuse-splits',type=Path)
    parser.add_argument('--candidate',default='modeljars-gliner25-decide-general-v1')
    args = parser.parse_args()
    for entry in json.loads((HERE/'finetune-sources.json').read_text()):
        path = args.sources/entry['file']
        if path.stat().st_size != entry['sizeBytes'] or digest(path) != entry['sha256']:
            raise ValueError('source mismatch: '+entry['file'])
    policy = json.loads((HERE/'policy.json').read_text())
    revision = subprocess.check_output(['git', '-C', str(args.publisher), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != policy['referenceCommit']:
        raise ValueError('wrong publisher revision')
    spec = importlib.util.spec_from_file_location('publisher_splitter', args.publisher/'gliner2/processing/word_splitter.py')
    splitter_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(splitter_module)
    splitter = splitter_module.WhitespaceTokenSplitter()
    from tokenizers import Tokenizer, __version__
    if __version__ != policy['tokenizersVersion']:
        raise ValueError('wrong tokenizer version')
    tokenizer_entry = next(e for e in json.loads((HERE/'snapshot.json').read_text())['files'] if e['path']=='tokenizer.json')
    if digest(args.tokenizer) != tokenizer_entry['sha256']:
        raise ValueError('wrong tokenizer identity')
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer.no_padding(); tokenizer.no_truncation()
    @functools.lru_cache(maxsize=50000)
    def piece(text):
        return tokenizer.encode(text, add_special_tokens=False).ids
    def encode(request):
        pieces = []
        for task in request['tasks']:
            if pieces: pieces.append('[SEP_STRUCT]')
            description = task['name']
            for label in task['labels']:
                if label['description']:
                    description += ' [DESCRIPTION] '+label['name']+': '+label['description']
            pieces.extend(['(', '[P]', description, '('])
            for label in task['labels']: pieces.extend(['[L]', label['name']])
            pieces.extend([')', ')'])
        pieces.append('[SEP_TEXT]')
        text = request['text']
        if not text.endswith(('.', '!', '?')): text += '.'
        pieces.extend(word for word,_,_ in splitter(text))
        return [token for word in pieces for token in piece(word)]
    # Check this preprocessing-only length counter against committed real publisher input IDs.
    oracle_path = HERE.parent.parent/'benchmark-results/2026-11-01-gliner25-decide/results-restart-1/parity-1-reference.jsonl'
    oracle = {r['id']:r for r in map(json.loads, oracle_path.read_text().splitlines()[1:])}
    for case in read_cases(HERE/'parity-cases.jsonl'):
        if encode(case['request']) != oracle[case['id']]['tokenIds']:
            raise ValueError('preflight token mismatch: '+case['id'])
    rows = []
    emotions = (args.sources/'emotions.txt').read_text().splitlines()
    for split in ['train', 'dev']:
        for line in (args.sources/f'go-{split}.tsv').read_text().splitlines():
            text, labels, id = line.split('\t')
            rows.append(make_case('go-'+split+'-'+id, text, 'emotions', emotions,
                [emotions[int(i)] for i in labels.split(',')], 'goemotions', split, True))
    domains = json.loads((args.sources/'domains.json').read_text())
    domain_of = {label:domain for domain,labels in domains.items() for label in labels}
    clinc = json.loads((args.sources/'clinc.json').read_text())
    for split in ['train', 'val', 'test']:
        for i,(text,label) in enumerate(clinc[split]):
            domain = domain_of[label]
            rows.append(make_case(f'clinc-{split}-{i}', text, 'intent',
                [l.replace('_',' ') for l in domains[domain]], [label.replace('_',' ')], 'clinc-'+domain, split))
    import pyarrow.parquet as pq
    for split in ['train', 'validation']:
        for i,row in enumerate(pq.read_table(args.sources/f'boolq-{split}.parquet').to_pylist()):
            question = row['question'].rstrip('?')+'?'
            rows.append(make_case(f'boolq-{split}-{i}', row['passage'], question, ['no','yes'],
                ['yes' if row['answer'] else 'no'], 'boolq', split))
    excluded = collections.Counter(); eligible = []
    old = {fingerprint(c['request']['text']) for c in read_cases(HERE/'quality-cases.jsonl')}
    if args.reuse_splits:
        for name in ['train','development','qualification']:
            old.update(fingerprint(c['request']['text']) for c in read_cases(args.reuse_splits/(name+'-cases.jsonl')))
    for row in rows:
        row['textFingerprint'] = fingerprint(row['request']['text'])
        row['encodedTokens'] = len(encode(row['request']))
        if row['textFingerprint'] in old:
            excluded[row['cohort']+':previous-qualification-text'] += 1
        elif row['encodedTokens'] > 512:
            excluded[row['cohort']+':overlength'] += 1
        else: eligible.append(row)
    # Holdouts have priority over development and training, including normalized text duplicates.
    final = stratified([r for r in eligible if r['cohort']=='goemotions' and r['sourceSplit']=='dev'],10)
    final += ordered([r for r in eligible if r['cohort']=='boolq' and r['sourceSplit']=='validation'])[:300]
    final += stratified([r for r in eligible if r['cohort'].startswith('clinc-') and r['sourceSplit']=='test'],3)
    def unique(rows, forbidden):
        chosen=[]
        for row in ordered(rows):
            if row['textFingerprint'] in forbidden:
                excluded[row['cohort']+':split-overlap-or-duplicate'] += 1
            else:
                forbidden.add(row['textFingerprint']); chosen.append(row)
        return chosen
    used = set(old); final=unique(final,used)
    available = [r for r in eligible if r['textFingerprint'] not in used]
    dev = stratified([r for r in available if r['cohort']=='goemotions' and r['sourceSplit']=='train'],10)
    dev += ordered([r for r in available if r['cohort']=='boolq' and r['sourceSplit']=='validation'])[:300]
    dev += stratified([r for r in available if r['cohort'].startswith('clinc-') and r['sourceSplit']=='val'],3)
    dev = unique(dev,used)
    train = unique([r for r in eligible if r['sourceSplit']=='train'],used)
    if args.reuse_splits:
        train=read_cases(args.reuse_splits/'train-cases.jsonl')
        dev=read_cases(args.reuse_splits/'development-cases.jsonl')
        assert not ({fingerprint(r['request']['text']) for r in final} & old)
    args.output.mkdir(parents=True,exist_ok=True)
    manifest = dict(candidate=args.candidate, selection='deterministic hash, no inference-based selection',
        maximumEncodedTokens=512, exclusions=dict(excluded), files={})
    if args.reuse_splits:
        manifest['previousSplitsSha256']={name:digest(args.reuse_splits/(name+'-cases.jsonl')) for name in ['train','development','qualification']}
    for name,data in [('train',train),('development',dev),('qualification',final)]:
        path=args.output/(name+'-cases.jsonl')
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data))
        manifest['files'][path.name]=dict(sha256=digest(path),cases=len(data),
            cohorts=dict(collections.Counter(r['cohort'] for r in data)))
        if name != 'qualification':
            records=[]
            for row in data:
                tasks=[]
                for task in row['request']['tasks']:
                    tasks.append(dict(task=task['name'], labels=[l['name'] for l in task['labels']],
                        true_label=row['expected'][task['name']], multi_label=task['multiLabel']))
                records.append(dict(input=row['request']['text'], output=dict(classifications=tasks)))
            path=args.output/(name+'.jsonl');path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
            manifest['files'][path.name]=dict(sha256=digest(path),cases=len(records))
    (args.output/'split-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()
