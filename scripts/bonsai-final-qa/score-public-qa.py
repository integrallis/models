"""Score the frozen windows, preserving missing/failed/truncated cases in the denominator."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import string
import sys

case_file, model_file, reference_file, output_file = map(Path, sys.argv[1:])
raw = case_file.read_bytes()
manifest = json.loads(raw)
cases = manifest['cases']
case_hash = hashlib.sha256(raw).hexdigest()

def normalize(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())

def score(case, row):
    if row is None or row.get('error'):
        return {'exact': 0., 'f1': 0., 'prediction': '', 'failed': True, 'truncated': False}
    prediction = row['output']
    if case['dataset'] == 'boolq':
        normalized = prediction.lower().strip().strip(string.punctuation).strip()
        exact = float(normalized == case['label'])
        return {'exact': exact, 'f1': exact, 'prediction': normalized,
                'failed': False, 'truncated': row.get('stopReason') in ('MAX_TOKENS', 'REPETITION_LOOP', 'UNKNOWN')}
    normalized = normalize(prediction)
    if normalized == 'unanswerable':
        normalized = ''
    em, f1 = 0., 0.
    for answer in case['answers']:
        gold = normalize(answer)
        em = max(em, float(normalized == gold))
        if not gold or not normalized:
            overlap_score = float(gold == normalized)
        else:
            common = sum((Counter(normalized.split()) & Counter(gold.split())).values())
            overlap_score = 2 * common / (len(normalized.split()) + len(gold.split()))
        f1 = max(f1, overlap_score)
    return {'exact': em, 'f1': f1, 'prediction': normalized, 'failed': False,
            'truncated': row.get('stopReason') in ('MAX_TOKENS', 'REPETITION_LOOP', 'UNKNOWN')}

def read(path):
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    header, *rows = records
    assert header['caseSha256'] == case_hash, 'case manifest mismatch'
    indexed = {}
    for row in rows:
        i = row['index']
        assert i not in indexed, 'duplicate case'
        assert row['dataset'] == cases[i]['dataset'] and row['id'] == cases[i]['id'], 'case identity mismatch'
        assert row['promptSha256'] == cases[i]['promptSha256'], 'prompt mismatch'
        indexed[i] = row
    return indexed

def wilson(successes, total):
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [center-radius, center+radius]

arms = {'models': read(model_file), 'publisher': read(reference_file)}
report = {'inputReports': {arm: {'file': file.name, 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()} for arm, file in [('models', model_file), ('publisher', reference_file)]}, 'protocol': manifest['protocol'], 'caseSha256': case_hash, 'complete': all(len(rows) == len(cases) for rows in arms.values()), 'datasets': {}, 'disagreements': []}
for dataset in manifest['sources']:
    selected = [(i, c) for i, c in enumerate(cases) if c['dataset'] == dataset]
    result = {}
    scored = {}
    for arm, rows in arms.items():
        scores = [(i, score(c, rows.get(i))) for i, c in selected]
        scored[arm] = dict(scores)
        n = len(scores)
        total_exact = sum(s['exact'] for _, s in scores)
        result[arm] = {'count': n, 'present': sum(i in rows for i, _ in selected),
            'exactMatch': total_exact/n, 'tokenF1': sum(s['f1'] for _, s in scores)/n,
            'exactMatchWilson95': wilson(total_exact,n), 'failures': sum(s['failed'] for _,s in scores),
            'truncations': sum(s['truncated'] for _,s in scores)}
        if dataset == 'squad2':
            impossible = [scored[arm][i] for i,c in selected if c['label'] == 'unanswerable']
            result[arm]['unanswerableAccuracy'] = sum(s['exact'] for s in impossible)/len(impossible)
    metric = 'tokenF1' if dataset == 'squad2' else 'exactMatch'
    result['modelsMinusPublisher'] = result['models'][metric] - result['publisher'][metric]
    result['descriptiveRegressionRulePassed'] = report['complete'] and result['models']['failures'] == 0 and result['publisher']['failures'] == 0 and result['models']['truncations'] <= result['publisher']['truncations'] and result['modelsMinusPublisher'] >= -.03000000001
    for i,c in selected:
        a,b = scored['models'][i],scored['publisher'][i]
        if a['prediction'] != b['prediction']:
            report['disagreements'].append({'index':i,'dataset':dataset,'id':c['id'],'models':a,'publisher':b})
    report['datasets'][dataset] = result
report['descriptiveRegressionRulePassed'] = all(r['descriptiveRegressionRulePassed'] for r in report['datasets'].values())
output_file.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k != 'disagreements'}, indent=2))
