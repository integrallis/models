"""Independent sklearn check of public development predictions; no model execution."""
import argparse
import json
from pathlib import Path

import sklearn
import numpy
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from common import digest, read_cases


def audit(cases_path, predictions):
    cases = read_cases(cases_path)
    raw = [json.loads(line) for line in Path(predictions).read_text().splitlines()]
    if raw[0].get('type') != 'ready' or raw[0].get('device') != ['cpu']:
        raise ValueError('missing CPU runtime identity')
    rows = {row['id']: row for row in raw[1:]}
    if len(rows) != len(raw)-1 or set(rows) != {case['id'] for case in cases}:
        raise ValueError('missing, duplicate or extra predictions')
    domains = {}
    for domain in sorted({case['cohort'] for case in cases}):
        subset = [case for case in cases if case['cohort'] == domain]
        labels = sorted({(task['name'], label['name']) for case in subset
                         for task in case['request']['tasks'] for label in task['labels']})
        gold = []
        chosen = []
        head_exact = []
        for case in subset:
            row = rows[case['id']]
            if 'error' in row or set(row.get('selected', {})) != set(case['expected']):
                raise ValueError('invalid prediction: ' + case['id'])
            gold.append([(task, label) for task, values in case['expected'].items() for label in values])
            chosen.append([(task, label) for task, values in row['selected'].items() for label in values])
            for task in case['request']['tasks']:
                selected = row['selected'][task['name']]
                allowed = {label['name'] for label in task['labels']}
                if (not selected or len(set(selected)) != len(selected) or not set(selected) <= allowed
                        or (not task['multiLabel'] and len(selected) != 1)):
                    raise ValueError('invalid head: ' + case['id'])
                head_exact.append(set(selected) == set(case['expected'][task['name']]))
        encoder = MultiLabelBinarizer(classes=labels)
        y = encoder.fit_transform(gold)
        p = encoder.transform(chosen)
        domains[domain] = dict(cases=len(subset), heads=len(head_exact),
            documentExactMatch=float(accuracy_score(y, p)),
            headExactMatch=sum(head_exact)/len(head_exact),
            microF1=float(f1_score(y, p, average='micro', zero_division=0)),
            macroF1=float(f1_score(y, p, average='macro', zero_division=0)))
    return dict(stage='independent-publisher-scoring-audit', split='public development',
        officialBenchmarkReproduction=False, sklearnVersion=sklearn.__version__, numpyVersion=numpy.__version__,
        caseSha256=digest(cases_path), predictionSha256=digest(predictions), domains=domains,
        cases=len(cases), errors=0,
        meanDomainDocumentExactMatch=sum(d['documentExactMatch'] for d in domains.values())/len(domains),
        meanDomainHeadExactMatch=sum(d['headExactMatch'] for d in domains.values())/len(domains))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['cases', 'predictions', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.cases, args.predictions)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'domains'}))
