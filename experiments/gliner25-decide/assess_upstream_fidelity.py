"""Bind disclosed upstream-v2 task scores to complete, identical-snapshot raw outcomes."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()

def assess(joint, single):
    policy = json.loads((HERE/'catalog-policy-v2.json').read_text())
    snapshot = json.loads((HERE/'snapshot.json').read_text())
    qualities = {'models': {}, 'reference': {}}
    evidence = {}
    disagreements = {}
    for cohort, folder, stem, count in [('publisher.singleHead', single, 'publisher', 2900),
            ('publisher.jointHead', joint, 'publisher', 1700), ('goemotions.fullTest', joint, 'go-full', 5427)]:
        predictions = {}
        for arm in qualities:
            raw_path = folder/f'{stem}-{arm}.jsonl'
            score_path = folder/f'{stem}-{arm}-independent-score.json'
            raw = [json.loads(line) for line in raw_path.read_text().splitlines()]
            ready, rows = raw[0], raw[1:]
            score = json.loads(score_path.read_text())
            if (ready.get('snapshot') != snapshot or ready.get('device') != ['cpu']
                    or ready.get('threads') != policy['threadsPerArm']
                    or ready.get('dtype') != (['float32'] if arm == 'models' else ['torch.float32'])):
                raise ValueError('runtime or original snapshot identity differs')
            if len(rows) != count or len({r['id'] for r in rows}) != count or any('error' in r for r in rows):
                raise ValueError('incomplete or failed inference cohort')
            if score['predictionSha256'] != sha(raw_path) or score['cases'] != count:
                raise ValueError('independent score does not bind these predictions')
            if arm == 'reference' and (ready.get('sourceRevision') != policy['referenceCommit'] or ready['versions']['tokenizers'] != policy['tokenizersVersion']):
                raise ValueError('publisher reference revision or tokenizer version differs')
            predictions[arm] = {r['id']: r['selected'] for r in rows}
            evidence[cohort+'.'+arm] = {'rawFile': str(raw_path), 'rawSha256': sha(raw_path),
                'scoreFile': str(score_path), 'scoreSha256': sha(score_path), 'casesSha256': score['caseSha256']}
            quality = qualities[arm]
            if cohort == 'goemotions.fullTest':
                if not score['annotationsAndLabelOrderExact'] or score['officialTestCases'] != count:
                    raise ValueError('official emotion labels not verified')
                for metric in ['microF1', 'macroF1']:
                    quality['goemotions.'+metric] = score[metric]
            else:
                mode = 'singleHead' if folder == single else 'jointHead'
                quality[f'publisher.{mode}.meanDomainExactMatch'] = score['meanDomainHeadExactMatch']
                for domain, result in score['domains'].items():
                    quality[f'publisher.{mode}.{domain}.exactMatch'] = result['headExactMatch']
                if mode == 'jointHead':
                    quality['publisher.jointDocument.meanDomainExactMatch'] = score['meanDomainDocumentExactMatch']
                    for domain, result in score['domains'].items():
                        quality[f'publisher.jointDocument.{domain}.exactMatch'] = result['documentExactMatch']
        if predictions['models'].keys() != predictions['reference'].keys():
            raise ValueError('paired request identities differ')
        if evidence[cohort+'.models']['casesSha256'] != evidence[cohort+'.reference']['casesSha256']:
            raise ValueError('paired request corpus differs')
        disagreements[cohort] = sum(predictions['models'][key] != predictions['reference'][key]
            for key in predictions['models'])
    quality, oracle = qualities['models'], qualities['reference']
    loss = max(0, *(oracle[key] - quality[key] for key in quality))
    parity = []
    for folder in [joint, single]:
        for name in ['variant-parity-report.json', 'variant-shape-report.json']:
            path = folder/name
            if not json.loads(path.read_text())['passed']:
                raise ValueError('numerical parity failed')
            parity.append({'path': str(path), 'sha256': sha(path)})
    passed = loss <= policy['quality']['maxOracleQualityLoss'] and not any(disagreements.values())
    return {'policy': policy['policy'], 'policySha256': sha(HERE/'catalog-policy-v2.json'),
        'quality': quality, 'oracleQuality': oracle, 'maximumOracleQualityLoss': loss,
        'cohortCases': policy['quality']['cohortCases'], 'uniqueSourceDocuments': 7127,
        'labelDisagreements': disagreements, 'inferenceErrors': 0, 'numericalParityPassed': True,
        'fidelityPassed': passed, 'snapshot': snapshot, 'evidence': evidence, 'parityEvidence': parity,
        'performanceClaim': False, 'releasedRuntimeVerified': False, 'catalogQualified': False,
        'limitations': ['Publisher public development split; not its private headline test.',
            'Publisher documents are reused for joint and single-head requests.',
            'Accuracy runs used concurrent workers; their timings are not performance qualification.',
            'Scores are not calibrated confidence; emotion fine-tuning is a separate candidate.']}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['joint', 'single', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = assess(args.joint, args.single)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ['fidelityPassed', 'maximumOracleQualityLoss', 'labelDisagreements']}))
