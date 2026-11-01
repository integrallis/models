"""Independent full-metric analysis; no original-cohort pooling or changed gates."""
import importlib.util
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('paired', HERE.parent / 'bonsai-paired/analyze.py')
paired = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paired)
CONFIG = json.loads((HERE / 'protocol.json').read_text())
paired.CONFIG = CONFIG


def compare_records(before, after, history):
    differences = []
    for b, c in zip(before, after):
        assert (b['id'], b['repeat'], b['inputTokens']) == (c['id'], c['repeat'], c['inputTokens'])
        signature = (c['fullLogitSha256'], c['greedyTokens'], c['prediction'])
        assert history.setdefault(c['id'], signature) == signature, 'candidate numerical drift'
        fields = [k for k in ['fullLogitSha256', 'greedyTokens', 'prediction'] if b[k] != c[k]]
        if fields:
            differences.append({'id':c['id'], 'repeat':c['repeat'], 'fields':fields})
    return differences


def analyze(root, arch):
    expected = json.loads((HERE / f'expected-{arch}.json').read_text())
    groups = {k: [] for k in CONFIG['limits']}
    history, differences, fixtures = {}, [], []
    quality = {arm:{d:{'count':0,'correct':0} for d in ['jevbench','snips']} for arm in ['baseline','candidate']}
    measured_ids = set()
    for shard, pairs in CONFIG['shardsByArchitecture'][arch].items():
        directory = root / f'precision-{arch}-{shard}'
        assert (directory / 'STATUS').read_text().strip() == 'COMPLETE'
        provenance = json.loads((directory / 'provenance.json').read_text())
        assert provenance['precisionProtocol'] == CONFIG and provenance['protocol'] == CONFIG
        assert provenance['jarEntries'] == expected['jarEntries']
        assert provenance['nativeSha256'] == expected['nativeSha256']
        assert provenance['model'] == 'qwen-bf16' and provenance['shard'] == shard
        assert provenance['harnessSha256'] == CONFIG['harnessSha256']
        fixture = json.loads((directory / 'fixtures.json').read_text())
        assert fixture['candidate'] == 0
        fixtures.append(fixture)
        logs = {k:[] for k in groups}
        for pair in pairs:
            assert json.loads((directory / f'pair{pair}.complete.json').read_text())['pair'] == pair
            b, br = paired.read(directory, 'qwen-bf16', pair, 'baseline')
            c, cr = paired.read(directory, 'qwen-bf16', pair, 'candidate')
            differences.extend(dict(pair=pair, **r) for r in compare_records(br, cr, history))
            for arm, records in [('baseline',br),('candidate',cr)]:
                for row in records:
                    if row['repeat'] >= 0:
                        measured_ids.add(row['id'])
                        q = quality[arm][row['dataset']]
                        q['count'] += 1
                        q['correct'] += row['prediction'] == row['expected']
            for metric in logs:
                logs[metric].append(math.log(c[metric]/b[metric]))
        for metric in logs:
            groups[metric].append(logs[metric])
    assert len(measured_ids) == 30
    for arm in quality:
        assert all(q['count'] == CONFIG['pairsByArchitecture'][arch]*3//2 for q in quality[arm].values())
    metrics = {k:paired.bootstrap(v, CONFIG['limits'][k]) for k,v in groups.items()}
    quality_pass = all(quality['candidate'][d]['correct'] >= quality['baseline'][d]['correct'] for d in quality['candidate'])
    return {'architecture':arch,'pairs':CONFIG['pairsByArchitecture'][arch], 'uniqueMeasuredTasks':30,
            'fixtureExitCodes':fixtures,'metrics':metrics,'quality':quality,'qualityPass':quality_pass,
            'baselineNumericDifferences':differences,'candidateExactAcrossContexts':True,
            'originalCohortReclassified':False,'pass':quality_pass and all(v['pass'] for v in metrics.values())}


def main():
    results=[]
    for arch in ['x86','arm']:
        try:
            results.append(analyze(Path(sys.argv[1]),arch))
        except Exception as error:
            results.append({'architecture':arch,'pass':False,'error':f'{type(error).__name__}: {error}'})
    Path(sys.argv[2]).write_text(json.dumps({'protocol':CONFIG,'results':results},indent=2)+'\n')
    return 0 if all(r['pass'] for r in results) else 1


if __name__ == '__main__':
    sys.exit(main())
