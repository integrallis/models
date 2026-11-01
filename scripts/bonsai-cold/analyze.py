"""Analyze all ten independent matched host pairs; preserve incomplete or failed gates."""
import importlib.util
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('precision', HERE.parent / 'qwen-precision/analyze.py')
precision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(precision)
paired = precision.paired
CONFIG = json.loads((HERE / 'protocol.json').read_text())
paired.CONFIG = CONFIG


def cold_allocation(rows):
    cold = [r for r in rows if r['repeat'] == -4]
    assert len(cold) == 1
    return cold[0]['javaAllocatedBytes']


def analyze(root, arch):
    expected = json.loads((HERE / f'expected-{arch}.json').read_text())
    logs = {k:[] for k in CONFIG['limits']}
    history, differences, fixtures = {}, [], []
    quality = {arm:{d:{'count':0,'correct':0} for d in ['jevbench','snips']} for arm in ['baseline','candidate']}
    measured_ids = set()
    for pair in CONFIG['hostPairs']:
        directory = root / f'cold-{arch}-{pair}'
        assert (directory / 'STATUS').read_text().strip() == 'COMPLETE'
        p = json.loads((directory / 'provenance.json').read_text())
        assert p['protocol'] == CONFIG and p['jarEntries'] == expected['jarEntries']
        assert p['nativeSha256'] == expected['nativeSha256']
        assert p['model'] == 'bonsai2' and p['shard'] == str(pair)
        assert p['harnessSha256'] == CONFIG['harnessSha256']
        fixture = json.loads((directory / 'fixtures.json').read_text())
        assert fixture['candidate'] == 0
        fixtures.append(fixture)
        assert json.loads((directory / f'pair{pair}.complete.json').read_text())['pair'] == pair
        b, br = paired.read(directory, 'bonsai2', pair, 'baseline')
        c, cr = paired.read(directory, 'bonsai2', pair, 'candidate')
        b['coldJavaAllocatedBytes'], c['coldJavaAllocatedBytes'] = cold_allocation(br), cold_allocation(cr)
        differences.extend(dict(pair=pair, **r) for r in precision.compare_records(br, cr, history))
        for arm, rows in [('baseline',br),('candidate',cr)]:
            for row in rows:
                if row['repeat'] >= 0:
                    measured_ids.add(row['id'])
                    q = quality[arm][row['dataset']]
                    q['count'] += 1
                    q['correct'] += row['prediction'] == row['expected']
        for metric in logs:
            logs[metric].append(math.log(c[metric]/b[metric]))
    assert len(measured_ids) == 30
    assert all(q['count'] == 15 for arm in quality.values() for q in arm.values())
    # Each host contains one matched pair. Resample whole paired host observations,
    # never singleton strata (which would incorrectly collapse uncertainty to zero).
    metrics = {k:paired.bootstrap([v], CONFIG['limits'][k]) for k,v in logs.items()}
    quality_pass = all(quality['candidate'][d]['correct'] >= quality['baseline'][d]['correct'] for d in quality['candidate'])
    return {'architecture':arch,'pairs':10,'uniqueMeasuredTasks':30,'fixtureExitCodes':fixtures,
            'metrics':metrics,'quality':quality,'qualityPass':quality_pass,
            'baselineNumericDifferences':differences,'candidateExactAcrossContexts':True,
            'pass':quality_pass and all(v['pass'] for v in metrics.values())}


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
