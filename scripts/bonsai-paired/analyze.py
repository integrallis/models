"""Analyze all declared pairs; incompleteness, regressions and imprecision fail acceptance."""
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys

CONFIG = json.loads((Path(__file__).parent / 'protocol.json').read_text())


def read(directory, model, pair, arm):
    prefix = directory / f'{model}-pair{pair}-{arm}'
    assert json.loads(prefix.with_suffix('.execution.json').read_text())['exitCode'] == 0
    rows = [json.loads(s) for s in prefix.with_suffix('.stdout').read_text().splitlines() if s.startswith('{')]
    requests = [r for r in rows if r['kind'] == 'request']
    assert len(requests) == 7 and sorted(r['repeat'] for r in requests) == [-4, -3, -2, -1, 0, 1, 2]
    for row in requests:
        assert len(row['fullLogitSha256']) == 9 and len(row['greedyTokens']) == 8
    warm = [r for r in requests if r['repeat'] >= 0]
    loads = [r for r in rows if r['kind'] == 'load']
    assert len(loads) == 1
    rss = re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)', prefix.with_suffix('.stderr').read_text())
    assert rss
    values = {'loadNanos': loads[0]['nanos'], 'peakRssKiB': int(rss.group(1)),
              'coldRequestNanos': next(r['requestNanos'] for r in requests if r['repeat'] == -4)}
    values.update({metric: statistics.mean(r[metric] for r in warm)
                   for metric in ['prefillNanos', 'decodeNanos', 'requestNanos', 'javaAllocatedBytes']})
    return values, requests


def bootstrap(groups, limit):
    logs = [x for group in groups for x in group]
    rng = random.Random(CONFIG['bootstrapSeed'])
    samples = sorted(math.exp(statistics.mean([x for group in groups for x in rng.choices(group, k=len(group))]))
                     for _ in range(CONFIG['bootstrapResamples']))
    lo, hi = samples[500], samples[19499]
    precision = hi - lo <= CONFIG['maximumIntervalWidth']
    return {'ratio': math.exp(statistics.mean(logs)), 'ci95': [lo, hi], 'limit': limit,
            'precisionTargetMet': precision, 'pass': hi <= limit and precision,
            'status': 'REGRESSION' if lo > limit else 'PASS' if hi <= limit and precision else 'UNRESOLVED',
            'processRatios': [math.exp(x) for x in logs]}


def analyze(root, arch, model):
    samples = {arm: [] for arm in ['baseline', 'candidate']}
    measured = {arm: [] for arm in samples}
    candidate_history = {}
    numeric_changes = []
    fixtures = []
    artifact_identities = []
    for shard in ['0', '1']:
        directory = root / f'paired-{arch}-{model}-{shard}'
        assert (directory / 'STATUS').read_text().strip() == 'COMPLETE'
        provenance = json.loads((directory / 'provenance.json').read_text())
        assert provenance['protocol'] == CONFIG
        assert provenance['harnessSha256'] == CONFIG['harnessSha256']
        assert provenance['model'] == model and provenance['shard'] == shard
        artifact_identities.append(provenance['jarEntries'])
        fixture = json.loads((directory / 'fixtures.json').read_text())
        assert fixture['candidate'] == 0
        fixtures.append(fixture)
        for pair in CONFIG['shards'][shard]:
            records = {}
            for arm in samples:
                values, records[arm] = read(directory, model, pair, arm)
                samples[arm].append(values)
                measured[arm].extend(r for r in records[arm] if r['repeat'] >= 0)
            for before, after in zip(records['baseline'], records['candidate']):
                assert (before['id'], before['repeat'], before['inputTokens']) == (after['id'], after['repeat'], after['inputTokens'])
                fields = [k for k in ['fullLogitSha256', 'greedyTokens', 'prediction'] if before[k] != after[k]]
                if fields:
                    numeric_changes.append({'pair': pair, 'id': after['id'], 'repeat': after['repeat'], 'fields': fields})
                signature = (after['fullLogitSha256'], after['greedyTokens'], after['prediction'])
                prior = candidate_history.setdefault(after['id'], signature)
                assert prior == signature, f'candidate compilation/session nondeterminism: {after["id"]}'
    # Archive timestamps can differ across builds; every entry's bytes must still match.
    assert artifact_identities[0] == artifact_identities[1], 'artifact content differs between paired shards'
    for arm in measured:
        assert len(measured[arm]) == 30 and len({r['id'] for r in measured[arm]}) == 30
    quality = {arm: {dataset: {'count': sum(r['dataset'] == dataset for r in rows),
                              'correct': sum(r['dataset'] == dataset and r['prediction'] == r['expected'] for r in rows)}
                     for dataset in ['jevbench', 'snips']} for arm, rows in measured.items()}
    assert all(quality[arm][dataset]['count'] == 15 for arm in quality for dataset in quality[arm])
    quality_pass = all(quality['candidate'][d]['correct'] >= quality['baseline'][d]['correct'] for d in quality['candidate'])
    metrics = {}
    for metric, limit in CONFIG['limits'].items():
        logs = [math.log(c[metric] / b[metric]) for c, b in zip(samples['candidate'], samples['baseline'])]
        metrics[metric] = bootstrap([logs[:5], logs[5:]], limit)
    return {'architecture': arch, 'model': model, 'pairs': 10, 'fixtureExitCodes': fixtures,
            'baselineNumericDifferences': numeric_changes, 'candidateRepeatedCaseHashesExact': True,
            'quality': quality, 'qualityPass': quality_pass, 'metrics': metrics,
            'decodeThroughputRatio': 1 / metrics['decodeNanos']['ratio'],
            'pass': quality_pass and all(m['pass'] for m in metrics.values())}


def main():
    root, output = map(Path, sys.argv[1:3])
    results, errors = [], []
    for arch in ['x86', 'arm']:
        for model in ['qwen-bf16', 'bonsai2']:
            try:
                results.append(analyze(root, arch, model))
            except (AssertionError, OSError, ValueError, KeyError) as error:
                errors.append({'architecture': arch, 'model': model, 'error': str(error), 'type': type(error).__name__})
    passed = not errors and len(results) == 4 and all(r['pass'] for r in results)
    output.write_text(json.dumps({'pass': passed, 'errors': errors, 'results': results,
                      'scope': 'Exact pinned Models, native library and candidate core; no publication authorization'}, indent=2) + '\n')
    print(output.read_text())
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
