"""Same-host CPU performance evidence; a screening pair cannot qualify the catalog."""
import argparse
import collections
import json
import math
from pathlib import Path
import statistics

from common import HERE, read_cases


def load(cases, path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if not rows or rows[0].get('type') != 'ready' or rows[0].get('device') != ['cpu']:
        raise ValueError('missing CPU readiness')
    ready, results = rows[0], rows[1:]
    by_id = {row['id']: row for row in results}
    if len(by_id) != len(results) or set(by_id) != {case['id'] for case in cases}:
        raise ValueError('missing, extra, or duplicate outcomes')
    buckets = collections.defaultdict(lambda: dict(warmup=[], timed=[]))
    for case in cases:
        row = by_id[case['id']]
        if 'error' in row or not row.get('selected'):
            raise ValueError('inference failure: '+case['id'])
        for metric in ['wallMs', 'elapsedMs', 'peakRssKiB']:
            if metric not in row or not math.isfinite(row[metric]) or row[metric] <= 0:
                raise ValueError('invalid '+metric+': '+case['id'])
        buckets[case['bucket']][case['phase']].append(row)
    return ready, buckets


def summarize(rows):
    values = sorted(row['wallMs'] for row in rows)
    return dict(samples=len(values), p50Ms=statistics.median(values),
                p95Ms=values[math.ceil(len(values)*.95)-1],
                requestsPerSecond=1000*len(values)/sum(values),
                meanInferenceMs=statistics.mean(row['elapsedMs'] for row in rows),
                peakRssKiB=max(row['peakRssKiB'] for row in rows))


def compare_pair(cases, candidate, reference):
    policy = json.loads((HERE/'policy.json').read_text())['performance']
    ca, left = load(cases, candidate)
    ra, right = load(cases, reference)
    for key in ['snapshot', 'threads', 'driverHostname', 'driverCpuAffinity']:
        if key not in ca or key not in ra or ca[key] is None or ca[key] != ra[key]:
            raise ValueError(key+' differs or is missing')
    results = {}; failures = []
    for bucket in left:
        for arm in [left, right]:
            if len(arm[bucket]['warmup']) < policy['warmups'] or len(arm[bucket]['timed']) < policy['minTimedSamplesPerBucket']:
                raise ValueError('insufficient warmup/timed samples: '+bucket)
        a, b = summarize(left[bucket]['timed']), summarize(right[bucket]['timed'])
        throughput = a['requestsPerSecond']/b['requestsPerSecond']
        latency = a['p95Ms']/b['p95Ms']
        passed = throughput >= policy['minThroughputRatio'] and latency <= policy['maxP95LatencyRatio']
        results[bucket] = dict(candidate=a, reference=b, throughputRatio=throughput,
                               p95LatencyRatio=latency, passed=passed)
        if not passed:
            failures.append(bucket)
    return dict(policy='schema-classification-v1', stage='performance-pair', passed=not failures,
                candidateRuntime=ca, referenceRuntime=ra, buckets=results, failures=failures,
                catalogQualified=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['cases', 'candidate', 'reference', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    report = compare_pair(read_cases(a.cases), a.candidate, a.reference)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['candidateRuntime','referenceRuntime']}, indent=2))
    raise SystemExit(0 if report['passed'] else 1)
