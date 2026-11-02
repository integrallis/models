"""Measure the effect of joint versus publisher-example single-head requests."""
import argparse
import collections
import json
from pathlib import Path

from common import digest, read_cases


def outcomes(path, cases):
    raw = [json.loads(line) for line in path.read_text().splitlines()]
    rows = {row['id']: row for row in raw[1:]}
    if (raw[0].get('type') != 'ready' or raw[0].get('device') != ['cpu']
            or len(rows) != len(raw)-1 or set(rows) != {case['id'] for case in cases}
            or any('error' in row for row in rows.values())):
        raise ValueError('incomplete or invalid results')
    return raw[0], rows


def compare(joint_cases, single_cases, joint_results, single_results):
    joint = read_cases(joint_cases)
    single = read_cases(single_cases)
    a_ready, a = outcomes(joint_results, joint)
    b_ready, b = outcomes(single_results, single)
    if a_ready['snapshot'] != b_ready['snapshot']:
        raise ValueError('snapshot identity differs')
    by_id = {case['id']: case for case in single}
    domains = collections.defaultdict(lambda: dict(heads=0, changedHeads=0,
        jointCorrect=0, singleCorrect=0, improved=0, regressed=0))
    controls = dict(heads=0, disagreements=0)
    matched = set()
    for case in joint:
        for index, task in enumerate(case['request']['tasks']):
            id = case['id']+'-head-'+str(index)
            paired = by_id[id]
            name = task['name']
            if (paired['request'] != dict(text=case['request']['text'], tasks=[task])
                    or paired['expected'] != {name: case['expected'][name]}):
                raise ValueError('input changed beyond head composition')
            matched.add(id)
            before = set(a[case['id']]['selected'][name])
            after = set(b[id]['selected'][name])
            truth = set(case['expected'][name])
            row = domains[case['cohort']]
            row['heads'] += 1
            row['changedHeads'] += before != after
            row['jointCorrect'] += before == truth
            row['singleCorrect'] += after == truth
            row['improved'] += before != truth and after == truth
            row['regressed'] += before == truth and after != truth
            if len(case['request']['tasks']) == 1:
                controls['heads'] += 1
                controls['disagreements'] += before != after
    if matched != set(by_id):
        raise ValueError('unmatched single-head inputs')
    return dict(stage='publisher-request-composition-audit', domains=dict(domains),
        unchangedSingleHeadControls=controls,
        rawSha256={str(path): digest(path) for path in [joint_cases, single_cases, joint_results, single_results]},
        performanceClaim=False, catalogQualified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['joint-cases', 'single-cases', 'joint-results', 'single-results', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.joint_cases, args.single_cases, args.joint_results, args.single_results)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
