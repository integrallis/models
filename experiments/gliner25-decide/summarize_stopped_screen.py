"""Retain completed buckets from an early-stopped screen without claiming full coverage."""
import argparse
import collections
import json
from pathlib import Path
import tempfile

from common import HERE, digest, read_cases
from compare_performance import compare_pair


def summarize(root):
    case_path = HERE/'performance-cases.jsonl'
    cases = read_cases(case_path)
    expected = {case['id'] for case in cases}
    grouped = collections.defaultdict(set)
    for case in cases:
        grouped[case['bucket']].add(case['id'])
    arms = {}
    sources = {}
    for name in ['models', 'reference']:
        path = root/f'dense-performance-{name}.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        by_id = {row['id']:row for row in rows[1:]}
        if len(by_id) != len(rows)-1 or not set(by_id) <= expected:
            raise ValueError('duplicate or unexpected '+name+' outcomes')
        if any('error' in row for row in by_id.values()):
            raise ValueError('inference error in '+name)
        arms[name] = (rows[0], by_id)
        sources[path.name] = digest(path)
    previous = 0
    if arms['models'][0]['backend'] != 'java-deberta-rust-dense-tokenizer':
        raise ValueError('experimental backend was not executed')
    for row in arms['models'][1].values():
        current = row['denseProjectionCalls']
        if current <= previous:
            raise ValueError('native projection calls did not increase')
        previous = current
    completed = [name for name,ids in grouped.items()
                 if all(ids <= set(arm[1]) for arm in arms.values())]
    if not completed:
        raise ValueError('no complete paired buckets')
    chosen = [case for case in cases if case['bucket'] in completed]
    with tempfile.TemporaryDirectory() as temp:
        paths = {}
        for name,(ready,rows) in arms.items():
            path = Path(temp)/name
            path.write_text(''.join(json.dumps(row)+'\n' for row in [ready]+[rows[c['id']] for c in chosen]))
            paths[name] = path
        report = compare_pair(chosen,paths['models'],paths['reference'])
    if report['passed']:
        raise ValueError('no completed bucket justifies a relative-gate early stop')
    report.update(stage='early-stopped-performance-screen',passed=False,
        fullCoverage=len(completed)==len(grouped),
        completedBuckets=completed,incompleteBuckets=[name for name in grouped if name not in completed],
        recordedRequests={name:len(arm[1]) for name,arm in arms.items()},
        nativeProjectionCalls=previous,
        stopReason='completed bucket failed fixed relative gate; remaining buckets not evaluated',
        sources={**sources,'performance-cases.jsonl':digest(case_path),
                 'summarize_stopped_screen.py':digest(Path(__file__))},catalogQualified=False)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    args = parser.parse_args()
    report = summarize(args.root)
    (args.root/'stopped-performance-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ['completedBuckets','incompleteBuckets','failures','fullCoverage','catalogQualified']}))
