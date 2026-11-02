"""Fail closed on missing evidence, mismatched snapshots or non-finite/unequal outputs."""
import argparse
import json
import math
from pathlib import Path

from common import HERE, read_cases


def compare(cases, candidate, reference):
    policy = json.loads((HERE / 'policy.json').read_text())['parity']
    def load(path):
        rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
        ready, rows = rows[0], rows[1:]
        if ready.get('type') != 'ready' or ready.get('device') != ['cpu']:
            raise ValueError('missing CPU attestation')
        by_id = {row['id']: row for row in rows}
        if len(by_id) != len(rows) or set(by_id) != {row['id'] for row in cases}:
            raise ValueError('missing/extra/duplicate outcomes')
        return ready, by_id
    ca, left = load(candidate); ra, right = load(reference)
    if ca['snapshot'] != ra['snapshot'] or ca['threads'] != ra['threads']:
        raise ValueError('snapshot or thread budgets differ')
    failures = []; max_error = 0.0
    for case in cases:
        a, b = left[case['id']], right[case['id']]
        if 'error' in a or 'error' in b:
            failures.append(dict(id=case['id'], reason='worker error', candidate=a, reference=b)); continue
        for key in ['tokenIds', 'selected']:
            if key not in a or key not in b or a[key] != b[key]:
                failures.append(dict(id=case['id'], reason=key + ' mismatch'))
        x, y = a.get('logits', []), b.get('logits', [])
        if not x or len(x) != len(y):
            failures.append(dict(id=case['id'], reason='logit shape mismatch')); continue
        for i, (v, w) in enumerate(zip(x, y)):
            error = abs(v-w); max_error = max(max_error, error)
            if not math.isfinite(v) or not math.isfinite(w) or error > policy['atol'] + policy['rtol']*abs(w):
                failures.append(dict(id=case['id'], reason='logit tolerance', index=i, candidate=v, reference=w))
    return dict(policy='schema-classification-v1', stage='implementation-parity',
                passed=not failures, maxAbsoluteLogitError=max_error, cases=len(cases), failures=failures,
                catalogQualified=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['cases','candidate','reference','output']: p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); report = compare(read_cases(a.cases), a.candidate, a.reference)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['passed'] else 1)
