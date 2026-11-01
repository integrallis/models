"""Account for every completed rental before authorizing another model experiment."""
import json
from pathlib import Path


def prior_rented_seconds(evidence):
    evidence = Path(evidence)
    initial = json.loads((evidence/'deployment.json').read_text())
    total = initial['cleanup']['rentedSecondsUpperBound']
    if type(total) is not int or total < 0:
        raise ValueError('invalid initial rental receipt')
    for pattern, identifier, final in [
        ('gliner25-training-*/control.json', 'podId', 'deleted'),
        ('gliner25-eval-*/control.json', 'instanceId', 'terminated'),
    ]:
        for path in evidence.glob(pattern):
            record = json.loads(path.read_text())
            if identifier not in record:
                continue
            if record.get('state') != final:
                raise ValueError('unresolved rental: '+path.parent.name)
            seconds = record.get('rentedSecondsUpperBound')
            if type(seconds) is not int or seconds <= 0:
                raise ValueError('missing or invalid rental receipt: '+path.parent.name)
            total += seconds
    return total
