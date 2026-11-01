"""Account for every completed rental before authorizing another model experiment."""
import json
from pathlib import Path
from datetime import datetime, timezone


def aggregate_limit_seconds():
    path = Path(__file__).parent/'continuation-budget.json'
    if not path.exists():
        return 7200
    record = json.loads(path.read_text())
    return validate_limit(record)


def validate_limit(record):
    limit=record['previousAggregateLimitSeconds']
    additions=[dict(previousLimitSeconds=limit,additionalAllocationSeconds=record['additionalAllocationSeconds'],
                    newLimitSeconds=limit+record['additionalAllocationSeconds'])]+record.get('continuations',[])
    for addition in additions:
        extra=addition['additionalAllocationSeconds']
        if type(extra) is not int or extra<=0 or addition['previousLimitSeconds']!=limit:
            raise ValueError('invalid continuation budget chain')
        limit+=extra
        if addition['newLimitSeconds']!=limit:raise ValueError('invalid continuation increment')
    if type(record['maximumAggregateRentedSeconds']) is not int or record['maximumAggregateRentedSeconds']!=limit:
        raise ValueError('invalid continuation budget')
    return limit


def prior_rented_seconds(evidence, reserve_active=False):
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
                if reserve_active:
                    deadline=datetime.fromisoformat(record['deletionDeadline'])
                    seconds=record.get('maximumRentalSeconds')
                    if deadline<=datetime.now(timezone.utc) or type(seconds) is not int or seconds<=0:
                        raise ValueError('invalid or expired active reservation: '+path.parent.name)
                    total+=seconds
                    continue
                raise ValueError('unresolved rental: '+path.parent.name)
            seconds = record.get('rentedSecondsUpperBound')
            if type(seconds) is not int or seconds <= 0:
                raise ValueError('missing or invalid rental receipt: '+path.parent.name)
            total += seconds
    return total
