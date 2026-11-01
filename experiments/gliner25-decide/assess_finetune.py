"""Fresh holdout, base regression, and matched-runtime quality gates for a new checkpoint."""
import argparse
import json
from pathlib import Path

from assess_quality import assess, measure
from common import HERE, digest, read_cases


def assess_variant(cases, candidate, reference, base, regression):
    policy = json.loads((HERE/'finetune-policy.json').read_text())
    gate = policy['quality']
    models, publisher, original = [measure(cases,p) for p in [candidate,reference,base]]
    if models['runtime']['snapshot'] != publisher['runtime']['snapshot']:
        raise ValueError('Models and publisher did not load the same merged checkpoint')
    if models['runtime']['snapshot']['id'] != policy['candidate']:
        raise ValueError('wrong fine-tuned candidate identity')
    if original['runtime']['snapshot'] != json.loads((HERE/'snapshot.json').read_text()):
        raise ValueError('wrong original checkpoint')
    old = assess(read_cases(HERE/'quality-cases.jsonl'),regression)
    if old['runtime']['snapshot'] != models['runtime']['snapshot']:
        raise ValueError('regression results refer to another checkpoint')
    required = {'boolq','goemotions'} | {'clinc-'+domain for domain in [
        'banking','credit_cards','auto_and_commute','travel','home','work','kitchen_and_dining','small_talk','utility','meta']}
    if set(models['cohorts']) != required:
        raise ValueError('fresh cohort coverage differs')
    failures = ['original-suite: '+f for f in old['failures']]
    for arm in [models,publisher,original]:
        for name,metrics in arm['cohorts'].items():
            if metrics['errors']: failures.append(name+': inference errors')
    for name,metrics in models['cohorts'].items():
        floors = {'accuracy':gate['boolqAccuracy']} if name=='boolq' else (
            {'microF1':gate['goEmotionsMicroF1'],'macroF1':gate['goEmotionsMacroF1']} if name=='goemotions'
            else {'macroF1':gate['eachClincDomainMacroF1']})
        for metric,floor in floors.items():
            if metrics[metric]<floor: failures.append(f'{name}.{metric}: {metrics[metric]:.6f} < {floor}')
        for metric in ['accuracy','microF1','macroF1']:
            if original['cohorts'][name][metric]-metrics[metric]>gate['maximumBaseAccuracyOrF1RegressionPerCohort']:
                failures.append(name+'.'+metric+': regression against base')
            if publisher['cohorts'][name][metric]-metrics[metric]>.005:
                failures.append(name+'.'+metric+': loss against matched publisher')
    return dict(policy='gliner-general-v1',stage='fine-tuned-quality',passed=not failures,
                candidate=models,reference=publisher,base=original,regression=old,
                failures=failures,catalogQualified=False)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['candidate','reference','base','regression','output']:
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    path=HERE/'finetune-qualification-cases.jsonl'
    expected=json.loads((HERE/'finetune-splits.json').read_text())['files']['qualification-cases.jsonl']['sha256']
    if digest(path)!=expected:raise ValueError('fresh cohort digest mismatch')
    report=assess_variant(read_cases(path),args.candidate,args.reference,args.base,args.regression)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['candidate','reference','base','regression']},indent=2))
    raise SystemExit(0 if report['passed'] else 1)
