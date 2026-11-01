"""Independent sklearn audit of retained predictions and official GoEmotions annotations."""
import argparse
import collections
import json
from pathlib import Path
import sklearn
import numpy
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
from common import digest, read_cases


def audit(cases_path, predictions, raw_test, emotions):
    labels=Path(emotions).read_text().splitlines()
    source={}
    for line in Path(raw_test).read_text().splitlines():
        text,numbers,id=line.split('\t')
        source[id]=(text,[labels[int(n)] for n in numbers.split(',')])
    cases=[c for c in read_cases(cases_path) if c['cohort']=='goemotions']
    raw=[json.loads(line) for line in Path(predictions).read_text().splitlines()]
    if not raw or raw[0].get('type')!='ready' or raw[0].get('device')!=['cpu']:
        raise ValueError('missing CPU runtime identity')
    outcomes={r['id']:r for r in raw[1:]}
    if len(outcomes)!=len(raw)-1:
        raise ValueError('duplicate predictions')
    if not {case['id'] for case in cases}<=set(outcomes):
        raise ValueError('missing emotion predictions')
    expected=[];selected=[]
    for case in cases:
        id=case['id'].removeprefix('goemotions-');text,gold=source[id]
        task=case['request']['tasks'][0]
        assert case['request']['text']==text
        assert task['name']=='emotions' and task['multiLabel'] and task['threshold']==.5
        assert [l['name'] for l in task['labels']]==labels
        assert set(case['expected']['emotions'])==set(gold)
        row=outcomes[case['id']]
        assert 'error' not in row
        prediction=row['selected']['emotions']
        if set(row['selected'])!={'emotions'} or not prediction or len(prediction)!=len(set(prediction)) or not set(prediction)<=set(labels):
            raise ValueError('invalid emotion prediction')
        expected.append(gold);selected.append(prediction)
    encoder=MultiLabelBinarizer(classes=labels)
    y=encoder.fit_transform(expected);p=encoder.transform(selected)
    full_counts=collections.Counter(l for _,gold in source.values() for l in gold)
    sample_counts=collections.Counter(l for gold in expected for l in gold)
    precision,recall,f1,support=precision_recall_fscore_support(y,p,average=None,zero_division=0)
    non_neutral=[i for i,label in enumerate(labels) if label!='neutral']
    result=dict(stage='independent-emotion-scoring-audit',cases=len(cases),officialTestCases=len(source),
        sourceSha256=digest(raw_test),caseSha256=digest(cases_path),predictionSha256=digest(predictions),
        sklearnVersion=sklearn.__version__,numpyVersion=numpy.__version__,annotationsAndLabelOrderExact=True,
        labelOrder=labels,
        microF1ExcludingNeutralDiagnostic=float(f1_score(y[:,non_neutral],p[:,non_neutral],average="micro",zero_division=0)),
        perLabel={label:dict(precision=float(precision[i]),recall=float(recall[i]),f1=float(f1[i]),support=int(support[i])) for i,label in enumerate(labels)},
        microF1=float(f1_score(y,p,average='micro',zero_division=0)),
        macroF1=float(f1_score(y,p,average='macro',zero_division=0)),
        samplesF1=float(f1_score(y,p,average='samples',zero_division=0)),
        exactSetAccuracy=float(accuracy_score(y,p)),
        microPrecision=float(precision_score(y,p,average='micro',zero_division=0)),
        microRecall=float(recall_score(y,p,average='micro',zero_division=0)),
        labelPrevalence={l:dict(sample=sample_counts[l]/len(cases),fullTest=full_counts[l]/len(source),
            sampleCount=sample_counts[l],fullTestCount=full_counts[l]) for l in labels},catalogQualified=False)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['cases','predictions','raw-test','emotions','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();r=audit(a.cases,a.predictions,a.raw_test,a.emotions)
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ['labelPrevalence','perLabel','labelOrder']}))
