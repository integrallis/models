"""Development-only classification diagnostics; never reads qualification inputs."""
import collections


def measure_development(model, cases, bf16):
    import torch
    counts = collections.defaultdict(lambda: collections.defaultdict(lambda:[0,0,0]))
    correct = collections.Counter()
    totals = collections.Counter()
    with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16):
        for case in cases:
            request=case['request']
            schema={'classifications':[dict(task=t['name'],labels=[l['name'] for l in t['labels']],
                multi_label=t['multiLabel'],cls_threshold=t['threshold']) for t in request['tasks']]}
            result=model.extract(request['text'],schema,format_results=False,include_confidence=True)
            task=request['tasks'][0]; value=result[task['name']]
            prediction={v[0] for v in value} if task['multiLabel'] else {value[0]}
            truth=set(case['expected'][task['name']]); cohort=case['cohort']
            correct[cohort]+=prediction==truth; totals[cohort]+=1
            names={label['name'] for label in task['labels']}
            if not prediction or not prediction<=names:
                raise ValueError('invalid development prediction')
            for label in names:
                row=counts[cohort][label]
                row[0]+=label in truth and label in prediction
                row[1]+=label not in truth and label in prediction
                row[2]+=label in truth and label not in prediction
    def f1(row):
        tp,fp,fn=row
        return 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
    metrics={}
    for cohort,labels in counts.items():
        prefix='dev_'+cohort.replace('-','_')+'_'
        metrics[prefix+'accuracy']=correct[cohort]/totals[cohort]
        metrics[prefix+'micro_f1']=f1([sum(row[i] for row in labels.values()) for i in range(3)])
        metrics[prefix+'macro_f1']=sum(map(f1,labels.values()))/len(labels)
    return metrics
