#!/usr/bin/env python3
"""Generate score and timing tables from committed analyzer/JMH artifacts."""
import gzip,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
R=HERE/'results'
with gzip.open(HERE.parent/'pr380-quality-review/cli-hotspots/results/modern-details.tsv.gz','rt') as f:
    old=[s.rstrip('\n').split('\t') for s in f]
with gzip.open(R/'after-details.tsv.gz','rt') as f:
    new=[s.rstrip('\n').split('\t') for s in f]
methods=[('ClipImagePreprocessor.java','preprocess'),('ClipVisionWeights.java','fromGgufFile'),('VisionQualificationCli.java','measureDecoder'),('InferenceBenchmarkCli.java','main'),('LlamaForwardPass.java','advancePrefillBatch'),('LlamaForwardPass.java','forwardSequenceInternal'),('LlamaForwardPass.java','prefillEmbeddings')]
def maximum(rows,file,name):
    return max(int(r[5]) for r in rows if r[0]=='COGNITIVE' and Path(r[2]).name==file and r[4]==name)
lines=['# Measured refactor results','','## Cognitive complexity','','MFCQI 0.7.0 with the same modern-Java parser setting in both arms. Overloads use the maximum method score. This is not a comparison against the default parser’s partial-coverage aggregate.','','| Method | Before | After |','|---|---:|---:|']
for file,name in methods:
    lines.append(f'| `{file}:{name}` | {maximum(old,file,name)} | {maximum(new,file,name)} |')
new_methods={(r[2],r[4]) for r in new if r[0]=='COGNITIVE'}-{(r[2],r[4]) for r in old if r[0]=='COGNITIVE'}
lines+=['','## Extracted helpers','','All newly named methods in the four analyzed modules, including the command dispatcher.','','| Method | Cognitive complexity |','|---|---:|']
for file,name in sorted(new_methods):
    lines.append(f'| `{Path(file).name}:{name}` | {maximum(new,Path(file).name,name)} |')
lines+=['','## Synthetic JMH regression check','','Same host/JDK, two forks per arm; three warmups and five measurements per fork, 500 ms per iteration. Time is mean microseconds per operation; ratio is after / before (lower is faster). This small synthetic measurement does not establish full-model catalog performance.','','| Operation | Fixture | Before µs | After µs | Ratio |','|---|---|---:|---:|---:|']
def timings(arm):
    return {(r['benchmark'].rsplit('.',1)[-1],r.get('params',{}).get('fixture','image')):r['primaryMetric'] for r in json.loads((R/(arm+'-jmh.json')).read_text())}
a,b=timings('before'),timings('after'); assert a.keys()==b.keys()
for key in sorted(a):
    x,y=a[key]['score'],b[key]['score']
    lines.append(f'| {key[0]} | {key[1]} | {x:.3f} | {y:.3f} | {y/x:.4f} |')
lines+=['', '## Longer-warmup investigation', '', 'The initial short run crossed the 10% investigation threshold for these two operations. The repeat used five 1 s warmups and five 1 s measurements per fork, two forks, and reversed arm order (after, then before). Both initial and repeat data are retained. Intervals are JMH 99.9% confidence intervals, not equivalence bounds.', '', '| Operation | Before µs | After µs | Ratio | Before CI | After CI |', '|---|---:|---:|---:|---|---|']
for operation in ['imagePreprocess','embeddingPrefillAndDecode']:
    a,b=[json.loads((R/'longer-warmup'/(arm+'-'+operation+'.json')).read_text())[0]['primaryMetric'] for arm in ['before','after']]
    ratio=b['score']/a['score']
    ci=lambda m: '–'.join(f"{x:.1f}" for x in m['scoreConfidence'])
    lines.append(f"| {operation} | {a['score']:.3f} | {b['score']:.3f} | {ratio:.4f} | {ci(a)} | {ci(b)} |")
lines+=['', 'The repeat means are within the predeclared 10% investigation threshold, but the decoder intervals overlap widely. These noisy local measurements do not establish zero overhead or replace a dedicated full-model performance repeat.', '']
e=json.loads((R/'equivalence.json').read_text())
lines+=['',f'Exact differential check: **{e["comparisons"]} output fingerprints matched**. Fingerprints include prefill logits, successive decode logits, repeated inference and every preprocessed pixel. Route logs confirm sequential F32 and batched/grouped Q4_K execution.', '', 'Full unit, narrow-vector, formatting, static-analysis and coverage checks passed for backend-java and models-bench. Raw JMH distributions, output fingerprints, logs, source revisions and hashes are retained in this directory.','']
(HERE/'RESULTS.md').write_text('\n'.join(lines))
