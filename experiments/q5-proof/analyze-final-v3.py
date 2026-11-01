import pathlib,json,math,statistics,random,re,sys
root=pathlib.Path(sys.argv[1]);out=[]
def rows(f):return [json.loads(s) for s in f.read_text().splitlines() if s.startswith('{')]
def ci(values):
 rng=random.Random(724);means=[math.log(v) for v in values];boot=sorted(math.exp(statistics.mean(rng.choices(means,k=len(means)))) for _ in range(10000));return {'pairs':len(values),'ratio':math.exp(statistics.mean(means)),'ci95':[boot[250],boot[9750]]}
for arch in ('x86','arm'):
 for phase,kind,baseline,candidate,metrics in [('bert','embedding','baseline','candidate',['nanos','allocatedBytes']),('causal','request','java','native',['prefillNanos','decodeNanos','modelNanos','javaAllocatedBytes'])]:
  vals={m:[] for m in metrics+['loadNanos','peakRssKb']}
  for i in range(10):
   files=[root/arch/f'{phase}-timing-{i}-{a}.log' for a in (baseline,candidate)]
   if not all(f.exists() and 'Maximum resident set size' in f.read_text() for f in files):continue
   data=[rows(f) for f in files];requests=[{(r['id'],r.get('repeat',0)):r for r in d if r['kind']==kind and r.get('repeat',0)>=0} for d in data]
   assert requests[0].keys()==requests[1].keys()
   for m in metrics:vals[m].append(math.exp(statistics.mean(math.log(requests[1][k][m]/requests[0][k][m]) for k in requests[0])))
   loads=[next(r for r in d if r['kind']=='load') for d in data];key='nanos' if 'nanos' in loads[0] else 'modelLoadNanos';vals['loadNanos'].append(loads[1][key]/loads[0][key])
   rss=[int(re.search(r'Maximum resident set size \(kbytes\): (\d+)',f.read_text())[1]) for f in files];vals['peakRssKb'].append(rss[1]/rss[0])
  for metric,v in vals.items():
   if v:out.append(dict(arch=arch,phase=phase,metric=metric,**ci(v)))
print(json.dumps(out,indent=2))
