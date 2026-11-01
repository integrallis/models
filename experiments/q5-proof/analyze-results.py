"""Process-clustered paired summaries; raw request observations remain authoritative."""
import collections,hashlib,json,math,pathlib,random,statistics,sys
root=pathlib.Path(sys.argv[1]);output=pathlib.Path(sys.argv[2]);output.mkdir(parents=True,exist_ok=True)
def load(phase,arch,arm,kind,prefix='timing'):
 result={}
 for file in (root/('remote-'+phase)/arch).glob(prefix+'-*-'+arm+'.log'):
  number=int(file.name[len(prefix)+1:].split('-')[0])
  for line in file.read_text().splitlines():
   if not line.startswith('{'):continue
   row=json.loads(line)
   if row.get('kind')!=kind or row.get('repeat',0)<0:continue
   result[(number,row['id'],row.get('repeat',0))]=row
 return result
def compare(phase,arch,base,candidate,kind,metric,prefix='timing'):
 left=load(phase,arch,base,kind,prefix);right=load(phase,arch,candidate,kind,prefix);clusters=collections.defaultdict(list)
 for key,row in left.items():
  if key in right:clusters[key[0]].append(math.log(right[key][metric]/row[metric]))
 means=[statistics.mean(values) for _,values in sorted(clusters.items())]
 if not means:return None
 rng=random.Random(724);boot=sorted(math.exp(statistics.mean(rng.choices(means,k=len(means)))) for _ in range(10000))
 return dict(phase=phase,arch=arch,baseline=base,candidate=candidate,metric=metric,pairs=len(means),requests=sum(map(len,clusters.values())),geometricRatio=math.exp(statistics.mean(means)),bootstrap95=[boot[250],boot[9750]],resampling='whole process-pair clusters; seed724;10000draws')
results=[]
for arch in ('x86','arm'):
 for metric in ('modelNanos','decodeNanos'):
  results.append(compare('q50-wholemodel-v1',arch,'baseline','candidate','request',metric))
 for metric in ('nanos','allocatedBytes'):results.append(compare('q51-simd-v2',arch,'native','java','embedding',metric))
 for metric in ('nanos','allocatedBytes'):results.append(compare('q5-final-v1',arch,'baseline','candidate','embedding',metric,'bert-timing'))
 for metric in ('modelNanos','decodeNanos','javaAllocatedBytes'):results.append(compare('q5-final-v1',arch,'java','native','request',metric,'causal-timing'))
for arm in ('dual','triple'):
 for metric in ('modelNanos','decodeNanos'):results.append(compare('q50-ablation-v2','arm','baseline',arm,'request',metric))
(output/'paired-summary.json').write_text(json.dumps([r for r in results if r],indent=2)+'\n')
files=[]
for phase in ['q5-screen-v1','q5-embedding-v1','q50-wholemodel-v1','q50-ablation-v2','q51-simd-v1','q51-simd-v2','q5-final-v1']:
 for file in sorted((root/('remote-'+phase)).rglob('*')):
  if file.is_file():files.append(hashlib.sha256(file.read_bytes()).hexdigest()+'  '+str(file.relative_to(root)))
(output/'raw-files.sha256').write_text('\n'.join(files)+'\n')
