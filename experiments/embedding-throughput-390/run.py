#!/usr/bin/env python3
"""Run on the dedicated Linux worker. Archive even failures; never reinterpret failed gates."""
import argparse,json,pathlib,statistics,subprocess
ROOT=pathlib.Path('/opt/embedding390')
OUT=ROOT/'results'
CANDIDATE_DIRECTORY='candidate'
JAVA=ROOT/'java/bin/java'
def run(arm,model,length=32,mode='latency',batch=1,warmups=5,repetitions=11,label=None):
 name=label or f'{model}-{arm}-{length}-{batch}-{mode}'
 target=OUT/(name+'.json')
 reference=ROOT/'references'/({'eg2':'reference-embeddinggemma2-q8_0.json','eg300':'reference-embeddinggemma-300m-q8_0.json'}[model])
 args=[str(JAVA),'--add-modules=jdk.incubator.vector','--enable-native-access=ALL-UNNAMED','-Xmx4g','-Dvectors.gguf.threads=8','-Dmodels.native.kernels.threads=8','-Dmodels.purejava.maxContextLength=1024',f'-Dembedding.reference={reference}',f'-Dembedding.probes={ROOT}/references/probes.txt',f'-Dembedding.profiler.library={ROOT}/profiler/lib/libasyncProfiler.so','-cp',f'{ROOT}/{CANDIDATE_DIRECTORY if arm=="candidate" else "baseline"}/lib/*:{ROOT}/profiler/lib/async-profiler.jar','EmbeddingProfile',str(ROOT/'models'/(model+'.gguf')),str(target),str(length),str(batch),str(warmups),str(repetitions),'mean',mode]
 (OUT/(name+'-command.json')).write_text(json.dumps(args,indent=2)+'\n')
 print('START',name,flush=True)
 with (OUT/(name+'.log')).open('w') as log:
  subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
 result=json.loads(target.read_text());print('DONE',name,flush=True);return result

def main():
 global OUT,CANDIDATE_DIRECTORY
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--candidate-directory',default='candidate')
 parser.add_argument('--output-directory',default='results')
 config=parser.parse_args()
 CANDIDATE_DIRECTORY=config.candidate_directory
 OUT=ROOT/config.output_directory
 OUT.mkdir(exist_ok=True)
 oracle={}
 for model,arm in [('eg300','baseline'),('eg2','baseline'),('eg2','candidate')]:
  oracle[(model,arm)]=run(arm,model,mode='oracle')
 assert oracle[('eg2','baseline')]['vectorBitsSha256']==oracle[('eg2','candidate')]['vectorBitsSha256'],'Oracle vectors changed'
 for arm in ['baseline','candidate']:
  run(arm,'eg2',mode='trace',warmups=1,repetitions=1)
 timings={}
 for length in [32,128,512]:
  results=[run(a,'eg2',length,label=f'eg2-{a}-{length}') for a in ['baseline-1','candidate','baseline-2']]
  hashes={r['vectorBitsSha256'] for r in results};assert len(hashes)==1,'Timed vectors changed'
  medians=[statistics.median(s['milliseconds'] for s in r['runs'] if s['phase']=='measured') for r in results]
  control=run('baseline','eg300',length)
  control_ms=statistics.median(s['milliseconds'] for s in control['runs'] if s['phase']=='measured')
  noise=abs(medians[0]-medians[2]);gain=min(medians[0],medians[2])-medians[1]
  timings[length]={'baseline1MedianMs':medians[0],'candidateMedianMs':medians[1],'baseline2MedianMs':medians[2],'baselineDriftMs':noise,'conservativeGainMs':gain,'gainExceedsDrift':gain>noise,'default300mMedianMs':control_ms,'candidateVsDefaultRatio':medians[1]/control_ms}
  (OUT/'comparison.json').write_text(json.dumps(timings,indent=2)+'\n')
 for arm in ['baseline','candidate']:
  run(arm,'eg2',length=128,mode='profile',warmups=5,repetitions=20)
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
