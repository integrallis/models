#!/usr/bin/env python3
"""Second diagnostic, after the EmbeddingGemma experiment has exited (no concurrent inference)."""
import hashlib,json,pathlib,statistics,subprocess
ROOT=pathlib.Path('/opt/embedding390')
OUT=ROOT/'qwen-results'
MODEL={'url':'https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/370f27d7550e0def9b39c1f16d3fbaa13aa67728/Qwen3-Embedding-0.6B-Q8_0.gguf','sha256':'06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439'}
def run(arm,length,mode='latency'):
 name=f'qwen-{arm}-{length}-{mode}';output=OUT/(name+'.json')
 cp=ROOT/('qwen-candidate' if arm=='candidate' else 'baseline')/'lib'
 args=[str(ROOT/'java/bin/java'),'--add-modules=jdk.incubator.vector','--enable-native-access=ALL-UNNAMED','-Xmx4g','-Dvectors.gguf.threads=8','-Dmodels.native.kernels.threads=8','-Dmodels.purejava.maxContextLength=1024',f'-Dembedding.reference={ROOT}/references/reference-qwen3-embedding-0.6b-q8_0.json',f'-Dembedding.probes={ROOT}/references/probes.txt','-cp',str(cp)+'/*','EmbeddingProfile',str(ROOT/'models/qwen.gguf'),str(output),str(length),'1','5','11','last-token',mode]
 (OUT/(name+'-command.json')).write_text(json.dumps(args,indent=2)+'\n')
 print('START',name,flush=True)
 with (OUT/(name+'.log')).open('w') as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
 print('DONE',name,flush=True);return json.loads(output.read_text())
def main():
 assert (ROOT/'worker-exit.txt').read_text().strip()=='0','Complete the first experiment before running this one'
 OUT.mkdir(exist_ok=True);(OUT/'model.json').write_text(json.dumps(MODEL,indent=2)+'\n')
 path=ROOT/'models/qwen.gguf'
 subprocess.run(['curl','--retry','3','--connect-timeout','20','--retry-max-time','180','-fsSL',MODEL['url'],'-o',str(path)],check=True)
 assert hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==MODEL['sha256']
 a=run('baseline',32,'oracle');b=run('candidate',32,'oracle');assert a['vectorBitsSha256']==b['vectorBitsSha256'],'Oracle vectors changed'
 run('baseline',32,'trace');run('candidate',32,'trace')
 comparison={}
 for length in [32,128,512]:
  rows=[run(a,length) for a in ['baseline-1','candidate','baseline-2']]
  assert len({r['vectorBitsSha256'] for r in rows})==1,'Timed vectors changed'
  medians=[statistics.median(s['milliseconds'] for s in r['runs'] if s['phase']=='measured') for r in rows]
  drift=abs(medians[0]-medians[2]);gain=min(medians[0],medians[2])-medians[1]
  comparison[length]={'baseline1MedianMs':medians[0],'candidateMedianMs':medians[1],'baseline2MedianMs':medians[2],'baselineDriftMs':drift,'conservativeGainMs':gain,'gainExceedsDrift':gain>drift}
  (OUT/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
if __name__=='__main__':main()
