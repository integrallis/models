"""Actual composed GDN artifacts, a one-method recurrence control, and retained publisher baseline."""
import argparse,hashlib,json,os,shutil,signal,subprocess,tarfile,time,urllib.request,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
P=json.loads((HERE/'protocol.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def entries(p):
 with zipfile.ZipFile(p) as z:
  assert z.testzip() is None
  return {n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if not n.endswith('/')}
def execute(command,prefix,timeout):
 assert not prefix.with_suffix('.execution.json').exists(),'never overwrite an observation'
 start=time.monotonic();capped=False
 with prefix.with_suffix('.stdout').open('w') as stdout,prefix.with_suffix('.stderr').open('w') as stderr:
  p=subprocess.Popen(command,stdout=stdout,stderr=stderr,start_new_session=True)
  try:code=p.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   capped=True;os.killpg(p.pid,signal.SIGTERM)
   try:code=p.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
 save(prefix.with_suffix('.execution.json'),{'command':command,'exitCode':code,'timedOut':capped,'wallSeconds':time.monotonic()-start})
 return code==0 and not capped

def ablate_provider(source):
 old='public boolean supportsGatedDeltaNet() {\n    return gatedDeltaNetSupported;\n  }'
 assert source.count(old)==1,'provider ablation no longer matches reviewed source'
 return source.replace(old,'public boolean supportsGatedDeltaNet() {\n    return false;\n  }')

def prepare(args,out):
 models=args.models.resolve();core=args.core.resolve()
 assert subprocess.check_output(['git','-C',str(models),'rev-parse','HEAD'],text=True).strip()==P['modelsRevision']
 assert sha(core)==P['vectorsSha256']
 assert sha(HERE/'ProductionRouteProof.java')==P['sourceSha256']
 assert sha(HERE/'data/tasks.json')==P['dataSha256']
 source=models/'backend-native/build/paired-runtime';native=models/'backend-native/build/rust-target/release/libjmodels_kernels.so'
 assert native.is_file()
 for arm in ('native','baseline'):
  folder=out/'lib'/arm;folder.mkdir(parents=True)
  for p in source.glob('*.jar'):
   if p.name.startswith('vectors-core-'):p=core
   shutil.copy2(p,folder/p.name)
 payload=out/'published-reference.tar.gz'
 urllib.request.urlretrieve(os.environ['GDN_REFERENCE_URL'],payload)
 assert sha(payload)==P['historicalPublishedPayloadSha256']
 reference=out/'historical-reference';reference.mkdir()
 with tarfile.open(payload) as archive:
  metadata=[m.name for m in archive.getmembers() if any(p.startswith('._') or p=='__MACOSX' for p in Path(m.name).parts)]
  archive.extractall(reference,members=[m for m in archive.getmembers() if m.name not in metadata],filter='data')
 shutil.copytree(reference/'lib/published',out/'lib/published')
 sourcefile=models/'backend-native/src/main/java/com/integrallis/models/backend/nativekernel/RustGgufBatchedMatrixKernel.java'
 patched=out/'RustGgufBatchedMatrixKernel.java';patched.write_text(ablate_provider(sourcefile.read_text()))
 classes=out/'ablation-classes';classes.mkdir()
 subprocess.run(['javac','--release','25','--add-modules','jdk.incubator.vector','-cp',str(out/'lib/native/*'),'-d',str(classes),str(patched)],check=True)
 target=next((out/'lib/baseline').glob('backend-native-*.jar'));original=target.read_bytes();tmp=target.with_suffix('.tmp')
 classname='com/integrallis/models/backend/nativekernel/RustGgufBatchedMatrixKernel.class'
 with zipfile.ZipFile(target) as old,zipfile.ZipFile(tmp,'w') as new:
  for entry in old.infolist():new.writestr(entry,(classes/classname).read_bytes() if entry.filename==classname else old.read(entry))
 tmp.replace(target)
 nativejar=next((out/'lib/native').glob('backend-native-*.jar'));before,after=entries(nativejar),entries(target)
 assert {n for n in before.keys()|after.keys() if before.get(n)!=after.get(n)}=={classname}
 probe=out/'classes';probe.mkdir()
 subprocess.run(['javac','--release','25','--add-modules','jdk.incubator.vector','-cp',str(out/'lib/native/*'),'-d',str(probe),str(HERE/'ProductionRouteProof.java')],check=True)
 model=next(m for m in P['models'] if m['key']==args.model);weights=out/'model.gguf'
 urllib.request.urlretrieve(model['downloadUri'],weights)
 assert weights.stat().st_size==model['sizeBytes'] and sha(weights)==model['sha256']
 manifest={'protocol':P,'model':args.model,'java':subprocess.check_output(['java','-version'],stderr=subprocess.STDOUT,text=True),'cpu':subprocess.check_output(['lscpu'],text=True),'affinity':sorted(os.sched_getaffinity(0)),'memory':Path('/proc/meminfo').read_text(),'nativeSha256':sha(native),'publishedNativeSha256':sha(reference/'rust/published/target/release/libjmodels_kernels.so') if (reference/'rust/published/target/release/libjmodels_kernels.so').exists() else None,'metadataExcluded':metadata,'artifacts':{a:{j.name:{'sha256':sha(j),'entries':entries(j)} for j in (out/'lib'/a).glob('*.jar')} for a in ('native','baseline','published')},'ablationClassOnly':classname,'sourceSha256':sha(sourcefile),'patchedSourceSha256':sha(patched)}
 # The historical archive contains source, not a host-specific native binary.
 published_rust=reference/'rust/published'
 subprocess.run(['cargo','+1.96.0','build','--locked','--release','--manifest-path',str(published_rust/'Cargo.toml')],check=True)
 published_native=published_rust/'target/release/libjmodels_kernels.so'
 manifest['publishedNativeSha256']=sha(published_native)
 expected=json.loads((HERE.parent/'bonsai-cold'/f'expected-{args.arch}.json').read_text())
 assert {n:v['entries'] for n,v in manifest['artifacts']['native'].items()}==expected['jarEntries']['candidate']
 assert manifest['nativeSha256']==expected['nativeSha256']
 save(out/'provenance.json',manifest)
 save(out/'prepared.json',{'native':str(native),'publishedNative':str(published_native),'model':str(weights)})

def records(path):return [json.loads(l) for l in path.read_text().splitlines() if l.startswith('{')]
def signatures(rows):
 requests=[r for r in rows if r['kind']=='request']
 assert len(requests)==P['cases'] and len({r['id'] for r in requests})==P['cases']
 assert all(len(r['fullLogitSha256'])==P['vectorsPerCase'] and len(r['greedyTokens'])==P['continuationTokens'] for r in requests)
 return {r['id']:(r['inputTokens'],r['fullLogitSha256'],r['greedyTokens'],r['prediction']) for r in requests}
def analyze(out):
 for arm in ('native','baseline','published'):
  execution=json.loads((out/f'cohort-{arm}.execution.json').read_text())
  assert execution['exitCode']==0 and not execution['timedOut'],'incomplete cohort execution'
 for arm in ('native','baseline'):
  execution=json.loads((out/f'routing-{arm}.execution.json').read_text())
  assert execution['exitCode']==0 and not execution['timedOut'],'current routing gate failed'
 rows={a:records(out/f'cohort-{a}.stdout') for a in ('native','baseline','published')}
 sig={a:signatures(v) for a,v in rows.items()};assert sig['native']==sig['baseline'],'current native and Java recurrence numerical difference'
 quality={}
 for arm,values in rows.items():
  counts={}
  for r in values:
   if r['kind']=='request':
    d=counts.setdefault(r['dataset'],{'correct':0,'total':0});d['correct']+=int(r['prediction']==r['expected']);d['total']+=1
  quality[arm]=counts
 assert quality['native']==quality['baseline']
 gate=all(quality['native'][d]['correct']>=v['correct'] for d,v in quality['published'].items())
 differences=[{'id':i,'current':sig['native'][i],'published':sig['published'][i]} for i in sig['native'] if sig['native'][i]!=sig['published'][i]]
 published_route=json.loads((out/'routing-published.execution.json').read_text())
 result={'publishedRoutingPass':published_route['exitCode']==0 and not published_route['timedOut'],'currentRecurrenceExact':True,'cases':P['cases'],'quality':quality,'publishedQualityNonRegression':gate,'publishedDifferences':differences,'scope':'Numerical and task-quality only. No performance gate inferred.'}
 save(out/'summary.json',result);return result

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','baseline','native','published','analyze']);p.add_argument('--out',type=Path,required=True);p.add_argument('--arch',choices=['x86','arm']);p.add_argument('--models',type=Path);p.add_argument('--core',type=Path);p.add_argument('--model',choices=['qwen-0.8b','qwen-4b']);a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
 if a.stage=='prepare':prepare(a,out);return
 if a.stage=='analyze':assert analyze(out)['publishedQualityNonRegression'],'task quality regressed';return
 state=json.loads((out/'prepared.json').read_text());native=state['publishedNative'] if a.stage=='published' else state['native']
 base=['java','--add-modules','jdk.incubator.vector','--enable-native-access=ALL-UNNAMED','-Xms2g','-Xmx10g','-cp',f'{out}/lib/{a.stage}/*:{out}/classes','com.integrallis.models.backend.purejava.qwen35.ProductionRouteProof']
 args=[state['model'],native,str(HERE/'data/tasks.json')]
 routed=execute([*base,'routing',*args,str(a.stage!='baseline').lower()],out/f'routing-{a.stage}',P['processTimeoutSeconds'])
 if a.stage!='published':assert routed,'current routing correctness failed'
 assert execute([*base,'cohort',*args,'0'],out/f'cohort-{a.stage}',P['processTimeoutSeconds'])
 signatures(records(out/f'cohort-{a.stage}.stdout'))
if __name__=='__main__':main()
