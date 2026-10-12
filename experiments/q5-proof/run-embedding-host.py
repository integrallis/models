import json,os,pathlib,subprocess,time,urllib.request,hashlib
out=pathlib.Path('/work/out');commands=[]
def run(name,cmd,timeout):
 (out/'STATUS').write_text(name+'\n');start=time.monotonic()
 with (out/(name+'.log')).open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
 commands.append(dict(name=name,command=cmd,exitCode=r.returncode,wallSeconds=time.monotonic()-start));(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
 subprocess.run(['aws','s3','cp',str(out),os.environ['RESULT_URI'],'--recursive','--only-show-errors'],check=True)
 assert r.returncode==0,name
run('rust-build',['/root/.cargo/bin/cargo','build','--release','--manifest-path','rust/Cargo.toml'],300)
model=pathlib.Path('/work/model.gguf');url='https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/544f204f2eaa2d71361ffc74d6df7170285b286a/all-MiniLM-L6-v2-Q5_K_S.gguf';urllib.request.urlretrieve(url,model)
assert hashlib.sha256(model.read_bytes()).hexdigest()=='0de5c3d1343904debb305a2f6c55f57ccdc68786968b5d4726dd404f6c4c4244'
base=['/opt/jdk/bin/java','--add-modules','jdk.incubator.vector','--enable-native-access=ALL-UNNAMED','-Xmx2g','-Dmodels.native.kernels.threads=16','-Dvectors.gguf.threads=16','-cp','probe.jar:lib/*','experiments.q5.EmbeddingRouteProof']
arms=['java','parallel','native']
for arm in arms:run('correctness-'+arm,base+[arm,str(model),'rust/target/release/libjmodels_kernels.so','data/cases.json','correctness','1'],360)
reference=None
for arm in arms:
 rows=[json.loads(s) for s in (out/('correctness-'+arm+'.log')).read_text().splitlines() if s.startswith('{')];actual={r['id']:r['values'] for r in rows if r['kind']=='embedding'}
 if reference is None:reference=actual
 else:assert actual==reference,'embedding parity '+arm
for pair in range(6):
 for step in range(3):
  arm=arms[(pair+step)%3];run('timing-'+str(pair)+'-'+arm,base+[arm,str(model),'rust/target/release/libjmodels_kernels.so','data/timing.json','timing','3'],120)
(out/'STATUS').write_text('COMPLETE\n')
