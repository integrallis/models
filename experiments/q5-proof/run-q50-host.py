import json,os,pathlib,subprocess,time,urllib.request,hashlib
out=pathlib.Path('/work/out');commands=[]
def run(name,cmd,timeout):
 (out/'STATUS').write_text(name+'\n');start=time.monotonic()
 with (out/(name+'.log')).open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
 commands.append(dict(name=name,command=cmd,exitCode=r.returncode,wallSeconds=time.monotonic()-start));(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
 subprocess.run(['aws','s3','cp',str(out),os.environ['RESULT_URI'],'--recursive','--only-show-errors'],check=True)
 assert r.returncode==0,name
run('rust-build',['/root/.cargo/bin/cargo','build','--release','--manifest-path','rust/Cargo.toml'],300)
model=pathlib.Path('/work/model.gguf');urllib.request.urlretrieve('https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-q5_0.gguf',model)
assert hashlib.sha256(model.read_bytes()).hexdigest()=='925946ab27801fea3eead42933f857b6449c6d299f062e374cee2ac48426f2a5'
base=['/opt/jdk/bin/java','--add-modules','jdk.incubator.vector','--enable-native-access=ALL-UNNAMED','-Xmx3g','-Dmodels.native.kernels.threads=16','-Dvectors.gguf.threads=16']
def cmd(arm,phase,value):return base+['-cp',arm+'.jar:probe.jar:lib/*','experiments.q5.Q50RouteProof',phase,str(model),'rust/target/release/libjmodels_kernels.so','tasks.json',str(value)]
for arm in ['baseline','candidate']:run('routing-'+arm,cmd(arm,'routing','true' if arm=='candidate' else 'false'),180)
for arm in ['baseline','candidate']:run('correctness-'+arm,cmd(arm,'cohort',0),900)
def rows(file):return [json.loads(s) for s in file.read_text().splitlines() if s.startswith('{')]
a={r['id']:r for r in rows(out/'correctness-baseline.log') if r['kind']=='request'};b={r['id']:r for r in rows(out/'correctness-candidate.log') if r['kind']=='request'}
assert a.keys()==b.keys()
for key in a:assert a[key]['fullLogitSha256']==b[key]['fullLogitSha256'],key
for pair in range(10):
 for arm in (['baseline','candidate'] if pair%2==0 else ['candidate','baseline']):run('timing-'+str(pair)+'-'+arm,cmd(arm,'timing',pair),150)
(out/'STATUS').write_text('COMPLETE\n')
