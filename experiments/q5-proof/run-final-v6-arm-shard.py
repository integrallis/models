"""Final BERT fix and prior-native Q5 route control; diagnostics stay outside decisive timing."""
import json,os,pathlib,subprocess,time,urllib.request,hashlib,platform
out=pathlib.Path('/work/out');commands=[]
def run(name,cmd,timeout):
 (out/'STATUS').write_text(name+'\n');start=time.monotonic()
 with (out/(name+'.log')).open('w') as log:r=subprocess.run(['/usr/bin/time','-v',*cmd],stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
 commands.append(dict(name=name,command=cmd,exitCode=r.returncode,wallSeconds=time.monotonic()-start));(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');subprocess.run(['aws','s3','cp',str(out),os.environ['RESULT_URI'],'--recursive','--only-show-errors'],check=True);assert r.returncode==0,name
def file_sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as source:
  for chunk in iter(lambda:source.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def download(url,path,sha):
 urllib.request.urlretrieve(url,path);assert file_sha(path)==sha
def rows(name):return [json.loads(s) for s in (out/(name+'.log')).read_text().splitlines() if s.startswith('{')]
base=['/opt/jdk/bin/java','--add-modules','jdk.incubator.vector','--enable-native-access=ALL-UNNAMED','-Xmx4g','-Dmodels.native.kernels.threads=16','-Dvectors.gguf.threads=16']
run('rust-build',['/root/.cargo/bin/cargo','build','--locked','--release','--manifest-path','rust/Cargo.toml'],300)
pathlib.Path('oracle-source').mkdir();run('oracle-extract',['tar','xf','oracle-source.tar','-C','oracle-source'],60)
run('oracle-configure',['cmake','-S','oracle-source','-B','oracle-build','-DCMAKE_BUILD_TYPE=Release','-DBUILD_SHARED_LIBS=OFF','-DGGML_NATIVE=OFF','-DGGML_CUDA=OFF','-DGGML_METAL=OFF','-DLLAMA_CURL=OFF','-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_SERVER=OFF'],120)
run('oracle-build',['cmake','--build','oracle-build','--target','llama-quantize','-j','8'],600)
download('https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-q5_0.gguf','source-q50.gguf','925946ab27801fea3eead42933f857b6449c6d299f062e374cee2ac48426f2a5')
run('convert-q51',['oracle-build/bin/llama-quantize','--allow-requantize','--pure','--output-tensor-type','q5_1','--token-embedding-type','q5_1','source-q50.gguf','causal-q51.gguf','Q5_1','8'],300)
(out/'derived-fixture.json').write_text(json.dumps(dict(q51Sha256=file_sha('causal-q51.gguf'),bytes=pathlib.Path('causal-q51.gguf').stat().st_size),indent=2)+'\n')
def causal(arm,phase,value):
 cp=('prior-native.jar:' if arm=='prior' else '')+'candidate.jar:probe.jar:lib/*'
 return base+['-cp',cp,'experiments.q5.Q51CausalProof',phase,'causal-q51.gguf','rust/target/release/libjmodels_kernels.so','tasks.json',str(value),'native']
run('causal-routing-prior',causal('prior','routing-prior','true'),180)
run('causal-routing-candidate',causal('candidate','routing',str(platform.machine()=='x86_64').lower()),180)
for pair in range(5,10):
 for arm in (['prior','candidate'] if pair%2==0 else ['candidate','prior']):run('causal-timing-'+str(pair)+'-'+arm,causal(arm,'timing',pair),400)
(out/'STATUS').write_text('COMPLETE\n')
