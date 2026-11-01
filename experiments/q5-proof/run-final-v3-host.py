"""Frozen Q5 production paths, real BERT and derived Q5_1 causal fixture."""
import json,os,pathlib,subprocess,time,urllib.request,hashlib,platform
out=pathlib.Path('/work/out');commands=[]
def run(name,cmd,timeout):
 (out/'STATUS').write_text(name+'\n');start=time.monotonic()
 with (out/(name+'.log')).open('w') as log:r=subprocess.run(['/usr/bin/time','-v',*cmd],stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
 commands.append(dict(name=name,command=cmd,exitCode=r.returncode,wallSeconds=time.monotonic()-start));(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
 subprocess.run(['aws','s3','cp',str(out),os.environ['RESULT_URI'],'--recursive','--only-show-errors'],check=True)
 assert r.returncode==0,name
def download(url,path,sha):
 urllib.request.urlretrieve(url,path);assert hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()==sha
base=['/opt/jdk/bin/java','--add-modules','jdk.incubator.vector','--enable-native-access=ALL-UNNAMED','-Xmx4g','-Dmodels.native.kernels.threads=16','-Dvectors.gguf.threads=16']
def embedding(arm,phase,data,reps):return base+['-cp',arm+'.jar:probe.jar:lib/*','experiments.q5.EmbeddingRouteProof','java','bert.gguf','unused',data,phase,str(reps)]
def rows(name):return [json.loads(s) for s in (out/(name+'.log')).read_text().splitlines() if s.startswith('{')]
# Separate correctness probes use explicit SIMD widths, never inference tuning flags.
for bits in (64,128,256):
 activation=['/opt/jdk/bin/java','--add-modules','jdk.incubator.vector','-Xmx512m',f'-Dvectors.maxBits={bits}','-cp','activation-probe.jar:lib/*']
 run('activation-cold-hot-'+str(bits),activation+['com.integrallis.vectors.core.ActivationColdHotProbe'],180)
 run('activation-random-'+str(bits),activation+['com.integrallis.vectors.core.SwiGluPublicProbe','10000000'],180)
 run('activation-exhaustive-'+str(bits),activation+['com.integrallis.vectors.core.SwiGluPublicProbe','0','exhaustive'],900)
download('https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/544f204f2eaa2d71361ffc74d6df7170285b286a/all-MiniLM-L6-v2-Q5_K_S.gguf','bert.gguf','0de5c3d1343904debb305a2f6c55f57ccdc68786968b5d4726dd404f6c4c4244')
for arm in ('baseline','candidate'):run('bert-correctness-'+arm,embedding(arm,'correctness','data/cases.json',1),360)
a={r['id']:r['values'] for r in rows('bert-correctness-baseline') if r['kind']=='embedding'};b={r['id']:r['values'] for r in rows('bert-correctness-candidate') if r['kind']=='embedding'};assert a==b and len(a)==82
if platform.machine()=='aarch64':
 for arm in ('baseline','candidate'):
  command=embedding(arm,'timing','data/timing.json',1);command.insert(1,f'-XX:StartFlightRecording=filename=/work/out/bert-{arm}.jfr,settings=profile,dumponexit=true');command.insert(1,'-XX:FlightRecorderOptions=stackdepth=64');run('bert-jfr-'+arm,command,180)
for pair in range(10):
 for arm in (('baseline','candidate') if pair%2==0 else ('candidate','baseline')):run('bert-timing-'+str(pair)+'-'+arm,embedding(arm,'timing','data/timing.json',3),150)
run('rust-build',['/root/.cargo/bin/cargo','build','--release','--manifest-path','rust/Cargo.toml'],300)
pathlib.Path('oracle-source').mkdir();run('oracle-extract',['tar','xf','oracle-source.tar','-C','oracle-source'],60)
run('oracle-configure',['cmake','-S','oracle-source','-B','oracle-build','-DCMAKE_BUILD_TYPE=Release','-DBUILD_SHARED_LIBS=OFF','-DGGML_NATIVE=OFF','-DGGML_CUDA=OFF','-DGGML_METAL=OFF','-DLLAMA_CURL=OFF','-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_SERVER=OFF'],120)
run('oracle-build',['cmake','--build','oracle-build','--target','llama-quantize','-j','8'],600)
download('https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-q5_0.gguf','source-q50.gguf','925946ab27801fea3eead42933f857b6449c6d299f062e374cee2ac48426f2a5')
run('convert-q51',['oracle-build/bin/llama-quantize','--allow-requantize','--pure','--output-tensor-type','q5_1','--token-embedding-type','q5_1','source-q50.gguf','causal-q51.gguf','Q5_1','8'],300)
(out/'derived-fixture.json').write_text(json.dumps(dict(sourceSha256='925946ab27801fea3eead42933f857b6449c6d299f062e374cee2ac48426f2a5',q51Sha256=hashlib.sha256(pathlib.Path('causal-q51.gguf').read_bytes()).hexdigest(),bytes=pathlib.Path('causal-q51.gguf').stat().st_size),indent=2)+'\n')
def causal(arm,phase,value):return base+['-cp','candidate.jar:probe.jar:lib/*','experiments.q5.Q51CausalProof',phase,'causal-q51.gguf','rust/target/release/libjmodels_kernels.so','tasks.json',str(value),arm]
run('causal-routing-native',causal('native','routing',str(platform.machine()=='x86_64').lower()),180)
for arm in ('java','native'):run('causal-correctness-'+arm,causal(arm,'cohort',0),900)
a={r['id']:r['fullLogitSha256'] for r in rows('causal-correctness-java') if r['kind']=='request'};b={r['id']:r['fullLogitSha256'] for r in rows('causal-correctness-native') if r['kind']=='request'};assert a==b and len(a)==148
for pair in range(10):
 for arm in (('java','native') if pair%2==0 else ('native','java')):run('causal-timing-'+str(pair)+'-'+arm,causal(arm,'timing',pair),300)
(out/'STATUS').write_text('COMPLETE\n')
