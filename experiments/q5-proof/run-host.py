"""Q5 screening preserves published routing until same-host evidence is complete."""
import json,os,pathlib,subprocess,time
out=pathlib.Path('/work/out');commands=[]
def run(name,cmd,timeout):
 (out/'STATUS').write_text(name+'\n');start=time.monotonic()
 with (out/(name+'.log')).open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
 commands.append(dict(name=name,command=cmd,exitCode=r.returncode,wallSeconds=time.monotonic()-start));(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
 subprocess.run(['aws','s3','cp',str(out),os.environ['RESULT_URI'],'--recursive','--only-show-errors'],check=True)
 assert r.returncode==0,name
run('rust-tests',['/root/.cargo/bin/cargo','test','--release','--manifest-path','rust/Cargo.toml'],300)
run('rust-build',['/root/.cargo/bin/cargo','build','--release','--manifest-path','rust/Cargo.toml'],300)
base=['/opt/jdk/bin/java','--add-modules','jdk.incubator.vector','--enable-native-access=ALL-UNNAMED','-Xmx2g','-cp','probe.jar:lib/*','experiments.q5.Q5ProjectionProof','rust/target/release/libjmodels_kernels.so']
for threads in (1,4,16):
 for mode in ('q51','q50'):run(mode+'-threads-'+str(threads),base+[mode,str(threads)],240)
(out/'STATUS').write_text('COMPLETE\n')
