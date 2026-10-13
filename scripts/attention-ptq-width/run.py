import hashlib,json,os,subprocess,sys,time,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
P=json.loads((HERE/'protocol.json').read_text())

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def run(command,prefix,timeout=180):
 start=time.monotonic()
 try:
  with prefix.with_suffix('.stdout').open('w') as out,prefix.with_suffix('.stderr').open('w') as err:
   result=subprocess.run(command,stdout=out,stderr=err,timeout=timeout)
  code=result.returncode
 except subprocess.TimeoutExpired:
  code=124
 save(prefix.with_suffix('.execution.json'),{'command':command,'exitCode':code,'wallSeconds':time.monotonic()-start})
 if code:raise RuntimeError(f'command failed ({code}), retained {prefix}')

def prepare(out):
 source=out/'source';source.mkdir(exist_ok=True);classes=out/'classes';classes.mkdir(exist_ok=True);libs=out/'lib';libs.mkdir(exist_ok=True)
 for dep in P['dependencies']:
  target=libs/dep['name']
  if not target.exists():urllib.request.urlretrieve(dep['url'],target)
  assert sha(target)==dep['sha256'],f'dependency mismatch: {target}'
 paths=['ops/GroupedQueryAttentionKernel.java','quant/TernaryDequantizer.java'];files=[]
 for label,revision in [('current',P['sourceRevision']),('candidate',P['candidateRevision'])]:
  src=source/label;src.mkdir(exist_ok=True);dest=classes/label;dest.mkdir(exist_ok=True);arm_files=[]
  for relative in paths:
   target=src/Path(relative).name
   target.write_bytes(subprocess.check_output(['git','show',revision+':backend-java/src/main/java/com/integrallis/models/backend/purejava/'+relative],cwd=ROOT));files.append(target);arm_files.append(target)
  run(['javac','--release','25','-g','--add-modules','jdk.incubator.vector','-cp',str(libs/'*'),'-d',str(dest),*[str(f) for f in arm_files],str(HERE/'WidthRiskProbe.java'),str(HERE/'WidthParityProbe.java'),str(HERE/'WidthCompilerProbe.java')],out/f'compile-{label}')
 for width,flags in [('default',[]),('128',['-XX:MaxVectorSize=16'])]:
  run(['java',*flags,'--add-modules','jdk.incubator.vector','--enable-native-access=ALL-UNNAMED','-cp',str(classes/'candidate'),'com.integrallis.models.backend.purejava.ops.WidthParityProbe',str(classes/'current'),str(classes/'candidate'),str(libs)],out/f'parity-{width}')

 cpu=subprocess.check_output(['lscpu'],text=True) if sys.platform=='linux' else subprocess.check_output(['uname','-a'],text=True)
 save(out/'provenance.json',{'protocol':P,'sourceSha256':{str(p.relative_to(source)):sha(p) for p in files},'probeSha256':sha(HERE/'WidthRiskProbe.java'),'classSha256':{str(f.relative_to(classes)):sha(f) for f in classes.rglob('*.class')},'dependencySha256':{p.name:sha(p) for p in libs.glob('*.jar')},'cpu':cpu,'java':subprocess.check_output(['java','-version'],stderr=subprocess.STDOUT,text=True),'flags':subprocess.check_output(['java','-XX:+PrintFlagsFinal','-version'],stderr=subprocess.STDOUT,text=True)})

def command(out,kernel,arm):
 return ['java','--add-modules','jdk.incubator.vector','--enable-native-access=ALL-UNNAMED','-Xms256m','-Xmx1g','-cp',str(out/'classes'/('candidate' if arm=='candidate' else 'current'))+os.pathsep+str(out/'lib/*'),'com.integrallis.models.backend.purejava.ops.WidthRiskProbe',kernel,arm]

def pair(out,kernel,index):
 signatures=[]
 for arm in (P['arms'] if index%2==0 else list(reversed(P['arms']))):
  prefix=out/f'{kernel}-{index}-{arm}';run(command(out,kernel,arm),prefix)
  records=[json.loads(line) for line in prefix.with_suffix('.stdout').read_text().splitlines() if line.startswith('{')]
  cold=[x for x in records if x['kind']=='cold'];output=[x for x in records if x['kind']=='output'];assert len(cold)==len(output)==1
  measured=[x for x in records if x['kind']=='measure']
  assert len(measured)==P['blocks']
  assert all(x['elapsedNanos']>=P['minimumMeasuredBlockNanos'] and x['calls']%P['timerBatchCalls'][kernel]==0 for x in measured)
  signatures.append((cold[0]['signature'],output[0]['signature']))
 assert all(signature==signatures[0] for signature in signatures),f'{kernel} reference/candidate output differs'
 save(out/f'{kernel}-{index}.complete.json',{'pair':index,'kernel':kernel,'numericalExact':True})

def diagnostic(out,kernel):
 cmd=command(out,kernel,'candidate');cmd[1:1]=[f'-XX:StartFlightRecording=filename={out}/{kernel}.jfr,settings=profile,dumponexit=true']
 run(cmd,out/f'jfr-{kernel}')
 run(['jfr','print','--events','jdk.ObjectAllocationSample','--json',str(out/f'{kernel}.jfr')],out/f'allocations-{kernel}')

def compiler(out):
 for arm in ['current','candidate']:
  cmd=command(out,'score',arm)
  cmd[cmd.index('com.integrallis.models.backend.purejava.ops.WidthRiskProbe')]='com.integrallis.models.backend.purejava.ops.WidthCompilerProbe'
  cmd[1:1]=['-XX:+UnlockDiagnosticVMOptions','-XX:+LogCompilation',f'-XX:LogFile={out}/compiler-{arm}.xml','-XX:CompileCommand=print,com.integrallis.models.backend.purejava.ops.GroupedQueryAttentionKernel::scoreGroup']
  run(cmd,out/f'compiler-{arm}')

out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=True)
if sys.argv[1]=='prepare':prepare(out)
elif sys.argv[1]=='pair':pair(out,sys.argv[3],int(sys.argv[4]))
elif sys.argv[1]=='compiler':compiler(out)
else:diagnostic(out,sys.argv[3])
