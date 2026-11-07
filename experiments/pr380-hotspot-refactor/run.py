#!/usr/bin/env python3
"""Run isolated before/after synthetic differential and JMH checks after Gradle checks pass.

Arguments: exported backend-java test runtime classpath file; fresh output directory.
Uses JAVA_HOME (JDK 25), pinned JMH artifacts from Gradle cache, and pinned original sources.
"""
from pathlib import Path
import hashlib,json,os,subprocess,sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
BEFORE='8a34f31a'
cp_file=Path(sys.argv[1]); out=Path(sys.argv[2]).resolve(); out.mkdir(parents=True,exist_ok=False)
java=Path(os.environ['JAVA_HOME'])/'bin'
cache=Path.home()/'.gradle/caches/modules-2/files-2.1'
def jar(group,artifact,version):
    matches=list((cache/group/artifact/version).glob('*/*.jar'))
    assert len(matches)==1,matches
    return matches[0]
jars=[jar('org.openjdk.jmh','jmh-core','1.37'),jar('org.openjdk.jmh','jmh-generator-annprocess','1.37'),jar('net.sf.jopt-simple','jopt-simple','5.0.4'),jar('org.apache.commons','commons-math3','3.6.1')]
cp=cp_file.read_text()+os.pathsep+os.pathsep.join(map(str,jars))
base=out/'baseline-classes'; compiled=out/'benchmark-classes'
base.mkdir(); compiled.mkdir()
paths=['backend-java/src/main/java/com/integrallis/models/backend/purejava/'+p for p in ['llama/LlamaForwardPass.java','vision/ClipImagePreprocessor.java','vision/ClipVisionWeights.java']]
original=[]
for path in paths:
    p=out/'original'/path; p.parent.mkdir(parents=True,exist_ok=True)
    p.write_bytes(subprocess.check_output(['git','show',BEFORE+':'+path],cwd=ROOT));original.append(str(p))
def run(cmd,log,**kwargs):
    with (out/log).open('w') as f:subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,check=True,**kwargs)
run([java/'javac','--add-modules=jdk.incubator.vector','-cp',cp,'-d',base,*original],'baseline-compile.log')
run([java/'javac','--add-modules=jdk.incubator.vector','-cp',cp,'-processorpath',os.pathsep.join(map(str,jars[:2])),'-d',compiled,HERE/'RefactorBenchmark.java'],'benchmark-compile.log')
classpaths={'before':os.pathsep.join([str(base),str(compiled),cp]),'after':os.pathsep.join([str(compiled),cp])}
for arm,classpath in classpaths.items():
    run([java/'java','--add-modules=jdk.incubator.vector','--enable-native-access=ALL-UNNAMED','-Xmx512m','-cp',classpath,'com.integrallis.models.experiments.RefactorBenchmark'],arm+'-outputs.log')
def outputs(arm):return [line for line in (out/(arm+'-outputs.log')).read_text().splitlines() if '\t' in line]
a,b=outputs('before'),outputs('after')
assert len(a)==39,(len(a),len(b))
assert a==b,'Output fingerprints differ; do not benchmark until explained'
(out/'equivalence.json').write_text(json.dumps(dict(equal=True,comparisons=len(a),outputs=a),indent=2)+'\n')
for arm,classpath in classpaths.items():
    run([java/'java','-cp',classpath,'org.openjdk.jmh.Main','.*RefactorBenchmark.*','-wi','3','-i','5','-w','500ms','-r','500ms','-f','2','-t','1','-rf','json','-rff',out/(arm+'-jmh.json')],arm+'-jmh.log')
meta=dict(before=subprocess.check_output(['git','rev-parse',BEFORE],cwd=ROOT,text=True).strip(),after=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),jdk=subprocess.check_output([java/'java','--version'],text=True),host=subprocess.check_output(['uname','-a'],text=True).strip(),jars={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in jars},source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},benchmark_sha256=hashlib.sha256((HERE/'RefactorBenchmark.java').read_bytes()).hexdigest(),protocol=dict(forks=2,warmups=3,measurements=5,iteration_ms=500,threads=1),scope='synthetic fixtures only; no full-model performance qualification')
(out/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps({'equal_outputs':len(a),'results':str(out)}))
