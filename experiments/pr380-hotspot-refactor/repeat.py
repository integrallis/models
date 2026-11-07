from pathlib import Path
import os,subprocess,json,sys
p=Path(sys.argv[1]).resolve(); out=p/'longer-warmup'; out.mkdir()
cache=Path.home()/'.gradle/caches/modules-2/files-2.1'
patterns=['org.openjdk.jmh/jmh-core/1.37/*/*.jar','net.sf.jopt-simple/jopt-simple/5.0.4/*/*.jar','org.apache.commons/commons-math3/3.6.1/*/*.jar']
jars=[str(f) for pattern in patterns for f in cache.glob(pattern)]
cp=Path(sys.argv[2]).read_text()+':'+':'.join(jars)
classpaths={'before':':'.join([str(p/'baseline-classes'),str(p/'benchmark-classes'),cp]),'after':':'.join([str(p/'benchmark-classes'),cp])}
commands=[]
java=Path(os.environ['JAVA_HOME'])/'bin/java'
for arm in ['after','before']:
 for operation in ['imagePreprocess','embeddingPrefillAndDecode']:
  name=arm+'-'+operation
  cmd=[str(java),'-cp',classpaths[arm],'org.openjdk.jmh.Main','.*RefactorBenchmark.'+operation+'$','-wi','5','-i','5','-w','1s','-r','1s','-f','2','-t','1','-rf','json','-rff',str(out/(name+'.json'))]
  if operation=='embeddingPrefillAndDecode':cmd+=['-p','fixture=buildQ4KNanoModel']
  commands.append(cmd)
  with (out/(name+'.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
print('completed')
