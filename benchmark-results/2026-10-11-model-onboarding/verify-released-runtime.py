"""Run the existing oracle harness with every runtime JAR fetched from Maven Central."""
from pathlib import Path
import argparse, datetime, hashlib, json, shutil, subprocess, urllib.request
p=argparse.ArgumentParser()
p.add_argument('--models',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--model',type=Path,required=True)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
lib=a.output/'lib';lib.mkdir(exist_ok=True)
records=[]
for source in sorted((a.models/'models-bench/build/install/models-bench/lib').glob('*.jar')):
 name=source.name
 target=lib/name
 if name.startswith('models-bench-'):
  shutil.copy2(source,target)
  records.append({'file':name,'source':'local unpublished measurement harness','sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
  continue
 if name.startswith(('models-','backend-','vectors-')):group='com.integrallis'
 elif name.startswith('jackson-datatype-'):group='com.fasterxml.jackson.datatype'
 elif name.startswith('jackson-'):group='com.fasterxml.jackson.core'
 elif name.startswith('slf4j-'):group='org.slf4j'
 else:raise ValueError(name)
 artifact,version=name[:-4].rsplit('-',1)
 url='https://repo.maven.apache.org/maven2/'+group.replace('.','/')+'/'+artifact+'/'+version+'/'+name
 with urllib.request.urlopen(url,timeout=120) as r: raw=r.read()
 with urllib.request.urlopen(url+'.sha1',timeout=120) as r: expected=r.read().decode().strip().split()[0]
 assert hashlib.sha1(raw).hexdigest()==expected,name
 target.write_bytes(raw)
 records.append({'file':name,'coordinate':group+':'+artifact+':'+version,'url':url,'sha256':hashlib.sha256(raw).hexdigest(),'centralSha1':expected})
 print('Verified '+name,flush=True)
report=a.output/'embeddinggemma2-released-equivalence.json'
command=['java','--add-modules','jdk.incubator.vector','--enable-native-access=ALL-UNNAMED','-Xmx4g','-cp',str(lib/'*'),'com.integrallis.models.bench.InferenceBenchmarkCli','embedding-equivalence','--model',str(a.model),'--report',str(report)]
with (a.output/'embeddinggemma2-released.log').open('w') as out:r=subprocess.run(command,stdout=out,stderr=subprocess.STDOUT)
receipt={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':command,'exitCode':r.returncode,'classpath':records,'scope':'The unpublished benchmark harness is local. All executable Models/Vectors runtime and external dependency JARs are fetched from Central, checksum-verified, and isolated from the source build classpath.'}
(a.output/'released-classpath.json').write_text(json.dumps(receipt,indent=2)+'\n')
raise SystemExit(r.returncode)
