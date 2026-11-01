"""Frozen score-layout diagnosis; metadata admission precedes any timing."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
P = json.loads((HERE / 'score-boundary-protocol.json').read_text())
OUT = Path(sys.argv[1]).resolve()
OUT.mkdir(parents=True, exist_ok=True)

def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2) + '\n')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run(command, name):
    start = time.monotonic()
    code = 124
    try:
        with (OUT / (name + '.stdout')).open('w') as stdout, (OUT / (name + '.stderr')).open('w') as stderr:
            code = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=180).returncode
    finally:
        save(name + '.execution.json', dict(command=command, exitCode=code, wallSeconds=time.monotonic()-start))
    if code:
        raise RuntimeError(f'{name} failed: {code}')

cpu = subprocess.check_output(['lscpu'], text=True) if sys.platform == 'linux' else subprocess.check_output(['uname', '-a'], text=True)
flags = subprocess.check_output(['java', '-XX:+PrintFlagsFinal', '-version'], stderr=subprocess.STDOUT, text=True)
actual = {}
for key in ['UseAVX', 'MaxVectorSize', 'UseFMA', 'UseSSE']:
    match = re.search(r'\b' + key + r'\s+=\s+(\S+)', flags)
    actual[key] = match.group(1) if match else None
model = re.search(r'^Model name:\s+(.+)$', cpu, re.MULTILINE)
actual['cpuModel'] = model.group(1).strip() if model else None
admitted = actual == P['admission']
save('admission.json', dict(admitted=admitted, required=P['admission'], actual=actual, cpu=cpu, flags=flags, protocol=P, harnessRevision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
if not admitted and '--local-preflight' not in sys.argv:
    save('status.json', dict(status='NOT_ADMITTED', reason='Predeclared CPU/JVM metadata mismatch; no timing run'))
    sys.exit(0)
libs = OUT/'lib'
libs.mkdir(exist_ok=True)
for dep in P['dependencies']:
    path = libs/dep['name']
    if not path.exists():
        urllib.request.urlretrieve(dep['url'], path)
    assert sha(path) == dep['sha256']
paths = ['ops/GroupedQueryAttentionKernel.java', 'quant/TernaryDequantizer.java']
for arm, revision in P['arms'].items():
    source = OUT/'source'/arm
    classes = OUT/'classes'/arm
    source.mkdir(parents=True, exist_ok=True)
    classes.mkdir(parents=True, exist_ok=True)
    files = []
    for relative in paths:
        path = source/Path(relative).name
        path.write_bytes(subprocess.check_output(['git','show',revision+':backend-java/src/main/java/com/integrallis/models/backend/purejava/'+relative], cwd=ROOT))
        files.append(path)
    run(['javac','--release','25','-g','--add-modules','jdk.incubator.vector','-cp',str(libs/'*'),'-d',str(classes),*[str(f) for f in files],str(HERE/'WidthRiskProbe.java'),str(HERE/'WidthParityProbe.java')], 'compile-'+arm)
for arm in list(P['arms'])[1:]:
    for width, options in [('default',[]),('128',['-XX:MaxVectorSize=16'])]:
        run(['java',*options,'--add-modules','jdk.incubator.vector','--enable-native-access=ALL-UNNAMED','-cp',str(OUT/'classes'/arm),'com.integrallis.models.backend.purejava.ops.WidthParityProbe',str(OUT/'classes/baseline'),str(OUT/'classes'/arm),str(libs)], 'parity-'+arm+'-'+width)
save('artifacts.json', {str(p.relative_to(OUT)):sha(p) for folder in ['source','classes','lib'] for p in (OUT/folder).rglob('*') if p.is_file()})
if '--local-preflight' in sys.argv:
    save('status.json', dict(status='LOCAL_PREFLIGHT_ONLY', admitted=admitted))
    sys.exit(0)
for pair in range(P['pairs']):
    arms = list(P['arms'])
    rotation = pair % len(arms)
    arms = arms[rotation:] + arms[:rotation]
    if (pair // len(arms)) % 2:
        arms.reverse()
    signatures = []
    for arm in arms:
        name = f'score-{pair}-{arm}'
        # Probe label controls only primitive-reference selection. Actual arm is frozen classpath.
        run(['java','--add-modules','jdk.incubator.vector','--enable-native-access=ALL-UNNAMED','-Xms256m','-Xmx1g','-cp',str(OUT/'classes'/arm)+os.pathsep+str(libs/'*'),'com.integrallis.models.backend.purejava.ops.WidthRiskProbe','score',arm],name)
        records=[json.loads(line) for line in (OUT/(name+'.stdout')).read_text().splitlines() if line.startswith('{')]
        cold=[r for r in records if r['kind']=='cold']
        output=[r for r in records if r['kind']=='output']
        blocks=[r for r in records if r['kind']=='measure']
        assert len(cold)==len(output)==1 and len(blocks)==P['blocks']
        assert all(r['elapsedNanos']>=P['minimumMeasuredBlockNanos'] and r['calls']%P['timerBatchCalls']['score']==0 for r in blocks)
        signatures.append((cold[0]['signature'], output[0]['signature']))
    assert all(s==signatures[0] for s in signatures)
    save(f'pair-{pair}.json', dict(pair=pair,order=arms,numericalExact=True))
save('status.json', dict(status='COMPLETE', pairs=P['pairs'], scope=P['scope']))
