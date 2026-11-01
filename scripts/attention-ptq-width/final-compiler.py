"""Capture final/baseline generated code; instrumented observations are not timing evidence."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]).resolve()
subprocess.run([sys.executable, str(HERE / 'attention-final.py'), str(OUT), '--local-preflight'], check=True)
admission = json.loads((OUT / 'admission.json').read_text())
if not admission['admitted']:
    (OUT / 'status.json').write_text(json.dumps({'status': 'NOT_ADMITTED', 'reason': 'Frozen 9V74 CPU/JVM metadata mismatch; compiler capture not run'}) + '\n')
    sys.exit(0)

diagnostic_classes = OUT / 'diagnostic-classes'
diagnostic_classes.mkdir(exist_ok=True)
diagnostic_source = OUT / 'WidthCompilerProbe.java'
diagnostic_source.write_bytes((HERE / 'WidthCompilerProbe.java').read_bytes())
subprocess.run([
    'javac', '--release', '25', '-g', '-cp', str(OUT / 'classes/candidate'),
    '-d', str(diagnostic_classes), str(diagnostic_source),
], check=True)

for kernel in ['score', 'softmax']:
    for arm in ['baseline', 'candidate']:
        name = f'compiler-{kernel}-{arm}'
        command = [
            'java', '--add-modules', 'jdk.incubator.vector', '--enable-native-access=ALL-UNNAMED',
            '-Xms256m', '-Xmx1g', '-XX:+UnlockDiagnosticVMOptions', '-XX:+LogCompilation',
            '-XX:LogFile=' + str(OUT / (name + '.xml')),
            '-XX:CompileCommand=print,com.integrallis.models.backend.purejava.ops.GroupedQueryAttentionKernel::scoreGroup',
            '-XX:CompileCommand=print,com.integrallis.models.backend.purejava.ops.GroupedQueryAttentionKernel::scoreGroup256',
            '-XX:CompileCommand=print,com.integrallis.models.backend.purejava.ops.GroupedQueryAttentionKernel::softmax',
            '-cp', str(diagnostic_classes) + os.pathsep + str(OUT / 'classes' / arm) + os.pathsep + str(OUT / 'lib/*'),
            'com.integrallis.models.backend.purejava.ops.WidthCompilerProbe', kernel, arm,
        ]
        (OUT / (name + '.command.json')).write_text(json.dumps(command, indent=2) + '\n')
        with (OUT / (name + '.stdout')).open('w') as stdout, (OUT / (name + '.stderr')).open('w') as stderr:
            subprocess.run(command, stdout=stdout, stderr=stderr, timeout=180, check=True)

(OUT / 'status.json').write_text(json.dumps({
    'status': 'COMPLETE', 'scope': 'Generated-code diagnosis only; no latency or release verdict',
    'attempts': 8, 'configuration': 'Original ordinary-compilation flags plus diagnostic logging; no -Xbatch or altered compiler thresholds',
}) + '\n')

(OUT / 'compiler-manifest.json').write_text(json.dumps({
    str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in OUT.rglob('*') if p.is_file() and p.name != 'compiler-manifest.json'
}, indent=2) + '\n')
