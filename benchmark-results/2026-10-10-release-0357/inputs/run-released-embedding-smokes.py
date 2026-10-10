"""Exercise all eight embeddings qualified on 0.3.55 with the verified Central runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--runtime', type=Path, required=True,
                    help='Successful run-released-qwen-smoke output, containing Central libraries')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
workspace = Path(__file__).resolve().parent.parent
root = workspace / 'projects/models'
runtime = args.runtime.resolve()
prior = json.loads((runtime / 'run-inputs.json').read_text())
if prior.get('exitCode') != 0 or prior.get('verdict') != 'PASS':
    raise ValueError('Successful released runtime verification is required')
source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
if source != prior['sourceSha']:
    raise ValueError('Benchmark source must match the verified release source')
java = Path('/Users/briansam-bodden/.sdkman/candidates/java/25.0.3-tem/bin/java')
env = dict(os.environ, JAVA_HOME=str(java.parent.parent))
env['PATH'] = str(java.parent) + os.pathsep + env['PATH']
for key in ('JAVA_TOOL_OPTIONS', 'JAVA_OPTS', 'JDK_JAVA_OPTIONS', '_JAVA_OPTIONS'):
    if 'models.' in env.get(key, ''):
        raise ValueError('Refusing tuned environment: ' + key)
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
with (out / 'harness-build.log').open('w') as log:
    subprocess.run([str(root / 'gradlew'), ':models-bench:jar', '--max-workers=4'],
                   cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
harness = root / 'models-bench/build/libs/models-bench-0.3.57.jar'

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

artifacts = []
for item in prior['runtimeArtifacts']:
    path = runtime / 'lib' / Path(item['file']).name
    if digest(path) != item['sha256']:
        raise ValueError('Runtime artifact changed: ' + str(path))
    artifacts.append(item)
artifacts.append({'file': str(harness), 'sha256': digest(harness), 'sourceSha': source,
                  'origin': 'source-built unpublished embedding harness'})
catalog = json.loads((workspace / 'projects/model-jars/catalog/models.json').read_text())
qualifications = json.loads((workspace / 'projects/model-jars/catalog/embedding-qualifications.json').read_text())
ids = {e['modelId'] for e in qualifications['entries']
       if e.get('qualified') and e.get('backendVersion') == 'models-0.3.55'}
models = [m for m in catalog['models'] if m['id'] in ids]
if len(models) != 8:
    raise ValueError('Expected exactly eight embeddings qualified on Models 0.3.55')
results = []
for model in sorted(models, key=lambda m: m['sizeBytes']):
    filename = model['downloadUri'].rsplit('/', 1)[-1]
    directory = 'lfm2.5-embedding-350m' if 'lfm2_5_embedding' in model['id'] else 'embedding-runtime-upgrade'
    fixture = workspace / '.model-fixtures' / directory / filename
    if fixture.stat().st_size != model['sizeBytes'] or digest(fixture) != model['sha256']:
        raise ValueError('Fixture mismatch: ' + str(fixture))
    report = out / (model['id'] + '.json')
    command = [str(java), '--add-modules', 'jdk.incubator.vector', '--enable-native-access=ALL-UNNAMED',
        '-cp', str(harness) + os.pathsep + str(runtime / 'lib/*'),
        'com.integrallis.models.bench.EmbeddingEquivalenceCli', '--model', str(fixture),
        '--report', str(report)]
    started = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    with (out / (model['id'] + '.log')).open('w') as log:
        completed = subprocess.run(command, cwd=root, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, timeout=1800)
    result = {'modelId': model['id'], 'artifactSha256': model['sha256'],
        'startedAt': started, 'exitCode': completed.returncode, 'command': command,
        'completedAt': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    if report.exists():
        measured = json.loads(report.read_text())
        result.update(reportSha256=digest(report), qualified=measured['qualified'],
                      minimumOracleCosine=measured['minimumOracleCosine'], probes=measured['probes'])
    results.append(result)
    (out / 'run-inputs.json').write_text(json.dumps({
        'sourceSha': source, 'backendVersion': prior['backendVersion'],
        'kind': 'released-library comparison against committed exact-artifact oracle vectors',
        'runtimeArtifacts': artifacts, 'tuningSystemProperties': [], 'results': results}, indent=2) + '\n')
    print(model['id'], result, flush=True)
if any(r['exitCode'] != 0 or r.get('qualified') is not True or r.get('probes') != 8 for r in results):
    raise SystemExit(1)
