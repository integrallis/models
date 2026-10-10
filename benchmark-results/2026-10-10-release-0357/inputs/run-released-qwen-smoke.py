"""Run the pinned Qwen smoke with Central libraries and a source-built benchmark harness."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-sha', required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
workspace = Path(__file__).resolve().parent.parent
root = workspace / 'projects/models'
out = args.output.resolve()
java = Path('/Users/briansam-bodden/.sdkman/candidates/java/25.0.3-tem/bin/java')
env = dict(os.environ, JAVA_HOME=str(java.parent.parent))
env['PATH'] = str(java.parent) + os.pathsep + env['PATH']
for key in ('JAVA_TOOL_OPTIONS', 'JAVA_OPTS', 'JDK_JAVA_OPTIONS', '_JAVA_OPTIONS'):
    if 'models.' in env.get(key, ''):
        raise ValueError('Refusing tuned environment: ' + key)

def git(*command):
    return subprocess.check_output(['git', *command], cwd=root, text=True).strip()

if git('rev-parse', 'HEAD') != args.source_sha or git('status', '--porcelain'):
    raise ValueError('A clean checkout of the exact released source is required')
if git('rev-parse', 'v0.3.57^{commit}') != args.source_sha:
    raise ValueError('Source does not match the release tag')
out.mkdir(parents=True, exist_ok=False)
consumer = out / 'consumer'
consumer.mkdir()
(consumer / 'settings.gradle').write_text("rootProject.name = 'released-qwen-smoke'\n")
(consumer / 'build.gradle').write_text('''
import groovy.json.JsonOutput
plugins { id 'java' }
repositories { mavenCentral() }
dependencies {
    implementation 'com.integrallis:models-rag:0.3.57'
    implementation 'com.integrallis:models-runtime:0.3.57'
    implementation 'com.integrallis:backend-java:0.3.57'
    implementation 'com.integrallis:backend-native:0.3.57'
    implementation 'com.integrallis:models-langchain4j:0.3.57'
    implementation 'com.integrallis:models-spring-ai:0.3.57'
    implementation 'dev.langchain4j:langchain4j:1.17.2'
    implementation 'org.apache.lucene:lucene-core:10.4.0'
    implementation 'com.fasterxml.jackson.core:jackson-databind:2.22.3'
    implementation 'org.springframework.ai:spring-ai-rag:2.0.0'
}
tasks.register('collectRuntime') {
    doLast {
        def artifacts = configurations.runtimeClasspath.resolvedConfiguration.resolvedArtifacts
        file('runtime.json').text = JsonOutput.prettyPrint(JsonOutput.toJson(artifacts.collect {
            [coordinate: it.moduleVersion.id.group + ':' + it.name,
             version: it.moduleVersion.id.version, file: it.file.absolutePath]
        }.sort { it.coordinate }))
    }
}
''')

def run(command, log, timeout=1800):
    with (out / log).open('w') as stream:
        subprocess.run(command, cwd=root, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, timeout=timeout, check=True)

run([str(root / 'gradlew'), '-p', str(consumer), '-g', str(out / 'gradle-cache'),
     '--no-daemon', '--max-workers=4', 'collectRuntime'], 'central-resolution.log')
run([str(root / 'gradlew'), ':models-rag-bench:jar', '--max-workers=4'], 'harness-build.log')
policy = json.loads((root / 'gradle/dependency-policy.json').read_text())
lib = out / 'lib'
lib.mkdir()
artifacts = json.loads((consumer / 'runtime.json').read_text())

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

for node in artifacts:
    path = Path(node['file'])
    group, artifact = node['coordinate'].split(':')
    expected_external = policy.get('modules', {}).get(node['coordinate']) or policy.get('families', {}).get(group)
    if group == 'com.fasterxml.jackson' or group.startswith('com.fasterxml.jackson.'):
        expected_external = policy['jacksonAnnotations'] if artifact == 'jackson-annotations' else policy['jacksonBom']
    elif group == 'org.slf4j':
        expected_external = policy['slf4j']
    if expected_external is not None and node['version'] != expected_external:
        raise ValueError('Combined runtime violates external policy: ' + str(node))
    if node['coordinate'].startswith('com.integrallis:'):
        expected = '0.1.29' if node['coordinate'].split(':')[1].startswith('vectors') else '0.3.57'
        if node['version'] != expected:
            raise ValueError('Unexpected internal version: ' + str(node))
        with zipfile.ZipFile(path) as jar:
            manifest = json.loads(jar.read('META-INF/jvm-ai/release-dependencies.json'))
        if manifest['externalPolicy'] != policy or any(
            {'models': '0.3.57', 'vectors': '0.1.29'}.get(k) != v
            for k, v in manifest['versions'].items()
        ):
            raise ValueError('Inconsistent released manifest: ' + str(node))
    target = lib / path.name
    if target.exists():
        raise ValueError('Duplicate runtime filename: ' + target.name)
    shutil.copyfile(path, target)
    node['sha256'] = digest(path)
    node['sizeBytes'] = path.stat().st_size
    node['origin'] = 'Maven Central'
harness = root / 'models-rag-bench/build/libs/models-rag-bench-0.3.57.jar'
shutil.copyfile(harness, lib / harness.name)
artifacts.append({'file': harness.name, 'sourceSha': args.source_sha,
    'sha256': digest(harness),
    'sizeBytes': harness.stat().st_size, 'origin': 'source-built unpublished benchmark harness'})
job = json.loads((root / 'scripts/fleet/smoke-qwen-bf16.json').read_text())[0]
snapshot = workspace / '.model-fixtures' / job['id']
for item in job['files']:
    path = snapshot / item['path']
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if path.stat().st_size != item['sizeBytes'] or digest != item['sha256']:
        raise ValueError('Fixture mismatch: ' + str(path))
label = 'models@0.3.57+' + args.source_sha[:12]
command = [str(java), '--add-modules', 'jdk.incubator.vector', '--enable-native-access=ALL-UNNAMED',
    '-cp', str(lib / '*'), 'com.integrallis.models.rag.RagBenchmarkCli',
    '--framework', 'plain-java', '--backend', job['backend'], '--backend-version', label,
    '--model', str(snapshot), '--model-id', job['id'], '--workload', job['wl'],
    '--prompt-template', job['tpl'], '--context', str(job['context']),
    '--threads', str(job['threads']), '--max-tokens', str(job['mt']), '--top-k', str(job['topK']),
    '--warmups', '0', '--iterations', '1', '--output', str(out / 'models-pure-java.json')]
receipt = {'kind': 'released-library correctness; unpublished harness built from release source',
    'sourceSha': args.source_sha, 'backendVersion': label, 'runtimeArtifacts': artifacts,
    'job': job, 'command': command, 'tuningSystemProperties': [],
    'javaVersion': subprocess.check_output([str(java), '-version'], stderr=subprocess.STDOUT, text=True),
    'startedAt': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
(out / 'run-inputs.json').write_text(json.dumps(receipt, indent=2) + '\n')
run(command, 'console.log')
worker = (root / 'scripts/fleet/smoke-worker.sh').read_text()
gate = worker.split("<<'GATE'\n", 1)[1].split('\nGATE\n', 1)[0]
(out / 'gate.py').write_text(gate + '\n')
verdict = subprocess.check_output([sys.executable, str(out / 'gate.py'),
                                   str(out / 'models-pure-java.json')], text=True).strip()
(out / 'gate.txt').write_text(verdict + '\n')
receipt.update(exitCode=0, verdict=verdict,
               completedAt=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
(out / 'run-inputs.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(verdict, flush=True)
if verdict != 'PASS':
    raise SystemExit(1)
