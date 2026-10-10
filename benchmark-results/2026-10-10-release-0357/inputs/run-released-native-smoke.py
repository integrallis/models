"""Exercise the published native bundle under the existing default-correctness protocol."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--runtime', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
workspace = Path(__file__).resolve().parent.parent
root = workspace / 'projects/models'
runtime = args.runtime.resolve()
prior = json.loads((runtime / 'run-inputs.json').read_text())
if prior.get('exitCode') != 0 or prior.get('verdict') != 'PASS':
    raise ValueError('Successful Central runtime verification is required')
source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
if source != prior['sourceSha']:
    raise ValueError('Smoke worker must match the release source')
for key in ('JAVA_TOOL_OPTIONS', 'JAVA_OPTS', 'JDK_JAVA_OPTIONS', '_JAVA_OPTIONS'):
    if 'models.' in os.environ.get(key, ''):
        raise ValueError('Refusing tuned environment: ' + key)

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

for item in prior['runtimeArtifacts']:
    path = runtime / 'lib' / Path(item['file']).name
    if digest(path) != item['sha256']:
        raise ValueError('Runtime changed: ' + str(path))
model_id = 'qwen2_5_coder_0_5b_instruct_q4_0'
catalog = json.loads((workspace / 'projects/model-jars/catalog/models.json').read_text())
model = next(m for m in catalog['models'] if m['id'] == model_id)
preflight = json.loads((workspace / 'audit-2026-10-10/smoke-settings-preflight.json').read_text())
record = next(r for r in preflight if r['modelId'] == model_id)
settings = record['settings']
# Qwen2.5 uses ChatML and this coder's workload is coding. Retain the qualified artifact's
# explicit context/thread/output budget; the current runtime supplies the grounding policy.
if record['backend'] != 'rust-ffm' or settings['promptTemplate'] != 'chatml' or settings['workload'] != 'coding':
    raise ValueError('Unexpected native smoke protocol')
fixture = Path.home() / '.jvllm/models/qwen2.5-coder-0.5b-instruct-q4_0.gguf'
if fixture.stat().st_size != model['sizeBytes'] or digest(fixture) != model['sha256']:
    raise ValueError('Fixture does not match the pinned catalog artifact')
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
java = '/Users/briansam-bodden/.sdkman/candidates/java/25.0.3-tem/bin/java'
command = [java, '--add-modules', 'jdk.incubator.vector', '--enable-native-access=ALL-UNNAMED',
    '-cp', str(runtime / 'lib/*'), 'com.integrallis.models.rag.RagBenchmarkCli',
    '--framework', 'plain-java', '--backend', 'rust-ffm', '--backend-version', prior['backendVersion'],
    '--model', str(fixture), '--model-id', model_id, '--workload', 'coding', '--prompt-template', 'chatml',
    '--context', str(settings['contextLength']), '--threads', str(settings['threads']),
    '--max-tokens', str(settings['maxOutputTokens']), '--top-k', str(settings['retrievalTopK']),
    '--warmups', '0', '--iterations', '1', '--output', str(out / 'models-rust-ffm.json')]
receipt = {'kind': 'published native-bundle default correctness, not comparative qualification',
    'sourceSha': source, 'backendVersion': prior['backendVersion'], 'runtimeArtifacts': prior['runtimeArtifacts'],
    'modelId': model_id, 'artifactSha256': model['sha256'], 'artifactSizeBytes': model['sizeBytes'],
    'settingsSource': {'report': record['report'], 'sha256': digest(root / record['report'])},
    'command': command, 'tuningSystemProperties': [],
    'startedAt': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
(out / 'run-inputs.json').write_text(json.dumps(receipt, indent=2) + '\n')
with (out / 'console.log').open('w') as log:
    result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
receipt.update(exitCode=result.returncode, completedAt=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
if result.returncode == 0:
    worker = (root / 'scripts/fleet/smoke-worker.sh').read_text()
    gate = worker.split("<<'GATE'\n", 1)[1].split('\nGATE\n', 1)[0]
    (out / 'gate.py').write_text(gate + '\n')
    receipt['verdict'] = subprocess.check_output([sys.executable, str(out / 'gate.py'),
        str(out / 'models-rust-ffm.json')], text=True).strip()
(out / 'run-inputs.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(receipt.get('verdict', 'RUN FAILED'), flush=True)
if result.returncode or receipt.get('verdict') != 'PASS':
    raise SystemExit(1)
