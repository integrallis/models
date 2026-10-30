"""Prepare a bounded check using released Central product JARs only; never launch here."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

source = Path('/private/tmp/flint-merge-worktree')
harness = Path('/Users/briansam-bodden/Code/java-ai/projects/models')
release_revision = sys.argv[1]
assert len(release_revision) == 40 and all(c in '0123456789abcdef' for c in release_revision)
campaign = 'flint-released-058-20261030-' + release_revision[:8]
base = Path('/private/tmp') / campaign
base.mkdir(exist_ok=False)
deployment = base / 'deployment'
deployment.mkdir()
subprocess.run([
    sys.executable,
    str(harness / 'benchmark-results/2026-10-11-bonsai-production/prepare-released-payload.py'),
    '--models-version', '0.5.8', '--vectors-version', '0.1.32',
    '--output', str(base / 'released')], check=True)
archive = base / 'released.tar.gz'
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
# The benchmark scorer is a source-built harness; every product JAR remains a Central download.
bench = harness / 'models-bench/build/install/models-bench/lib/models-bench-0.5.8.jar'
with zipfile.ZipFile(bench) as jar:
    classes = [name for name in jar.namelist() if name.endswith('.class')]
    assert classes and all(name.startswith('com/integrallis/models/bench/') for name in classes)
shutil.copy2(bench, base / 'released/lib' / bench.name)
classpath = base / 'released/classpath.json'
records = json.loads(classpath.read_text())
records.append({'file': bench.name, 'origin': 'source-built measurement harness',
                'sourceRevision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=harness, text=True).strip(),
                'sha256': digest(bench), 'sizeBytes': bench.stat().st_size})
classpath.write_text(json.dumps(records, indent=2) + '\n')
with tarfile.open(archive, 'w:gz') as output:
    output.add(base / 'released', arcname='released')
payload_uri = 's3://models-qual-077051030817/payload/' + campaign
results_uri = 's3://models-qual-077051030817/results/' + campaign
worker = deployment / 'released-worker.sh'
shutil.copy2(source / 'scripts/flint/released-worker.sh', worker)
probe_sources = {}
for name in ['FlintKernelParity.java', 'FlintReleasedProfileProbe.java', 'reference.properties',
             'FlintToolTaskProbe.java', 'tool-task-held-out-v1.json']:
    original = (Path('/private/tmp/flint-ordered-20261030-3336aed1/payload/reference.properties')
                if name == 'reference.properties' else source / 'scripts/flint' / name)
    shutil.copy2(original, deployment / name)
    probe_sources[name] = {'uri': payload_uri + '/' + name, 'sha256': digest(deployment / name)}
report = json.loads((source / 'benchmark-results/2026-10-30-flint-qualification/results/native-candidate.json').read_text())
outputs = [r['grounding']['rawText'] for r in report['runs'] if r['caseId'] == 'auto-glass-deadline']
assert len(outputs) == 3 and len(set(outputs)) == 1
manifest = {
    'schemaVersion': 1, 'campaign': campaign, 'releaseRevision': release_revision,
    'workerRevision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip(),
    'modelsVersion': '0.5.8', 'vectorsVersion': '0.1.32',
    'payloadUri': payload_uri + '/released.tar.gz', 'payloadSha256': digest(archive),
    'payloadBytes': archive.stat().st_size, 'workerUri': payload_uri + '/' + worker.name,
    'workerSha256': digest(worker), 'probeSources': probe_sources,
    'expectedRawSha256': hashlib.sha256(outputs[0].encode()).hexdigest(),
    'resultsUri': results_uri, 'deadlineSeconds': 5400,
    'instanceType': 'm6a.4xlarge', 'rootVolumeThroughputMiBps': 500,
    'protocol': 'Released-library compatibility: numerical parity, untuned default smoke, public threads profile round trip. Full RAG qualification is separately retained.',
    'candidateNativeThreads': 8, 'baselineNativeThreads': 16,
    'toolDiagnostic': True,
    'modelSha256': report['artifactSha256'],
}
(deployment / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
bootstrap = f'''#!/bin/bash
set -euo pipefail
exec > >(tee -a /var/log/flint-bootstrap.log /dev/console) 2>&1
# Independent bound applies during installation, download, inference and upload.
shutdown -h +90
finish() {{
  rc=$?
  trap - EXIT
  if command -v aws >/dev/null; then
    aws s3 cp /var/log/flint-bootstrap.log '{results_uri}/bootstrap.log' --region us-east-1 --only-show-errors || true
  fi
  shutdown -h now
  exit "$rc"
}}
trap finish EXIT
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq curl unzip jq python3
mkdir -p /work
curl -fL --retry 3 --max-time 300 https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /work/awscli.zip
unzip -q /work/awscli.zip -d /work
/work/aws/install
aws s3 cp '{payload_uri}/manifest.json' /work/manifest.json --region us-east-1 --only-show-errors
aws s3 cp '{payload_uri}/released-worker.sh' /work/released-worker.sh --region us-east-1 --only-show-errors
printf '%s  %s\\n' '{digest(worker)}' /work/released-worker.sh | sha256sum -c -
timeout --signal=TERM --kill-after=30 5100 bash /work/released-worker.sh /work/manifest.json
'''
(deployment / 'bootstrap.sh').write_text(bootstrap)
launch = json.loads(Path('/private/tmp/flint-qualification-20261030-fb5d7a68/launch-request.json').read_text())
launch['ClientToken'] = campaign
for item in launch['TagSpecifications']:
    for tag in item['Tags']:
        if tag['Key'] in ('Name', 'campaign'):
            tag['Value'] = campaign
(deployment / 'launch-request.json').write_text(json.dumps(launch, indent=2) + '\n')
shutil.copy2(__file__, deployment / Path(__file__).name)
for script in ['bootstrap.sh', 'released-worker.sh']:
    subprocess.run(['bash', '-n', str(deployment / script)], check=True)
print(json.dumps({'directory': str(base), 'manifest': manifest}, indent=2))
