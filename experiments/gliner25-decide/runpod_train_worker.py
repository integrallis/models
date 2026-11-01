"""Remote-only training worker. Uses object-scoped URLs, never cloud API credentials."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import urllib.request

ROOT = Path('/work/gliner')
ROOT.mkdir(parents=True, exist_ok=True)
MANIFEST = json.loads(Path(sys.argv[1]).read_text())
LOG = ROOT/'worker.log'
STOP = threading.Event()


def upload(path, name):
    url = MANIFEST['puts'][name]
    # curl streams large weights instead of materializing them in Python memory.
    result = subprocess.run(['curl', '--silent', '--show-error', '--fail', '--retry', '2',
                             '--max-time', '240', '--upload-file', str(path), url],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode:
        raise RuntimeError('object upload failed: '+name)


def status(stage, **values):
    path = ROOT/'status.json'
    path.write_text(json.dumps(dict(stage=stage, utc=time.time(), **values))+'\n')
    upload(path, 'status.json')
    print('stage='+stage, flush=True)


def download(url, path, expected=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=180) as source, path.open('wb') as target:
        while block := source.read(1024*1024):
            target.write(block); h.update(block)
    if expected and h.hexdigest() != expected:
        raise ValueError('download digest mismatch: '+path.name)


def command(args, timeout=1200):
    with LOG.open('ab') as log:
        # Arguments containing signed URLs are never logged.
        subprocess.run(args, check=True, stdout=log, stderr=subprocess.STDOUT,
                       timeout=timeout, env=dict(os.environ, PYTHONUNBUFFERED='1'))


def stream_logs():
    while not STOP.wait(20):
        try:
            if LOG.exists(): upload(LOG, 'worker.log')
        except Exception:
            pass


def main():
    # Base image includes Python, while apt supplies git/curl for pinned source and streaming uploads.
    command(['apt-get', 'update', '-qq'], 180)
    command(['apt-get', 'install', '-y', '-qq', 'git', 'curl', 'ca-certificates', 'libgomp1'], 180)
    threading.Thread(target=stream_logs, daemon=True).start()
    status('installing')
    payload = ROOT/'payload.tar.gz'
    download(MANIFEST['payloadUrl'], payload, MANIFEST['payloadSha256'])
    with tarfile.open(payload) as archive:
        archive.extractall(ROOT/'payload', filter='data')
    experiment = ROOT/'payload/experiment'
    command([sys.executable, '-m', 'venv', str(ROOT/'venv')], 60)
    python = str(ROOT/'venv/bin/python')
    command([python, '-m', 'pip', 'install', '--disable-pip-version-check', '--require-hashes',
             '--extra-index-url', 'https://download.pytorch.org/whl/cu128',
             '-r', str(experiment/'requirements-training.lock')], 600)
    command(['git', 'clone', str(ROOT/'payload/publisher.bundle'), str(ROOT/'publisher')], 60)
    command(['git', '-C', str(ROOT/'publisher'), 'checkout', '--detach', MANIFEST['publisherCommit']], 30)
    status('downloading-snapshot')
    snapshot = json.loads((experiment/'snapshot.json').read_text())
    for entry in snapshot['files']:
        download(snapshot['sourceUri']+'/resolve/'+snapshot['revision']+'/'+entry['path'],
                 ROOT/'base-snapshot'/entry['path'], entry['sha256'])
    command([python, '-c', 'import torch,json; print(json.dumps(dict(torch=torch.__version__, cuda=torch.version.cuda, available=torch.cuda.is_available(), gpu=torch.cuda.get_device_name(0), bf16=torch.cuda.is_bf16_supported())))'], 60)
    status('training')
    policy=json.loads((experiment/'finetune-policy.json').read_text())
    command([python, str(experiment/'train_general.py'), '--snapshot', str(ROOT/'base-snapshot'),
             '--source', str(ROOT/'publisher'), '--data', str(ROOT/'payload/data'),
             '--output', str(ROOT/'training-output')], policy['maximumTrainingSeconds']+600)
    status('uploading')
    output = ROOT/'training-output'
    for name in ['snapshot.json', 'training-record.json']:
        upload(output/name, name)
    for entry in snapshot['files']:
        upload(output/'snapshot'/entry['path'], 'snapshot/'+entry['path'])
    adapter = ROOT/'selected-adapter.tar.gz'
    with tarfile.open(adapter, 'w:gz') as archive:
        archive.add(output/'checkpoints/best', arcname='best')
    upload(adapter, 'selected-adapter.tar.gz')
    upload(LOG, 'worker.log')
    result = json.loads((output/'training-record.json').read_text())
    status('complete', steps=result['steps'], trainingSeconds=result['trainingSeconds'],
           mergedWeightSha256=result['mergedSnapshot']['revision'], catalogQualified=False)


try:
    main()
except Exception as exc:
    with LOG.open('a') as log:
        # Full tracebacks are safe here: downloads don't embed signed URLs in their errors.
        traceback.print_exc(file=log)
    try:
        upload(LOG, 'worker.log')
        status('failed', errorType=type(exc).__name__)
    except Exception:
        pass
    raise SystemExit(1)
finally:
    STOP.set()
