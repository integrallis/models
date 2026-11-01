"""Run immutable same-host pairs; preserve a failed baseline fixture without hiding it."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import urllib.parse
import zipfile

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / 'protocol.json').read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def run(command, prefix, timeout=7200):
    print(f'RUN {prefix.name}', flush=True)
    start = time.monotonic()
    code = None
    try:
        with prefix.with_suffix('.stdout').open('w') as stdout, prefix.with_suffix('.stderr').open('w') as stderr:
            code = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=timeout).returncode
    finally:
        write(prefix.with_suffix('.execution.json'), {'command': command, 'exitCode': code,
              'wallSeconds': time.monotonic() - start})
    print(f'FINISHED {prefix.name}: exit={code}', flush=True)
    return code


def origins(prefix, core):
    rows = [json.loads(line) for line in prefix.with_suffix('.stdout').read_text().splitlines()
            if line.startswith('{')]
    origin = [r for r in rows if r['kind'] == 'runtime-origin']
    assert len(origin) == 1, 'missing runtime-origin receipt'
    assert Path(urllib.parse.urlsplit(origin[0]['bf16']).path).resolve() == core.resolve()
    actual_models = Path(urllib.parse.urlsplit(origin[0]['models']).path).resolve()
    assert actual_models.parent == core.resolve().parent and actual_models.name.startswith('backend-java-')


def prepare(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    models = args.models.resolve()
    assert subprocess.check_output(['git', '-C', str(models), 'rev-parse', 'HEAD'], text=True).strip() == CONFIG['modelsRevision']
    assert sha(HERE / 'Bf16ModelProof.java') == CONFIG['harnessSha256']
    source = models / 'backend-native/build/paired-runtime'
    baseline = list(source.glob('vectors-core-*.jar'))
    assert len(baseline) == 1 and baseline[0].name == 'vectors-core-0.1.29.jar'
    assert sha(baseline[0]) == CONFIG['baselineCoreSha256']
    for name, expected in CONFIG['dataSha256'].items():
        assert sha(HERE / 'data' / name) == expected, name
    native = models / 'backend-native/build/rust-target/release/libjmodels_kernels.so'
    assert native.is_file()
    cores = {}
    for arm in ['baseline', 'candidate']:
        directory = out / 'lib' / arm
        directory.mkdir(parents=True, exist_ok=True)
        for jar in source.glob('*.jar'):
            if jar == baseline[0]:
                jar = args.candidate_core.resolve() if arm == 'candidate' else jar
            shutil.copy2(jar, directory / jar.name)
        cores[arm] = next(directory.glob('vectors-core-*.jar'))
    inventories = {arm: {p.name: sha(p) for p in (out / 'lib' / arm).glob('*.jar')}
                   for arm in cores}
    assert {k: v for k, v in inventories['baseline'].items() if not k.startswith('vectors-core-')} == {
        k: v for k, v in inventories['candidate'].items() if not k.startswith('vectors-core-')}
    jar_entries = {arm: {jar.name: {name: hashlib.sha256(archive.read(name)).hexdigest()
                   for name in archive.namelist() if not name.endswith('/')}
                   for jar in (out / 'lib' / arm).glob('*.jar')
                   for archive in [zipfile.ZipFile(jar)]} for arm in cores}
    write(out / 'provenance.json', {'jarEntries': jar_entries, 'protocol': CONFIG, 'harnessSha256': sha(HERE / 'Bf16ModelProof.java'),
          'java': subprocess.check_output(['java', '-version'], stderr=subprocess.STDOUT, text=True),
          'cpu': subprocess.check_output(['lscpu'], text=True), 'platform': platform.platform(),
          'nativeSha256': sha(native), 'artifacts': inventories, 'githubRunId': os.getenv('GITHUB_RUN_ID'),
          'githubJob': os.getenv('GITHUB_JOB'), 'runnerName': os.getenv('RUNNER_NAME'),
          'model': args.model, 'shard': args.shard})
    classes = out / 'classes'
    classes.mkdir(exist_ok=True)
    subprocess.run(['javac', '--add-modules', 'jdk.incubator.vector', '-cp', str(out / 'lib/candidate/*'),
                    '-d', str(classes), str(HERE / 'Bf16ModelProof.java')], check=True)
    entry = next(m for m in CONFIG['models'] if m['key'] == args.model)
    model_dir = Path(os.environ['RUNNER_TEMP']) / 'paired-models' / args.model
    for file in entry['files']:
        path = model_dir / file['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or sha(path) != file['sha256']:
            subprocess.run(['curl', '-fL', '--retry', '3', file['uri'], '-o', str(path)], check=True)
        assert path.stat().st_size == file['sizeBytes'] and sha(path) == file['sha256']
    model = model_dir / entry['entrypoint']
    base = ['java', '--add-modules', 'jdk.incubator.vector', '--enable-native-access=ALL-UNNAMED', '-Xms2g', '-Xmx10g']
    return out, classes, cores, native, model, base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['bonsai2', 'qwen-bf16'], required=True)
    parser.add_argument('--shard', choices=['0', '1'], required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--candidate-core', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out, classes, cores, native, model, base = prepare(args)
    fixture_codes = {}
    for arm in ['baseline', 'candidate']:
        prefix = out / f'{args.model}-{arm}-fixture'
        command = [*base, '-cp', f'{out}/lib/{arm}/*:{classes}',
                   'com.integrallis.models.backend.purejava.Bf16ModelProof', 'fixture', str(model), str(native)]
        fixture_codes[arm] = run(command, prefix)
        origins(prefix, cores[arm])
    write(out / 'fixtures.json', fixture_codes)
    assert fixture_codes['candidate'] == 0, 'candidate fixture failed; timing blocked'
    for pair in CONFIG['shards'][args.shard]:
        order = ['baseline', 'candidate'] if pair % 2 == 0 else ['candidate', 'baseline']
        for arm in order:
            prefix = out / f'{args.model}-pair{pair}-{arm}'
            command = ['/usr/bin/time', '-v', *base, '-cp', f'{out}/lib/{arm}/*:{classes}',
                       'com.integrallis.models.backend.purejava.Bf16ModelProof', 'timing', str(model),
                       str(native), str(HERE / 'data/tasks.json'), str(pair)]
            assert run(command, prefix) == 0, prefix
            origins(prefix, cores[arm])
    (out / 'STATUS').write_text('COMPLETE\n')


if __name__ == '__main__':
    main()
