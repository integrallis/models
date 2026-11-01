"""Frozen public-QA blocks with exact artifact and same-host provenance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import urllib.request

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / 'protocol.json').read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def run(command, prefix, timeout=5400, env=None):
    started = time.time()
    code = None
    status = 'RUNNING'
    receipt = {'command': command, 'exitCode': None, 'status': status, 'startedAt': started, 'timeoutSeconds': timeout}
    write(prefix.with_suffix('.execution.json'), receipt)
    try:
        with prefix.with_suffix('.stdout').open('w') as stdout, prefix.with_suffix('.stderr').open('w') as stderr:
            code = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=timeout, env=env).returncode
        status = 'COMPLETE' if code == 0 else 'FAILED'
    except subprocess.TimeoutExpired:
        status = 'TIMED_OUT'
        raise
    finally:
        receipt.update(exitCode=code, status=status, completedAt=time.time())
        write(prefix.with_suffix('.execution.json'), receipt)
    if code != 0:
        raise RuntimeError(f'{prefix.name} exit={code}')


def rows(path):
    header, *data = [json.loads(line) for line in path.read_text().splitlines()]
    return header, data


def check_rows(path, first, count):
    header, data = rows(path)
    if header['caseSha256'] != P['inputs']['public-qa-cases.json']:
        raise ValueError('case manifest changed')
    if len(data) != count or [r['index'] for r in data] != list(range(first, first + count)):
        raise ValueError('missing, duplicate or reordered cases')
    expected = {r['index']: r for r in map(json.loads, (HERE / 'public-qa-reference-tokens.jsonl').read_text().splitlines())}
    for row in data:
        if row.get('error') or row['inputTokenIds'] != expected[row['index']]['inputTokenIds']:
            raise ValueError(f'case {row["index"]} failed or prompt tokens changed')
    return header, data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'block'])
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--block', type=int)
    args = parser.parse_args()
    if not 0 <= args.shard < P['shards']:
        raise ValueError('shard outside frozen range')
    root = Path.cwd()
    out = root / 'out' / f'shard-{args.shard}'
    out.mkdir(parents=True, exist_ok=True)
    work = Path(os.environ['RUNNER_TEMP']) / 'bonsai-final-qa'
    work.mkdir(exist_ok=True)
    model = Path.home() / '.jvllm/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf'
    native = root / 'models/backend-native/build/rust-target/release/libjmodels_kernels.so'
    reference_bin = work / 'prism/build/bin'
    reference_env = dict(os.environ, LD_LIBRARY_PATH=str(reference_bin))
    classes = work / 'classes'
    libs = work / 'lib'
    java = ['java', '--add-modules', 'jdk.incubator.vector', '--enable-native-access=ALL-UNNAMED', '-XX:ActiveProcessorCount=4', '-Xmx4g', f'-Dmodels.native.kernels.library={native}', '-cp', f'{libs}/*:{classes}', 'PublicQaProbe']
    for name, expected in P['inputs'].items():
        if sha(HERE / name) != expected:
            raise ValueError(f'harness input changed: {name}')
    if args.stage == 'prepare':
        for folder, expected in [('models', P['modelsRevision']), ('vectors', P['vectorsRevision'])]:
            if subprocess.check_output(['git', '-C', folder, 'rev-parse', 'HEAD'], text=True).strip() != expected:
                raise ValueError('runtime source changed')
        core = root / 'vectors/vectors-core/build/libs/vectors-core-0.1.30.jar'
        if sha(core) != P['coreSha256']:
            raise ValueError('core binary changed')
        libs.mkdir(exist_ok=True)
        source = root / 'models/backend-native/build/paired-runtime'
        for jar in source.glob('*.jar'):
            if not jar.name.startswith('vectors-core-'):
                shutil.copy2(jar, libs / jar.name)
        shutil.copy2(core, libs / core.name)
        if len(list(libs.glob('vectors-core-*.jar'))) != 1:
            raise ValueError('ambiguous core classpath')
        classes.mkdir(exist_ok=True)
        run(['javac', '--add-modules', 'jdk.incubator.vector', '-cp', f'{libs}/*', '-d', str(classes), str(HERE / 'PublicQaProbe.java')], out / 'compile')
        archive = work / 'reference.tar.gz'
        # Do not put signed URLs in recorded command arguments or logs.
        with urllib.request.urlopen(os.environ['REFERENCE_ARCHIVE_URL'], timeout=120) as response, archive.open('wb') as destination:
            shutil.copyfileobj(response, destination)
        if sha(archive) != P['referenceArchiveSha256']:
            raise ValueError('publisher archive changed')
        with tarfile.open(archive) as tar:
            members = [m for m in tar.getmembers() if m.name in P['referenceFiles']]
            tar.extractall(work, members=members, filter='data')
        for name, expected in P['referenceFiles'].items():
            if sha(work / name) != expected:
                raise ValueError(f'publisher binary changed: {name}')
        run(['ldd', str(reference_bin / 'llama-server')], out / 'publisher-ldd', env=reference_env)
        if 'not found' in (out / 'publisher-ldd.stdout').read_text():
            raise ValueError('publisher shared library missing')
        run([str(reference_bin / 'llama-server'), '--version'], out / 'publisher-version', env=reference_env)
        model.parent.mkdir(parents=True, exist_ok=True)
        if not model.exists() or sha(model) != P['modelSha256']:
            run(['curl', '-fL', '--retry', '3', P['modelUrl'], '-o', str(model)], out / 'download-model')
        if model.stat().st_size != P['modelSizeBytes'] or sha(model) != P['modelSha256']:
            raise ValueError('model artifact changed')
        write(out / 'manifest.json', {'protocol': P, 'shard': args.shard, 'runId': os.environ.get('GITHUB_RUN_ID'), 'runnerName': os.environ.get('RUNNER_NAME'), 'host': subprocess.check_output(['lscpu'], text=True), 'java': subprocess.check_output(['java', '-version'], stderr=subprocess.STDOUT, text=True), 'rustc': subprocess.check_output(['rustup', 'run', '1.96.0', 'rustc', '-vV'], text=True), 'cargo': subprocess.check_output(['rustup', 'run', '1.96.0', 'cargo', '--version'], text=True), 'publisherLibraryPath': str(reference_bin), 'nativeSha256': sha(native), 'jars': {p.name: sha(p) for p in sorted(libs.glob('*.jar'))}, 'harnessRevision': subprocess.check_output(['git', '-C', str(HERE), 'rev-parse', 'HEAD'], text=True).strip()})
        first = args.shard * P['casesPerShard']
        token_file = out / 'token-preflight.jsonl'
        run(java + [str(model), str(HERE / 'public-qa-cases.json'), str(token_file), 'validate', '2048', str(first), str(P['casesPerShard']), P['modelsRevision']], out / 'token-preflight')
        check_rows(token_file, first, P['casesPerShard'])
        (out / 'PREPARED').write_text('Exact inputs and every prompt token verified\n')
        return
    if args.block is None or not 0 <= args.block < P['blocks'] or not (out / 'PREPARED').exists():
        raise ValueError('unprepared or invalid block')
    block = out / f'block-{args.block}'
    block.mkdir()
    first = args.shard * P['casesPerShard'] + args.block * P['casesPerBlock']
    count = P['casesPerBlock']
    write(block / 'identity.json', {'shard': args.shard, 'block': args.block, 'first': first, 'count': count, 'runnerName': os.environ.get('RUNNER_NAME'), 'runId': os.environ.get('GITHUB_RUN_ID'), 'manifestSha256': sha(out / 'manifest.json')})
    (block / 'STATUS').write_text('RUNNING\n')
    deadline = time.monotonic() + P['blockTimeoutSeconds']
    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError('paired block exhausted its frozen time budget')
        return seconds
    try:
        arms = ['models', 'publisher'] if (args.shard + args.block) % 2 == 0 else ['publisher', 'models']
        for arm in arms:
            output = block / f'{arm}.jsonl'
            if arm == 'models':
                run(java + [str(model), str(HERE / 'public-qa-cases.json'), str(output), 'rust-ffm', '2048', str(first), str(count), P['modelsRevision']], block / arm, timeout=remaining())
            else:
                command = [str(work / 'prism/build/bin/llama-server'), '-m', str(model), '-c', '2048', '-t', '4', '-tb', '4', '-ngl', '0', '--parallel', '1', '--host', '127.0.0.1', '--port', '8080', '--no-warmup']
                write(block / 'reference-command.json', command)
                with (block / 'publisher-server.log').open('w') as log:
                    server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=reference_env)
                    try:
                        for _ in range(180):
                            remaining()
                            if server.poll() is not None:
                                raise RuntimeError('publisher server exited')
                            try:
                                with urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2) as response:
                                    if response.status == 200:
                                        break
                            except OSError:
                                time.sleep(1)
                        else:
                            raise RuntimeError('publisher startup timeout')
                        run(['python3', str(HERE / 'run-public-reference.py'), str(HERE / 'public-qa-cases.json'), str(output), 'http://127.0.0.1:8080', str(first), str(count)], block / arm, timeout=remaining())
                    finally:
                        server.terminate()
                        try:
                            server.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            server.kill()
                            server.wait()
            check_rows(output, first, count)
        (block / 'STATUS').write_text('COMPLETE\n')
    except BaseException as error:
        (block / 'STATUS').write_text('FAILED ' + repr(error) + '\n')
        raise


if __name__ == '__main__':
    main()
