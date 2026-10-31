"""One SHA-pinned artifact: released default smoke, then a serial same-host pair.

Provisioning, deadlines and reference-server startup belong to the disposable worker.
This runner never builds a product or downloads a substitute model.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_payload(payload):
    provenance = json.loads((payload / 'provenance.json').read_text())
    records = json.loads((payload / 'classpath.json').read_text())
    if {r['file'] for r in records} != {p.name for p in (payload / 'lib').glob('*.jar')}:
        raise ValueError('unrecorded or missing classpath JAR')
    for row in records:
        name = row['file']
        if Path(name).name != name or digest(payload / 'lib' / name) != row['sha256']:
            raise ValueError('classpath identity mismatch: ' + name)
        product = name.startswith(('models-', 'backend-', 'vectors-')) and not name.startswith('models-rag-bench-')
        if product and row.get('origin') != 'Maven Central':
            raise ValueError('product is not released: ' + name)
    return provenance


def require_default_smoke(report, job):
    summary = report['summary']
    if not (report['modelId'] == job['id'] and report['artifactSha256'] == job['sha256']
            and summary['totalAttempts'] > 0
            and summary['totalAttempts'] == summary['successfulAttempts']
            and summary['correctAnswerRate'] == 1 and summary['abstentionAccuracy'] == 1
            and not report['failures'] and report['settings']['warmups'] == 0
            and report['settings']['iterations'] == 1
            and report['settings']['generationControls']['promptCache'] == 'longest-common-prefix'
            and report['backendDiagnostics']['environment']['native-quantized-decode'] == 'true'):
        raise ValueError('released library-default smoke failed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--java', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-version', required=True)
    parser.add_argument('--reference-model', required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    job = json.loads(args.job.read_text())
    commands = []
    status = out / 'STATUS'

    def run(name, command, timeout=3600):
        row = dict(name=name, command=command, startedAt=time.time())
        commands.append(row)
        (out / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        with (out / (name + '.log')).open('w') as log:
            result = subprocess.run(['/usr/bin/time', '-v', *command], stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        row.update(exitCode=result.returncode, completedAt=time.time())
        (out / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        if result.returncode:
            raise RuntimeError(name + ' failed; see retained log')

    try:
        for variable in ['JAVA_TOOL_OPTIONS', 'JDK_JAVA_OPTIONS', '_JAVA_OPTIONS', 'JAVA_OPTS']:
            if 'models.' in os.environ.get(variable, ''):
                raise ValueError(variable + ' contains Models tuning properties')
        if args.model.stat().st_size != job['sizeBytes'] or digest(args.model) != job['sha256']:
            raise ValueError('model identity mismatch')
        provenance = verify_payload(args.payload)
        for name in ['provenance.json', 'classpath.json']:
            shutil.copyfile(args.payload / name, out / name)
        shutil.copyfile(args.job, out / 'job.json')
        shutil.copyfile(__file__, out / 'runner.py')
        for name, command in [('hostname', ['hostname']), ('lscpu', ['lscpu']), ('processes-before', ['ps', 'aux'])]:
            (out / (name + '.txt')).write_bytes(subprocess.check_output(command))
        (out / 'java-version.txt').write_bytes(subprocess.check_output([args.java, '-version'], stderr=subprocess.STDOUT))
        label = 'models@' + provenance['modelsVersion'] + '+' + provenance['harnessRevision']
        java = [args.java, '--add-modules', 'jdk.incubator.vector', '--enable-native-access=ALL-UNNAMED', '-Xmx16g']
        classpath = ['-cp', str(args.payload / 'lib/*')]
        common = ['com.integrallis.models.rag.RagBenchmarkCli', '--framework', 'plain-java', '--model-id', job['id'],
                  '--workload', job['workload'], '--prompt-template', job['template'], '--top-k', '1',
                  '--max-tokens', str(job['maxTokens']), '--context', str(job['contextLength']), '--threads', str(job['threads'])]
        candidate = ['--backend', 'rust-ffm', '--backend-version', label, '--model', str(args.model)]
        status.write_text('RUNNING_RELEASED_DEFAULT_SMOKE\n')
        smoke_dir = out / 'default-correctness'
        smoke_dir.mkdir()
        smoke_path = smoke_dir / 'models-rust-ffm.json'
        run('default-smoke', java + classpath + common + candidate + ['--warmups', '0', '--iterations', '1', '--output', str(smoke_path)])
        require_default_smoke(json.loads(smoke_path.read_text()), job)
        (smoke_dir / 'configuration.json').write_text(json.dumps(dict(configuration='library-defaults',
            tuningSystemProperties=[], backendVersion=label, reportSha256=digest(smoke_path)), indent=2) + '\n')
        status.write_text('RUNNING_PAIRED_RAG\n')
        trials = ['--warmups', '1', '--iterations', '3']
        tuned = [] if job.get('decodeThreads') is None else ['-Dmodels.native.kernels.decodeThreads=' + str(job['decodeThreads'])]
        run('models-rag', java + tuned + classpath + common + candidate + trials + ['--output', str(out / 'models-rag.json')])
        if job['comparator'] == 'ollama':
            reference = ['--backend', 'ollama', '--endpoint', 'http://127.0.0.1:11434']
            policy = []
        else:
            raise ValueError('use the separate publisher runner for non-Ollama artifacts')
        run('reference-rag', java + classpath + common + reference + ['--backend-version', args.reference_version,
            '--model', args.reference_model, '--artifact', str(args.model)] + trials + ['--output', str(out / 'reference-rag.json')])
        # A negative verdict is a completed measurement, not an infrastructure exception.
        run('qualification', java + classpath + ['com.integrallis.models.rag.RagQualificationCli',
            '--candidate', str(out / 'models-rag.json'), '--comparator', str(out / 'reference-rag.json'),
            *policy, '--output', str(out / 'verdict.json')], 120)
        status.write_text('COMPLETE ' + json.loads((out / 'verdict.json').read_text())['qualification']['verdict'] + '\n')
    except BaseException as error:
        status.write_text('FAILED ' + repr(error) + '\n')
        raise


if __name__ == '__main__':
    main()
