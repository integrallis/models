"""Controlled source-native experiment. Results cannot qualify a released artifact."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import time


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    protocol = json.loads((a.work / 'protocol.json').read_text())
    job = protocol['job']
    pair = load_module('pair', a.work / 'run-released-pair.py')
    sweep = load_module('sweep', a.work / 'run-decode-thread-experiment.py')
    pair.verify_payload(a.work / 'released')
    model = a.work / 'model.gguf'
    assert model.stat().st_size == job['sizeBytes'] and pair.digest(model) == job['sha256']
    kernels = {}
    for variant in ['baseline', 'candidate']:
        library = a.work / variant / 'target/release/libjmodels_kernels.so'
        kernels[variant] = dict(path=str(library), sha256=pair.digest(library))
    (a.output / 'native-libraries.json').write_text(json.dumps(kernels, indent=2) + '\n')
    java = [str(a.work / 'jdk/bin/java'), '--add-modules', 'jdk.incubator.vector',
            '--enable-native-access=ALL-UNNAMED', '-Xmx16g']
    cp = ['-cp', str(a.work / 'released/lib/*')]
    common = ['com.integrallis.models.rag.RagBenchmarkCli', '--framework', 'plain-java',
              '--model-id', job['id'], '--workload', job['workload'], '--prompt-template', job['template'],
              '--threads', str(job['threads']), '--context', str(job['contextLength']),
              '--max-tokens', str(job['maxTokens']), '--top-k', '1']
    commands = []

    def run(name, command):
        record = dict(name=name, command=command, startedAt=time.time())
        commands.append(record)
        (a.output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        with (a.output / (name + '.log')).open('w') as log:
            result = subprocess.run(['/usr/bin/time', '-v', *command], stdout=log,
                                    stderr=subprocess.STDOUT, timeout=900)
        record.update(exitCode=result.returncode, completedAt=time.time())
        (a.output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        if result.returncode:
            raise RuntimeError(name + ' failed')

    def benchmark(name, variant, smoke=False):
        # Unload the reference between arms, including reference-first repetitions.
        subprocess.run(['runuser', '-u', 'root', '--', 'ollama', 'stop', 'qualification'], check=True)
        tuning = []
        label = 'ollama@0.40.13'
        if variant == 'reference':
            backend = ['--backend', 'ollama', '--endpoint', 'http://127.0.0.1:11434',
                       '--model', 'qualification', '--artifact', str(model)]
        else:
            backend = ['--backend', 'rust-ffm', '--model', str(model)]
            label = 'models@0.5.9+released-native'
            if variant != 'released':
                tuning = ['-Dmodels.native.kernels.library=' + kernels[variant]['path']]
                label = 'models@0.5.9+EXPERIMENTAL-native-' + variant + '-' + kernels[variant]['sha256']
            if not smoke:
                tuning += ['-Dmodels.native.kernels.threads=16', '-Dmodels.native.kernels.decodeThreads=8']
        report = a.output / (name + '.json')
        command = java + tuning + cp + common + backend + ['--backend-version', label,
            '--warmups', '0' if smoke else '1', '--iterations', '1' if smoke else '3', '--output', str(report)]
        run(name, command)
        data = json.loads(report.read_text())
        if smoke:
            pair.require_default_smoke(data, job)
        else:
            assert data['summary']['totalAttempts'] == data['summary']['successfulAttempts'] == 27
            if variant != 'reference':
                env = data['backendDiagnostics']['environment']
                assert env['native-kernel-threads'] == '16' and env['native-kernel-decode-threads'] == '8'
        return data

    smoke_signature = None
    for variant in ['released', 'baseline', 'candidate']:
        report = benchmark('smoke-' + variant, variant, True)
        signature = sweep.answer_signature(report)
        if smoke_signature is None:
            smoke_signature = signature
        assert signature == smoke_signature, 'native override changed default smoke answers'
    rows = []
    baseline_signature = None
    for i, order in enumerate(protocol['orders'], 1):
        reports = {variant: benchmark(f'repeat-{i}-{variant}', variant) for variant in order}
        if baseline_signature is None:
            baseline_signature = sweep.answer_signature(reports['baseline'])
        for variant in ['baseline', 'candidate']:
            name = f'repeat-{i}-{variant}'
            run(name + '-verdict', java + cp + ['com.integrallis.models.rag.RagQualificationCli',
                '--candidate', str(a.output / (name + '.json')), '--comparator',
                str(a.output / f'repeat-{i}-reference.json'), '--output', str(a.output / (name + '-verdict.json'))])
            q = json.loads((a.output / (name + '-verdict.json')).read_text())['qualification']
            assert len(q['comparisons']) == 1 and not q['exclusions']
            c = q['comparisons'][0]
            row = dict(repeat=i, variant=variant, order=order,
                answersStable=sweep.answer_signature(reports[variant]) == baseline_signature,
                qualified=q['qualified'], verdict=q['verdict'], modelAnswerRate=q['modelAnswerRate'],
                modelAnswerCorrectRate=q['modelAnswerCorrectRate'],
                tokensPerSecond=reports[variant]['summary']['p50DecodeTokensPerSecond'],
                decodeRatio=c['decodeThroughputRatio'], latencyRatio=c['endToEndLatencyRatio'])
            rows.append(row)
            print(json.dumps(row), flush=True)
        (a.output / 'summary.json').write_text(json.dumps(dict(complete=False, trials=rows), indent=2) + '\n')
    candidate = [r for r in rows if r['variant'] == 'candidate']
    summary = dict(complete=True, trials=rows, releasedQualification=False,
        experimentalRepeatGatePassed=all(r['qualified'] and r['answersStable'] for r in candidate),
        experimentalMarginTargetMet=all(r['qualified'] and r['answersStable'] and r['decodeRatio'] >= .85 for r in candidate),
        scope='Source native overrides only; released qualification requires a released-runtime repeat. Answer equality means raw text, counts, prompts and evaluations, not token-ID equality.')
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
