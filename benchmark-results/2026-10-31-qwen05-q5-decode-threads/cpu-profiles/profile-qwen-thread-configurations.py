"""CPU sampling only; instrumented timings are never qualification evidence."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location('pair', a.work / 'run-released-pair.py')
    pair = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pair)
    pair.verify_payload(a.work / 'released')
    protocol = json.loads((a.work / 'protocol.json').read_text())
    job = protocol['job']
    model = a.work / 'model.gguf'
    assert model.stat().st_size == job['sizeBytes'] and pair.digest(model) == job['sha256']
    for native in [16, 8]:
        out = a.output / ('native-' + str(native))
        out.mkdir()
        command = [str(a.work / 'jdk/bin/java'), '--add-modules', 'jdk.incubator.vector',
            '--enable-native-access=ALL-UNNAMED', '-Xmx16g', '-XX:+UnlockDiagnosticVMOptions',
            '-XX:+DebugNonSafepoints', '-Dmodels.native.kernels.decodeThreads=8',
            '-Dmodels.native.kernels.threads=' + str(native), '-cp', str(a.work / 'released/lib/*'),
            'com.integrallis.models.rag.RagBenchmarkCli', '--framework', 'plain-java',
            '--backend', 'rust-ffm', '--backend-version', 'models@0.5.9+instrumented-diagnostic',
            '--model-id', job['id'], '--model', str(model), '--workload', job['workload'],
            '--prompt-template', job['template'], '--threads', '16', '--context', '2048',
            '--max-tokens', '256', '--top-k', '1', '--warmups', '3', '--iterations', '5',
            '--output', str(out / 'instrumented-rag.json')]
        (out / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
        with (out / 'java.log').open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                time.sleep(15)
                if process.poll() is not None:
                    raise RuntimeError('benchmark exited before profiling began')
                profile = [str(a.work / 'profiler/bin/asprof'), '-e', 'cpu', '-d', '20',
                    '-i', '1000000', '-t', '--cstack', 'dwarf', '-f', str(out / 'cpu.collapsed'), str(process.pid)]
                (out / 'profiler-command.json').write_text(json.dumps(profile, indent=2) + '\n')
                with (out / 'profiler.log').open('w') as prof_log:
                    subprocess.run(profile, stdout=prof_log, stderr=subprocess.STDOUT, check=True, timeout=60)
                if process.wait(timeout=600):
                    raise RuntimeError('instrumented benchmark failed')
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        print('profile complete', native, flush=True)


if __name__ == '__main__':
    main()
