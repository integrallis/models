"""One immutable Bonsai pair per host, with independent stages for retained checkpoints."""
import argparse
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('paired', HERE.parent / 'bonsai-paired/run.py')
paired = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paired)
CONFIG = json.loads((HERE / 'protocol.json').read_text())
paired.CONFIG = CONFIG


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', choices=['x86', 'arm'], required=True)
    p.add_argument('--shard', type=int, choices=range(10), required=True)
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--candidate-core', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=['prepare', 'fixture-baseline', 'fixture-candidate', 'first', 'second'], required=True)
    args = p.parse_args()
    args.model = 'bonsai2'
    args.shard = str(args.shard)
    out = args.output.resolve()
    if args.stage == 'prepare':
        assert paired.sha(args.candidate_core) == CONFIG['candidateCoreSha256']
        out, classes, cores, native, model, base = paired.prepare(args)
        provenance = json.loads((out / 'provenance.json').read_text())
        expected = json.loads((HERE / f'expected-{args.arch}.json').read_text())
        assert provenance['jarEntries'] == expected['jarEntries']
        assert provenance['nativeSha256'] == expected['nativeSha256']
        assert '25.0.3' in provenance['java']
        provenance['hostResources'] = {
            'cpuAffinity': sorted(os.sched_getaffinity(0)),
            'logicalCpuCount': os.cpu_count(),
            'linuxFiles': {name: Path(name).read_text().strip() if Path(name).is_file() else None
                for name in ['/proc/meminfo', '/sys/fs/cgroup/memory.max',
                             '/sys/fs/cgroup/cpu.max', '/sys/fs/cgroup/cpuset.cpus.effective']}}
        paired.write(out / 'provenance.json', provenance)
        paired.write(out / 'prepared.json', {'native': str(native), 'model': str(model), 'base': base,
                     'cores': {k: str(v) for k, v in cores.items()}})
        return
    state = json.loads((out / 'prepared.json').read_text())
    if args.stage.startswith('fixture-'):
        arm = args.stage.removeprefix('fixture-')
        prefix = out / f'bonsai2-{arm}-fixture'
        assert not prefix.with_suffix('.execution.json').exists(), 'do not replace fixture observation'
        command = [*state['base'], '-cp', f'{out}/lib/{arm}/*:{out}/classes',
                   'com.integrallis.models.backend.purejava.Bf16ModelProof', 'fixture', state['model'], state['native']]
        code = paired.run(command, prefix, timeout=CONFIG['processTimeoutSeconds'])
        paired.origins(prefix, Path(state['cores'][arm]))
        if arm == 'candidate':
            assert code == 0, 'candidate fixture failed; timing blocked'
        return
    candidate_fixture = out / 'bonsai2-candidate-fixture.execution.json'
    assert json.loads(candidate_fixture.read_text())['exitCode'] == 0
    pair = int(args.shard)
    order = ['baseline', 'candidate'] if pair % 2 == 0 else ['candidate', 'baseline']
    arm = order[0 if args.stage == 'first' else 1]
    prefix = out / f'bonsai2-pair{pair}-{arm}'
    assert not prefix.with_suffix('.execution.json').exists(), 'do not replace measured observation'
    command = ['/usr/bin/time', '-v', *state['base'], '-cp', f'{out}/lib/{arm}/*:{out}/classes',
               'com.integrallis.models.backend.purejava.Bf16ModelProof', 'timing', state['model'],
               state['native'], str(HERE.parent / 'bonsai-paired/data/tasks.json'), str(pair)]
    assert paired.run(command, prefix, timeout=CONFIG['processTimeoutSeconds']) == 0
    paired.origins(prefix, Path(state['cores'][arm]))
    if args.stage == 'second':
        fixtures = {a:json.loads((out / f'bonsai2-{a}-fixture.execution.json').read_text())['exitCode'] for a in order}
        paired.write(out / 'fixtures.json', fixtures)
        paired.write(out / f'pair{pair}.complete.json', {'pair': pair, 'order': order})
        (out / 'STATUS').write_text('COMPLETE\n')
    print(f'PROCESS RETAINED pair={pair} arm={arm}', flush=True)


if __name__ == '__main__':
    main()
