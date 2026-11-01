"""Fixed 320/160-pair Qwen cohort; no optional stopping or replacing failed measurements."""
import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAIRED = HERE.parent / 'bonsai-paired'
spec = importlib.util.spec_from_file_location('paired', PAIRED / 'run.py')
paired = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paired)
CONFIG = json.loads((HERE / 'protocol.json').read_text())
paired.CONFIG = CONFIG


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', choices=['x86', 'arm'], required=True)
    p.add_argument('--shard', required=True)
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--candidate-core', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--block', type=int, choices=range(4), required=True)
    args = p.parse_args()
    args.model = 'qwen-bf16'
    pairs = CONFIG['shardsByArchitecture'][args.arch][args.shard]
    out = args.output.resolve()
    if args.block == 0:
        assert paired.sha(args.candidate_core) == CONFIG['candidateCoreSha256']
        out, classes, cores, native, model, base = paired.prepare(args)
        provenance = json.loads((out / 'provenance.json').read_text())
        expected = json.loads((HERE / f'expected-{args.arch}.json').read_text())
        assert provenance['jarEntries'] == expected['jarEntries']
        assert provenance['nativeSha256'] == expected['nativeSha256']
        assert '25.0.3' in provenance['java']
        provenance.update(precisionProtocol=CONFIG, precisionRunnerSha256=paired.sha(HERE / 'run.py'))
        paired.write(out / 'provenance.json', provenance)
        fixture_codes = {}
        for arm in ['baseline', 'candidate']:
            prefix = out / f'{args.model}-{arm}-fixture'
            command = [*base, '-cp', f'{out}/lib/{arm}/*:{classes}',
                       'com.integrallis.models.backend.purejava.Bf16ModelProof', 'fixture', str(model), str(native)]
            fixture_codes[arm] = paired.run(command, prefix)
            paired.origins(prefix, cores[arm])
        paired.write(out / 'fixtures.json', fixture_codes)
        assert fixture_codes['candidate'] == 0, 'candidate fixture failed; timing blocked'
        paired.write(out / 'prepared.json', {'native': str(native), 'model': str(model), 'base': base,
                     'cores': {k: str(v) for k, v in cores.items()}})
    state = json.loads((out / 'prepared.json').read_text())
    for pair in pairs[args.block*20:(args.block+1)*20]:
        order = ['baseline', 'candidate'] if pair % 2 == 0 else ['candidate', 'baseline']
        for arm in order:
            prefix = out / f'{args.model}-pair{pair}-{arm}'
            assert not prefix.with_suffix('.execution.json').exists(), 'do not replace measured observations'
            command = ['/usr/bin/time', '-v', *state['base'], '-cp', f'{out}/lib/{arm}/*:{out}/classes',
                       'com.integrallis.models.backend.purejava.Bf16ModelProof', 'timing', state['model'],
                       state['native'], str(PAIRED / 'data/tasks.json'), str(pair)]
            assert paired.run(command, prefix, timeout=300) == 0
            paired.origins(prefix, Path(state['cores'][arm]))
        paired.write(out / f'pair{pair}.complete.json', {'pair': pair, 'order': order})
        print(f'PAIR COMPLETE {pair}', flush=True)
    paired.write(out / f'block{args.block}.complete.json', {'block': args.block})
    if args.block == 3:
        (out / 'STATUS').write_text('COMPLETE\n')


if __name__ == '__main__':
    main()
