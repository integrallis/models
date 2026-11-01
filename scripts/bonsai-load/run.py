"""Fixed prospective load cohort; upload checkpoints between blocks, never during timing."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
PAIRED = HERE.parent / 'bonsai-paired'
spec = importlib.util.spec_from_file_location('paired', PAIRED / 'run.py')
paired = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paired)
CONFIG = json.loads((HERE / 'protocol.json').read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['qwen-bf16', 'bonsai2'], required=True)
    parser.add_argument('--shard', choices=['0', '1'], required=True)
    parser.add_argument('--arch', choices=['x86', 'arm'], required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--candidate-core', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--block', type=int, choices=range(10), required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.block == 0:
        assert paired.sha(args.candidate_core) == CONFIG['candidateCoreSha256']
        out, classes, cores, native, model, base = paired.prepare(args)
        provenance = json.loads((out / 'provenance.json').read_text())
        expected = json.loads((HERE / f'expected-{args.arch}.json').read_text())
        assert provenance['jarEntries'] == expected['jarEntries'], 'runtime differs from original macro'
        assert provenance['nativeSha256'] == expected['nativeSha256'], 'native binary differs'
        assert '25.0.3' in provenance['java']
        provenance.update(loadProtocol=CONFIG, loadHarnessSha256=paired.sha(HERE / 'Bf16LoadProof.java'),
                          loadRunnerSha256=paired.sha(HERE / 'run.py'))
        paired.write(out / 'provenance.json', provenance)
        subprocess.run(['javac', '--add-modules', 'jdk.incubator.vector', '-cp',
                        f'{out}/lib/candidate/*:{classes}', '-d', str(classes),
                        str(HERE / 'Bf16LoadProof.java')], check=True)
        paired.write(out / 'prepared.json', {'native': str(native), 'model': str(model),
                     'base': base, 'cores': {k: str(v) for k, v in cores.items()}})
    state = json.loads((out / 'prepared.json').read_text())
    assert paired.sha(HERE / 'Bf16LoadProof.java') == CONFIG['loadHarnessSha256']
    for pair in CONFIG['shards'][args.shard][args.block * 20:(args.block + 1) * 20]:
        order = ['baseline', 'candidate'] if pair % 2 == 0 else ['candidate', 'baseline']
        for arm in order:
            prefix = out / f'{args.model}-pair{pair}-{arm}'
            assert not prefix.with_suffix('.execution.json').exists(), 'never retry a measured process'
            command = ['/usr/bin/time', '-v', *state['base'], '-cp', f'{out}/lib/{arm}/*:{out}/classes',
                       'com.integrallis.models.backend.purejava.Bf16LoadProof', state['model'],
                       state['native'], str(PAIRED / 'data/tasks.json'), str(pair)]
            assert paired.run(command, prefix, CONFIG['processTimeoutSeconds']) == 0, prefix
            paired.origins(prefix, Path(state['cores'][arm]))
        # Complete pair receipts, no data selection or in-flight measurement perturbation.
        paired.write(out / f'pair{pair}.complete.json', {'pair': pair, 'order': order})
        print(f'PAIR COMPLETE {pair}', flush=True)
    paired.write(out / f'block{args.block}.complete.json', {'block': args.block})
    if args.block == 9:
        (out / 'STATUS').write_text('COMPLETE\n')


if __name__ == '__main__':
    main()
