"""Keep startup follow-up separate; all 400 predeclared pairs are mandatory."""
import importlib.util
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('paired_analysis', HERE.parent / 'bonsai-paired/analyze.py')
paired = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paired)
CONFIG = json.loads((HERE / 'protocol.json').read_text())


def observation(directory, model, pair, arm):
    prefix = directory / f'{model}-pair{pair}-{arm}'
    assert json.loads(prefix.with_suffix('.execution.json').read_text())['exitCode'] == 0
    rows = [json.loads(line) for line in prefix.with_suffix('.stdout').read_text().splitlines()
            if line.startswith('{')]
    loads = [row for row in rows if row['kind'] == 'load']
    assert len(loads) == 1 and loads[0]['pair'] == pair and loads[0]['phase'] == 'load-only'
    assert loads[0]['nanos'] > 0
    return loads[0]['nanos']


def analyze(root, arch, model):
    groups = []
    expected = json.loads((HERE / f'expected-{arch}.json').read_text())
    for shard, pairs in CONFIG['shards'].items():
        directory = root / f'load-{arch}-{model}-{shard}'
        assert (directory / 'STATUS').read_text().strip() == 'COMPLETE'
        provenance = json.loads((directory / 'provenance.json').read_text())
        assert provenance['loadProtocol'] == CONFIG
        assert provenance['loadHarnessSha256'] == CONFIG['loadHarnessSha256']
        assert provenance['model'] == model and provenance['shard'] == shard
        assert provenance['jarEntries'] == expected['jarEntries']
        assert provenance['nativeSha256'] == expected['nativeSha256']
        logs = []
        for pair in pairs:
            assert json.loads((directory / f'pair{pair}.complete.json').read_text())['pair'] == pair
            logs.append(math.log(observation(directory, model, pair, 'candidate') /
                                 observation(directory, model, pair, 'baseline')))
        groups.append(logs)
    result = paired.bootstrap(groups, CONFIG['limits']['loadNanos'])
    return {'architecture': arch, 'model': model, 'pairs': 400, 'loadNanos': result,
            'pass': result['pass'], 'scope': CONFIG['scope'], 'originalMacroReclassified': False}


def main():
    root, output = Path(sys.argv[1]), Path(sys.argv[2])
    models = sys.argv[3:] or ['qwen-bf16']
    results = []
    for model in models:
        for arch in ['x86', 'arm']:
            try:
                results.append(analyze(root, arch, model))
            except Exception as error:
                results.append({'architecture': arch, 'model': model, 'pass': False,
                                'error': f'{type(error).__name__}: {error}'})
    output.write_text(json.dumps({'protocol': CONFIG, 'results': results}, indent=2) + '\n')
    return 0 if all(row['pass'] for row in results) else 1


if __name__ == '__main__':
    sys.exit(main())
