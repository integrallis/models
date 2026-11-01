"""Negative checks: incomplete observations and imprecise/regressing samples cannot pass."""
import importlib.util
import math
import json
from unittest.mock import patch
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('paired_analysis', Path(__file__).with_name('analyze.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GateTests(unittest.TestCase):
    def test_regression_fails_even_when_precise(self):
        result = module.bootstrap([[math.log(1.05)] * 5] * 2, 1.01)
        self.assertFalse(result['pass'])
        self.assertEqual('REGRESSION', result['status'])

    def test_wide_interval_cannot_pass(self):
        result = module.bootstrap([[math.log(x) for x in [.6, .8, 1., 1.2, 1.4]]] * 2, 1.01)
        self.assertFalse(result['pass'])
        self.assertFalse(result['precisionTargetMet'])

    def test_precise_improvement_passes(self):
        result = module.bootstrap([[math.log(.9)] * 5] * 2, 1.)
        self.assertTrue(result['pass'])

    def test_missing_shard_cannot_qualify(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                module.analyze(Path(folder), 'arm', 'bonsai2')


class CompleteObservationTests(unittest.TestCase):
    def populate(self, root):
        for shard in ['0', '1']:
            folder = root / f'paired-arm-bonsai2-{shard}'
            folder.mkdir()
            (folder / 'STATUS').write_text('COMPLETE\n')
            (folder / 'fixtures.json').write_text(json.dumps({'baseline': 1, 'candidate': 0}))
            (folder / 'provenance.json').write_text(json.dumps({
                'protocol': module.CONFIG, 'harnessSha256': module.CONFIG['harnessSha256'],
                'model': 'bonsai2', 'shard': shard, 'jarEntries': {'fixed': 'same'}}))
            for pair in module.CONFIG['shards'][shard]:
                for arm in ['baseline', 'candidate']:
                    prefix = folder / f'bonsai2-pair{pair}-{arm}'
                    prefix.with_suffix('.execution.json').write_text('{"exitCode":0}')
                    prefix.with_suffix('.stderr').write_text('Maximum resident set size (kbytes): 100\n')
                    rows = [{'kind': 'load', 'nanos': 100}]
                    for repeat in [-4, -3, -2, -1, 0, 1, 2]:
                        index = pair * 3 + repeat if repeat >= 0 else repeat + 4
                        rows.append({'kind': 'request', 'repeat': repeat, 'id': str(index),
                                     'dataset': 'jevbench' if index % 2 == 0 else 'snips',
                                     'fullLogitSha256': [str(index)] * 9, 'greedyTokens': [1] * 8,
                                     'inputTokens': [index], 'prediction': 'yes', 'expected': 'yes',
                                     **{metric: 100 for metric in ['prefillNanos', 'decodeNanos',
                                                                  'requestNanos', 'javaAllocatedBytes']}})
                    prefix.with_suffix('.stdout').write_text('\n'.join(map(json.dumps, rows)))

    def test_failed_baseline_fixture_does_not_hide_complete_comparison(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.populate(root)
            with patch.object(module, 'bootstrap', return_value={'pass': True, 'ratio': 1.}):
                result = module.analyze(root, 'arm', 'bonsai2')
            self.assertTrue(result['pass'])
            self.assertEqual(1, result['fixtureExitCodes'][0]['baseline'])
            self.assertEqual(15, result['quality']['candidate']['snips']['count'])

    def test_candidate_nondeterminism_blocks_acceptance(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.populate(root)
            path = root / 'paired-arm-bonsai2-0/bonsai2-pair1-candidate.stdout'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[1]['fullLogitSha256'][0] = 'changed-after-compilation'
            path.write_text('\n'.join(map(json.dumps, rows)))
            with self.assertRaisesRegex(AssertionError, 'nondeterminism'):
                module.analyze(root, 'arm', 'bonsai2')


if __name__ == '__main__':
    unittest.main()
