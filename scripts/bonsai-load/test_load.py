import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import analyze


class LoadProtocolTest(unittest.TestCase):
    def test_fixed_balanced_cohort(self):
        groups = analyze.CONFIG['shards']
        self.assertEqual(list(range(400)), groups['0'] + groups['1'])
        for group in groups.values():
            self.assertEqual(100, sum(pair % 2 == 0 for pair in group))
        self.assertEqual(1.01, analyze.CONFIG['limits']['loadNanos'])
        self.assertEqual(.02, analyze.CONFIG['maximumIntervalWidth'])

    def test_failed_or_wrong_process_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prefix = root / 'qwen-bf16-pair0-baseline'
            prefix.with_suffix('.execution.json').write_text('{"exitCode":1}')
            with self.assertRaises(AssertionError):
                analyze.observation(root, 'qwen-bf16', 0, 'baseline')
            prefix.with_suffix('.execution.json').write_text('{"exitCode":0}')
            prefix.with_suffix('.stdout').write_text(json.dumps({'kind':'load','pair':1,'phase':'load-only','nanos':1}))
            with self.assertRaises(AssertionError):
                analyze.observation(root, 'qwen-bf16', 0, 'baseline')

    def test_no_partial_cohort_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                analyze.analyze(Path(tmp), 'arm', 'qwen-bf16')


if __name__ == '__main__':
    unittest.main()
