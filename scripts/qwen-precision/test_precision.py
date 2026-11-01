import copy
import json
from pathlib import Path
import tempfile
import unittest
import analyze


class PrecisionTest(unittest.TestCase):
    def test_counts_and_limits_fixed(self):
        original = json.loads((Path(__file__).parent.parent / 'bonsai-paired/protocol.json').read_text())
        self.assertEqual(original['limits'], analyze.CONFIG['limits'])
        self.assertEqual(original['maximumIntervalWidth'], analyze.CONFIG['maximumIntervalWidth'])
        for arch, count in [('x86',320),('arm',160)]:
            groups=analyze.CONFIG['shardsByArchitecture'][arch]
            self.assertEqual(list(range(count)),[p for g in groups.values() for p in g])
            self.assertTrue(all(len(g)==80 and sum(p%2==0 for p in g)==40 for g in groups.values()))

    def test_compilation_drift_fails(self):
        row={'id':'case','repeat':0,'inputTokens':3,'fullLogitSha256':['a']*9,
             'greedyTokens':[1]*8,'prediction':'ok'}
        history={}
        self.assertEqual([],analyze.compare_records([row],[row],history))
        changed=copy.deepcopy(row); changed['fullLogitSha256'][0]='b'
        with self.assertRaises(AssertionError):
            analyze.compare_records([row],[changed],history)

    def test_baseline_difference_retained(self):
        b={'id':'case','repeat':0,'inputTokens':3,'fullLogitSha256':['a']*9,'greedyTokens':[1]*8,'prediction':'ok'}
        c=copy.deepcopy(b); c['fullLogitSha256'][0]='b'
        result=analyze.compare_records([b],[c],{})
        self.assertEqual(['fullLogitSha256'],result[0]['fields'])

    def test_missing_shard_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                analyze.analyze(Path(tmp),'x86')


if __name__=='__main__':
    unittest.main()
