import json
import math
from pathlib import Path
import unittest
import analyze

class ColdProofTest(unittest.TestCase):
    def test_preserves_limits_and_adds_cold_allocation(self):
        old=json.loads((Path(__file__).parent.parent/'bonsai-paired/protocol.json').read_text())
        limits=analyze.CONFIG['limits'].copy()
        self.assertEqual(1.01,limits.pop('coldJavaAllocatedBytes'))
        self.assertEqual(old['limits'],limits)
        self.assertEqual(old['maximumIntervalWidth'],analyze.CONFIG['maximumIntervalWidth'])
        self.assertEqual(list(range(10)),analyze.CONFIG['hostPairs'])

    def test_cold_allocation_is_first_request_not_warm_mean(self):
        self.assertEqual(900,analyze.cold_allocation([{'repeat':-4,'javaAllocatedBytes':900},{'repeat':0,'javaAllocatedBytes':1}]))
        with self.assertRaises(AssertionError):analyze.cold_allocation([{'repeat':0,'javaAllocatedBytes':1}])

    def test_host_variation_does_not_collapse_interval(self):
        r=analyze.paired.bootstrap([[math.log(x) for x in [.9,1.1]*5]],1.01)
        self.assertGreater(r['ci95'][1]-r['ci95'][0],.02)
        self.assertFalse(r['pass'])

    def test_cold_regression_fails(self):
        r=analyze.paired.bootstrap([[math.log(1.1)]*10],1.01)
        self.assertEqual('REGRESSION',r['status'])
        self.assertFalse(r['pass'])

if __name__=='__main__':unittest.main()
