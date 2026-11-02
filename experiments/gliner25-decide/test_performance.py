import json
import tempfile
import unittest
from pathlib import Path

from compare_performance import compare_pair


class PerformanceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cases = [dict(id=str(i), bucket='shape', phase='warmup' if i < 3 else 'timed') for i in range(33)]

    def write(self, name, ms=1.0, missing=False, host='host'):
        rows = [dict(type='ready', device=['cpu'], snapshot='pinned', threads=8,
                     driverHostname=host, driverCpuAffinity=list(range(8)))]
        rows += [dict(id=c['id'], wallMs=ms, elapsedMs=ms, peakRssKiB=1024, selected={'task':['label']}) for c in self.cases]
        if missing: rows.pop()
        path = self.root/name
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return path

    def test_passing_pair_never_grants_catalog_admission(self):
        r = compare_pair(self.cases, self.write('a'), self.write('b'))
        self.assertTrue(r['passed'])
        self.assertFalse(r['catalogQualified'])

    def test_slow_candidate_fails(self):
        r = compare_pair(self.cases, self.write('a', ms=2), self.write('b'))
        self.assertFalse(r['passed'])
        self.assertEqual(.5, r['buckets']['shape']['throughputRatio'])

    def test_no_missing_nonfinite_or_cross_host_evidence(self):
        reference = self.write('reference')
        for args in [dict(ms=float('nan')), dict(missing=True), dict(host='other')]:
            with self.assertRaises(ValueError):
                compare_pair(self.cases, self.write('candidate', **args), reference)


if __name__ == '__main__':
    unittest.main()
