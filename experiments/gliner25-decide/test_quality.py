import json
import tempfile
import unittest
from pathlib import Path

from assess_quality import assess, compare_quality


class QualityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cases = []
        for cohort, kind in [('jevbench', 'choice'), ('jevbench', 'noul'),
                             ('clinc-banking', 'single'), ('goemotions', 'multi')]:
            self.cases.append(dict(id=cohort+kind, cohort=cohort, kind=kind,
                request=dict(tasks=[dict(name='task', labels=[dict(name='yes')],
                                        multiLabel=kind == 'multi')]), expected={'task': ['yes']}))

    def write(self, name, transform=lambda r: r):
        ready = dict(type='ready', device=['cpu'], snapshot={'revision': 'pinned'}, threads=8)
        rows = [ready] + [transform(dict(id=c['id'], selected={'task': ['yes']})) for c in self.cases]
        path = self.root/name
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return path

    def test_identical_wrong_arms_do_not_qualify(self):
        wrong = lambda r: dict(r, selected={'task': []})
        report = compare_quality(self.cases, self.write('a', wrong), self.write('b', wrong))
        self.assertFalse(report['passed'])
        self.assertFalse(report['catalogQualified'])
        self.assertEqual(0, report['oracleLoss']['clinc-banking']['macroF1'])
        self.assertEqual(1, report['candidate']['cohorts']['clinc-banking']['errors'])

    def test_perfect_outcomes_pass_quality_only(self):
        report = compare_quality(self.cases, self.write('a'), self.write('b'))
        self.assertTrue(report['passed'])
        self.assertFalse(report['catalogQualified'])
        self.assertEqual([], report['disagreements'])

    def test_missing_or_unknown_labels_are_errors(self):
        for selected in [{}, {'task': ['unknown']}, {'task': ['yes', 'yes']}]:
            report = assess(self.cases, self.write('bad', lambda r: dict(r, selected=selected)))
            self.assertFalse(report['passed'])
            self.assertEqual(2, report['cohorts']['jevbench']['errors'])

    def test_missing_outcomes_cannot_be_dropped(self):
        path = self.write('missing')
        path.write_text('\n'.join(path.read_text().splitlines()[:-1])+'\n')
        with self.assertRaises(ValueError):
            assess(self.cases, path)


if __name__ == '__main__':
    unittest.main()
