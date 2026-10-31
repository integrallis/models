"""Selection must reject a fast configuration that changes answers or fails quality."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('experiment', Path(__file__).with_name('run-decode-thread-experiment.py'))
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


class SelectionTest(unittest.TestCase):
    def test_speed_cannot_override_answer_stability_or_quality(self):
        rows = [dict(decodeThreads=16, answersStable=True, qualityPassed=True, decodeRatio=.74, latencyRatio=1.4),
                dict(decodeThreads=8, answersStable=True, qualityPassed=True, decodeRatio=.90, latencyRatio=1.2),
                dict(decodeThreads=4, answersStable=False, qualityPassed=True, decodeRatio=1.2, latencyRatio=.9)]
        self.assertEqual(8, experiment.select_winner(rows)['decodeThreads'])
        rows[1]['qualityPassed'] = False
        self.assertEqual(16, experiment.select_winner(rows)['decodeThreads'])
        rows[0]['answersStable'] = False
        with self.assertRaises(ValueError):
            experiment.select_winner(rows)

    def test_a_latency_failure_cannot_outrank_a_complete_pass(self):
        base = dict(answersStable=True, qualityPassed=True, decodeThreads=8)
        rows = [dict(base, decodeRatio=.95, latencyRatio=1.6, qualified=False),
                dict(base, decodeRatio=.90, latencyRatio=1.2, qualified=True)]
        self.assertEqual(1.2, experiment.select_winner(rows)['latencyRatio'])


if __name__ == '__main__':
    unittest.main()
