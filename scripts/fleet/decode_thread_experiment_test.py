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


if __name__ == '__main__':
    unittest.main()
