"""Reject misleading decoder comparisons before reporting performance closure."""
import json
import unittest
import numpy as np
from test_compare import ComparisonTest
from compare_decoder import compare_decoder
from decoder_reference import sha


class DecoderComparisonTest(unittest.TestCase):
    def setUp(self):
        ComparisonTest.setUp(self)
        self.report.update(textSha256='text', prefixTokens=[1, 2], suffixTokens=[3],
                           imageTextPrefillMs=[3., 3., 3.], decodeMs=[2., 2., 2.],
                           generatedTokenRuns=[[4, 5]] * 3, context=1024, kvType='f32', maximumOutputTokens=32)
        (self.candidate / 'report.json').write_text(json.dumps(self.report))
        (self.oracle / 'report.json').write_text(json.dumps(self.report | {'preprocessMs': 0.}))
        self.reference = self.candidate.parent / 'reference'
        self.reference.mkdir()
        self.ref = {key: self.report[key] for key in ('host', 'threads', 'textSha256', 'mmprojSha256', 'imageSha256', 'warmups', 'repetitions', 'prefixTokens', 'suffixTokens')}
        self.ref.update(configuration=dict(context=1024, maximumOutputTokens=32, threads=8, kvType='f32'),
                        embeddingsSha256=sha(self.oracle / 'embeddings.f32'),
                        runs=[dict(run=i, prefillMs=1., decodeMs=2., tokens=[4, 5]) for i in range(3)])
        np.array([1, 2, 3], dtype='<f4').tofile(self.candidate / 'first-logits.f32')
        np.array([1, 2, 3], dtype='<f4').tofile(self.reference / 'oracle-first-logits.f32')
        self.save()

    def save(self):
        (self.reference / 'report.json').write_text(json.dumps(self.ref))

    def compare(self):
        return compare_decoder(self.candidate, self.oracle, self.reference)

    def test_paired_stages_include_reference_image_encoding(self):
        result = self.compare()
        self.assertTrue(result['numericalPass'])
        self.assertTrue(result['performancePass'])
        self.assertEqual(result['imageTextPrefillRatio'], 1.)
        self.assertEqual(result['totalRatio'], 1.)
        self.assertFalse(result['catalogQualified'])

    def test_different_answer_fails_even_when_fast(self):
        self.ref['runs'][1]['tokens'] = [7]
        self.save()
        self.assertFalse(self.compare()['numericalPass'])

    def test_wrong_prompt_and_missing_or_invalid_timings_fail_closed(self):
        for key, value in [('prefixTokens', [9]), ('threads', 4), ('embeddingsSha256', 'wrong'), ('runs', [])]:
            original = self.ref[key]
            self.ref[key] = value
            self.save()
            with self.assertRaises(ValueError):
                self.compare()
            self.ref[key] = original
        for invalid in (float('nan'), 0., -1.):
            self.ref['runs'][0]['decodeMs'] = invalid
            self.save()
            with self.assertRaises(ValueError):
                self.compare()

    def test_correctness_does_not_excuse_slow_prefill(self):
        self.report['imageTextPrefillMs'] = [6., 6., 6.]
        (self.candidate / 'report.json').write_text(json.dumps(self.report))
        result = self.compare()
        self.assertTrue(result['numericalPass'])
        self.assertFalse(result['performancePass'])


if __name__ == '__main__':
    unittest.main()
