import json
import tempfile
import unittest
from pathlib import Path

import compare
import finalize
import test_protocol


class FinalizeTest(unittest.TestCase):
    def test_missing_integration_is_a_failed_result_not_a_pass_or_crash(self):
        with tempfile.TemporaryDirectory() as temp:
            result = finalize.integration(Path(temp), {})
            self.assertFalse(result['passed'])
            self.assertEqual(5, len(result['missingEvidence']))

    def fixture(self, root):
        results = root / 'results'
        results.mkdir()
        corpus, manifest, variant = test_protocol.ProtocolTest().fixture(results)
        task = compare.evaluate(corpus, manifest, results, variant, test_protocol.Evaluator())
        path = results / 'receipt.json'
        receipt = compare.read(path)
        receipt['oracleCommit'] = 'c' * 40
        path.write_text(json.dumps(receipt))
        f16receipt = dict(receipt, referenceCommit=receipt['oracleCommit'], kvType='f16')
        directory = results / variant
        (directory / 'reference-performance-f16-receipt.json').write_text(json.dumps(f16receipt))
        rows = [json.loads(line) for line in (directory / 'reference.jsonl').read_text().splitlines()]
        rows[0]['configuration']['kvType'] = 'f16'
        selected = {case['id'] for case in corpus['cases'] if case['repetitions'] == 3}
        rows = [rows[0]] + [row for row in rows[1:] if row['id'] in selected]
        (directory / 'reference-performance-f16.jsonl').write_text('\n'.join(map(json.dumps, rows)))
        return corpus, variant, task

    def test_strongest_reference_and_generation_lengths_are_enforced(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            corpus, variant, task = self.fixture(root)
            result = finalize.stronger_performance(root, corpus, variant, task)
            self.assertTrue(all(row['passed'] for row in result))
            task['performance'][0]['ratios']['decodeMs'] = 1.11
            result = finalize.stronger_performance(root, corpus, variant, task)
            self.assertFalse(result[0]['passed'])
            task['performance'][0]['ratios']['decodeMs'] = 1
            path = root / 'results' / variant / 'reference-performance-f16.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[2]['tokens'] = [1, 2]
            path.write_text('\n'.join(map(json.dumps, rows)))
            result = finalize.stronger_performance(root, corpus, variant, task)
            self.assertFalse(result[0]['generationLengthsMatch'])
            self.assertFalse(result[0]['passed'])

    def test_wrong_host_rejected_before_ratio_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            corpus, variant, task = self.fixture(root)
            path = root / 'results' / variant / 'reference-performance-f16-receipt.json'
            receipt = compare.read(path)
            receipt['host'] = 'another-worker'
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'host mismatch'):
                finalize.stronger_performance(root, corpus, variant, task)


if __name__ == '__main__':
    unittest.main()
