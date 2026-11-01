import json
from pathlib import Path
import tempfile
import unittest

from rental_budget import prior_rented_seconds


class RentalBudgetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'deployment.json').write_text(json.dumps({'cleanup':{'rentedSecondsUpperBound':2602}}))

    def receipt(self, name, **record):
        path = self.root/name/'control.json'
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(record))

    def test_training_and_failed_evaluations_share_one_budget(self):
        self.receipt('gliner25-training-1',podId='pod',state='deleted',rentedSecondsUpperBound=457)
        self.receipt('gliner25-eval-1',instanceId='failed',state='terminated',rentedSecondsUpperBound=225)
        self.receipt('gliner25-eval-2',instanceId='complete',state='terminated',rentedSecondsUpperBound=1876)
        self.assertEqual(prior_rented_seconds(self.root),5160)

    def test_live_evaluation_blocks_new_training(self):
        self.receipt('gliner25-eval-1',instanceId='live',state='running')
        with self.assertRaisesRegex(ValueError,'unresolved rental'):
            prior_rented_seconds(self.root)

    def test_missing_receipt_cannot_be_counted_as_zero(self):
        self.receipt('gliner25-training-1',podId='gone',state='deleted')
        with self.assertRaisesRegex(ValueError,'invalid rental receipt'):
            prior_rented_seconds(self.root)

    def test_unlaunched_preparation_is_not_a_rental(self):
        self.receipt('gliner25-eval-1',state='prepared')
        self.assertEqual(prior_rented_seconds(self.root),2602)


if __name__ == '__main__':
    unittest.main()
