import unittest
from balance_training import repeat_counts


class BalanceTest(unittest.TestCase):
    def test_only_rare_training_labels_repeat_and_every_row_remains(self):
        rows=[dict(id=str(i),cohort='goemotions',expected={'emotions':['neutral']}) for i in range(100)]
        rows += [dict(id='rare',cohort='goemotions',expected={'emotions':['grief']})]
        rows += [dict(id='intent',cohort='clinc-banking',expected={'intent':['balance']})]
        counts=repeat_counts(rows,20,8)
        self.assertEqual([1]*100+[8,1],counts)
        self.assertEqual(counts,repeat_counts(rows,20,8))

    def test_multilabel_uses_one_bounded_repetition_count(self):
        row=dict(id='mixed',cohort='goemotions',expected={'emotions':['grief','sadness']})
        self.assertEqual([4],repeat_counts([row],100,4))
        with self.assertRaises(ValueError):repeat_counts([row],100,0)


if __name__=='__main__':unittest.main()
