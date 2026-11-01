import contextlib
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from development_metrics import measure_development


class DevelopmentMetricsTest(unittest.TestCase):
    def test_multilabel_counts_include_false_positives_and_missing_labels(self):
        cases=[dict(request=dict(text=str(i),tasks=[dict(name='tags',labels=[dict(name=x) for x in 'abc'],
            multiLabel=True,threshold=.5)]),expected={'tags':truth},cohort='emotions')
            for i,truth in enumerate([['a','b'],['b']])]
        responses=[{'tags':[('a',.9)]},{'tags':[('b',.8),('c',.7)]}]
        model=SimpleNamespace(extract=lambda *a,**kw:responses.pop(0))
        torch=SimpleNamespace(bfloat16=None,inference_mode=contextlib.nullcontext,
            autocast=lambda *a,**kw:contextlib.nullcontext())
        with patch.dict(sys.modules,torch=torch):
            measured=measure_development(model,cases,True)
        self.assertAlmostEqual(measured['dev_emotions_micro_f1'],2/3)
        self.assertAlmostEqual(measured['dev_emotions_macro_f1'],5/9)
        self.assertEqual(measured['dev_emotions_accuracy'],0)


if __name__ == '__main__':
    unittest.main()
