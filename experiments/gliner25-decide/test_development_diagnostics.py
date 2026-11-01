import json
import math
import tempfile
import unittest
from pathlib import Path
from development_diagnostics import analyze

class DevelopmentDiagnosticsTest(unittest.TestCase):
    def test_offsets_are_diagnostics_and_qualification_inputs_are_rejected(self):
        case=dict(id='one',cohort='goemotions',sourceSplit='train',kind='multi',expected={'emotions':['joy']},
            request=dict(text='example',tasks=[dict(name='emotions',labels=[dict(name='joy'),dict(name='sadness')],multiLabel=True,threshold=.5)]))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'out.jsonl'
            row=dict(id='one',selected={'emotions':['joy']},logits=[1.,-1.])
            path.write_text(json.dumps(dict(type='ready',device=['cpu']))+'\n'+json.dumps(row)+'\n')
            report=analyze([case],path)
            self.assertFalse(report['catalogQualified'])
            self.assertEqual(report['measured']['cohorts']['goemotions']['microF1'],1.)
            zero=next(v for v in report['globalLogitOffsetSweep'] if v['logitOffset']==0)
            self.assertEqual(zero['microF1'],1.)
            self.assertEqual(report['globalLogitOffsetSweep'][-1]['fp'],1)
            with self.assertRaises(ValueError):analyze([dict(case,sourceSplit='dev')],path)
            row['logits'][0]=math.nan
            path.write_text(json.dumps(dict(type='ready',device=['cpu']))+'\n'+json.dumps(row)+'\n')
            with self.assertRaises(ValueError):analyze([case],path)

if __name__=='__main__':unittest.main()
