import json
import tempfile
import unittest
from pathlib import Path
from compare_parity import compare
from common import read_cases

class HarnessTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cases = [{'id':'one'}]
        self.ready = dict(type='ready',device=['cpu'],snapshot={'revision':'fixed'},threads=8)
        self.result = dict(id='one',tokenIds=[1,2],selected={'task':['yes']},logits=[1.0,0.0])
    def write(self,name,result=None,ready=None):
        path=self.root/name
        path.write_text(json.dumps(ready or self.ready)+'\n'+json.dumps(result or self.result)+'\n')
        return path
    def test_agreement_is_only_parity_not_catalog_admission(self):
        r=compare(self.cases,self.write('left'),self.write('right'))
        self.assertTrue(r['passed']); self.assertFalse(r['catalogQualified'])
    def test_nonfinite_and_missing_outcomes_fail(self):
        r=dict(self.result,logits=[float('nan'),0])
        self.assertFalse(compare(self.cases,self.write('left',r),self.write('right'))['passed'])
        with self.assertRaises(ValueError):
            compare([{'id':'absent'}],self.write('left'),self.write('right'))
    def test_identity_cannot_be_substituted(self):
        with self.assertRaises(ValueError):
            compare(self.cases,self.write('left',ready=dict(self.ready,snapshot={'revision':'different'})),self.write('right'))
        with self.assertRaises(ValueError):
            compare(self.cases,self.write('left',ready=dict(self.ready,device=['cuda:0'])),self.write('right'))
    def test_duplicate_case_ids_fail(self):
        p=self.root/'cases';p.write_text('{"id":"x"}\n{"id":"x"}\n')
        with self.assertRaises(ValueError):read_cases(p)

if __name__=='__main__': unittest.main()
