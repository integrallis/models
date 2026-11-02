import json
from pathlib import Path
import tempfile
import unittest
from report_publisher_audit import score

class PublisherAuditTest(unittest.TestCase):
    def test_row_exact_match_is_not_head_exact_match_or_f1(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'out.jsonl'
            case=dict(id='x',cohort='domain',expected={'intent':['refund'],'tone':['urgent','negative']},
                request=dict(tasks=[dict(name='intent',multiLabel=False,labels=[dict(name='refund'),dict(name='shipping')]),
                    dict(name='tone',multiLabel=True,labels=[dict(name='urgent'),dict(name='negative')])]))
            ready=dict(type='ready',device=['cpu'])
            row=dict(id='x',selected={'intent':['refund'],'tone':['urgent']})
            p.write_text(json.dumps(ready)+'\n'+json.dumps(row)+'\n')
            r=score([case],p);d=r['domains']['domain']
            self.assertEqual(d['documentExactMatch'],0)
            self.assertEqual(d['headExactMatch'],.5)
            self.assertEqual(d['microF1'],.8)
            self.assertEqual(d['errors'],0)
            p.write_text(json.dumps(ready)+'\n')
            with self.assertRaises(ValueError):score([case],p)

if __name__=='__main__':unittest.main()
