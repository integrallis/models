import json
from pathlib import Path
import tempfile
import unittest

from assess_finetune import assess_variant
from common import HERE, read_cases
from prepare_finetune import fingerprint, ordered, stratified


class FineTuneTest(unittest.TestCase):
    def test_normalized_duplicate_detection_and_deterministic_selection(self):
        self.assertEqual(fingerprint('Ｃafé\n  TEXT'),fingerprint('café text'))
        rows=[dict(id='a',expected={'tags':['rare','common']}),dict(id='b',expected={'tags':['common']})]
        self.assertEqual(ordered(rows),ordered(list(reversed(rows))))
        selected=stratified(rows,1)
        self.assertEqual(len(selected),len({r['id'] for r in selected}))
        self.assertIn('rare',{l for r in selected for l in r['expected']['tags']})

    def test_regression_cannot_be_hidden_by_passing_other_cohorts(self):
        base=json.loads((HERE/'snapshot.json').read_text())
        variant=dict(base,id='modeljars-gliner25-decide-general-v1',revision='trained')
        cases=read_cases(HERE/'finetune-qualification-cases.jsonl')
        old=read_cases(HERE/'quality-cases.jsonl')
        with tempfile.TemporaryDirectory() as temp:
            def write(name, cohort, snapshot, wrong=False):
                rows=[dict(type='ready',device=['cpu'],snapshot=snapshot,threads=8)]
                rows += [dict(id=c['id'],selected=c['expected']) for c in cohort]
                if wrong: rows[1]=dict(rows[1],selected={})
                path=Path(temp)/name
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                return path
            reference=write('reference',cases,variant)
            original=write('base',cases,base)
            regression=write('old',old,variant)
            good=assess_variant(cases,write('good',cases,variant),reference,original,regression)
            self.assertTrue(good['passed'])
            self.assertFalse(good['catalogQualified'])
            bad=assess_variant(cases,write('bad',cases,variant,True),reference,original,regression)
            self.assertFalse(bad['passed'])
            with self.assertRaises(ValueError):
                assess_variant(cases,write('wrong-checkpoint',cases,base),reference,original,regression)


if __name__=='__main__':
    unittest.main()
