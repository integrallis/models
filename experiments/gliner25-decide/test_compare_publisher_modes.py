import copy
import json
from pathlib import Path
import tempfile
import unittest

from compare_publisher_modes import compare


class CompositionAuditTest(unittest.TestCase):
    def test_counts_improvements_and_regressions_and_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tasks = [dict(name='intent'), dict(name='tone')]
            joint = [dict(id='x', cohort='domain', request=dict(text='original', tasks=tasks),
                          expected=dict(intent=['refund'], tone=['urgent', 'negative']))]
            single = [dict(id='x-head-'+str(i), cohort='domain',
                           request=dict(text='original', tasks=[task]),
                           expected={task['name']: joint[0]['expected'][task['name']]})
                      for i, task in enumerate(tasks)]
            ready = dict(type='ready', device=['cpu'], snapshot='same')
            records = dict(joint=joint, single=single,
                before=[ready, dict(id='x', selected=dict(intent=['refund'], tone=['urgent']))],
                after=[ready, dict(id='x-head-0', selected=dict(intent=['shipping'])),
                       dict(id='x-head-1', selected=dict(tone=['urgent', 'negative']))])
            for name, rows in records.items():
                (root/name).write_text(''.join(json.dumps(row)+'\n' for row in rows))
            result = compare(root/'joint', root/'single', root/'before', root/'after')
            self.assertEqual(result['domains']['domain'], dict(heads=2, changedHeads=2,
                jointCorrect=1, singleCorrect=1, improved=1, regressed=1))
            modified = copy.deepcopy(single)
            modified[0]['request']['text'] = 'different'
            (root/'single').write_text(''.join(json.dumps(row)+'\n' for row in modified))
            with self.assertRaisesRegex(ValueError, 'input changed'):
                compare(root/'joint', root/'single', root/'before', root/'after')


if __name__ == '__main__':
    unittest.main()
