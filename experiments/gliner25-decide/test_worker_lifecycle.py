import ast
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rental_budget import validate_limit
from development_metrics import admission_margin


class WorkerLifecycleTest(unittest.TestCase):
    def test_provider_restart_does_not_execute_completed_training_again(self):
        tree=ast.parse((Path(__file__).parent/'runpod_train_controller.py').read_text())
        bootstrap=next(node.value for node in ast.walk(tree) if isinstance(node,ast.Constant)
            and isinstance(node.value,str) and node.value.startswith('import os,json,urllib.request,pathlib'))
        worker=b'print("stub worker")'
        manifest=dict(workerUrl='https://worker.invalid',workerSha256=hashlib.sha256(worker).hexdigest())
        def download(url,**kwargs):return io.BytesIO(worker if 'worker.invalid' in url else json.dumps(manifest).encode())
        class DeletedByController(Exception):pass
        with tempfile.TemporaryDirectory() as directory:
            code=bootstrap.replace("pathlib.Path('/work')",'pathlib.Path('+repr(directory)+')')
            with patch.dict('os.environ',GLINER_MANIFEST_URL='https://manifest.invalid'), patch('urllib.request.urlopen',download), patch('subprocess.call',return_value=0) as run, patch('time.sleep',side_effect=DeletedByController):
                for _ in range(2):
                    with self.assertRaises(DeletedByController):exec(code,{})
            self.assertEqual(run.call_count,1)
            self.assertEqual(json.loads((Path(directory)/'worker-exit.json').read_text()),{'exitCode':0})

    def test_continuations_must_preserve_each_previous_limit(self):
        record=dict(previousAggregateLimitSeconds=7200,additionalAllocationSeconds=7200,maximumAggregateRentedSeconds=18000,
            continuations=[dict(previousLimitSeconds=14400,newLimitSeconds=18000,additionalAllocationSeconds=3600)])
        self.assertEqual(validate_limit(record),18000)
        record['continuations'][0]['previousLimitSeconds']=7200
        with self.assertRaises(ValueError):validate_limit(record)

    def test_selection_score_cannot_hide_a_failed_cohort_behind_an_average(self):
        metrics=dict(dev_boolq_accuracy=.69,dev_goemotions_micro_f1=.8,dev_goemotions_macro_f1=.7)
        metrics.update({f'dev_clinc_domain{i}_macro_f1':.99 for i in range(10)})
        gates=dict(boolqAccuracy=.7,goEmotionsMicroF1=.7,goEmotionsMacroF1=.6,eachClincDomainMacroF1=.7)
        self.assertAlmostEqual(admission_margin(metrics,gates),.69/.7)
        del metrics['dev_clinc_domain9_macro_f1']
        with self.assertRaises(ValueError):admission_margin(metrics,gates)


if __name__=='__main__':unittest.main()
