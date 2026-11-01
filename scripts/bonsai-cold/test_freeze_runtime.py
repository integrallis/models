import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('freeze',Path(__file__).with_name('freeze_runtime.py'))
freeze=importlib.util.module_from_spec(spec);spec.loader.exec_module(freeze)

class FreezeTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.protocol={'vectorsRevision':'a'*40,'candidateCoreSha256':'b'*64,'phase':'old','limits':{'latency':1.0},'hostPairs':list(range(10)),'maximumIntervalWidth':.02}
        self.root.joinpath('protocol.json').write_text(json.dumps(self.protocol))
        for arch in ['arm','x86']:
            self.root.joinpath(f'expected-{arch}.json').write_text(json.dumps({'nativeSha256':'fixed','jarEntries':{'baseline':{'core':'fixed'},'candidate':{'vectors-core-0.1.30.jar':{'old':'old'},'backend.jar':{'fixed':'fixed'}}}}))
    def tearDown(self):self.temp.cleanup()
    def test_only_runtime_identity_changes(self):
        with patch.object(freeze,'HERE',self.root),patch.object(freeze,'digest',side_effect=['c'*64,freeze.REFERENCE_SHA]),patch.object(freeze,'entries',side_effect=[{'same':'1','changed':'2'},{'same':'1','changed':'3'}]):
            freeze.freeze('d'*40,Path('candidate'),'c'*64,Path('reference'),'new')
        result=json.loads(self.root.joinpath('protocol.json').read_text())
        self.assertEqual({k for k in result if result[k]!=self.protocol[k]},{'vectorsRevision','candidateCoreSha256','phase'})
        self.assertEqual(result['limits'],self.protocol['limits'])
        for arch in ['arm','x86']:
            manifest=json.loads(self.root.joinpath(f'expected-{arch}.json').read_text())
            self.assertEqual(manifest['jarEntries']['baseline'],{'core':'fixed'})
            self.assertEqual(manifest['nativeSha256'],'fixed')
    def test_wrong_hash_makes_no_changes(self):
        old=self.root.joinpath('protocol.json').read_bytes()
        with patch.object(freeze,'HERE',self.root),patch.object(freeze,'digest',return_value='wrong'):
            with self.assertRaises(ValueError):freeze.freeze('d'*40,Path('candidate'),'c'*64,Path('reference'),'new')
        self.assertEqual(self.root.joinpath('protocol.json').read_bytes(),old)

    def model_fixture(self):
        self.protocol['modelsRevision']='a'*40
        self.root.joinpath('protocol.json').write_text(json.dumps(self.protocol))
        names=['com/integrallis/models/backend/purejava/ops/PinnedReduction.class',
               'com/integrallis/models/backend/purejava/quant/Q5_1Dequantizer.class',
               'com/integrallis/models/backend/purejava/quant/Q5_1Dequantizer$Narrow.class']
        before={names[0]:'old-dot',names[1]:'old-decoder','unchanged':'fixed'}
        after={names[0]:'new-dot',names[1]:'new-decoder',names[2]:'narrow','unchanged':'fixed'}
        for arch in ['arm','x86']:
            self.root.joinpath(f'expected-{arch}.json').write_text(json.dumps({'nativeSha256':'fixed','jarEntries':{arm:{'backend-java-0.3.59.jar':before,'other.jar':{'same':'same'}} for arm in ('baseline','candidate')}}))
        return after
    def test_models_changes_are_limited_and_shared_by_both_arms(self):
        after=self.model_fixture()
        with patch.object(freeze,'HERE',self.root),patch.object(freeze,'digest',return_value='c'*64),patch.object(freeze,'entries',return_value=after):
            freeze.freeze_models('d'*40,Path('backend'),'c'*64)
        protocol=json.loads(self.root.joinpath('protocol.json').read_text())
        self.assertEqual({k for k in protocol if protocol[k]!=self.protocol[k]},{'modelsRevision'})
        for arch in ['arm','x86']:
            value=json.loads(self.root.joinpath(f'expected-{arch}.json').read_text())
            self.assertEqual(value['nativeSha256'],'fixed')
            for arm in ('baseline','candidate'):
                self.assertEqual(value['jarEntries'][arm]['backend-java-0.3.59.jar'],after)
                self.assertEqual(value['jarEntries'][arm]['other.jar'],{'same':'same'})
    def test_unreviewed_models_class_is_rejected_before_any_write(self):
        after=self.model_fixture();after['unreviewed.class']='changed'
        before={p:p.read_bytes() for p in self.root.glob('*.json')}
        with patch.object(freeze,'HERE',self.root),patch.object(freeze,'digest',return_value='c'*64),patch.object(freeze,'entries',return_value=after):
            with self.assertRaises(AssertionError):freeze.freeze_models('d'*40,Path('backend'),'c'*64)
        self.assertEqual(before,{p:p.read_bytes() for p in self.root.glob('*.json')})

if __name__=='__main__':unittest.main()
