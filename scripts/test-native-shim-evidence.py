#!/usr/bin/env python3
"""Mutation checks for the proof gate, including the historical unexecuted-native-arm failure."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('verifier', ROOT / 'scripts/verify-native-shim-evidence.py')
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)


class EvidenceGateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.registry = json.loads((ROOT / 'docs/native-shims/registry.json').read_text())
        paths = {V.CAPABILITIES}
        for folder in [V.RUST, V.CUDA]:
            paths.update(str(p.relative_to(ROOT)) for p in (ROOT / folder).rglob('*.rs'))
        for op in self.registry['operations']:
            paths.update(row['path'] for row in op['dispatch'] + op['evidence'])
            paths.update(self.registry['sourceScopes'][op['sourceScope']])
        for name in paths:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, path)

    def verify(self):
        return V.verify(self.root, self.registry)

    def proof(self, **updates):
        op = self.registry['operations'][0]
        raw = self.root / 'raw.json'
        raw.write_text('{"diagnostic": "validator fixture, not production evidence"}\n')
        reference = {'path': 'raw.json', 'sha256': V.sha(raw)}
        self.decision = {'gates': {gate: 'PASS' for gate in V.GATES},
            'sources': [{'path': name, 'sha256': V.sha(self.root / name)} for name in self.registry['sourceScopes'][op['sourceScope']]],
            'profiles': [{'host': host, 'jvm': 'fixture-jvm', 'javaCalls': 1, 'nativeCalls': 1,
                'datasets': ['public-a', 'public-b'], 'decision': 'PASS', 'rawEvidence': [reference]}
                for host in op['requiredProfiles']],
            'jvmLimitation': 'fixture', 'controlledExperiment': 'fixture',
            'workloadRequirement': 'fixture', 'retirementCommand': 'fixture', 'review': 'fixture'}
        self.decision.update(updates)
        op['status'] = 'PROVEN_REQUIRED'
        self.save_proof()

    def save_proof(self):
        path = self.root / 'decision.json'
        path.write_text(json.dumps(self.decision))
        self.registry['operations'][0]['proof'] = {'path': 'decision.json', 'sha256': V.sha(path)}

    def test_inventory_reports_gaps_instead_of_certifying_them(self):
        self.assertEqual(17, len(self.verify()['unresolved']))

    def test_new_export_requires_inventory_and_operation(self):
        with (self.root / V.RUST / 'lib.rs').open('a') as stream:
            stream.write('\npub extern "C" fn unreviewed_operation() {}\n')
        with self.assertRaisesRegex(ValueError, 'exported symbol'):
            self.verify()

    def test_capability_bit_drift_is_rejected(self):
        path = self.root / V.CAPABILITIES
        path.write_text(path.read_text().replace('Q4_0_F32_BATCHED_MATMUL(1L)', 'Q4_0_F32_BATCHED_MATMUL(1L << 30)'))
        with self.assertRaisesRegex(ValueError, 'masks differ'):
            self.verify()

    def test_retired_capability_bit_cannot_be_reassigned(self):
        path = self.root / V.CAPABILITIES
        path.write_text(path.read_text().replace('Q4_0_F32_BATCHED_MATMUL(1L)', 'Q4_0_F32_BATCHED_MATMUL(1L << 13)'))
        path = self.root / V.RUST / 'lib.rs'
        path.write_text(path.read_text().replace('CAPABILITY_Q4_0_F32_BATCHED_MATMUL: u64 = 1;', 'CAPABILITY_Q4_0_F32_BATCHED_MATMUL: u64 = 1 << 13;'))
        self.registry['capabilities']['Q4_0_F32_BATCHED_MATMUL'] = 1 << 13
        with self.assertRaisesRegex(ValueError, 'Retired capability bit reused'):
            self.verify()

    def test_dispatch_anchor_drift_is_rejected(self):
        self.registry['operations'][0]['dispatch'][0]['anchor'] = 'no such dispatch method'
        with self.assertRaisesRegex(ValueError, 'dispatch anchor'):
            self.verify()

    def test_new_unproven_operation_cannot_enter_unlisted(self):
        self.registry['migrationChecklist'].remove(self.registry['operations'][0]['id'])
        with self.assertRaisesRegex(ValueError, 'New unproven operation'):
            self.verify()

    def test_proven_status_requires_proof(self):
        self.registry['operations'][0]['status'] = 'PROVEN_REQUIRED'
        with self.assertRaisesRegex(ValueError, 'Missing proof'):
            self.verify()

    def test_structurally_complete_fixture_is_accepted(self):
        self.proof()
        self.assertEqual(16, len(self.verify()['unresolved']))

    def test_native_jar_presence_without_native_calls_is_rejected(self):
        self.proof()
        self.decision['profiles'][0]['nativeCalls'] = 0
        self.save_proof()
        with self.assertRaisesRegex(ValueError, 'Unexecuted comparison arm'):
            self.verify()

    def test_missing_host_is_rejected(self):
        self.proof()
        self.decision['profiles'].pop()
        self.save_proof()
        with self.assertRaisesRegex(ValueError, 'host profile'):
            self.verify()

    def test_changed_dependency_fingerprint_is_rejected(self):
        self.proof()
        with (self.root / V.CAPABILITIES).open('a') as stream:
            stream.write('\n// dependency changed\n')
        with self.assertRaisesRegex(ValueError, 'Stale evidence hash'):
            self.verify()

    def test_missing_raw_comparator_evidence_is_rejected(self):
        self.proof()
        (self.root / 'raw.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing or external file'):
            self.verify()

    def test_failed_causal_gate_is_rejected(self):
        self.proof()
        self.decision['gates']['jvmCause'] = 'UNPROVEN'
        self.save_proof()
        with self.assertRaisesRegex(ValueError, 'Incomplete proof gates'):
            self.verify()

    def test_retired_operation_cannot_still_be_exported(self):
        self.registry['operations'][0]['status'] = 'RETIRED'
        with self.assertRaisesRegex(ValueError, 'still exported'):
            self.verify()


if __name__ == '__main__':
    unittest.main()
