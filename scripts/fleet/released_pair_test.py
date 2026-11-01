"""Reject misleading release and default-smoke evidence before qualification."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('runner', Path(__file__).with_name('run-released-pair.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ReleaseEvidenceTest(unittest.TestCase):
    def test_rejects_source_product_even_when_hash_is_correct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'lib').mkdir()
            artifact = root / 'lib/backend-java-0.5.10.jar'
            artifact.write_bytes(b'candidate')
            (root / 'provenance.json').write_text('{}')
            row = dict(file=artifact.name, sha256=runner.digest(artifact), origin='source-built candidate')
            (root / 'classpath.json').write_text(json.dumps([row]))
            with self.assertRaisesRegex(ValueError, 'not released'):
                runner.verify_payload(root)
            row['origin'] = 'Maven Central'
            (root / 'classpath.json').write_text(json.dumps([row]))
            self.assertEqual({}, runner.verify_payload(root))
            artifact.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                runner.verify_payload(root)

    def test_successful_subset_does_not_pass_default_smoke(self):
        report = dict(modelId='model', artifactSha256='digest', summary=dict(totalAttempts=9,
            successfulAttempts=9, correctAnswerRate=1, abstentionAccuracy=1), failures=[],
            settings=dict(warmups=0, iterations=1, generationControls=dict(promptCache='longest-common-prefix')),
            backendDiagnostics=dict(environment={'native-quantized-decode': 'true'}))
        job = dict(id='model', sha256='digest')
        runner.require_default_smoke(report, job)
        report['summary']['successfulAttempts'] = 8
        with self.assertRaisesRegex(ValueError, 'smoke failed'):
            runner.require_default_smoke(report, job)
        report['summary']['successfulAttempts'] = 9
        report['settings']['generationControls']['promptCache'] = 'none'
        with self.assertRaisesRegex(ValueError, 'smoke failed'):
            runner.require_default_smoke(report, job)


if __name__ == '__main__':
    unittest.main()
