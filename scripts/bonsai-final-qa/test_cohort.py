"""Scheduling/receipt tests; synthetic answers are never qualification evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import run

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / 'protocol.json').read_text())
CASES = json.loads((HERE / 'public-qa-cases.json').read_text())['cases']
TOKENS = {r['index']: r['inputTokenIds'] for r in map(json.loads, (HERE / 'public-qa-reference-tokens.jsonl').read_text().splitlines())}
GENERATION = dict(temperature=0, topK=1, topP=1, seed=42, repetitionPenalty=1, rawPrompt=True, promptCache=False)


def write(path, value):
    path.write_text(json.dumps(value) + '\n')


class CohortTest(unittest.TestCase):
    def fixture(self, evidence):
        for shard in range(P['shards']):
            directory = evidence / str(shard)
            directory.mkdir(parents=True)
            manifest = dict(protocol=P, shard=shard, runId='test-run', runnerName=f'host-{shard}',
                            harnessRevision='test-source', nativeSha256='native', rustc='rustc', cargo='cargo',
                            jars={'vectors-core-0.1.30.jar': P['coreSha256']})
            write(directory / 'manifest.json', manifest)
            digest = hashlib.sha256((directory / 'manifest.json').read_bytes()).hexdigest()
            for block in range(P['blocks']):
                first = shard * P['casesPerShard'] + block * P['casesPerBlock']
                target = directory / f'block-{block}'
                target.mkdir()
                write(target / 'identity.json', dict(shard=shard, block=block, first=first, count=1,
                    runnerName=manifest['runnerName'], runId=manifest['runId'], manifestSha256=digest))
                (target / 'STATUS').write_text('COMPLETE\n')
                write(target / 'reference-command.json', ['llama-server', '-t', '4', '-tb', '4', '-c', '2048', '-ngl', '0', '--parallel', '1'])
                case = CASES[first]
                row = {k: case[k] for k in ['dataset', 'id', 'promptSha256', 'maxTokens']}
                row.update(index=first, inputTokenIds=TOKENS[first], output='yes', stopReason='EOS')
                for arm in ['models', 'publisher']:
                    header = dict(caseSha256=P['inputs']['public-qa-cases.json'], context=P['context'],
                                  generation=GENERATION, tokenizationMode='trusted-frozen-chatml-v2', first=first, count=1)
                    if arm == 'models':
                        header.update(processors=P['processors'], runtimeIdentity=P['modelsRevision'], javaVersion=P['javaVersion'],
                            classOrigins={key: 'file:/work/lib/' + jar for key, jar in [
                                ('core', 'vectors-core-0.1.30.jar'), ('models', 'backend-java-test.jar'),
                                ('native', 'backend-native-test.jar'), ('runtime', 'models-runtime-test.jar')]})
                    else:
                        header['backendVersion'] = 'PrismML-Eng/llama.cpp@' + P['referenceRevision']
                    (target / f'{arm}.jsonl').write_text(json.dumps(header) + '\n' + json.dumps(row) + '\n')
                    write(target / f'{arm}.execution.json', {'exitCode': 0})

    def analyze_fixture(self, mutation=None):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            evidence, out = root / 'evidence', root / 'summary'
            self.fixture(evidence)
            if mutation:
                mutation(evidence)
            result = subprocess.run([sys.executable, str(HERE / 'analyze.py'), str(evidence), str(out)], capture_output=True, text=True)
            return result.returncode, json.loads((out / 'summary.json').read_text())

    def test_exact_schedule_complete_cohort(self):
        self.assertEqual((P['shards'], P['casesPerShard'], P['blocks'], P['casesPerBlock']), (20, 10, 10, 1))
        self.assertEqual([s * 10 + b for s in range(20) for b in range(10)], list(range(200)))
        code, summary = self.analyze_fixture()
        self.assertEqual(code, 0, summary)
        self.assertEqual((summary['hosts'], summary['pairedBlocks'], summary['pairedCases']), (20, 200, 200))

    def test_workflow_schedule_and_revision_output(self):
        import ast
        import re
        workflow = (HERE.parents[1] / '.github/workflows/bonsai-final-public-qa.yml').read_text()
        matrix = re.search(r'shard: (\[[^\n]+\])', workflow).group(1)
        self.assertEqual(ast.literal_eval(matrix), list(range(P['shards'])))
        blocks = re.findall(r'--block (\d+)', workflow)
        self.assertEqual(list(map(int, blocks)), list(range(P['blocks'])))
        self.assertEqual(workflow.count('timeout-minutes: 30\n'), P['blocks'])
        self.assertIn('timeout-minutes: 300\n', workflow)
        line = next(line.strip() for line in workflow.splitlines() if 'out.write(' in line)
        newline = ast.parse(line).body[0].value.args[0].right.value
        self.assertEqual(newline, '\n')
        self.assertNotIn('  push:', workflow)

    def test_missing_host_rejected(self):
        code, summary = self.analyze_fixture(lambda e: (e / '19/manifest.json').unlink())
        self.assertNotEqual(code, 0)
        self.assertIn('incomplete host cohort', summary['error'])

    def test_missing_block_rejected(self):
        code, summary = self.analyze_fixture(lambda e: (e / '19/block-9/identity.json').unlink())
        self.assertNotEqual(code, 0)
        self.assertIn('missing paired blocks', summary['error'])

    def test_partial_arm_rejected(self):
        def partial(e):
            path = e / '3/block-2/publisher.jsonl'
            path.write_text(path.read_text().splitlines()[0] + '\n')
        code, summary = self.analyze_fixture(partial)
        self.assertNotEqual(code, 0)
        self.assertIn('missing, duplicate or unexpected cases', summary['error'])

    def test_failed_arm_rejected(self):
        code, summary = self.analyze_fixture(lambda e: write(e / '3/block-2/models.execution.json', {'exitCode': None, 'status': 'TIMED_OUT'}))
        self.assertNotEqual(code, 0)
        self.assertIn('arm failed', summary['error'])

    def test_timeout_retains_partial_output_and_execution_receipt(self):
        with tempfile.TemporaryDirectory() as name:
            prefix = Path(name) / 'arm'
            command = [sys.executable, '-c', 'import time; print("partial observation", flush=True); time.sleep(10)']
            with self.assertRaises(subprocess.TimeoutExpired):
                run.run(command, prefix, timeout=.3)
            self.assertIn('partial observation', prefix.with_suffix('.stdout').read_text())
            receipt = json.loads(prefix.with_suffix('.execution.json').read_text())
            self.assertEqual(receipt['status'], 'TIMED_OUT')
            self.assertIsNone(receipt['exitCode'])
            self.assertEqual(receipt['command'], command)


if __name__ == '__main__':
    unittest.main()
