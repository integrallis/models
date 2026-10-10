"""Test the deployed worker's embedded downloader with tiny synthetic bytes, never model weights."""

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest


WORKER = Path(__file__).with_name("smoke-worker.sh")
SOURCE = WORKER.read_text()
HELPER = SOURCE.split("<<'ARTIFACT_HELPER'\n", 1)[1].split("\nARTIFACT_HELPER\n", 1)[0]
MODULE = {"__name__": "smoke_artifact_test"}
exec(compile(HELPER, str(WORKER), "exec"), MODULE)


class SmokeWorkerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.receipt = self.root / "receipt.json"
        self.contents = {name: (name + " fixture").encode() for name in (
            "config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")}
        self.job = {"id": "qwen_bf16", "format": "safetensors", "backend": "pure-java",
                    "context": 2048, "threads": 8, "mt": 64, "topK": 1,
                    "wl": "general", "tpl": "chatml", "files": []}
        for name, data in self.contents.items():
            self.job["files"].append({"path": name, "uri": "https://example.test/" + name,
                "sha256": hashlib.sha256(data).hexdigest(), "sizeBytes": len(data)})
        self.fetched = []

    def open_fixture(self, uri, timeout):
        self.assertEqual(60, timeout)
        name = uri.rsplit("/", 1)[1]
        self.fetched.append(name)
        return io.BytesIO(self.contents[name])

    def fetch(self):
        return MODULE["fetch"](self.job, self.root / "models", self.receipt, self.open_fixture)

    def test_safetensors_returns_complete_directory_with_verified_receipt(self):
        artifact = self.fetch()
        self.assertTrue(artifact.is_dir())
        self.assertEqual(set(self.contents), {p.name for p in artifact.iterdir()})
        self.assertEqual(set(self.contents), set(self.fetched))
        receipt = json.loads(self.receipt.read_text())
        self.assertEqual(str(artifact), receipt["modelPath"])
        for entry in receipt["files"]:
            self.assertEqual(entry["sha256"], entry["verifiedSha256"])
            self.assertEqual(entry["sizeBytes"], entry["verifiedSizeBytes"])

    def test_corrupt_tokenizer_rejects_snapshot_without_success_receipt(self):
        original = self.contents["tokenizer.json"]
        self.contents["tokenizer.json"] = b"x" * len(original)
        with self.assertRaisesRegex(ValueError, "checksum or size mismatch: tokenizer.json"):
            self.fetch()
        self.assertFalse(self.receipt.exists())
        self.assertFalse(list(self.root.rglob("*.partial")))

    def test_missing_configuration_in_manifest_rejected_before_download(self):
        self.job["files"] = [f for f in self.job["files"] if f["path"] != "config.json"]
        with self.assertRaisesRegex(ValueError, "requires weights, config"):
            self.fetch()
        self.assertEqual([], self.fetched)

    def test_missing_configuration_download_cannot_return_model_path(self):
        del self.contents["config.json"]
        with self.assertRaises(KeyError):
            self.fetch()
        self.assertFalse(self.receipt.exists())

    def test_truncated_or_oversized_weights_rejected(self):
        for data in (b"x", b"x" * 100):
            with self.subTest(size=len(data)), tempfile.TemporaryDirectory() as directory:
                self.contents["model.safetensors"] = data
                receipt = Path(directory) / "receipt.json"
                with self.assertRaisesRegex(ValueError, "size"):
                    MODULE["fetch"](self.job, Path(directory) / "models", receipt, self.open_fixture)
                self.assertFalse(receipt.exists())

    def test_gguf_still_returns_file(self):
        self.job["format"] = "gguf"
        self.job["backend"] = "rust-ffm"
        self.contents = {"model.gguf": b"GGUF synthetic fixture"}
        data = self.contents["model.gguf"]
        self.job["files"] = [{"path": "model.gguf", "uri": "https://example.test/model.gguf",
                              "sha256": hashlib.sha256(data).hexdigest(), "sizeBytes": len(data)}]
        artifact = self.fetch()
        self.assertTrue(artifact.is_file())
        self.assertEqual(data, artifact.read_bytes())

    def test_invalid_paths_and_duplicate_files_rejected_before_network(self):
        for path in ("../escape", "/absolute", "./config.json", "a/../config.json", ".", "a\\b"):
            with self.subTest(path=path):
                job = copy.deepcopy(self.job)
                job["files"][0]["path"] = path
                with self.assertRaisesRegex(ValueError, "invalid or duplicate artifact path"):
                    MODULE["validate_job"](job)
        self.job["files"].append(self.job["files"][0])
        with self.assertRaisesRegex(ValueError, "duplicate artifact path"):
            self.fetch()
        self.assertEqual([], self.fetched)

    def test_missing_backend_and_settings_are_not_guessed(self):
        for field in ("backend", "context", "threads", "mt", "wl", "tpl", "topK"):
            with self.subTest(field=field):
                job = copy.deepcopy(self.job)
                del job[field]
                with self.assertRaises(KeyError):
                    MODULE["validate_job"](job)

    def test_snapshot_from_previous_run_is_not_reused(self):
        self.fetch()
        with self.assertRaises(FileExistsError):
            self.fetch()

    def test_shell_syntax(self):
        subprocess.run(["bash", "-n", str(WORKER)], check=True)

    def test_prepared_qwen_job_matches_hash_verified_settings_and_weights(self):
        jobs = json.loads(WORKER.with_name("smoke-qwen-bf16.json").read_text())
        self.assertEqual(1, len(jobs))
        job = jobs[0]
        MODULE["validate_job"](job)
        report_path = WORKER.parents[2] / job["settingsSource"]["report"]
        data = report_path.read_bytes()
        self.assertEqual(job["settingsSource"]["reportSha256"], hashlib.sha256(data).hexdigest())
        report = json.loads(data)
        # Historical reports used a display slug; the catalog now uses its canonical model ID.
        self.assertEqual(report["modelId"], job["settingsSource"]["modelId"])
        self.assertEqual(report["backend"], job["backend"])
        for field, setting in {"wl": "workload", "tpl": "promptTemplate", "context": "contextLength",
                               "threads": "threads", "mt": "maxOutputTokens", "topK": "retrievalTopK"}.items():
            self.assertEqual(report["settings"][setting], job[field])
        weights = next(f for f in job["files"] if f["path"] == "model.safetensors")
        self.assertEqual(report["artifactSha256"], weights["sha256"])
        for entry in job["files"]:
            self.assertIn("/resolve/" + job["revision"] + "/", entry["uri"])

    def test_payload_label_check_refuses_mismatched_published_native(self):
        checker = SOURCE.split("<<'VERSION_CHECK'\n", 1)[1].split("\nVERSION_CHECK\n", 1)[0]
        for label, jar, valid in (("models@0.3.57+abc123", "backend-native-0.3.57.jar", True),
                                  ("models@0.3.57+abc123", "backend-native-0.3.56.jar", False),
                                  ("local", "backend-native-0.3.57.jar", False)):
            with self.subTest(label=label, jar=jar):
                result = subprocess.run(["python3", "-c", checker, label,
                    "models-rag-bench-0.3.57-release.tar", jar], capture_output=True)
                self.assertEqual(valid, result.returncode == 0)

    def test_job_loop_passes_directory_to_java_only_after_all_files_verify(self):
        # Run the actual shell job loop. Replace network and Java with tiny local fixtures;
        # record what the CLI receives before the worker removes its downloaded snapshot.
        for corrupt in (False, True):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "out").mkdir()
                (root / "bin").mkdir()
                (root / "shard.json").write_text(json.dumps([self.job]))
                payloads = {key: value.decode() for key, value in self.contents.items()}
                if corrupt:
                    payloads["tokenizer.json"] = "x" * len(payloads["tokenizer.json"])
                fixture = ("import io, urllib.request\n"
                    f"payloads = {payloads!r}\n"
                    "urllib.request.urlopen = lambda uri, timeout: "
                    "io.BytesIO(payloads[uri.rsplit('/', 1)[1]].encode())\n")
                (root / "fetch-smoke-artifact.py").write_text(fixture + HELPER)
                gate = SOURCE.split("<<'GATE'\n", 1)[1].split("\nGATE\n", 1)[0]
                (root / "gate.py").write_text(gate)
                java = '''#!/usr/bin/env python3
import json, os, pathlib, sys
args = sys.argv[1:]
model = pathlib.Path(args[args.index('--model') + 1])
pathlib.Path(os.environ['CLI_RECEIPT']).write_text(json.dumps({
    'args': args, 'directory': model.is_dir(), 'files': sorted(p.name for p in model.iterdir())}))
output = pathlib.Path(args[args.index('--output') + 1])
output.write_text(json.dumps({'summary': {'correctAnswerRate': 1, 'abstentionAccuracy': 1},
    'backend': 'pure-java', 'settings': {'warmups': 0, 'iterations': 1,
    'generationControls': {'promptCache': 'longest-common-prefix'}}, 'failures': []}))
'''
                for name, script in {"java": java, "aws": "#!/bin/sh\nexit 0\n",
                                     "timeout": '#!/bin/sh\nshift 2\nexec "$@"\n'}.items():
                    command = root / "bin" / name
                    command.write_text(script)
                    command.chmod(0o755)
                loop = SOURCE[SOURCE.index("done_n=0\n"):].replace("/work/", str(root) + "/")
                runner = root / "run.sh"
                runner.write_text("#!/bin/bash\nsay() { :; }\n" + loop)
                receipt = root / "cli.json"
                env = {**os.environ, "PATH": str(root / "bin") + os.pathsep + os.environ["PATH"],
                       "CLI_RECEIPT": str(receipt), "JOBS": "1", "DEADLINE_SECONDS": "60",
                       "T_START": str(int(time.time())), "STATUS": "RUNNING", "CP": "fixture",
                       "BACKEND_VERSION": "models@0.3.57+test", "SHARD": "999", "BUCKET": "fixture"}
                subprocess.run(["bash", str(runner)], env=env, check=True, capture_output=True)
                progress = (root / "out/progress.tsv").read_text()
                if corrupt:
                    self.assertFalse(receipt.exists(), "Java must not start on a corrupt snapshot")
                    self.assertIn("DOWNLOAD_FAIL", progress)
                else:
                    invocation = json.loads(receipt.read_text())
                    self.assertTrue(invocation["directory"])
                    self.assertEqual(sorted(self.contents), invocation["files"])
                    args = invocation["args"]
                    for flag, value in {"--backend": "pure-java", "--context": "2048",
                                        "--threads": "8", "--max-tokens": "64"}.items():
                        self.assertEqual(value, args[args.index(flag) + 1])
                    self.assertIn("\tPASS\t", progress)


if __name__ == "__main__":
    unittest.main()
