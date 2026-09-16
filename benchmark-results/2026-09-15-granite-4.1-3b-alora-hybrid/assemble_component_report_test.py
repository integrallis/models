import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from assemble_component_report import assemble, bundle_identity, kernel_identity, window_suites


def window_report(backend, balanced, structured=1.0, shared=None, cases=110, complete=True):
    return {
        "backend": backend,
        "complete": complete,
        "limit": 0,
        "windowSha256": "9" * 64,
        "summary": {
            "cases": cases,
            "structuredRate": structured,
            "balancedAccuracy": balanced,
            "physicallyShared": cases if shared is None else shared,
            "answerableCorrect": 50,
            "answerableCases": 55,
            "unanswerableCorrect": 48,
            "unanswerableCases": 55,
        },
        "cases": [{"output": f'"{l}"', "sharedPrefixTokens": 100 + i} for i, l in enumerate(["answerable"] * 10)],
    }


class AssembleComponentReportTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.adapter = self.root / "adapter"
        self.adapter.mkdir()
        for name, content in (
            ("adapter_model.safetensors", b"weights"),
            ("NOTICE", b"notice"),
            ("LICENSE", b"license"),
        ):
            (self.adapter / name).write_bytes(content)
        (self.adapter / "models-activated-lora.json").write_text(
            json.dumps(
                {
                    "adapter": {"sha256": hashlib.sha256(b"weights").hexdigest()},
                    "invocation": {"tokens": [100264, 78191, 100265]},
                    "tokenizer": {"files": [{"name": "tokenizer.json", "sha256": "8" * 64}]},
                    "upstream": {
                        "repository": "ibm-granite/granitelib-rag-r1.0",
                        "revision": "3" * 40,
                        "adapterConfigSha256": "5" * 64,
                        "modelCardSha256": "6" * 64,
                        "license": "Apache-2.0",
                    },
                }
            )
        )
        self.base = self.root / "base.gguf"
        self.base.write_bytes(b"base")

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return path

    def junit(self, failures=0, name="junit.xml"):
        path = self.root / name
        path.write_text(
            f'<testsuite name="G" tests="2" skipped="0" failures="{failures}" errors="0">'
            '<testcase name="matchesTheLlamaCppOracleAndPinsTheMarkerToTheGraniteTokenizer()"/>'
            '<testcase name="opensTheUpstreamAnswerabilityAdapterAtTheAssistantMarker()"/></testsuite>'
        )
        return path

    def long_context(self, qualified=True, name="gate5.json"):
        return self.write(
            name,
            {
                "qualified": qualified,
                "targetPrefixTokens": 4096,
                "policyVersion": "activated-answerability-long-context-retention-v1",
                "summary": {
                    "attempts": 8,
                    "physicallyShared": 8,
                    "nativeCorrect": 7,
                    "retainedNativeCorrect": 7,
                    "exactOutputMatches": 8,
                    "specialistCorrect": 7,
                },
            },
        )

    def crossover(self, qualified=True, name="gate6.json"):
        summaries = [
            {
                "prefixTokens": tier,
                "sharedPhysicalContractPassed": True,
                "recomputedIndependencePassed": True,
                "tokenExactAcrossStrategies": True,
            }
            for tier in (256, 1024, 4096)
        ]
        return self.write(
            name,
            {
                "qualified": qualified,
                "policyVersion": "activated-prefix-sharing-crossover-v3",
                "summaries": summaries,
                "measurements": [{"strategy": "shared", "physicallyShared": True, "jvmAfter": {"nativeMemory": {"available": True}}}],
                "verdict": {"crossoverPrefixTokens": 256, "fourKImprovement": 0.31, "passed": qualified},
                "peakRssBytes": 4_000_000_000,
                "memoryAccountingComplete": True,
            },
        )

    def inputs(self, **overrides):
        windows = {
            "mtrag-human-rag|specialist": self.write("w1.json", window_report("rust-ffm", 0.88)),
            "mtrag-human-rag|base": self.write("w2.json", window_report("rust-ffm", 0.70)),
            "squad-v2-dev|specialist": self.write("w3.json", window_report("rust-ffm", 0.84, cases=200)),
            "squad-v2-dev|base": self.write("w4.json", window_report("rust-ffm", 0.60, cases=200)),
        }
        java = self.write("ij.json", window_report("pure-java", 0.9))
        kernel = self.write("ik.json", window_report("rust-ffm", 0.9))
        identities = {
            "mtrag-human-rag|specialist": (java, kernel),
            "mtrag-human-rag|base": (java, kernel),
            "squad-v2-dev|specialist": (java, kernel),
            "squad-v2-dev|base": (java, kernel),
        }
        arguments = dict(
            adapter_directory=self.adapter,
            base_model_id="ibm_granite_granite_4_1_3b_gguf_q4_k_m",
            base_revision="a" * 40,
            base_artifact=self.base,
            window_reports=windows,
            identity_reports=identities,
            junit_xml=self.junit(),
            long_context_report=self.long_context(),
            crossover_report=self.crossover(),
            models_revision="f" * 40,
        )
        arguments.update(overrides)
        return arguments

    def test_assembles_a_qualified_upstream_specialist_report_from_raw_evidence(self):
        report = assemble(**self.inputs())

        self.assertEqual("upstream-rag-specialist", report["specialistKind"])
        self.assertTrue(report["evaluation"]["qualified"])
        gates = report["evaluation"]["gates"]
        self.assertEqual("rust-ffm", gates["taskCorrectness"]["backend"])
        self.assertEqual(2, len(gates["taskCorrectness"]["suites"]))
        self.assertEqual(0.70, gates["taskCorrectness"]["suites"][0]["baseBalancedAccuracy"])
        self.assertTrue(gates["kernelIdentity"]["pass"])
        self.assertEqual(10, gates["kernelIdentity"]["casesPerSuitePerArm"])
        self.assertEqual(256, gates["performanceAndMemory"]["crossoverPrefixTokens"])
        self.assertTrue(gates["performanceAndMemory"]["tokenExactAtAllTiers"])
        self.assertEqual(7, gates["longContext"]["specialistCorrectCases"])
        self.assertEqual("3" * 40, gates["provenance"]["upstreamRevision"])
        size, sha = bundle_identity(report["evaluation"]["artifact"]["files"])
        self.assertEqual(size, report["evaluation"]["artifact"]["bundleSizeBytes"])
        self.assertEqual(sha, report["evaluation"]["artifact"]["bundleSha256"])

    def test_refuses_a_window_below_the_floor_or_worse_than_base(self):
        low = self.inputs()
        low["window_reports"]["squad-v2-dev|specialist"] = self.write("low.json", window_report("rust-ffm", 0.79, cases=200))
        with self.assertRaisesRegex(SystemExit, "taskCorrectness"):
            assemble(**low)
        worse = self.inputs()
        worse["window_reports"]["mtrag-human-rag|base"] = self.write("worse.json", window_report("rust-ffm", 0.95))
        with self.assertRaisesRegex(SystemExit, "taskCorrectness"):
            assemble(**worse)

    def test_refuses_a_kernel_window_without_identity_and_an_incomplete_run(self):
        divergent = self.inputs()
        kernel = window_report("rust-ffm", 0.9)
        kernel["cases"][3]["output"] = '"unanswerable"'
        divergent["identity_reports"]["squad-v2-dev|base"] = (self.write("dj.json", window_report("pure-java", 0.9)), self.write("dk.json", kernel))
        with self.assertRaisesRegex(SystemExit, "kernelIdentity"):
            assemble(**divergent)
        partial = self.inputs()
        partial["window_reports"]["mtrag-human-rag|base"] = self.write("partial.json", window_report("rust-ffm", 0.70, complete=False))
        with self.assertRaisesRegex(ValueError, "not a complete run"):
            assemble(**partial)

    def test_refuses_failed_long_context_crossover_or_integration_gates(self):
        with self.assertRaisesRegex(SystemExit, "longContext"):
            assemble(**self.inputs(long_context_report=self.long_context(qualified=False, name="gate5-fail.json")))
        with self.assertRaisesRegex(SystemExit, "performanceAndMemory"):
            assemble(**self.inputs(crossover_report=self.crossover(qualified=False, name="gate6-fail.json")))
        with self.assertRaisesRegex(SystemExit, "plainJava"):
            assemble(**self.inputs(junit_xml=self.junit(failures=1, name="junit-fail.xml")))

    def test_kernel_identity_requires_ten_identical_cases_per_comparison(self):
        java = window_report("pure-java", 0.9)
        kernel = window_report("rust-ffm", 0.9)
        self.assertTrue(kernel_identity({"a": (java, kernel)}, "rust-ffm")["pass"])
        short = window_report("rust-ffm", 0.9)
        short["cases"] = short["cases"][:9]
        self.assertFalse(kernel_identity({"a": (java, short)}, "rust-ffm")["pass"])

    def test_window_suites_reject_mixed_backends(self):
        reports = {
            "s|specialist": window_report("rust-ffm", 0.9),
            "s|base": window_report("pure-java", 0.7),
        }
        with self.assertRaisesRegex(ValueError, "backend differs"):
            window_suites(reports)


if __name__ == "__main__":
    unittest.main()
