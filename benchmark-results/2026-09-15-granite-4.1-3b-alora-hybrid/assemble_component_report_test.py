import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from assemble_component_report import (
    assemble,
    bundle_identity,
    kernel_identity,
    main,
    validate_clean_host_run,
    validate_models_artifact,
    window_suites,
)

MODULES = ["backend-java", "models-runtime", "models-spring-ai", "models-langchain4j"]


def labeled_window(backend, predictions, dataset_labels):
    """A complete window report whose summary is scored from its own per-case predictions."""
    ids = sorted(dataset_labels)
    counts = {label: [0, 0] for label in ("answerable", "unanswerable")}
    for cid in ids:
        counts[dataset_labels[cid]][1] += 1
        counts[dataset_labels[cid]][0] += predictions[cid] == dataset_labels[cid]
    balanced = 0.5 * sum(correct / total for correct, total in counts.values())
    return {
        "backend": backend,
        "complete": True,
        "limit": 0,
        "windowSha256": "9" * 64,
        "summary": {
            "cases": len(ids),
            "structuredRate": 1.0,
            "balancedAccuracy": balanced,
            "physicallyShared": len(ids),
            "answerableCorrect": counts["answerable"][0],
            "answerableCases": counts["answerable"][1],
            "unanswerableCorrect": counts["unanswerable"][0],
            "unanswerableCases": counts["unanswerable"][1],
        },
        "cases": [
            {
                "id": cid,
                "label": dataset_labels[cid],
                "prediction": predictions[cid],
                "structured": True,
                "physicallyShared": True,
                "output": f'"{predictions[cid]}"',
                "sharedPrefixTokens": 100,
            }
            for cid in ids
        ],
    }


def confirmed_labels(suite, labels, dataset_labels):
    """A label file in the shape confirm_disputed_labels.py writes."""
    cases = sorted(
        (
            {
                "id": cid,
                "label": labels[cid],
                "datasetLabel": dataset_labels[cid],
                "source": "audit-kept" if labels[cid] == dataset_labels[cid] else "evidence-confirmed",
            }
            for cid in labels
        ),
        key=lambda case: case["id"],
    )
    canon = json.dumps(cases, sort_keys=True, separators=(",", ":"))
    return {"suite": suite, "cases": cases, "casesSha256": hashlib.sha256(canon.encode()).hexdigest()}


def models_artifact(version="0.3.42"):
    return {
        "pass": True,
        "version": version,
        "artifacts": [
            {
                "module": module,
                "coordinate": f"com.integrallis:{module}:{version}",
                **{
                    kind: {
                        "uri": f"https://repo1.maven.org/maven2/com/integrallis/{module}/{version}/{module}-{version}.{kind}",
                        "sha256": "c" * 64,
                        "sizeBytes": 1234,
                    }
                    for kind in ("jar", "pom")
                },
            }
            for module in MODULES
        ],
    }


def clean_host_run(version="0.3.42"):
    return {
        "pass": True,
        "freshMachine": True,
        "externalInference": False,
        "javaMajor": 25,
        "exitCode": 0,
        "modelsVersion": version,
        "startedAt": "2026-09-17T10:00:00Z",
        "completedAt": "2026-09-17T10:30:00Z",
        "command": ["java", "-jar", "qualify.jar"],
        "resolvedClasspathSha256": "d" * 64,
        "outputLog": {
            "uri": "https://raw.githubusercontent.com/integrallis/models/" + "e" * 40 + "/benchmark-results/run.log",
            "sha256": "f" * 64,
            "sizeBytes": 4096,
        },
    }


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
            model_id="integrallis_granite_4_1_3b_answerability_alora",
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

    def squad_window_with_disputed_labels(self):
        """200 cases; the dataset calls c100-c139 unanswerable but confirmation says answerable.

        The specialist answers c000-c089 and c100-c139: 0.75 on dataset labels (below the floor),
        (130/140 + 60/60) / 2 on confirmed labels. The base answers everything: 0.50 either way.
        """
        dataset = {f"c{i:03d}": "answerable" if i < 100 else "unanswerable" for i in range(200)}
        confirmed = {cid: ("answerable" if 100 <= int(cid[1:]) < 140 else label) for cid, label in dataset.items()}
        specialist = {
            cid: ("answerable" if int(cid[1:]) < 90 or 100 <= int(cid[1:]) < 140 else "unanswerable") for cid in dataset
        }
        base = {cid: "answerable" for cid in dataset}
        inputs = self.inputs()
        inputs["window_reports"]["squad-v2-dev|specialist"] = self.write("ls.json", labeled_window("rust-ffm", specialist, dataset))
        inputs["window_reports"]["squad-v2-dev|base"] = self.write("lb.json", labeled_window("rust-ffm", base, dataset))
        return inputs, dataset, confirmed

    def test_dataset_labels_are_recorded_as_the_label_source_without_confirmed_labels(self):
        report = assemble(**self.inputs())
        suites = report["evaluation"]["gates"]["taskCorrectness"]["suites"]
        self.assertEqual(["dataset", "dataset"], [s["labelSource"] for s in suites])
        self.assertNotIn("labelsSha256", suites[0])
        self.assertEqual("integrallis_granite_4_1_3b_answerability_alora", report["evaluation"]["artifact"]["modelId"])
        self.assertNotIn("modelsArtifact", report["evaluation"]["gates"])
        self.assertNotIn("cleanHostRun", report["evaluation"]["gates"])

    def test_confirmed_labels_rescore_both_arms_from_per_case_predictions(self):
        inputs, dataset, confirmed = self.squad_window_with_disputed_labels()
        with self.assertRaisesRegex(SystemExit, "taskCorrectness"):
            assemble(**inputs)
        document = confirmed_labels("squad-v2-dev", confirmed, dataset)
        inputs["labels"] = {"squad-v2-dev": self.write("confirmed.json", document)}

        report = assemble(**inputs)

        suites = report["evaluation"]["gates"]["taskCorrectness"]["suites"]
        suite = next(s for s in suites if s["name"] == "squad-v2-dev")
        self.assertEqual("confirmed", suite["labelSource"])
        self.assertEqual(document["casesSha256"], suite["labelsSha256"])
        self.assertAlmostEqual(0.5 * (130 / 140 + 1.0), suite["balancedAccuracy"])
        self.assertAlmostEqual(0.5, suite["baseBalancedAccuracy"])
        self.assertAlmostEqual(0.75, suite["originalBalancedAccuracy"])
        self.assertAlmostEqual(0.5, suite["originalBaseBalancedAccuracy"])
        low, high = suite["balancedInterval95"]
        self.assertLess(low, suite["balancedAccuracy"])
        self.assertGreater(high, suite["balancedAccuracy"])
        self.assertEqual(
            (130, 140, 60, 60),
            (suite["answerableCorrect"], suite["answerableCases"], suite["unanswerableCorrect"], suite["unanswerableCases"]),
        )
        self.assertEqual((1.0, 200, 200), (suite["structuredRate"], suite["cases"], suite["physicallySharedCases"]))
        other = next(s for s in suites if s["name"] != "squad-v2-dev")
        self.assertEqual("dataset", other["labelSource"])

    def test_confirmed_labels_fail_closed_on_case_set_hash_or_suite_mismatch(self):
        inputs, dataset, confirmed = self.squad_window_with_disputed_labels()
        missing = dict(confirmed)
        missing.pop("c000")
        inputs["labels"] = {"squad-v2-dev": self.write("missing.json", confirmed_labels("squad-v2-dev", missing, dataset))}
        with self.assertRaisesRegex(ValueError, "case ids"):
            assemble(**inputs)
        tampered = confirmed_labels("squad-v2-dev", confirmed, dataset)
        tampered["cases"][0]["label"] = "unanswerable"
        inputs["labels"] = {"squad-v2-dev": self.write("tampered.json", tampered)}
        with self.assertRaisesRegex(ValueError, "casesSha256"):
            assemble(**inputs)
        inputs["labels"] = {"squad-v2-dev": self.write("other.json", confirmed_labels("msmarco-v2.1-validation", confirmed, dataset))}
        with self.assertRaisesRegex(ValueError, "suite"):
            assemble(**inputs)
        inputs["labels"] = {"no-such-suite": self.write("nosuite.json", confirmed_labels("no-such-suite", confirmed, dataset))}
        with self.assertRaisesRegex(ValueError, "no-such-suite"):
            assemble(**inputs)

    def test_confirmed_labels_still_apply_the_floor(self):
        inputs, dataset, _ = self.squad_window_with_disputed_labels()
        # Confirmed labels that agree with the dataset leave the specialist at 0.75: still refused.
        inputs["labels"] = {"squad-v2-dev": self.write("same.json", confirmed_labels("squad-v2-dev", dataset, dataset))}
        with self.assertRaisesRegex(SystemExit, "taskCorrectness"):
            assemble(**inputs)

    def test_models_artifact_and_clean_host_run_are_validated_and_copied(self):
        report = assemble(
            **self.inputs(
                models_artifact=self.write("ma.json", models_artifact()),
                clean_host_run=self.write("ch.json", clean_host_run()),
            )
        )
        self.assertEqual(models_artifact(), report["evaluation"]["gates"]["modelsArtifact"])
        self.assertEqual(clean_host_run(), report["evaluation"]["gates"]["cleanHostRun"])
        with self.assertRaisesRegex(ValueError, "cleanHostRun"):
            assemble(
                **self.inputs(
                    models_artifact=self.write("ma2.json", models_artifact()),
                    clean_host_run=self.write("ch2.json", clean_host_run("0.3.41")),
                )
            )
        with self.assertRaisesRegex(ValueError, "modelsArtifact"):
            assemble(**self.inputs(clean_host_run=self.write("ch3.json", clean_host_run())))

    def test_models_artifact_validation_mirrors_the_gate(self):
        validate_models_artifact(models_artifact())
        mutations = [
            lambda g: g.pop("pass"),
            lambda g: g.update(version="0.3"),
            lambda g: g["artifacts"].pop(),
            lambda g: g["artifacts"][1].update(module="backend-java", coordinate="com.integrallis:backend-java:0.3.42"),
            lambda g: g["artifacts"][0].update(coordinate="com.integrallis:backend-java:0.3.41"),
            lambda g: g["artifacts"][0]["jar"].update(uri="https://example.com/backend-java-0.3.42.jar"),
            lambda g: g["artifacts"][0]["pom"].update(sha256="C" * 64),
            lambda g: g["artifacts"][0]["jar"].update(sizeBytes=0),
            lambda g: g["artifacts"][0]["jar"].update(sizeBytes=True),
            lambda g: g["artifacts"][0].pop("pom"),
        ]
        for index, mutate in enumerate(mutations):
            gate = models_artifact()
            mutate(gate)
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "modelsArtifact"):
                validate_models_artifact(gate)

    def test_clean_host_run_validation_mirrors_the_gate(self):
        validate_clean_host_run(clean_host_run(), "0.3.42")
        mutations = [
            lambda r: r.pop("pass"),
            lambda r: r.update(freshMachine=False),
            lambda r: r.update(externalInference=None),
            lambda r: r.update(javaMajor=21),
            lambda r: r.update(exitCode=1),
            lambda r: r.update(exitCode=False),
            lambda r: r.update(completedAt="2026-09-17T09:00:00Z"),
            lambda r: r.update(startedAt="2026-09-17 10:00:00"),
            lambda r: r.update(command=[]),
            lambda r: r.update(command=["java", ""]),
            lambda r: r.update(resolvedClasspathSha256="d" * 63),
            lambda r: r["outputLog"].update(uri="https://raw.githubusercontent.com/integrallis/models/main/run.log"),
            lambda r: r["outputLog"].update(sizeBytes=-1),
        ]
        for index, mutate in enumerate(mutations):
            run = clean_host_run()
            mutate(run)
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "cleanHostRun"):
                validate_clean_host_run(run, "0.3.42")

    def test_command_line_passes_labels_model_id_and_release_evidence_through(self):
        inputs, dataset, confirmed = self.squad_window_with_disputed_labels()
        labels = self.write("cli-labels.json", confirmed_labels("squad-v2-dev", confirmed, dataset))
        output = self.root / "report.json"
        argv = [
            "--adapter-directory", str(inputs["adapter_directory"]),
            "--base-model-id", inputs["base_model_id"],
            "--base-revision", inputs["base_revision"],
            "--base-artifact", str(inputs["base_artifact"]),
            "--junit-xml", str(inputs["junit_xml"]),
            "--long-context-report", str(inputs["long_context_report"]),
            "--crossover-report", str(inputs["crossover_report"]),
            "--models-revision", inputs["models_revision"],
            "--model-id", "integrallis_answerability",
            "--labels", f"squad-v2-dev={labels}",
            "--models-artifact", str(self.write("cli-ma.json", models_artifact())),
            "--clean-host-run", str(self.write("cli-ch.json", clean_host_run())),
            "--output", str(output),
        ]
        for key, path in inputs["window_reports"].items():
            argv += ["--window", f"{key}={path}"]
        for key, (java, kernel) in inputs["identity_reports"].items():
            argv += ["--identity", f"{key}={java},{kernel}"]
        main(argv)
        report = json.loads(output.read_text())
        gates = report["evaluation"]["gates"]
        self.assertEqual("integrallis_answerability", report["evaluation"]["artifact"]["modelId"])
        self.assertIn("confirmed", [s["labelSource"] for s in gates["taskCorrectness"]["suites"]])
        self.assertEqual("0.3.42", gates["modelsArtifact"]["version"])
        self.assertTrue(gates["cleanHostRun"]["pass"])


if __name__ == "__main__":
    unittest.main()
