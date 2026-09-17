#!/usr/bin/env python3
"""Unit tests for write_composition_clean_host_run.py.

Run from this directory:  python3 -m unittest write_composition_clean_host_run_test -v
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import write_composition_clean_host_run as writer

JAVA_25 = 'openjdk version "25.0.4" 2026-07-21 LTS\nOpenJDK Runtime Environment Temurin-25.0.4+1\n'
JAVA_21 = 'openjdk version "21.0.2" 2024-01-16 LTS\n'
FRESH = "absent /root/.m2\nabsent /root/.jbang\nabsent /root/.gradle\n"
BASE_SHA = "662b0626cd58f443baea23559b469df6576a81d349649c59413b36a9fb32eb29"
ADAPTER_SHA = "67533dff14cd0cfa9ae0d00e83a9955226426ff1f98f1fd28790b2fd8757eea6"
WINDOW_SHA = "dfb8cd7203418c79bae27306ffc4857f0ab02be7f9f8ddcae793af556e9cab37"
CASES_PER_SUITE = 3
MEASUREMENTS = CASES_PER_SUITE * 2 * 2


def measurements(cases_per_suite: int = CASES_PER_SUITE) -> list[dict]:
    rows = []
    for suite in writer.SUITES:
        for index in range(cases_per_suite):
            for arm in writer.ARMS:
                rows.append(
                    {
                        "arm": arm,
                        "suite": suite,
                        "id": f"{suite}-{index}",
                        "promptSha256": "a" * 64,
                        "output": '"answerable"',
                        "label": "ANSWERABLE",
                        "structured": True,
                        "physicallySharesPrefix": arm == "composite",
                        "sharedPrefixTokens": 300,
                        "millis": 30_000 if arm == "composite" else 60_000,
                        "pass": True,
                    }
                )
    return rows


def program_report(**overrides) -> dict:
    report = {
        "schemaVersion": 1,
        "pass": True,
        "modeljarsVersion": "0.1.47",
        "modelsVersion": "0.3.42",
        "windowSha256": WINDOW_SHA,
        "casesPerSuite": CASES_PER_SUITE,
        "armOrder": "composite-first",
        "minimumSharedPrefixTokens": 256,
        "chatTemplate": "GRANITE",
        "controlMedianMillis": 60_000.0,
        "compositeMedianMillis": 30_000.0,
        "latencyImprovement": 0.5,
        "publishedArtifacts": [
            {
                "modelId": "ibm_granite_granite_4_1_3b_gguf_q4_k_m",
                "coordinate": (
                    "org.modeljars.huggingface:ibm-granite.granite-4.1-3b-gguf.q4_k_m:"
                    "4.1.0-q4_k_m.2"
                ),
                "sha256": BASE_SHA,
                "resolvedFromCentral": True,
                "runViaPublicApi": True,
                "markerResourceUrl": "jar:file:/root/.m2/repository/x.jar!/",
            },
            {
                "modelId": "modeljars_granite_4_1_3b_answerability_alora_integrallis_f32",
                "coordinate": (
                    "org.modeljars.github:modeljars.activated-adapters."
                    "granite-4.1-3b-answerability-alora-integrallis.f32:1.0.0-f32.1"
                ),
                "sha256": ADAPTER_SHA,
                "resolvedFromCentral": True,
                "runViaPublicApi": True,
                "markerResourceUrl": "jar:file:/root/.m2/repository/y.jar!/",
            },
        ],
        "measurements": measurements(),
    }
    report.update(overrides)
    return report


class WriteCompositionCleanHostRunTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.classpath = self.root / "resolved-classpath.txt"
        self.classpath.write_text("abc  modeljars-0.1.47.jar\n", encoding="utf-8")
        self.log = self.root / "composition-clean-host-output.log"
        self.log.write_text(f"running\n{writer.pass_line(MEASUREMENTS)}\n", encoding="utf-8")

    def build(self, **overrides):
        arguments = {
            "started_at": "2026-09-18T10:00:00Z",
            "completed_at": "2026-09-18T10:45:00Z",
            "exit_code": 0,
            "models_version": "0.3.42",
            "modeljars_version": "0.1.47",
            "command": ["jbang", "CompositionCleanHost.java"],
            "java_version_output": JAVA_25,
            "fresh_check": FRESH,
            "classpath_file": self.classpath,
            "output_log": self.log,
            "output_log_repo_path": "benchmark-results/x/composition-clean-host-output.log",
            "program_report": program_report(),
        }
        arguments.update(overrides)
        return writer.build(**arguments)

    def test_records_a_passing_composition_run(self) -> None:
        run = self.build()

        self.assertTrue(run["pass"])
        self.assertTrue(run["freshMachine"])
        self.assertFalse(run["externalInference"])
        self.assertEqual(25, run["javaMajor"])
        self.assertEqual(0, run["exitCode"])
        self.assertEqual("0.3.42", run["modelsVersion"])
        self.assertEqual("0.1.47", run["modeljarsVersion"])
        self.assertEqual(list(writer.SUITES), run["suites"])
        self.assertEqual(CASES_PER_SUITE, run["casesPerSuite"])
        self.assertEqual(MEASUREMENTS // 2, run["casesPerArm"])
        self.assertEqual("composite-first", run["armOrder"])
        self.assertEqual(WINDOW_SHA, run["windowSha256"])
        self.assertEqual(60_000.0, run["controlMedianMillis"])
        self.assertEqual(30_000.0, run["compositeMedianMillis"])
        self.assertEqual(0.5, run["latencyImprovement"])
        self.assertEqual(2, len(run["publishedArtifacts"]))
        self.assertEqual(
            hashlib.sha256(self.classpath.read_bytes()).hexdigest(),
            run["resolvedClasspathSha256"],
        )
        self.assertEqual(
            hashlib.sha256(self.log.read_bytes()).hexdigest(), run["outputLog"]["sha256"]
        )
        self.assertEqual(self.log.stat().st_size, run["outputLog"]["sizeBytes"])
        self.assertIn(writer.REVISION_PLACEHOLDER, run["outputLog"]["uri"])

    def test_fails_when_a_member_was_not_resolved_from_central(self) -> None:
        report = program_report()
        report["publishedArtifacts"][0]["resolvedFromCentral"] = False

        self.assertFalse(self.build(program_report=report)["pass"])

    def test_fails_when_a_member_was_not_run_through_the_public_api(self) -> None:
        report = program_report()
        report["publishedArtifacts"][1]["runViaPublicApi"] = False

        self.assertFalse(self.build(program_report=report)["pass"])

    def test_rejects_a_truthy_resolution_flag_that_is_not_true(self) -> None:
        report = program_report()
        report["publishedArtifacts"][0]["resolvedFromCentral"] = "true"

        self.assertFalse(self.build(program_report=report)["pass"])

    def test_rejects_an_artifact_without_a_real_sha256(self) -> None:
        report = program_report()
        report["publishedArtifacts"][0]["sha256"] = "not-a-digest"

        with self.assertRaises(ValueError):
            self.build(program_report=report)

    def test_rejects_an_artifact_coordinate_that_is_not_gav(self) -> None:
        report = program_report()
        report["publishedArtifacts"][0]["coordinate"] = "org.modeljars:only-two"

        with self.assertRaises(ValueError):
            self.build(program_report=report)

    def test_rejects_duplicate_members(self) -> None:
        report = program_report()
        report["publishedArtifacts"][1]["modelId"] = report["publishedArtifacts"][0]["modelId"]

        with self.assertRaises(ValueError):
            self.build(program_report=report)

    def test_fails_when_the_improvement_does_not_follow_from_the_medians(self) -> None:
        report = program_report(latencyImprovement=0.9)

        self.assertFalse(self.build(program_report=report)["pass"])

    def test_fails_when_the_composite_arm_is_not_the_faster_one(self) -> None:
        report = program_report(
            controlMedianMillis=30_000.0, compositeMedianMillis=60_000.0, latencyImprovement=-1.0
        )

        self.assertFalse(self.build(program_report=report)["pass"])

    def test_accepts_the_improvement_within_floating_point_tolerance(self) -> None:
        report = program_report(
            controlMedianMillis=823_087.348929,
            compositeMedianMillis=289_346.722502,
            latencyImprovement=(823_087.348929 - 289_346.722502) / 823_087.348929,
        )

        self.assertTrue(self.build(program_report=report)["pass"])

    def test_fails_when_a_composite_measurement_did_not_physically_share(self) -> None:
        report = program_report()
        for measurement in report["measurements"]:
            if measurement["arm"] == "composite":
                measurement["physicallySharesPrefix"] = False
                break

        self.assertFalse(self.build(program_report=report)["pass"])

    def test_fails_when_a_control_measurement_shared_physically(self) -> None:
        report = program_report()
        for measurement in report["measurements"]:
            if measurement["arm"] == "control":
                measurement["physicallySharesPrefix"] = True
                break

        self.assertFalse(self.build(program_report=report)["pass"])

    def test_fails_when_a_measurement_did_not_pass(self) -> None:
        report = program_report()
        report["measurements"][0]["pass"] = False

        self.assertFalse(self.build(program_report=report)["pass"])

    def test_fails_when_measurements_are_missing_for_the_declared_configuration(self) -> None:
        report = program_report()
        report["measurements"] = report["measurements"][:-2]

        self.assertFalse(self.build(program_report=report)["pass"])

    def test_fails_when_the_program_reported_failure(self) -> None:
        self.assertFalse(self.build(program_report=program_report(**{"pass": False}))["pass"])

    def test_fails_on_a_dirty_machine(self) -> None:
        self.assertFalse(self.build(fresh_check="present /root/.m2\nabsent /root/.jbang\n")["pass"])

    def test_fails_on_the_wrong_java(self) -> None:
        run = self.build(java_version_output=JAVA_21)

        self.assertFalse(run["pass"])
        self.assertEqual(21, run["javaMajor"])

    def test_fails_on_a_nonzero_exit(self) -> None:
        self.assertFalse(self.build(exit_code=1)["pass"])

    def test_fails_when_the_log_does_not_end_with_the_verdict(self) -> None:
        self.log.write_text("running\nFAIL measurements=12 passed=11\n", encoding="utf-8")

        self.assertFalse(self.build()["pass"])

    def test_fails_when_the_log_verdict_counts_fewer_measurements(self) -> None:
        self.log.write_text(f"running\n{writer.pass_line(MEASUREMENTS - 2)}\n", encoding="utf-8")

        self.assertFalse(self.build()["pass"])

    def test_fails_when_the_run_did_not_advance_the_clock(self) -> None:
        self.assertFalse(
            self.build(
                started_at="2026-09-18T10:00:00Z", completed_at="2026-09-18T10:00:00Z"
            )["pass"]
        )

    def test_rejects_a_program_report_from_another_release(self) -> None:
        with self.assertRaises(ValueError):
            self.build(program_report=program_report(modelsVersion="0.3.41"))
        with self.assertRaises(ValueError):
            self.build(program_report=program_report(modeljarsVersion="0.1.46"))

    def test_rejects_an_unknown_arm_order(self) -> None:
        with self.assertRaises(ValueError):
            self.build(program_report=program_report(armOrder="interleaved"))

    def test_rejects_an_absolute_output_log_path(self) -> None:
        with self.assertRaises(ValueError):
            self.build(output_log_repo_path="/etc/passwd")

    def test_rejects_an_empty_command(self) -> None:
        with self.assertRaises(ValueError):
            self.build(command=[])

    def test_rejects_a_non_positive_cases_per_suite(self) -> None:
        with self.assertRaises(ValueError):
            self.build(program_report=program_report(casesPerSuite=0))

    def test_cli_writes_both_records_and_reports_the_verdict(self) -> None:
        java_file = self.root / "java-version.txt"
        java_file.write_text(JAVA_25, encoding="utf-8")
        fresh_file = self.root / "fresh-check.txt"
        fresh_file.write_text(FRESH, encoding="utf-8")
        report_file = self.root / "program-report.json"
        report_file.write_text(json.dumps(program_report()), encoding="utf-8")
        out = self.root / "composition-clean-host-run.json"
        artifacts_out = self.root / "published-artifacts.json"

        status = writer.main(
            [
                "--started-at", "2026-09-18T10:00:00Z",
                "--completed-at", "2026-09-18T10:45:00Z",
                "--exit-code", "0",
                "--models-version", "0.3.42",
                "--modeljars-version", "0.1.47",
                "--command-json", json.dumps(["jbang", "CompositionCleanHost.java"]),
                "--java-version-file", str(java_file),
                "--fresh-check-file", str(fresh_file),
                "--classpath-file", str(self.classpath),
                "--output-log", str(self.log),
                "--output-log-repo-path", "benchmark-results/x/log.txt",
                "--program-report", str(report_file),
                "--out", str(out),
                "--published-artifacts-out", str(artifacts_out),
            ]
        )

        self.assertEqual(0, status)
        written = json.loads(out.read_text(encoding="utf-8"))
        self.assertTrue(written["pass"])
        artifacts = json.loads(artifacts_out.read_text(encoding="utf-8"))
        self.assertIsInstance(artifacts, list)
        self.assertEqual(
            {
                "ibm_granite_granite_4_1_3b_gguf_q4_k_m",
                "modeljars_granite_4_1_3b_answerability_alora_integrallis_f32",
            },
            {artifact["modelId"] for artifact in artifacts},
        )

    def test_cli_returns_one_when_the_run_did_not_pass(self) -> None:
        java_file = self.root / "java-version.txt"
        java_file.write_text(JAVA_25, encoding="utf-8")
        fresh_file = self.root / "fresh-check.txt"
        fresh_file.write_text(FRESH, encoding="utf-8")
        report_file = self.root / "program-report.json"
        report_file.write_text(json.dumps(program_report(**{"pass": False})), encoding="utf-8")
        out = self.root / "composition-clean-host-run.json"

        status = writer.main(
            [
                "--started-at", "2026-09-18T10:00:00Z",
                "--completed-at", "2026-09-18T10:45:00Z",
                "--exit-code", "0",
                "--models-version", "0.3.42",
                "--modeljars-version", "0.1.47",
                "--command-json", json.dumps(["jbang", "CompositionCleanHost.java"]),
                "--java-version-file", str(java_file),
                "--fresh-check-file", str(fresh_file),
                "--classpath-file", str(self.classpath),
                "--output-log", str(self.log),
                "--output-log-repo-path", "benchmark-results/x/log.txt",
                "--program-report", str(report_file),
                "--out", str(out),
            ]
        )

        self.assertEqual(1, status)
        self.assertFalse(json.loads(out.read_text(encoding="utf-8"))["pass"])


if __name__ == "__main__":
    unittest.main()
