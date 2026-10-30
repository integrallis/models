# Copyright 2025-2026 Integrallis Software, LLC
# SPDX-License-Identifier: Apache-2.0
"""Offline admission checks must fail closed on partial or mismatched evidence."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("assessment", Path(__file__).with_name("assess-improvement.py"))
assessment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assessment)


class ImprovementAssessmentTest(unittest.TestCase):
    def test_complete_bound_bundle_passes_runtime_but_does_not_admit_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def put(name, value):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
            revision = "a" * 40
            put("inputs.json", {"sourceRevision": revision})
            put("parity.json", {"kind": "flint-native-parity-v1", "passed": True,
                "cases": [{"backend": backend, "case": case, "passed": True,
                    "tokenizerMatches": True, "continuationMatches": True,
                    "finiteLogits": True, "maxAbsoluteLogitDifferenceFromJava": 0.01,
                    "relativeRmsLogitDifferenceFromJava": 0.001}
                    for backend in ("pure-java", "rust-ffm") for case in ("chat", "completion")]})
            report = {"artifactSha256": assessment.MODEL_SHA, "backend": "rust-ffm",
                      "backendVersion": "models@0.5.7+flint-" + revision,
                      "settings": {"maxOutputTokens": 256}, "failures": [],
                      "summary": {"successfulAttempts": 9, "totalAttempts": 9,
                                  "correctAnswerRate": 1, "abstentionAccuracy": 1}}
            for label in ("native-default", "native-candidate", "native-reliability",
                          "ollama-candidate", "ollama-reliability"):
                put(label + ".json", report)
            put("default-correctness/smoke.json", {"reportSha256": assessment.digest(root / "native-default.json"),
                "artifactSha256": assessment.MODEL_SHA, "backend": "rust-ffm"})
            put("selected-profile.json", {"maxOutputTokens": 256, "pilotVisibleGatePassed": True,
                                           "heldOutConsulted": False})
            for label, ids, workload in (("native-default", [str(i) for i in range(9)], "general"),
                                         ("native-reliability", assessment.HELD_OUT_IDS, "answer-reliability")):
                put("visible/" + label + ".json", {
                    "sourceReportSha256": assessment.digest(root / (label + ".json")),
                    "artifactSha256": assessment.MODEL_SHA,
                    "policyId": "visible-answer-completion-and-bounded-facts-v1",
                    "assessmentGroundingPolicy": "bounded-context-injection-screened-citation-safe-statement-grounding-v24",
                    "workload": workload, "attempts": len(ids), "distinctQuestions": len(ids),
                    "requiredFactsAndCitationsPasses": len(ids),
                    "assessments": [{"caseId": key, "requiredFactsAndCitationsPass": True,
                        "polarityConsistent": True,
                        "visibleEvaluation": {"abstained": key.startswith("unknown-")}} for key in ids]})
            put("qualification.json", {"policyId": "production-rag-model-contribution-v6",
                "candidate": {"sha256": assessment.digest(root / "native-candidate.json")},
                "comparators": [{"sha256": assessment.digest(root / "ollama-candidate.json")}],
                "qualification": {"qualified": True}})
            result = assessment.assess(root)
            self.assertTrue(result["runtimeEvidencePassed"], result["checks"])
            self.assertFalse(result["catalogAdmitted"])
            self.assertTrue(result["releasedLibraryEvidenceRequired"])
            put("native-reliability.json", {"changed": True})
            result = assessment.assess(root)
            self.assertFalse(result["runtimeEvidencePassed"])
            self.assertIn("heldOutAccuracy", result["failedChecks"])

    def test_missing_reports_never_qualify(self):
        with tempfile.TemporaryDirectory() as directory:
            result = assessment.assess(Path(directory))
            self.assertFalse(result["runtimeEvidencePassed"])
            self.assertFalse(result["catalogAdmitted"])
            self.assertTrue(result["manualAnswerReviewRequired"])
            self.assertIn("numericalParity", result["failedChecks"])

    def test_hash_binding_rejects_stale_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "native-reliability.json").write_text('{"changed":true}')
            audit = {"sourceReportSha256": "0" * 64}
            with self.assertRaises(ValueError):
                assessment.bind_audit(root, "native-reliability", audit)

    def test_repeated_questions_cannot_replace_held_out_coverage(self):
        rows = [{"caseId": "unknown-moon", "requiredFactsAndCitationsPass": True,
                 "polarityConsistent": True, "visibleEvaluation": {"abstained": True}}
                for _ in range(16)]
        audit = {"assessments": rows, "attempts": 16, "distinctQuestions": 16,
                 "requiredFactsAndCitationsPasses": 16}
        with self.assertRaises(ValueError):
            assessment.check_held_out(audit)

    def test_one_missed_abstention_fails_even_with_fifteen_passes(self):
        ids = assessment.HELD_OUT_IDS
        rows = [{"caseId": key, "requiredFactsAndCitationsPass": key != "unknown-moon",
                 "polarityConsistent": True,
                 "visibleEvaluation": {"abstained": key == "unknown-satellite"}}
                for key in ids]
        audit = {"assessments": rows, "attempts": 16, "distinctQuestions": 16,
                 "requiredFactsAndCitationsPasses": 15}
        with self.assertRaises(ValueError):
            assessment.check_held_out(audit)

    def test_held_out_allows_one_factual_miss_but_no_polarity_error(self):
        rows = [{"caseId": key, "requiredFactsAndCitationsPass": key != "route-identifier",
                 "polarityConsistent": True,
                 "visibleEvaluation": {"abstained": key.startswith("unknown-")}}
                for key in assessment.HELD_OUT_IDS]
        audit = {"assessments": rows, "attempts": 16, "distinctQuestions": 16,
                 "requiredFactsAndCitationsPasses": 15}
        assessment.check_held_out(audit)
        rows[0]["polarityConsistent"] = False
        with self.assertRaises(ValueError):
            assessment.check_held_out(audit)


if __name__ == "__main__":
    unittest.main()
