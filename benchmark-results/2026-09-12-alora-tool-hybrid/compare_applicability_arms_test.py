import copy
import unittest

from compare_applicability_arms import compare_arms


def manifest(arm, coefficient, call_correct, no_call_correct, lm_loss):
    call_accuracy = call_correct / 390
    no_call_accuracy = no_call_correct / 81
    return {
        "experiment": "qwen3-17b-applicability-decision-v13",
        "arm": arm,
        "base": {"model": "Qwen/Qwen3-1.7B", "revision": "pinned"},
        "continuedFrom": {"adapterSha256": "v9", "trainingManifestSha256": "v9-manifest"},
        "decision": {"coefficient": coefficient},
        "data": {"preparedManifestSha256": "v8", "train": {"sourceLinesSha256": "train"}},
        "training": {
            "epochs": 1.0,
            "screenSteps": 32,
            "optimizerStepsCompleted": 32,
            "learningRate": 2e-5,
            "gradientAccumulation": 16,
            "maxLength": 1024,
            "seed": 20260918,
            "initialLogitsSha256": "same-logits",
            "exactBaseLogitsSha256": "b" * 64,
            "initialDisabledAdapterLogitsSha256": "b" * 64,
            "finalLogitsSha256": f"{arm}-final",
            "finalDisabledAdapterLogitsSha256": "b" * 64,
            "reloadedLogitsSha256": f"{arm}-final",
            "finiteGradientChecks": 512,
            "teacherForcedEvaluation": {
                "languageModelLoss": lm_loss,
                "call": {"correct": call_correct, "total": 390, "accuracy": call_accuracy},
                "noCall": {"correct": no_call_correct, "total": 81, "accuracy": no_call_accuracy},
                "balancedAccuracy": (call_accuracy + no_call_accuracy) / 2,
            },
        },
        "environment": {"torch": "2.6.0", "cuda": "12.4", "gpu": "A40"},
    }


class CompareApplicabilityArmsTest(unittest.TestCase):
    def test_passes_only_when_every_frozen_directional_gate_passes(self):
        control = manifest("control", 0.0, 380, 60, 1.0)
        treatment = manifest("treatment", 0.1, 374, 69, 1.05)

        report = compare_arms(control, treatment, expected_steps=32)

        self.assertTrue(report["passed"])
        self.assertEqual(report["noCallCorrectGain"], 9)
        self.assertEqual(report["callCorrectLoss"], 6)

    def test_rejects_less_than_nine_extra_no_call_decisions(self):
        report = compare_arms(
            manifest("control", 0.0, 380, 60, 1.0),
            manifest("treatment", 0.1, 374, 68, 1.05),
            expected_steps=32,
        )
        self.assertFalse(report["passed"])
        self.assertIn("nine additional", " ".join(report["failures"]))

    def test_rejects_more_than_seven_lost_call_decisions(self):
        report = compare_arms(
            manifest("control", 0.0, 380, 60, 1.0),
            manifest("treatment", 0.1, 372, 70, 1.05),
            expected_steps=32,
        )
        self.assertFalse(report["passed"])
        self.assertIn("seven", " ".join(report["failures"]))

    def test_rejects_a_language_model_loss_regression_over_ten_percent(self):
        report = compare_arms(
            manifest("control", 0.0, 380, 60, 1.0),
            manifest("treatment", 0.1, 374, 70, 1.10001),
            expected_steps=32,
        )
        self.assertFalse(report["passed"])
        self.assertIn("10%", " ".join(report["failures"]))

    def test_rejects_nonidentical_initial_logits(self):
        control = manifest("control", 0.0, 380, 60, 1.0)
        treatment = manifest("treatment", 0.1, 374, 70, 1.05)
        treatment["training"]["initialLogitsSha256"] = "different"

        report = compare_arms(control, treatment, expected_steps=32)

        self.assertFalse(report["passed"])
        self.assertIn("initial logits", " ".join(report["failures"]))

    def test_rejects_any_nonobjective_contract_difference(self):
        control = manifest("control", 0.0, 380, 60, 1.0)
        treatment = copy.deepcopy(manifest("treatment", 0.1, 374, 70, 1.05))
        treatment["training"]["seed"] += 1

        report = compare_arms(control, treatment, expected_steps=32)

        self.assertFalse(report["passed"])
        self.assertIn("training contract", " ".join(report["failures"]))


if __name__ == "__main__":
    unittest.main()
