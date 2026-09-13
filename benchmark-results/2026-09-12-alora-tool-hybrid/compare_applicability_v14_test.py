import copy
import unittest

from compare_applicability_v14 import compare_arms


def manifest(experiment, arm, coefficient, call, no_call, lm_loss, decision_loss):
    call_accuracy = call / 390
    no_call_accuracy = no_call / 81
    return {
        "experiment": experiment,
        "arm": arm,
        "base": {"model": "Qwen/Qwen3-1.7B", "revision": "pinned"},
        "continuedFrom": {"adapterSha256": "v9"},
        "decision": {"coefficient": coefficient, "callToken": 1, "noCallToken": 2},
        "data": {"preparedManifestSha256": "v8"},
        "environment": {"torch": "2.6.0", "cuda": "12.4", "gpu": "A16"},
        "training": {
            "epochs": 1.0,
            "screenSteps": 32,
            "optimizerStepsCompleted": 32,
            "learningRate": 2e-5,
            "gradientAccumulation": 16,
            "maxLength": 1024,
            "seed": 20260918,
            "initialLogitsSha256": "initial",
            "exactBaseLogitsSha256": "b" * 64,
            "initialDisabledAdapterLogitsSha256": "b" * 64,
            "finalDisabledAdapterLogitsSha256": "b" * 64,
            "finalLogitsSha256": f"{arm}-final",
            "postSaveLogitsSha256": f"{arm}-final",
            "reloadedLogitsSha256": f"{arm}-final",
            "trainedAdapterStateSha256": f"{arm}-state",
            "reloadedAdapterStateSha256": f"{arm}-state",
            "finiteGradientChecks": 512,
            "teacherForcedEvaluation": {
                "languageModelLoss": lm_loss,
                "applicabilityLoss": decision_loss,
                "call": {"correct": call, "total": 390},
                "noCall": {"correct": no_call, "total": 81},
                "balancedAccuracy": (call_accuracy + no_call_accuracy) / 2,
            },
        },
    }


class CompareApplicabilityV14Test(unittest.TestCase):
    def setUp(self):
        self.control = manifest(
            "qwen3-17b-applicability-decision-v13", "control", 0.0, 362, 77, 0.047, 0.183
        )
        self.treatment = manifest(
            "qwen3-17b-applicability-decision-v14", "treatment", 0.1, 360, 80, 0.049, 0.170
        )

    def test_passes_a_material_decision_improvement_without_class_regression(self):
        report = compare_arms(self.control, self.treatment, expected_steps=32)

        self.assertTrue(report["passed"])
        self.assertEqual(report["noCallCorrectGain"], 3)
        self.assertEqual(report["callCorrectLoss"], 2)

    def test_rejects_no_call_regression(self):
        treatment = copy.deepcopy(self.treatment)
        treatment["training"]["teacherForcedEvaluation"]["noCall"]["correct"] = 76

        report = compare_arms(self.control, treatment, expected_steps=32)

        self.assertFalse(report["passed"])
        self.assertIn("no-call", " ".join(report["failures"]))

    def test_rejects_less_than_five_percent_decision_loss_reduction(self):
        treatment = copy.deepcopy(self.treatment)
        treatment["training"]["teacherForcedEvaluation"]["applicabilityLoss"] = 0.174

        report = compare_arms(self.control, treatment, expected_steps=32)

        self.assertFalse(report["passed"])
        self.assertIn("5%", " ".join(report["failures"]))

    def test_rejects_a_nonidentical_reloaded_adapter_state(self):
        treatment = copy.deepcopy(self.treatment)
        treatment["training"]["reloadedAdapterStateSha256"] = "different"

        report = compare_arms(self.control, treatment, expected_steps=32)

        self.assertFalse(report["passed"])
        self.assertIn("adapter state", " ".join(report["failures"]))

    def test_rejects_a_wrong_treatment_experiment(self):
        treatment = copy.deepcopy(self.treatment)
        treatment["experiment"] = "qwen3-17b-applicability-decision-v13"

        report = compare_arms(self.control, treatment, expected_steps=32)

        self.assertFalse(report["passed"])
        self.assertIn("V14", " ".join(report["failures"]))


if __name__ == "__main__":
    unittest.main()
