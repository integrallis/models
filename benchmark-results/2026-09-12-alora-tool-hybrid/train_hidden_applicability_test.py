import unittest

from train_hidden_applicability import (
    EXPERIMENT,
    ClassificationScore,
    artifact_sha256,
    balanced_score,
    decision_feature_tokens,
    require_cublas_deterministic_workspace,
    select_regularization,
    stratified_folds,
    validation_passes,
)


class FakeTokenizer:
    def encode(self, value, add_special_tokens=False):
        self.call = (value, add_special_tokens)
        return [151657, 198]


class HiddenApplicabilityTest(unittest.TestCase):
    def test_cuda_extraction_requires_the_frozen_deterministic_workspace(self):
        require_cublas_deterministic_workspace(
            {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
        )

        with self.assertRaisesRegex(RuntimeError, "CUBLAS_WORKSPACE_CONFIG=:4096:8"):
            require_cublas_deterministic_workspace({})

        with self.assertRaisesRegex(RuntimeError, "CUBLAS_WORKSPACE_CONFIG=:4096:8"):
            require_cublas_deterministic_workspace(
                {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}
            )

    def test_feature_ends_before_the_call_or_no_call_token(self):
        tokenizer = FakeTokenizer()

        tokens = decision_feature_tokens([10, 11], tokenizer)

        self.assertEqual([10, 11, 151657, 198], tokens)
        self.assertEqual(("<tool_call>\n", False), tokenizer.call)

    def test_rejects_a_tokenizer_with_a_different_decision_prefix(self):
        class WrongTokenizer(FakeTokenizer):
            def encode(self, value, add_special_tokens=False):
                return [151657, 199]

        with self.assertRaisesRegex(ValueError, "decision prefix token IDs"):
            decision_feature_tokens([10, 11], WrongTokenizer())

    def test_stratifies_hash_ordered_folds_without_using_input_order(self):
        rows = [
            {"source_line": value, "label": label}
            for label in (0, 1)
            for value in range(10)
        ]

        first = stratified_folds(rows, 5)
        second = stratified_folds(list(reversed(rows)), 5)

        self.assertEqual(first, second)
        for fold in range(5):
            assigned = [row for row in rows if first[row["source_line"], row["label"]] == fold]
            self.assertEqual(2, sum(row["label"] == 0 for row in assigned))
            self.assertEqual(2, sum(row["label"] == 1 for row in assigned))

    def test_balanced_score_uses_the_frozen_strict_zero_boundary(self):
        result = balanced_score(
            labels=[1, 1, 0, 0],
            scores=[0.1, 0.0, -0.1, 2.0],
        )

        self.assertEqual(1, result.correct_calls)
        self.assertEqual(1, result.correct_no_calls)
        self.assertEqual(0.5, result.balanced_accuracy)

    def test_regularization_ties_prefer_no_calls_then_calls_then_larger_lambda(self):
        selected = select_regularization(
            {
                0.001: balanced_score([1, 1, 0, 0], [2.0, 1.0, -1.0, 1.0]),
                0.01: balanced_score([1, 1, 0, 0], [2.0, -1.0, -1.0, -2.0]),
                0.1: balanced_score([1, 1, 0, 0], [2.0, -1.0, -1.0, -2.0]),
            }
        )

        self.assertEqual(0.1, selected)

    def test_experiment_name_is_frozen(self):
        self.assertEqual("qwen3-17b-hidden-applicability-v16", EXPERIMENT)

    def test_artifact_fingerprint_is_canonical_and_excludes_its_own_field(self):
        left = {"b": 2, "a": 1}
        right = {"artifactSha256": "not-input", "a": 1, "b": 2}

        self.assertEqual(artifact_sha256(left), artifact_sha256(right))

    def test_validation_requires_every_frozen_floor(self):
        passing = ClassificationScore(390, 81, 371, 77, 371 / 390, 77 / 81, 0.951)
        missed_calls = ClassificationScore(390, 81, 370, 81, 370 / 390, 1.0, 0.99)
        missed_no_calls = ClassificationScore(390, 81, 390, 76, 1.0, 76 / 81, 0.99)
        tied_balance = ClassificationScore(390, 81, 390, 81, 1.0, 1.0, 0.95)

        self.assertTrue(validation_passes(passing))
        self.assertFalse(validation_passes(missed_calls))
        self.assertFalse(validation_passes(missed_no_calls))
        self.assertFalse(validation_passes(tied_balance))


if __name__ == "__main__":
    unittest.main()
