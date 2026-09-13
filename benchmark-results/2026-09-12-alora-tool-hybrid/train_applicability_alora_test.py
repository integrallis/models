import math
from pathlib import Path
import unittest

from train_applicability_alora import (
    CALL_LABEL,
    NO_CALL_LABEL,
    auxiliary_loss,
    build_decision_example,
    class_weights,
    combined_loss,
    decision_contract,
    decision_prediction_position,
)
from train_alora import INVOCATION_TOKENS


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return [10, *INVOCATION_TOKENS]

    def encode(self, value, add_special_tokens=False):
        values = {
            "<tool_call>\n": [20, 21],
            '<tool_call>\n{"arguments":{},"name":"x"}\n</tool_call>': [
                20,
                21,
                30,
                40,
            ],
            "<tool_call>\n[]\n</tool_call>": [20, 21, 31, 41],
            '<tool_call>\n{"arguments":{},"name":"weather"}\n</tool_call><|im_end|>\n': [
                20,
                21,
                30,
                40,
                41,
            ],
            "<tool_call>\n[]\n</tool_call><|im_end|>\n": [20, 21, 31, 41],
        }
        return values[value]


class ApplicabilityAloraTest(unittest.TestCase):
    def test_identity_probes_precede_gradient_checkpointing_in_the_trainer_entrypoint(self):
        source = Path(__file__).with_name("train_applicability_alora.py").read_text()

        probe = source.index("initial_logits = initial_logits_sha256")
        checkpointing = source.index("model.gradient_checkpointing_enable")

        self.assertLess(probe, checkpointing)

    def test_resolves_the_single_contextual_call_no_call_decision(self):
        contract = decision_contract(FakeTokenizer())

        self.assertEqual(contract.call_prefix_tokens, (20, 21, 30))
        self.assertEqual(contract.no_call_prefix_tokens, (20, 21, 31))
        self.assertEqual(contract.shared_prefix_tokens, (20, 21))
        self.assertEqual(contract.call_token_id, 30)
        self.assertEqual(contract.no_call_token_id, 31)

    def test_rejects_a_context_that_changes_the_canonical_prefix(self):
        class BadTokenizer(FakeTokenizer):
            def encode(self, value, add_special_tokens=False):
                if value == '<tool_call>\n{"arguments":{},"name":"x"}\n</tool_call>':
                    return [20, 22, 30, 32]
                return super().encode(value, add_special_tokens)

        with self.assertRaisesRegex(ValueError, "preserve the canonical tool prefix"):
            decision_contract(BadTokenizer())

    def test_marks_the_first_completion_decision_and_its_causal_logit(self):
        record = {
            "sourceLine": 7,
            "user": "Weather?",
            "tools": [],
            "calls": [{"name": "weather", "arguments": {}}],
            "assistant": '<tool_call>\n{"arguments":{},"name":"weather"}\n</tool_call>',
        }

        example = build_decision_example(record, FakeTokenizer(), 32)

        self.assertEqual(example["decision_position"], 6)
        self.assertEqual(example["decision_label"], CALL_LABEL)
        self.assertEqual(example["input_ids"][6], 30)
        self.assertEqual(decision_prediction_position(example["decision_position"]), 5)

    def test_labels_an_empty_call_list_as_no_call(self):
        record = {
            "sourceLine": 8,
            "user": "Hello",
            "tools": [],
            "calls": [],
            "assistant": "<tool_call>\n[]\n</tool_call>",
        }

        example = build_decision_example(record, FakeTokenizer(), 32)

        self.assertEqual(example["decision_label"], NO_CALL_LABEL)
        self.assertEqual(example["input_ids"][6], 31)

    def test_uses_manual_balanced_weights_for_batch_size_one(self):
        call_weight, no_call_weight = class_weights(call_count=2356, no_call_count=484)

        self.assertAlmostEqual(call_weight, 0.602716468591, places=12)
        self.assertAlmostEqual(no_call_weight, 2.933884297521, places=12)
        self.assertAlmostEqual(
            auxiliary_loss(
                correct_logit=0.0,
                wrong_logit=0.0,
                example_weight=no_call_weight,
            ),
            math.log(2.0) * no_call_weight,
        )

    def test_increasing_the_correct_margin_lowers_the_auxiliary_loss(self):
        low_margin = auxiliary_loss(0.0, 0.0, 1.0)
        high_margin = auxiliary_loss(2.0, -1.0, 1.0)

        self.assertLess(high_margin, low_margin)
        epsilon = 1e-6
        correct_gradient = (
            auxiliary_loss(epsilon, 0.0, 1.0)
            - auxiliary_loss(-epsilon, 0.0, 1.0)
        ) / (2 * epsilon)
        wrong_gradient = (
            auxiliary_loss(0.0, epsilon, 1.0)
            - auxiliary_loss(0.0, -epsilon, 1.0)
        ) / (2 * epsilon)
        self.assertLess(correct_gradient, 0.0)
        self.assertGreater(wrong_gradient, 0.0)

    def test_zero_coefficient_is_the_unmodified_language_model_loss(self):
        language_model_loss = object()

        result = combined_loss(language_model_loss, object(), 0.0)

        self.assertIs(result, language_model_loss)

    def test_rejects_an_invalid_auxiliary_coefficient(self):
        with self.assertRaisesRegex(ValueError, "coefficient"):
            combined_loss(1.0, 2.0, -0.1)


if __name__ == "__main__":
    unittest.main()
