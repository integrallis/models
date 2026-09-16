import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from train_alora import (
    INVOCATION_TOKENS,
    build_example,
    last_subsequence,
    load_examples,
    prepare_for_evaluation,
    resolve_initial_adapter,
    resolve_candidate,
    verify_prepared_splits,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return [10, 11, *INVOCATION_TOKENS, 12, 13]

    def encode(self, value, add_special_tokens=False):
        return [20, 21, 22]


class TrainAloraTest(unittest.TestCase):
    def test_verifies_and_binds_an_initial_adapter_before_continuation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter = root / "adapter"
            adapter.mkdir()
            weights = adapter / "adapter_model.safetensors"
            weights.write_bytes(b"pinned adapter")
            weight_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
            (adapter / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "peft_type": "LORA",
                        "r": 16,
                        "lora_alpha": 32,
                        "alora_invocation_tokens": INVOCATION_TOKENS,
                        "target_modules": [
                            "q_proj",
                            "k_proj",
                            "v_proj",
                            "o_proj",
                            "gate_proj",
                            "up_proj",
                            "down_proj",
                        ],
                    }
                )
            )
            manifest = root / "training-manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "kind": "activated-lora-tool-specialist",
                        "base": {
                            "model": "Qwen/Qwen3-1.7B",
                            "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
                        },
                        "adapter": {
                            "rank": 16,
                            "alpha": 32,
                            "targetModules": [
                                "q_proj",
                                "k_proj",
                                "v_proj",
                                "o_proj",
                                "gate_proj",
                                "up_proj",
                                "down_proj",
                            ],
                            "files": [
                                {
                                    "name": weights.name,
                                    "bytes": weights.stat().st_size,
                                    "sha256": weight_hash,
                                }
                            ],
                        },
                        "invocation": {"tokens": INVOCATION_TOKENS},
                    }
                )
            )

            identity = resolve_initial_adapter(
                adapter,
                manifest,
                "Qwen/Qwen3-1.7B",
                "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
                16,
                32,
            )

            self.assertEqual(identity["adapterSha256"], weight_hash)
            self.assertEqual(
                identity["trainingManifestSha256"],
                hashlib.sha256(manifest.read_bytes()).hexdigest(),
            )
            weights.write_bytes(b"tampered value")
            with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
                resolve_initial_adapter(
                    adapter,
                    manifest,
                    "Qwen/Qwen3-1.7B",
                    "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
                    16,
                    32,
                )

    def test_disables_gradient_checkpointing_before_the_independent_evaluation_pass(self):
        class FakeModel:
            def __init__(self):
                self.disabled = False

            def gradient_checkpointing_disable(self):
                self.disabled = True

        model = FakeModel()
        prepare_for_evaluation(model, True)
        self.assertTrue(model.disabled)

    def test_resolves_only_pinned_training_candidates(self):
        self.assertEqual(
            resolve_candidate("qwen3-1.7b"),
            ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
        )
        with self.assertRaisesRegex(ValueError, "unsupported training candidate"):
            resolve_candidate("latest")

    def test_finds_the_last_complete_invocation(self):
        self.assertEqual(last_subsequence([1, 2, 1, 2, 3], [1, 2]), 2)
        self.assertEqual(last_subsequence([1, 2], [2, 3]), -1)

    def test_masks_every_token_before_the_completion(self):
        record = {
            "sourceLine": 4,
            "user": "Weather?",
            "tools": [],
            "calls": [],
            "assistant": "none",
        }
        example = build_example(record, FakeTokenizer(), 32)
        self.assertIsNotNone(example)
        self.assertEqual(example["invocation_start"], 2)
        self.assertEqual(example["labels"][:-3], [-100] * 7)
        self.assertEqual(example["labels"][-3:], [20, 21, 22])

    def test_refuses_to_truncate_a_training_example(self):
        record = {
            "sourceLine": 4,
            "user": "Weather?",
            "tools": [],
            "calls": [],
            "assistant": "none",
        }
        self.assertIsNone(build_example(record, FakeTokenizer(), 9))

    def test_reports_class_counts_only_for_examples_that_are_actually_trained(self):
        class VariableTokenizer(FakeTokenizer):
            def encode(self, value, add_special_tokens=False):
                return [20] * (20 if value.startswith("too long") else 3)

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "train.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "sourceLine": 1,
                        "user": "Skipped",
                        "tools": [],
                        "calls": [],
                        "assistant": "too long",
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "sourceLine": 2,
                        "user": "Used",
                        "tools": [],
                        "calls": [{"name": "weather", "arguments": {}}],
                        "assistant": "call",
                    }
                )
                + "\n"
            )

            examples, report = load_examples(source, VariableTokenizer(), 12, None)

            self.assertEqual(len(examples), 1)
            self.assertEqual(report["usable"], 1)
            self.assertEqual(report["skippedOverMaxLength"], 1)
            self.assertEqual(report["noCall"], 0)

    def test_rejects_an_ambiguous_invocation_boundary(self):
        class AmbiguousTokenizer(FakeTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                return [*INVOCATION_TOKENS, 1, *INVOCATION_TOKENS]

        record = {
            "sourceLine": 5,
            "user": "Weather?",
            "tools": [],
            "calls": [],
            "assistant": "none",
        }
        with self.assertRaisesRegex(ValueError, "multiple invocation"):
            build_example(record, AmbiguousTokenizer(), 32)

    def test_verifies_the_prepared_manifest_before_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            prepared = Path(temporary)
            train = prepared / "train.jsonl"
            validation = prepared / "validation.jsonl"
            train.write_text('{"calls":[]}\n')
            validation.write_text('{"calls":[{}]}\n')
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            (prepared / "manifest.json").write_text(
                json.dumps(
                    {
                        "schemaVersion": 2,
                        "train": {"sha256": digest(train)},
                        "validation": {"sha256": digest(validation)},
                    }
                )
            )

            manifest = verify_prepared_splits(prepared)

            self.assertEqual(manifest["schemaVersion"], 2)
            train.write_text('{"calls":[{}]}\n')
            with self.assertRaisesRegex(ValueError, "train split SHA-256 differs"):
                verify_prepared_splits(prepared)

    def test_rejects_an_unknown_preparation_schema_before_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            prepared = Path(temporary)
            (prepared / "manifest.json").write_text('{"schemaVersion":99}')

            with self.assertRaisesRegex(ValueError, "preparation schemaVersion"):
                verify_prepared_splits(prepared)


if __name__ == "__main__":
    unittest.main()
