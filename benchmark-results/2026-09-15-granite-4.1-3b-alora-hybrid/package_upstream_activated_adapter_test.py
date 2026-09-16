import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from package_upstream_activated_adapter import package_upstream_adapter


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PackageUpstreamActivatedAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.adapter = self.root / "adapter"
        self.adapter.mkdir()
        (self.adapter / "adapter_model.safetensors").write_bytes(b"weights")
        self.write_config(
            {
                "base_model_name_or_path": "ibm-granite/granite-4.1-3b",
                "peft_type": "LORA",
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "alora_invocation_tokens": [100264, 78191, 100265],
            }
        )
        self.card = self.root / "task-README.md"
        self.card.write_text("task-level publisher model card")
        self.base = self.root / "base.gguf"
        self.base.write_bytes(b"base")
        self.tokenizer = self.root / "tokenizer"
        self.tokenizer.mkdir()
        (self.tokenizer / "tokenizer.json").write_text("tokenizer")
        (self.tokenizer / "vocab.json").write_text("vocab")
        self.license = self.root / "LICENSE"
        self.license.write_text("Apache License 2.0")
        self.notice = self.root / "NOTICE"
        self.notice.write_text("upstream attribution")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_config(self, config: dict) -> None:
        (self.adapter / "adapter_config.json").write_text(json.dumps(config))

    def package(self, output: Path, **overrides) -> dict:
        arguments = dict(
            adapter_directory=self.adapter,
            base_artifact=self.base,
            base_model="ibm-granite/granite-4.1-3b",
            base_revision="a" * 40,
            tokenizer_directory=self.tokenizer,
            tokenizer_files=["vocab.json", "tokenizer.json"],
            publisher="IBM Research",
            repository="ibm-granite/granitelib-rag-r1.0",
            revision="b" * 40,
            license_name="Apache-2.0",
            license_file=self.license,
            notice_file=self.notice,
            invocation_tokens=[100264, 78191, 100265],
            invocation_text="<|start_of_role|>assistant<|end_of_role|>",
            model_card=self.card,
            output_directory=output,
        )
        arguments.update(overrides)
        return package_upstream_adapter(**arguments)

    def test_packages_a_token_declared_adapter_with_an_external_model_card(self) -> None:
        output = self.root / "runtime"
        metadata = self.package(output)

        self.assertEqual(5, metadata["schemaVersion"])
        self.assertEqual("activated-lora-specialist", metadata["kind"])
        self.assertEqual(sha256(self.base), metadata["base"]["artifactSha256"])
        self.assertEqual(
            "<|start_of_role|>assistant<|end_of_role|>", metadata["invocation"]["text"]
        )
        self.assertEqual([100264, 78191, 100265], metadata["invocation"]["tokens"])
        self.assertEqual(16, metadata["adapter"]["rank"])
        self.assertEqual(32, metadata["adapter"]["alpha"])
        self.assertEqual(sha256(self.card), metadata["upstream"]["modelCardSha256"])
        self.assertEqual(
            sha256(self.adapter / "adapter_config.json"),
            metadata["upstream"]["adapterConfigSha256"],
        )
        self.assertEqual(
            {"LICENSE", "NOTICE", "adapter_model.safetensors", "models-activated-lora.json"},
            {path.name for path in output.iterdir()},
        )

    def test_still_accepts_a_string_declared_adapter_with_an_inline_card(self) -> None:
        self.write_config(
            {
                "base_model_name_or_path": "ibm-granite/granite-4.1-3b",
                "peft_type": "LORA",
                "r": 32,
                "lora_alpha": 32,
                "target_modules": ["q_proj"],
                "invocation_string": "<marker>",
            }
        )
        (self.adapter / "README.md").write_text("inline card")

        metadata = self.package(
            self.root / "runtime", invocation_tokens=[1, 2], invocation_text=None, model_card=None
        )

        self.assertEqual("<marker>", metadata["invocation"]["text"])
        self.assertEqual([1, 2], metadata["invocation"]["tokens"])
        self.assertEqual(
            sha256(self.adapter / "README.md"), metadata["upstream"]["modelCardSha256"]
        )

    def test_rejects_pinned_tokens_that_differ_from_the_declared_tokens(self) -> None:
        with self.assertRaisesRegex(ValueError, "differ from the pinned invocation tokens"):
            self.package(self.root / "runtime", invocation_tokens=[100264, 78191])

    def test_requires_marker_text_when_only_tokens_are_declared(self) -> None:
        with self.assertRaisesRegex(ValueError, "--invocation-text is required"):
            self.package(self.root / "runtime", invocation_text=None)

    def test_rejects_marker_text_that_contradicts_a_declared_string(self) -> None:
        self.write_config(
            {
                "base_model_name_or_path": "ibm-granite/granite-4.1-3b",
                "peft_type": "LORA",
                "r": 32,
                "lora_alpha": 32,
                "target_modules": ["q_proj"],
                "invocation_string": "<marker>",
                "alora_invocation_tokens": [1, 2],
            }
        )

        with self.assertRaisesRegex(ValueError, "differs from adapter_config.invocation_string"):
            self.package(self.root / "runtime", invocation_tokens=[1, 2], invocation_text="<other>")

    def test_rejects_an_adapter_that_declares_no_activation_boundary(self) -> None:
        self.write_config(
            {
                "base_model_name_or_path": "ibm-granite/granite-4.1-3b",
                "peft_type": "LORA",
                "r": 32,
                "lora_alpha": 32,
                "target_modules": ["q_proj"],
            }
        )

        with self.assertRaisesRegex(ValueError, "must declare invocation_string or"):
            self.package(self.root / "runtime")

    def test_rejects_an_adapter_for_a_different_base(self) -> None:
        config = json.loads((self.adapter / "adapter_config.json").read_text())
        config["base_model_name_or_path"] = "someone-else/base"
        self.write_config(config)

        with self.assertRaisesRegex(ValueError, "base model differs"):
            self.package(self.root / "runtime")

    def test_rejects_an_unsupported_projection(self) -> None:
        config = json.loads((self.adapter / "adapter_config.json").read_text())
        config["target_modules"] = ["not_a_projection"]
        self.write_config(config)

        with self.assertRaisesRegex(ValueError, "unsupported projection"):
            self.package(self.root / "runtime")

    def test_rejects_a_missing_external_model_card(self) -> None:
        with self.assertRaisesRegex(ValueError, "adapter model card does not exist"):
            self.package(self.root / "runtime", model_card=self.root / "absent.md")


if __name__ == "__main__":
    unittest.main()
