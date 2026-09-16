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
        (self.adapter / "README.md").write_text("publisher model card")
        (self.adapter / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": "ibm-granite/granite-3.2-8b-instruct",
                    "peft_type": "LORA",
                    "r": 32,
                    "lora_alpha": 32,
                    "target_modules": ["q_proj", "k_proj", "v_proj"],
                    "invocation_string": "<marker>",
                }
            )
        )
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

    def package(self, output: Path) -> dict:
        return package_upstream_adapter(
            adapter_directory=self.adapter,
            base_artifact=self.base,
            base_model="ibm-granite/granite-3.2-8b-instruct",
            base_revision="a" * 40,
            tokenizer_directory=self.tokenizer,
            tokenizer_files=["vocab.json", "tokenizer.json"],
            publisher="IBM Research",
            repository="ibm-granite/granite-3.2-8b-alora-rag-query-rewrite",
            revision="b" * 40,
            license_name="Apache-2.0",
            license_file=self.license,
            notice_file=self.notice,
            invocation_tokens=[1, 2, 3],
            output_directory=output,
        )

    def test_packages_an_upstream_adapter_with_complete_pinned_provenance(self) -> None:
        output = self.root / "runtime"
        metadata = self.package(output)

        self.assertEqual(5, metadata["schemaVersion"])
        self.assertEqual(sha256(self.base), metadata["base"]["artifactSha256"])
        self.assertEqual("<marker>", metadata["invocation"]["text"])
        self.assertEqual([1, 2, 3], metadata["invocation"]["tokens"])
        self.assertEqual(sha256(self.adapter / "README.md"), metadata["upstream"]["modelCardSha256"])
        self.assertEqual(sha256(self.adapter / "adapter_config.json"), metadata["upstream"]["adapterConfigSha256"])
        self.assertEqual(
            {"LICENSE", "NOTICE", "adapter_model.safetensors", "models-activated-lora.json"},
            {path.name for path in output.iterdir()},
        )

    def test_rejects_an_adapter_for_a_different_base(self) -> None:
        config = json.loads((self.adapter / "adapter_config.json").read_text())
        config["base_model_name_or_path"] = "someone-else/base"
        (self.adapter / "adapter_config.json").write_text(json.dumps(config))

        with self.assertRaisesRegex(ValueError, "base model differs"):
            self.package(self.root / "runtime")

    def test_rejects_an_unsupported_projection(self) -> None:
        config = json.loads((self.adapter / "adapter_config.json").read_text())
        config["target_modules"] = ["not_a_projection"]
        (self.adapter / "adapter_config.json").write_text(json.dumps(config))

        with self.assertRaisesRegex(ValueError, "unsupported projection"):
            self.package(self.root / "runtime")
