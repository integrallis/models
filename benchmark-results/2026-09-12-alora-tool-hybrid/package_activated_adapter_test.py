import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from package_activated_adapter import package_adapter


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PackageActivatedAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.adapter = self.root / "adapter"
        self.adapter.mkdir()
        for name, content in {
            "adapter_model.safetensors": b"weights",
            "adapter_config.json": b"config",
            "README.md": b"readme",
            "tokenizer.json": b"tokenizer",
            "tokenizer_config.json": b"tokenizer config",
        }.items():
            (self.adapter / name).write_bytes(content)
        self.prepared = self.root / "prepared-manifest.json"
        self.prepared.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "trainingSource": {
                        "path": "data/train.javajs.messages.jsonl",
                        "sha256": "1" * 64,
                    },
                    "train": {"sha256": "2" * 64},
                    "validation": {"sha256": "3" * 64},
                }
            )
        )
        self.formatter = self.root / "train_alora.py"
        self.formatter.write_text("formatter")
        self.notice = self.root / "NOTICE"
        self.notice.write_text("Pinned base and training-data attribution\n")
        self.license = self.root / "LICENSE"
        self.license.write_text("Apache License 2.0\n")
        self.base = self.root / "base.gguf"
        self.base.write_bytes(b"base")
        self.training = self.root / "training-manifest.json"
        files = [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(self.adapter.iterdir())
        ]
        self.training.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "kind": "activated-lora-tool-specialist",
                    "base": {"model": "Qwen/Qwen3-0.6B", "revision": "a" * 40},
                    "invocation": {"tokens": [151644, 77091, 198]},
                    "adapter": {
                        "rank": 32,
                        "alpha": 64,
                        "targetModules": [
                            "q_proj",
                            "k_proj",
                            "v_proj",
                            "o_proj",
                            "gate_proj",
                            "up_proj",
                            "down_proj",
                        ],
                        "files": files,
                    },
                    "data": {
                        "preparedManifestSha256": sha256(self.prepared),
                        "train": {
                            "sha256": "2" * 64,
                            "usable": 4000,
                            "skippedOverMaxLength": 64,
                            "noCall": 760,
                            "multipleCall": 1709,
                            "sourceLinesSha256": "7" * 64,
                        },
                        "validation": {
                            "sha256": "3" * 64,
                            "usable": 979,
                            "skippedOverMaxLength": 21,
                            "noCall": 190,
                            "multipleCall": 420,
                            "sourceLinesSha256": "8" * 64,
                        },
                    },
                }
            )
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_packages_a_provenance_bound_runtime_adapter(self) -> None:
        output = self.root / "runtime"

        metadata = package_adapter(
            training_manifest=self.training,
            preparation_manifest=self.prepared,
            adapter_directory=self.adapter,
            base_artifact=self.base,
            dataset="edbuildingstuff/bfcl-ft-data",
            dataset_revision="b" * 40,
            source_file="train.javajs.messages.jsonl",
            formatter=self.formatter,
            notice=self.notice,
            output_directory=output,
        )

        self.assertEqual(4, metadata["schemaVersion"])
        self.assertEqual(sha256(self.base), metadata["base"]["artifactSha256"])
        self.assertEqual("b" * 40, metadata["training"]["sources"][0]["datasetRevision"])
        self.assertEqual(sha256(self.formatter), metadata["training"]["formatterSha256"])
        self.assertEqual(
            {
                "usable": 4000,
                "skippedOverMaxLength": 64,
                "noCall": 760,
                "multipleCall": 1709,
                "sourceLinesSha256": "7" * 64,
            },
            metadata["training"]["trainSelection"],
        )
        self.assertEqual(979, metadata["training"]["validationSelection"]["usable"])
        self.assertEqual(
            ["tokenizer.json", "tokenizer_config.json"],
            [entry["name"] for entry in metadata["tokenizer"]["files"]],
        )
        self.assertEqual(
            {"LICENSE", "NOTICE", "adapter_model.safetensors", "models-activated-lora.json"},
            {path.name for path in output.iterdir()},
        )
        self.assertEqual(
            metadata,
            json.loads((output / "models-activated-lora.json").read_text()),
        )

    def test_rejects_a_bundle_without_the_base_license(self) -> None:
        self.license.unlink()

        with self.assertRaisesRegex(ValueError, "base license does not exist"):
            package_adapter(
                training_manifest=self.training,
                preparation_manifest=self.prepared,
                adapter_directory=self.adapter,
                base_artifact=self.base,
                dataset="edbuildingstuff/bfcl-ft-data",
                dataset_revision="b" * 40,
                source_file="train.javajs.messages.jsonl",
                formatter=self.formatter,
                notice=self.notice,
                output_directory=self.root / "runtime",
            )

    def test_packages_every_pinned_schema_two_training_source(self) -> None:
        prepared = json.loads(self.prepared.read_text())
        prepared["schemaVersion"] = 2
        prepared["trainingSource"].update(
            {
                "path": "train.javajs.messages.jsonl",
                "repository": "edbuildingstuff/bfcl-ft-data",
                "revision": "b" * 40,
            }
        )
        prepared["irrelevanceSource"] = {
            "repository": "MadeAgents/xlam-irrelevance-7.5k",
            "revision": "c" * 40,
            "path": "xlam-7.5k-irrelevancek.json",
            "sha256": "5" * 64,
        }
        self.prepared.write_text(json.dumps(prepared))
        training = json.loads(self.training.read_text())
        training["data"]["preparedManifestSha256"] = sha256(self.prepared)
        training["data"]["preparedSchemaVersion"] = 2
        self.training.write_text(json.dumps(training))

        metadata = package_adapter(
            training_manifest=self.training,
            preparation_manifest=self.prepared,
            adapter_directory=self.adapter,
            base_artifact=self.base,
            dataset="edbuildingstuff/bfcl-ft-data",
            dataset_revision="b" * 40,
            source_file="train.javajs.messages.jsonl",
            formatter=self.formatter,
            notice=self.notice,
            output_directory=self.root / "runtime",
        )

        self.assertEqual(
            [
                {
                    "role": "tool-calls",
                    "dataset": "edbuildingstuff/bfcl-ft-data",
                    "datasetRevision": "b" * 40,
                    "sourceFile": "train.javajs.messages.jsonl",
                    "sourceSha256": "1" * 64,
                },
                {
                    "role": "no-call",
                    "dataset": "MadeAgents/xlam-irrelevance-7.5k",
                    "datasetRevision": "c" * 40,
                    "sourceFile": "xlam-7.5k-irrelevancek.json",
                    "sourceSha256": "5" * 64,
                },
            ],
            metadata["training"]["sources"],
        )
        self.assertEqual(2, metadata["training"]["preparedSchemaVersion"])

    def test_packages_a_merged_schema_two_preparation_with_both_pinned_sources(self) -> None:
        prepared = json.loads(self.prepared.read_text())
        prepared["schemaVersion"] = 2
        prepared.pop("trainingSource")
        prepared["trainingSources"] = [
            {
                "role": "tool-calls",
                "repository": "edbuildingstuff/bfcl-ft-data",
                "revision": "b" * 40,
                "path": "train.javajs.messages.jsonl",
                "sha256": "1" * 64,
            },
            {
                "role": "hard-no-call",
                "repository": "ShishirPatil/gorilla",
                "revision": "c" * 40,
                "path": "bfcl_eval/data/BFCL_v3_live_irrelevance.json",
                "sha256": "5" * 64,
            },
        ]
        self.prepared.write_text(json.dumps(prepared))
        training = json.loads(self.training.read_text())
        training["data"]["preparedManifestSha256"] = sha256(self.prepared)
        training["data"]["preparedSchemaVersion"] = 2
        self.training.write_text(json.dumps(training))

        metadata = package_adapter(
            training_manifest=self.training,
            preparation_manifest=self.prepared,
            adapter_directory=self.adapter,
            base_artifact=self.base,
            dataset="edbuildingstuff/bfcl-ft-data",
            dataset_revision="b" * 40,
            source_file="train.javajs.messages.jsonl",
            formatter=self.formatter,
            notice=self.notice,
            output_directory=self.root / "runtime",
        )

        self.assertEqual(
            ["tool-calls", "hard-no-call"],
            [source["role"] for source in metadata["training"]["sources"]],
        )
        self.assertEqual(
            "bfcl_eval/data/BFCL_v3_live_irrelevance.json",
            metadata["training"]["sources"][1]["sourceFile"],
        )

    def test_resolves_merged_sources_from_hash_bound_parent_manifests(self) -> None:
        positive_parent = self.root / "positive-parent.json"
        positive_parent.write_text(
            json.dumps(
                {
                    "schemaVersion": 2,
                    "trainingSource": {
                        "repository": "edbuildingstuff/bfcl-ft-data",
                        "revision": "b" * 40,
                        "path": "train.javajs.messages.jsonl",
                        "sha256": "1" * 64,
                    },
                    "train": {"sha256": "4" * 64},
                    "validation": {"sha256": "5" * 64},
                }
            )
        )
        negative_parent = self.root / "negative-parent.json"
        negative_parent.write_text(
            json.dumps(
                {
                    "schemaVersion": 2,
                    "source": {
                        "repository": "ShishirPatil/gorilla",
                        "revision": "c" * 40,
                        "path": "BFCL_v3_live_irrelevance.json",
                        "sha256": "6" * 64,
                    },
                    "train": {"sha256": "7" * 64},
                    "validation": {"sha256": "8" * 64},
                }
            )
        )
        prepared = json.loads(self.prepared.read_text())
        prepared["schemaVersion"] = 2
        prepared.pop("trainingSource")
        prepared["parents"] = {
            "positive": {
                "manifestSha256": sha256(positive_parent),
                "trainSha256": "4" * 64,
                "validationSha256": "5" * 64,
            },
            "negative": {
                "manifestSha256": sha256(negative_parent),
                "trainSha256": "7" * 64,
                "validationSha256": "8" * 64,
            },
        }
        self.prepared.write_text(json.dumps(prepared))
        training = json.loads(self.training.read_text())
        training["data"]["preparedManifestSha256"] = sha256(self.prepared)
        training["data"]["preparedSchemaVersion"] = 2
        self.training.write_text(json.dumps(training))

        metadata = package_adapter(
            training_manifest=self.training,
            preparation_manifest=self.prepared,
            adapter_directory=self.adapter,
            base_artifact=self.base,
            dataset="edbuildingstuff/bfcl-ft-data",
            dataset_revision="b" * 40,
            source_file="train.javajs.messages.jsonl",
            formatter=self.formatter,
            notice=self.notice,
            output_directory=self.root / "runtime",
            positive_parent_manifest=positive_parent,
            negative_parent_manifest=negative_parent,
        )

        self.assertEqual(
            ["tool-calls", "hard-no-call"],
            [source["role"] for source in metadata["training"]["sources"]],
        )
        self.assertEqual("6" * 64, metadata["training"]["sources"][1]["sourceSha256"])

    def test_rejects_a_merged_parent_manifest_that_does_not_match_its_hash(self) -> None:
        prepared = json.loads(self.prepared.read_text())
        prepared["schemaVersion"] = 2
        prepared.pop("trainingSource")
        prepared["parents"] = {
            "positive": {
                "manifestSha256": "4" * 64,
                "trainSha256": "5" * 64,
                "validationSha256": "6" * 64,
            },
            "negative": {
                "manifestSha256": "7" * 64,
                "trainSha256": "8" * 64,
                "validationSha256": "9" * 64,
            },
        }
        self.prepared.write_text(json.dumps(prepared))
        training = json.loads(self.training.read_text())
        training["data"]["preparedManifestSha256"] = sha256(self.prepared)
        training["data"]["preparedSchemaVersion"] = 2
        self.training.write_text(json.dumps(training))
        parent = self.root / "parent.json"
        parent.write_text("{}")

        with self.assertRaisesRegex(ValueError, "positive parent manifest SHA-256 differs"):
            package_adapter(
                training_manifest=self.training,
                preparation_manifest=self.prepared,
                adapter_directory=self.adapter,
                base_artifact=self.base,
                dataset="edbuildingstuff/bfcl-ft-data",
                dataset_revision="b" * 40,
                source_file="train.javajs.messages.jsonl",
                formatter=self.formatter,
                notice=self.notice,
                output_directory=self.root / "runtime",
                positive_parent_manifest=parent,
                negative_parent_manifest=parent,
            )

    def test_rejects_a_training_manifest_bound_to_another_preparation_schema(self) -> None:
        prepared = json.loads(self.prepared.read_text())
        prepared["schemaVersion"] = 2
        prepared["trainingSource"].update(
            {
                "path": "train.javajs.messages.jsonl",
                "repository": "edbuildingstuff/bfcl-ft-data",
                "revision": "b" * 40,
            }
        )
        self.prepared.write_text(json.dumps(prepared))
        training = json.loads(self.training.read_text())
        training["data"]["preparedSchemaVersion"] = 1
        training["data"]["preparedManifestSha256"] = sha256(self.prepared)
        self.training.write_text(json.dumps(training))

        with self.assertRaisesRegex(ValueError, "preparation schemaVersion"):
            package_adapter(
                training_manifest=self.training,
                preparation_manifest=self.prepared,
                adapter_directory=self.adapter,
                base_artifact=self.base,
                dataset="edbuildingstuff/bfcl-ft-data",
                dataset_revision="b" * 40,
                source_file="train.javajs.messages.jsonl",
                formatter=self.formatter,
                notice=self.notice,
                output_directory=self.root / "runtime",
            )

    def test_rejects_impossible_selected_row_counts(self) -> None:
        training = json.loads(self.training.read_text())
        training["data"]["train"]["noCall"] = 4001
        self.training.write_text(json.dumps(training))

        with self.assertRaisesRegex(ValueError, "noCall must not exceed usable"):
            package_adapter(
                training_manifest=self.training,
                preparation_manifest=self.prepared,
                adapter_directory=self.adapter,
                base_artifact=self.base,
                dataset="edbuildingstuff/bfcl-ft-data",
                dataset_revision="b" * 40,
                source_file="train.javajs.messages.jsonl",
                formatter=self.formatter,
                notice=self.notice,
                output_directory=self.root / "runtime",
            )

    def test_rejects_a_secondary_training_source_without_a_repository(self) -> None:
        prepared = json.loads(self.prepared.read_text())
        prepared["schemaVersion"] = 2
        prepared["trainingSource"].update(
            {
                "path": "train.javajs.messages.jsonl",
                "repository": "edbuildingstuff/bfcl-ft-data",
                "revision": "b" * 40,
            }
        )
        prepared["irrelevanceSource"] = {
            "repository": None,
            "revision": "c" * 40,
            "path": "xlam-7.5k-irrelevancek.json",
            "sha256": "5" * 64,
        }
        self.prepared.write_text(json.dumps(prepared))
        training = json.loads(self.training.read_text())
        training["data"]["preparedManifestSha256"] = sha256(self.prepared)
        training["data"]["preparedSchemaVersion"] = 2
        self.training.write_text(json.dumps(training))

        with self.assertRaisesRegex(ValueError, "irrelevance source repository"):
            package_adapter(
                training_manifest=self.training,
                preparation_manifest=self.prepared,
                adapter_directory=self.adapter,
                base_artifact=self.base,
                dataset="edbuildingstuff/bfcl-ft-data",
                dataset_revision="b" * 40,
                source_file="train.javajs.messages.jsonl",
                formatter=self.formatter,
                notice=self.notice,
                output_directory=self.root / "runtime",
            )

    def test_rejects_a_primary_source_path_that_only_matches_by_filename(self) -> None:
        prepared = json.loads(self.prepared.read_text())
        prepared["schemaVersion"] = 2
        prepared["trainingSource"].update(
            {
                "repository": "edbuildingstuff/bfcl-ft-data",
                "revision": "b" * 40,
            }
        )
        self.prepared.write_text(json.dumps(prepared))
        training = json.loads(self.training.read_text())
        training["data"]["preparedManifestSha256"] = sha256(self.prepared)
        training["data"]["preparedSchemaVersion"] = 2
        self.training.write_text(json.dumps(training))

        with self.assertRaisesRegex(ValueError, "source file differs"):
            package_adapter(
                training_manifest=self.training,
                preparation_manifest=self.prepared,
                adapter_directory=self.adapter,
                base_artifact=self.base,
                dataset="edbuildingstuff/bfcl-ft-data",
                dataset_revision="b" * 40,
                source_file="train.javajs.messages.jsonl",
                formatter=self.formatter,
                notice=self.notice,
                output_directory=self.root / "runtime",
            )

    def test_rejects_tampering_instead_of_packaging_it(self) -> None:
        (self.adapter / "tokenizer.json").write_text("tokemizer")

        with self.assertRaisesRegex(ValueError, "hash differs"):
            package_adapter(
                training_manifest=self.training,
                preparation_manifest=self.prepared,
                adapter_directory=self.adapter,
                base_artifact=self.base,
                dataset="edbuildingstuff/bfcl-ft-data",
                dataset_revision="b" * 40,
                source_file="train.javajs.messages.jsonl",
                formatter=self.formatter,
                notice=self.notice,
                output_directory=self.root / "runtime",
            )


if __name__ == "__main__":
    unittest.main()
