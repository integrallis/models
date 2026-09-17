"""First-party (Integrallis-trained) component and composition reports.

The upstream fixtures come from assemble_component_report_test; these tests add a real git
checkout holding the training and prepared-data manifests so the pinned URIs are byte-checked
against ``git show <revision>:<path>`` exactly as the assembler does in a real checkout.
"""

import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from assemble_component_report import assemble, main, pinned_training_file
from assemble_component_report_test import AssembleComponentReportTest, window_report
from assemble_composition_report import assemble as assemble_composition
from assemble_composition_report_test import AssembleCompositionReportTest

WEIGHTS_SHA = hashlib.sha256(b"weights").hexdigest()
CONFIG_SHA = "5" * 64
TRAINING_PATH = "benchmark-results/answerability/pilot2/training-manifest.json"
PREPARED_PATH = "benchmark-results/answerability/prepared-manifest.json"
LICENSES = [("SQuAD 2.0", "CC-BY-SA-4.0"), ("QuAC", "CC-BY-SA-4.0")]


def git(checkout, *args):
    return subprocess.run(
        ["git", "-C", str(checkout), "-c", "user.email=test@example.invalid", "-c", "user.name=test",
         "-c", "commit.gpgsign=false", *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


class FirstPartyFixture:
    """Mixin: a training checkout with committed manifests and a first-party packaged adapter."""

    def training_checkout(self):
        checkout = self.root / "training-checkout"
        checkout.mkdir()
        git(checkout, "init", "-q")
        manifest = {
            "adapterFiles": {
                "adapter_config.json": {"bytes": 1217, "sha256": CONFIG_SHA},
                "adapter_model.safetensors": {"bytes": 7, "sha256": WEIGHTS_SHA},
            }
        }
        for path, value in ((TRAINING_PATH, manifest), (PREPARED_PATH, {"salt": "fixture"})):
            target = checkout / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(value, indent=2) + "\n")
        git(checkout, "add", ".")
        git(checkout, "commit", "-q", "-m", "training evidence")
        return checkout, git(checkout, "rev-parse", "HEAD")

    def first_party_inputs(self, **overrides):
        checkout, revision = self.training_checkout()
        metadata_path = self.adapter / "models-activated-lora.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["upstream"] = {
            "publisher": "Integrallis",
            "repository": "integrallis/models",
            "revision": revision,
            "adapterConfigSha256": CONFIG_SHA,
            "modelCardSha256": "6" * 64,
            "license": "Apache-2.0",
        }
        metadata_path.write_text(json.dumps(metadata))
        arguments = self.inputs()
        arguments.update(
            specialist_kind="first-party-rag-specialist",
            publisher="Integrallis",
            training_repository="integrallis/models",
            training_revision=revision,
            training_manifest=checkout / TRAINING_PATH,
            prepared_data_manifest=checkout / PREPARED_PATH,
            training_data_licenses=LICENSES,
        )
        arguments.update(overrides)
        return arguments, checkout, revision


class AssembleFirstPartyComponentReportTest(FirstPartyFixture, AssembleComponentReportTest):
    def test_assembles_a_first_party_report_in_the_modeljars_shape(self):
        arguments, checkout, revision = self.first_party_inputs()

        report = assemble(**arguments)

        self.assertEqual("first-party-rag-specialist", report["specialistKind"])
        provenance = report["evaluation"]["gates"]["provenance"]
        training_bytes = (checkout / TRAINING_PATH).read_bytes()
        prepared_bytes = (checkout / PREPARED_PATH).read_bytes()
        self.assertEqual(
            {
                "pass": True,
                "upstream": False,
                "publisher": "Integrallis",
                "trainingRepository": "integrallis/models",
                "trainingRevision": revision,
                "trainingManifest": {
                    "uri": f"https://raw.githubusercontent.com/integrallis/models/{revision}/{TRAINING_PATH}",
                    "sha256": hashlib.sha256(training_bytes).hexdigest(),
                    "sizeBytes": len(training_bytes),
                },
                "preparedDataManifest": {
                    "uri": f"https://raw.githubusercontent.com/integrallis/models/{revision}/{PREPARED_PATH}",
                    "sha256": hashlib.sha256(prepared_bytes).hexdigest(),
                    "sizeBytes": len(prepared_bytes),
                },
                "adapterSha256": WEIGHTS_SHA,
                "adapterConfigSha256": CONFIG_SHA,
                "modelCardSha256": "6" * 64,
                "license": "Apache-2.0",
                "trainingDataLicenses": [
                    {"dataset": "SQuAD 2.0", "license": "CC-BY-SA-4.0"},
                    {"dataset": "QuAC", "license": "CC-BY-SA-4.0"},
                ],
                "tokenizerFiles": [{"name": "tokenizer.json", "sha256": "8" * 64}],
                "invocationTokens": [100264, 78191, 100265],
            },
            provenance,
        )
        self.assertNotIn("upstreamRepository", provenance)

    def test_upstream_remains_the_default_kind(self):
        report = assemble(**self.inputs())
        self.assertEqual("upstream-rag-specialist", report["specialistKind"])
        self.assertTrue(report["evaluation"]["gates"]["provenance"]["upstream"])

    def test_refuses_unknown_kinds_and_first_party_options_on_an_upstream_report(self):
        with self.assertRaisesRegex(ValueError, "specialist_kind"):
            assemble(**self.inputs(specialist_kind="distilled-router"))
        with self.assertRaisesRegex(ValueError, "first-party"):
            assemble(**self.inputs(publisher="Integrallis"))

    def test_refuses_a_suite_that_does_not_strictly_beat_the_base(self):
        arguments, _, _ = self.first_party_inputs()
        arguments["window_reports"]["mtrag-human-rag|base"] = self.write("tie.json", window_report("rust-ffm", 0.88))
        with self.assertRaisesRegex(SystemExit, "strictly beat"):
            assemble(**arguments)
        # The same tie is admissible for an upstream specialist ("no worse than base").
        upstream = self.inputs()
        upstream["window_reports"]["mtrag-human-rag|base"] = self.write("tie-up.json", window_report("rust-ffm", 0.88))
        assemble(**upstream)

    def test_refuses_local_manifest_bytes_that_differ_from_the_training_commit(self):
        arguments, checkout, _ = self.first_party_inputs()
        (checkout / PREPARED_PATH).write_text('{"salt": "edited after the commit"}\n')
        with self.assertRaisesRegex(ValueError, "differ from git show"):
            assemble(**arguments)

    def test_refuses_a_manifest_that_is_not_at_the_training_commit(self):
        arguments, checkout, revision = self.first_party_inputs()
        later = checkout / "benchmark-results/answerability/later-manifest.json"
        later.write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "does not exist at"):
            assemble(**{**arguments, "prepared_data_manifest": later})
        with self.assertRaisesRegex(ValueError, "does not exist at"):
            pinned_training_file(
                checkout / PREPARED_PATH, repository="integrallis/models", revision=revision,
                repo_path="benchmark-results/elsewhere.json", label="prepared-data manifest",
            )

    def test_outside_a_checkout_requires_an_explicit_repository_path(self):
        arguments, checkout, revision = self.first_party_inputs()
        outside = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, outside)
        copy = outside / "training-manifest.json"
        shutil.copyfile(checkout / TRAINING_PATH, copy)
        with self.assertRaisesRegex(ValueError, "training-manifest-repo-path"):
            assemble(**{**arguments, "training_manifest": copy})
        report = assemble(**{**arguments, "training_manifest": copy, "training_manifest_repo_path": TRAINING_PATH})
        self.assertEqual(
            f"https://raw.githubusercontent.com/integrallis/models/{revision}/{TRAINING_PATH}",
            report["evaluation"]["gates"]["provenance"]["trainingManifest"]["uri"],
        )
        with self.assertRaisesRegex(ValueError, "repository-relative"):
            assemble(**{**arguments, "training_manifest": copy, "training_manifest_repo_path": "../escape.json"})

    def test_refuses_an_adapter_the_training_manifest_did_not_record(self):
        arguments, _, _ = self.first_party_inputs()
        (self.adapter / "adapter_model.safetensors").write_bytes(b"other weights")
        metadata_path = self.adapter / "models-activated-lora.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["adapter"]["sha256"] = hashlib.sha256(b"other weights").hexdigest()
        metadata_path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "training manifest"):
            assemble(**arguments)

    def test_refuses_packaged_metadata_that_names_another_training_source(self):
        arguments, _, _ = self.first_party_inputs()
        for field, value in (("revision", "1" * 40), ("repository", "someone/else"), ("publisher", "IBM")):
            with self.subTest(field=field):
                metadata_path = self.adapter / "models-activated-lora.json"
                metadata = json.loads(metadata_path.read_text())
                original = metadata["upstream"][field]
                metadata["upstream"][field] = value
                metadata_path.write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError, field):
                    assemble(**arguments)
                metadata["upstream"][field] = original
                metadata_path.write_text(json.dumps(metadata))

    def test_refuses_incomplete_first_party_identity(self):
        arguments, _, _ = self.first_party_inputs()
        for override, message in (
            ({"publisher": " "}, "publisher"),
            ({"training_repository": "integrallis"}, "training_repository"),
            ({"training_revision": "abc"}, "training_revision"),
            ({"training_data_licenses": []}, "training_data_licenses"),
            ({"training_data_licenses": [("QuAC", "")]}, "training_data_licenses"),
            ({"training_manifest": None}, "training_manifest"),
        ):
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, message):
                assemble(**{**arguments, **override})

    def test_command_line_assembles_a_first_party_report(self):
        arguments, checkout, revision = self.first_party_inputs()
        output = self.root / "first-party.json"
        argv = [
            "--specialist-kind", "first-party-rag-specialist",
            "--publisher", "Integrallis",
            "--training-repository", "integrallis/models",
            "--training-revision", revision,
            "--training-manifest", str(checkout / TRAINING_PATH),
            "--prepared-data-manifest", str(checkout / PREPARED_PATH),
            "--training-data-license", "SQuAD 2.0=CC-BY-SA-4.0",
            "--training-data-license", "MS MARCO v2.1=Microsoft MS MARCO terms (use approved by publisher)",
            "--adapter-directory", str(arguments["adapter_directory"]),
            "--base-model-id", arguments["base_model_id"],
            "--base-revision", arguments["base_revision"],
            "--base-artifact", str(arguments["base_artifact"]),
            "--junit-xml", str(arguments["junit_xml"]),
            "--long-context-report", str(arguments["long_context_report"]),
            "--crossover-report", str(arguments["crossover_report"]),
            "--models-revision", arguments["models_revision"],
            "--model-id", "integrallis_answerability",
            "--output", str(output),
        ]
        for key, path in arguments["window_reports"].items():
            argv += ["--window", f"{key}={path}"]
        for key, (java, kernel) in arguments["identity_reports"].items():
            argv += ["--identity", f"{key}={java},{kernel}"]
        main(argv)
        report = json.loads(output.read_text())
        self.assertEqual("first-party-rag-specialist", report["specialistKind"])
        self.assertEqual(
            [
                {"dataset": "SQuAD 2.0", "license": "CC-BY-SA-4.0"},
                {"dataset": "MS MARCO v2.1", "license": "Microsoft MS MARCO terms (use approved by publisher)"},
            ],
            report["evaluation"]["gates"]["provenance"]["trainingDataLicenses"],
        )
        with self.assertRaises(SystemExit):
            main(argv[:-2] + ["--training-data-license", "no-separator", "--output", str(output)])


class AssembleFirstPartyCompositionReportTest(FirstPartyFixture, AssembleCompositionReportTest):
    def test_a_first_party_composition_carries_the_first_party_kind_and_provenance(self):
        arguments, _, revision = self.first_party_inputs()
        composition = self.composition_inputs()
        first_party_options = ("specialist_kind", "publisher", "training_repository", "training_revision",
                               "training_manifest", "prepared_data_manifest", "training_data_licenses")
        composition.update({name: arguments[name] for name in first_party_options})
        report = assemble_composition(**composition)
        self.assertEqual("first-party-rag-specialist", report["specialistKind"])
        provenance = report["evaluation"]["gates"]["provenance"]
        self.assertFalse(provenance["upstream"])
        self.assertEqual(revision, provenance["trainingRevision"])

    def test_upstream_composition_kind_is_unchanged(self):
        self.assertEqual("upstream-rag-specialist", assemble_composition(**self.composition_inputs())["specialistKind"])


def load_tests(loader, standard_tests, pattern):
    """Runs only the tests these classes add; the inherited upstream tests run in their own modules."""
    suite = unittest.TestSuite()
    for case in (AssembleFirstPartyComponentReportTest, AssembleFirstPartyCompositionReportTest):
        for name in sorted(name for name in vars(case) if name.startswith("test_")):
            suite.addTest(case(name))
    return suite


del AssembleComponentReportTest, AssembleCompositionReportTest

if __name__ == "__main__":
    unittest.main()
