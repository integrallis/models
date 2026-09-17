import json
import unittest

from assemble_component_report_test import (
    AssembleComponentReportTest,
    clean_host_run,
    confirmed_labels,
    models_artifact,
)
from assemble_composition_report import assemble
from assemble_composition_report import main as composition_main


class AssembleCompositionReportTest(AssembleComponentReportTest):
    """Reuses the component fixtures; adds the tier medians and the artifact resolution record."""

    def crossover_with_medians(self, qualified=True, improvement_4k=0.31, name="gate6-medians.json"):
        path = self.crossover(qualified=qualified, name=name)
        report = json.loads(path.read_text())
        medians = {256: (13_500.0, 26_600.0, 47_197_440, 125_859_840),
                   1024: (25_000.0, 60_000.0, 70_000_000, 200_000_000),
                   4096: (60_000.0, 60_000.0 / (1 - improvement_4k), 120_000_000, 400_000_000)}
        for summary in report["summaries"]:
            shared, recomputed, shared_bytes, recomputed_bytes = medians[summary["prefixTokens"]]
            summary.update(
                sharedMedianHandoffMillis=shared,
                recomputedMedianHandoffMillis=recomputed,
                improvement=1 - shared / recomputed,
                sharedMedianUniqueStateBytes=shared_bytes,
                recomputedMedianUniqueStateBytes=recomputed_bytes,
            )
        report["verdict"]["fourKImprovement"] = next(
            s["improvement"] for s in report["summaries"] if s["prefixTokens"] == 4096
        )
        path.write_text(json.dumps(report))
        return path

    def artifacts(self, resolved=True, name="artifacts.json", members=("base", "specialist")):
        return self.write(
            name,
            [
                {
                    "modelId": member,
                    "coordinate": f"org.modeljars.test:{member}:1.0.0",
                    "sha256": "b" * 64,
                    "resolvedFromCentral": resolved,
                    "runViaPublicApi": resolved,
                }
                for member in members
            ],
        )

    def composition_inputs(self, **overrides):
        arguments = self.inputs()
        arguments.update(
            composition_id="granite_4_1_3b_answerability_hybrid",
            base_member_id="base",
            specialist_member_id="specialist",
            published_artifacts_path=self.artifacts(),
            crossover_report=self.crossover_with_medians(),
        )
        arguments.update(overrides)
        return arguments

    def test_assembles_a_composition_report_on_top_of_the_component_gates(self):
        report = assemble(**self.composition_inputs())
        self.assertEqual(report["schemaVersion"], 2)
        self.assertEqual(report["specialistKind"], "upstream-rag-specialist")
        evaluation = report["evaluation"]
        self.assertTrue(evaluation["qualified"] and evaluation["correctnessPassed"])
        self.assertAlmostEqual(evaluation["improvement"], 0.31)
        self.assertAlmostEqual(evaluation["controlMedianMillis"], 60_000.0 / 0.69)
        self.assertEqual(evaluation["hybridMedianMillis"], 60_000.0)
        gates = evaluation["gates"]
        self.assertEqual(gates["performanceCrossover"]["improvement"], evaluation["improvement"])
        self.assertEqual(gates["longContextRetrieval"]["contextTokens"], 4096)
        self.assertEqual(gates["longContextRetrieval"]["tokenExactCases"], 8)
        self.assertLess(
            gates["memoryAccounting"]["sharedUniqueStateBytes"],
            gates["memoryAccounting"]["recomputedUniqueStateBytes"],
        )
        self.assertEqual([a["modelId"] for a in evaluation["publishedArtifacts"]], ["base", "specialist"])
        self.assertEqual(report["implementation"]["stateHandoffMechanism"], "exact-kv-block-sharing")

    def test_refuses_artifacts_that_were_not_resolved_from_central(self):
        with self.assertRaises(SystemExit):
            assemble(**self.composition_inputs(published_artifacts_path=self.artifacts(resolved=False, name="unresolved.json")))

    def test_refuses_an_artifact_record_that_does_not_cover_every_member(self):
        with self.assertRaises(ValueError):
            assemble(**self.composition_inputs(published_artifacts_path=self.artifacts(members=("base",), name="a2.json")))

    def test_confirmed_labels_and_release_evidence_pass_through_to_the_composition(self):
        inputs, dataset, confirmed = self.squad_window_with_disputed_labels()
        composition = self.composition_inputs(window_reports=inputs["window_reports"])
        with self.assertRaises(SystemExit):
            assemble(**composition)
        composition.update(
            labels={"squad-v2-dev": self.write("c-labels.json", confirmed_labels("squad-v2-dev", confirmed, dataset))},
            models_artifact=self.write("c-ma.json", models_artifact()),
            clean_host_run=self.write("c-ch.json", clean_host_run()),
        )
        report = assemble(**composition)
        gates = report["evaluation"]["gates"]
        suite = next(s for s in gates["taskCorrectness"]["suites"] if s["name"] == "squad-v2-dev")
        self.assertEqual("confirmed", suite["labelSource"])
        self.assertAlmostEqual(0.75, suite["originalBalancedAccuracy"])
        self.assertEqual(models_artifact(), gates["modelsArtifact"])
        self.assertEqual(clean_host_run(), gates["cleanHostRun"])
        self.assertEqual("integrallis_granite_4_1_3b_answerability_alora", report["evaluation"]["artifact"]["modelId"])

    def test_composition_command_line_defaults_the_model_id_to_the_specialist_member(self):
        inputs, dataset, confirmed = self.squad_window_with_disputed_labels()
        output = self.root / "composition.json"
        argv = [
            "--composition-id", "hybrid",
            "--base-member-id", "base",
            "--specialist-member-id", "specialist",
            "--published-artifacts", str(self.artifacts(name="cli-artifacts.json")),
            "--adapter-directory", str(inputs["adapter_directory"]),
            "--base-model-id", inputs["base_model_id"],
            "--base-revision", inputs["base_revision"],
            "--base-artifact", str(inputs["base_artifact"]),
            "--junit-xml", str(inputs["junit_xml"]),
            "--long-context-report", str(inputs["long_context_report"]),
            "--crossover-report", str(self.crossover_with_medians(name="cli-gate6.json")),
            "--models-revision", inputs["models_revision"],
            "--labels", f"squad-v2-dev={self.write('cli-c-labels.json', confirmed_labels('squad-v2-dev', confirmed, dataset))}",
            "--output", str(output),
        ]
        for key, path in inputs["window_reports"].items():
            argv += ["--window", f"{key}={path}"]
        for key, (java, kernel) in inputs["identity_reports"].items():
            argv += ["--identity", f"{key}={java},{kernel}"]
        composition_main(argv)
        report = json.loads(output.read_text())
        self.assertEqual("specialist", report["evaluation"]["artifact"]["modelId"])
        self.assertNotIn("modelsArtifact", report["evaluation"]["gates"])
        self.assertIn("confirmed", [s["labelSource"] for s in report["evaluation"]["gates"]["taskCorrectness"]["suites"]])

    def test_refuses_a_failing_crossover_through_the_component_gates(self):
        with self.assertRaises(SystemExit):
            assemble(**self.composition_inputs(crossover_report=self.crossover_with_medians(qualified=False, name="gate6-failed.json")))


if __name__ == "__main__":
    unittest.main()
