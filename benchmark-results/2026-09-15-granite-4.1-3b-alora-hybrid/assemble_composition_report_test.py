import json
import unittest

from assemble_component_report_test import AssembleComponentReportTest
from assemble_composition_report import assemble


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

    def test_refuses_a_failing_crossover_through_the_component_gates(self):
        with self.assertRaises(SystemExit):
            assemble(**self.composition_inputs(crossover_report=self.crossover_with_medians(qualified=False, name="gate6-failed.json")))


if __name__ == "__main__":
    unittest.main()
