import unittest

from evaluate_base_model import candidate, summarize_base


class EvaluateBaseModelTest(unittest.TestCase):
    def test_resolves_only_a_pinned_candidate(self):
        self.assertEqual(
            candidate("qwen3-1.7b"),
            ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
        )
        with self.assertRaisesRegex(ValueError, "unsupported base candidate"):
            candidate("latest")

    def test_summarizes_tool_and_irrelevance_cases_without_an_adapter_arm(self):
        records = [
            {"kind": "simple", "syntaxValid": True, "schemaValid": True, "exact": True},
            {"kind": "multiple", "syntaxValid": True, "schemaValid": True, "exact": False},
            {
                "kind": "irrelevance",
                "syntaxValid": True,
                "schemaValid": True,
                "exact": False,
                "parsedCalls": [],
            },
        ]

        self.assertEqual(
            summarize_base(records, 12.5),
            {
                "cases": 3,
                "syntaxRate": 1.0,
                "schemaRate": 1.0,
                "toolExactRate": 0.5,
                "irrelevanceFalseToolRate": 0.0,
                "elapsedSeconds": 12.5,
            },
        )


if __name__ == "__main__":
    unittest.main()
