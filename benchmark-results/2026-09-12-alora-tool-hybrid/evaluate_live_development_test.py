import json
import tempfile
import unittest
from pathlib import Path

from evaluate_live_development import (
    assert_disjoint,
    assert_disjoint_from_prepared,
    load_live_slice,
)


class EvaluateLiveDevelopmentTest(unittest.TestCase):
    def test_loads_tool_and_irrelevance_cases_without_changing_ground_truth(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "possible_answer").mkdir()
            row = {
                "id": "simple_1",
                "question": [[{"role": "user", "content": "weather"}]],
                "function": [
                    {
                        "name": "weather",
                        "parameters": {
                            "type": "dict",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    }
                ],
            }
            (root / "BFCL_v3_live_simple.json").write_text(json.dumps(row) + "\n")
            (root / "possible_answer" / "BFCL_v3_live_simple.json").write_text(
                json.dumps(
                    {
                        "id": "simple_1",
                        "ground_truth": [{"weather": {"city": ["Jal"]}}],
                    }
                )
                + "\n"
            )
            irrelevant = {**row, "id": "irrelevance_1"}
            (root / "BFCL_v3_live_irrelevance.json").write_text(
                json.dumps(irrelevant) + "\n"
            )

            simple = load_live_slice(root, "simple", 1, 7)
            no_call = load_live_slice(root, "irrelevance", 1, 7)

            self.assertEqual(simple[0]["expected"], [{"weather": {"city": ["Jal"]}}])
            self.assertEqual(no_call[0]["expected"], [])
            self.assertEqual(simple[0]["tools"][0]["function"]["name"], "weather")

    def test_rejects_any_development_query_reused_by_qualification(self):
        development = [
            {"id": "dev", "messages": [{"role": "user", "content": "same query"}]}
        ]
        qualification = [
            {"id": "held", "messages": [{"role": "user", "content": "same query"}]}
        ]

        with self.assertRaisesRegex(ValueError, "overlaps qualification"):
            assert_disjoint(development, qualification)

    def test_rejects_development_queries_present_in_any_prepared_training_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            prepared = Path(temporary)
            (prepared / "train.jsonl").write_text(
                json.dumps({"user": "same query", "calls": []}) + "\n"
            )
            (prepared / "validation.jsonl").write_text(
                json.dumps({"user": "a different query", "calls": []}) + "\n"
            )
            development = [
                {"id": "dev", "messages": [{"role": "user", "content": "same query"}]}
            ]

            with self.assertRaisesRegex(ValueError, "overlaps prepared training data"):
                assert_disjoint_from_prepared(development, [prepared])


if __name__ == "__main__":
    unittest.main()
