import json
import tempfile
import unittest
from pathlib import Path

from freeze_live_decision_window import (
    assert_disjoint_from_training_sources,
    assert_unique_cases,
    decision_records,
)


class FreezeLiveDecisionWindowTest(unittest.TestCase):
    def test_converts_live_cases_without_rendering_or_changing_untrusted_input(self):
        cases = [
            {
                "id": "simple_1",
                "kind": "simple",
                "messages": [{"role": "user", "content": "Weather in Jal?"}],
                "tools": [{"type": "function", "function": {"name": "weather"}}],
                "expected": [{"weather": {"zipcode": ["88252"]}}],
            }
        ]

        records = decision_records(cases)

        self.assertEqual(records[0]["phase"], "screen")
        self.assertEqual(records[0]["messages"], cases[0]["messages"])
        self.assertEqual(records[0]["tools"], cases[0]["tools"])
        self.assertEqual(records[0]["expected"], cases[0]["expected"])
        self.assertNotIn("prompt", records[0])

    def test_allows_the_same_query_with_different_tool_contexts(self):
        cases = [
            {
                "id": "simple_1",
                "messages": [{"role": "user", "content": "Same query"}],
                "tools": [{"function": {"name": "applicable"}}],
            },
            {
                "id": "multiple_1",
                "messages": [{"role": "user", "content": " same   QUERY "}],
                "tools": [{"function": {"name": "unrelated"}}],
            },
        ]

        assert_unique_cases(cases)

    def test_rejects_an_exact_duplicate_messages_and_tools_case(self):
        cases = [
            {
                "id": "simple_1",
                "messages": [{"role": "user", "content": "Same query"}],
                "tools": [{"function": {"name": "weather"}}],
            },
            {
                "id": "simple_2",
                "messages": [{"role": "user", "content": "Same query"}],
                "tools": [{"function": {"name": "weather"}}],
            },
        ]

        with self.assertRaisesRegex(ValueError, "duplicate messages and tools"):
            assert_unique_cases(cases)

    def test_rejects_a_query_present_in_either_complete_training_source(self):
        cases = [
            {
                "id": "simple_1",
                "messages": [{"role": "user", "content": "Same query"}],
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training = root / "training.jsonl"
            irrelevance = root / "irrelevance.json"
            training.write_text(
                json.dumps(
                    {
                        "messages": [
                            {"role": "system", "content": "tools"},
                            {"role": "user", "content": "same  QUERY"},
                        ]
                    }
                )
                + "\n"
            )
            irrelevance.write_text(json.dumps([{"query": "different"}]))

            with self.assertRaisesRegex(ValueError, "training sources"):
                assert_disjoint_from_training_sources(cases, training, irrelevance)


if __name__ == "__main__":
    unittest.main()
