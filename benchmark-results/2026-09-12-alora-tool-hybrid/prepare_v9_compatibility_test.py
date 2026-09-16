import unittest

from prepare_v9_compatibility import normalize_tools_v9


class PrepareV9CompatibilityTest(unittest.TestCase):
    def test_retains_the_duplicate_callable_behavior_used_by_the_frozen_v8_corpus(self):
        tools = [
            {
                "name": "lookup",
                "description": "first",
                "parameters": {"value": {"type": "string"}},
            },
            {
                "name": "lookup",
                "description": "second",
                "parameters": {"count": {"type": "integer"}},
            },
        ]

        normalized = normalize_tools_v9(tools)

        self.assertEqual(len(normalized), 2)
        self.assertEqual(normalized[0]["function"]["name"], "lookup")
        self.assertEqual(normalized[1]["function"]["name"], "lookup")


if __name__ == "__main__":
    unittest.main()
