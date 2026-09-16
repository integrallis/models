import json
import tempfile
import unittest
from pathlib import Path

from preflight_applicability import prepared_query_overlap


class PreflightApplicabilityTest(unittest.TestCase):
    def test_detects_normalized_query_overlap_with_bfcl(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / "prepared.jsonl"
            bfcl = root / "bfcl.jsonl"
            prepared.write_text(
                json.dumps({"user": "  What IS the weather?  "}) + "\n"
                + json.dumps({"user": "Play music"})
                + "\n"
            )
            bfcl.write_text(
                json.dumps(
                    {
                        "question": [
                            [{"role": "user", "content": "what is the weather?"}]
                        ]
                    }
                )
                + "\n"
            )

            overlap = prepared_query_overlap([prepared], [bfcl])

            self.assertEqual(overlap["preparedQueries"], 2)
            self.assertEqual(overlap["evaluationQueries"], 1)
            self.assertEqual(overlap["overlap"], 1)


if __name__ == "__main__":
    unittest.main()
