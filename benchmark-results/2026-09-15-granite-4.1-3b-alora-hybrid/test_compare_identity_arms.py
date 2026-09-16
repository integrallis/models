import copy
import unittest

from compare_identity_arms import compare

BASE = {
    "suite": "s", "arm": "specialist", "windowSha256": "w", "windowFileSha256": "f",
    "modelsRevision": "r", "backend": "pure-java",
    "cases": [
        {"id": "a", "output": "\"answerable\"", "prediction": "\"answerable\"", "structured": True,
         "physicallyShared": True, "sharedPrefixTokens": 900, "millis": 10},
        {"id": "b", "output": "\"unanswerable\"", "prediction": "\"unanswerable\"", "structured": True,
         "physicallyShared": True, "sharedPrefixTokens": 700, "millis": 12},
    ],
}


class CompareIdentityArmsTest(unittest.TestCase):
    def test_identical_reports_with_different_timing_pass(self):
        other = copy.deepcopy(BASE)
        other["backend"] = "rust-ffm"
        other["cases"][0]["millis"] = 1
        self.assertEqual(compare(BASE, other), [])

    def test_output_drift_and_order_drift_are_reported(self):
        drift = copy.deepcopy(BASE)
        drift["cases"][1]["output"] = "\"answerable\""
        self.assertEqual(len(compare(BASE, drift)), 1)
        self.assertIn("case b output", compare(BASE, drift)[0])
        reordered = copy.deepcopy(BASE)
        reordered["cases"].reverse()
        self.assertIn("case ids or order differ", compare(BASE, reordered))

    def test_window_identity_is_part_of_the_check(self):
        other = copy.deepcopy(BASE)
        other["windowSha256"] = "x"
        self.assertTrue(any(p.startswith("windowSha256") for p in compare(BASE, other)))


if __name__ == "__main__":
    unittest.main()
