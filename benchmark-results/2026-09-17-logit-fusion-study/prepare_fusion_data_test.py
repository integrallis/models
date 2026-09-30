import hashlib
import json
import unittest

import prepare_fusion_data as prep

GSM_SPEC = prep.SOURCES["gsm8k-test"]


class PrepareFusionDataTest(unittest.TestCase):
    def test_gsm8k_final_answer_strips_commas_and_takes_last_marker(self):
        self.assertEqual(prep.gsm8k_final_answer("a #### 1 then\n#### 1,234 "), "1234")
        with self.assertRaises(ValueError):
            prep.gsm8k_final_answer("no marker")

    def test_gsm8k_records_ids_and_source(self):
        rows = [{"question": "q0", "answer": "x\n#### 5"}, {"question": "q1", "answer": "#### -3"}]
        records = prep.gsm8k_records(rows, GSM_SPEC, "test")
        self.assertEqual([r["id"] for r in records], ["gsm8k-test-0000", "gsm8k-test-0001"])
        self.assertEqual(records[1]["answer"], "-3")
        self.assertIsNone(records[0]["choices"])
        self.assertEqual(records[1]["source"]["row"], 1)
        self.assertEqual(records[0]["source"]["revision"], GSM_SPEC["revision"])

    def test_dev_indices_deterministic_sorted_and_matches_contract(self):
        first = prep.dev_indices(7473)
        self.assertEqual(first, prep.dev_indices(7473))
        self.assertEqual(len(first), 500)
        self.assertEqual(first, sorted(first))
        import random

        self.assertEqual(first, sorted(random.Random(20260917).sample(range(7473), 500)))

    def test_dev_records_use_train_row_ids(self):
        rows = [{"question": f"q{i}", "answer": f"#### {i}"} for i in range(10)]
        records = prep.gsm8k_records(rows, GSM_SPEC, "train", [2, 7])
        self.assertEqual([r["id"] for r in records], ["gsm8k-train-0002", "gsm8k-train-0007"])
        self.assertEqual(records[1]["answer"], "7")

    def test_arc_records(self):
        rows = [
            {"id": "Mercury_1", "question": "q", "answerKey": "2",
             "choices": {"label": ["1", "2", "3", "4"], "text": ["a", "b", "c", "d"]}}
        ]
        record = prep.arc_records(rows, prep.SOURCES["arc-challenge-test"])[0]
        self.assertEqual(record["id"], "arc-challenge-test-Mercury_1")
        self.assertEqual(record["answer"], "2")
        self.assertEqual(record["choices"][1], {"label": "2", "text": "b"})
        rows[0]["answerKey"] = "E"
        with self.assertRaises(ValueError):
            prep.arc_records(rows, prep.SOURCES["arc-challenge-test"])

    def test_math500_records(self):
        rows = [{"problem": "p", "solution": "s \\boxed{\\frac{1}{2}}", "answer": "\\frac{1}{2}",
                 "subject": "Algebra", "level": 3, "unique_id": "test/algebra/1.json"}]
        record = prep.math500_records(rows, prep.SOURCES["math500-test"])[0]
        self.assertEqual(record["id"], "math500-test-000")
        self.assertEqual(record["answer"], "\\frac{1}{2}")
        self.assertEqual(record["uniqueId"], "test/algebra/1.json")
        self.assertEqual(record["level"], 3)

    def test_ids_sha256_no_trailing_newline(self):
        expected = hashlib.sha256(b"a\nb").hexdigest()
        self.assertEqual(prep.ids_sha256(["a", "b"]), expected)

    def test_jsonl_and_manifest_entry(self):
        records = [{"id": "x"}, {"id": "y"}]
        data = prep.to_jsonl(records)
        self.assertEqual([json.loads(l) for l in data.decode().splitlines()], records)
        entry = prep.manifest_entry(GSM_SPEC, "abc", data, records)
        self.assertEqual(entry["items"], 2)
        self.assertEqual(entry["outputSha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual(entry["idsSha256"], prep.ids_sha256(["x", "y"]))
        self.assertEqual(entry["repo"], "openai/gsm8k")

    def test_pins_are_exact_hex(self):
        for spec in prep.SOURCES.values():
            self.assertRegex(spec["revision"], r"^[0-9a-f]{40}$")
            self.assertRegex(spec["sourceSha256"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
