import unittest

import confirm_disputed_labels as c


class QuoteVerification(unittest.TestCase):
    def test_verbatim_modulo_whitespace_case_and_edge_punctuation(self):
        self.assertTrue(c.quote_verifies('"A simple  majority."', "Rule: a simple majority. More text"))

    def test_paraphrase_rejected(self):
        self.assertFalse(c.quote_verifies("simple plurality", "a simple majority"))

    def test_empty_and_tiny_rejected(self):
        self.assertFalse(c.quote_verifies(None, "abc"))
        self.assertFalse(c.quote_verifies("ab", "ab cd"))

    def test_named_passage_wrong_falls_back_to_passage_that_contains_quote(self):
        items = {"s:1": {"id": "s:1", "question": "q", "passages": [{"doc_id": 1, "text": "nothing"}, {"doc_id": 2, "text": "the answer is 42"}]}}
        v = c.verify_quotes(items, {"s:1": {"id": "s:1", "quote": "answer is 42", "doc_id": 1}})
        self.assertTrue(v["s:1"]["verified"])
        self.assertEqual(v["s:1"]["doc_id"], 2)

    def test_hallucinated_quote_not_verified(self):
        items = {"s:1": {"id": "s:1", "question": "q", "passages": [{"doc_id": 1, "text": "nothing here"}]}}
        self.assertFalse(c.verify_quotes(items, {"s:1": {"id": "s:1", "quote": "answer is 42", "doc_id": 1}})["s:1"]["verified"])

    def test_extraction_coverage_enforced(self):
        with self.assertRaises(ValueError):
            c.verify_quotes({"s:1": {"passages": []}}, {})


class Decision(unittest.TestCase):
    def test_answerable_needs_verified_quote_and_both_yes(self):
        self.assertEqual(c.decide_label({"verified": True}, ["yes", "yes"]), "answerable")
        self.assertEqual(c.decide_label({"verified": True}, ["yes", "no"]), "unanswerable")
        self.assertEqual(c.decide_label({"verified": False}, [None, None]), "unanswerable")

    def key(self):
        k = {}
        for i in range(20):
            k[f"u{i}"] = {"group": "control-unanswerable", "suite": "squad-v2-dev", "caseId": f"u{i}", "datasetLabel": "unanswerable"}
            k[f"a{i}"] = {"group": "control-answerable", "suite": "squad-v2-dev", "caseId": f"a{i}", "datasetLabel": "answerable"}
        return k

    def run_decide(self, false_ans, true_ans):
        key = self.key()
        ver = {u: {"verified": False} for u in key}
        for i in range(false_ans):
            ver[f"u{i}"] = {"verified": True}
        for i in range(true_ans):
            ver[f"a{i}"] = {"verified": True}
        yes = {u: "yes" for u, v in ver.items() if v["verified"]}
        return c.decide(key, ver, {"A": yes, "B": dict(yes)})

    def test_admissible_at_edges(self):
        self.assertTrue(self.run_decide(2, 18)["protocolAdmissible"])

    def test_too_many_false_answerables_inadmissible(self):
        self.assertFalse(self.run_decide(3, 20)["protocolAdmissible"])

    def test_too_few_true_answerables_inadmissible(self):
        self.assertFalse(self.run_decide(0, 17)["protocolAdmissible"])

    def test_verifier_coverage_enforced(self):
        key = self.key()
        ver = {u: {"verified": u == "a0"} for u in key}
        with self.assertRaises(ValueError):
            c.decide(key, ver, {"A": {}, "B": {"a0": "yes"}})


class ConfirmedSuite(unittest.TestCase):
    def test_kept_unchanged_disputed_replaced_nothing_excluded(self):
        adj = {"cases": [{"id": "1", "label": "answerable", "datasetLabel": "answerable"},
                         {"id": "2", "label": "answerable", "datasetLabel": "unanswerable"}]}
        key = {"s:2": {"suite": "s", "caseId": "2", "datasetLabel": "unanswerable", "group": "disputed-flipped"},
               "s:3": {"suite": "s", "caseId": "3", "datasetLabel": "unanswerable", "group": "disputed-excluded"}}
        out = c.confirmed_suite("s", adj, key, {"s:2": "unanswerable", "s:3": "answerable"})
        self.assertEqual({x["id"]: x["label"] for x in out["cases"]}, {"1": "answerable", "2": "unanswerable", "3": "answerable"})


if __name__ == "__main__":
    unittest.main()
