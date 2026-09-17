import unittest

import audit_suite_labels as a


def case(i, label):
    return {"id": str(i), "label": label, "messages": [{"role": "user", "text": f"q{i}"}], "documents": "[]"}


class AdjudicationRule(unittest.TestCase):
    def test_one_agreeing_judge_keeps_label(self):
        self.assertEqual(a.adjudicate_case("unanswerable", "answerable", "unanswerable"), "unanswerable")
        self.assertEqual(a.adjudicate_case("answerable", "ambiguous", "answerable"), "answerable")

    def test_two_definite_contradictions_flip(self):
        self.assertEqual(a.adjudicate_case("unanswerable", "answerable", "answerable"), "answerable")

    def test_ambiguity_without_agreement_excludes(self):
        self.assertIsNone(a.adjudicate_case("unanswerable", "answerable", "ambiguous"))
        self.assertIsNone(a.adjudicate_case("answerable", "ambiguous", "ambiguous"))

    def test_unknown_judgement_rejected(self):
        with self.assertRaises(ValueError):
            a.adjudicate_case("answerable", "yes", "answerable")


class Admission(unittest.TestCase):
    def build(self, flips_un, excl_un, n=100):
        cases = [case(i, "answerable") for i in range(n)] + [case(n + i, "unanswerable") for i in range(n)]
        ja = {c["id"]: c["label"] for c in cases}
        jb = dict(ja)
        for i in range(flips_un):
            cid = str(n + i); ja[cid] = jb[cid] = "answerable"
        for i in range(flips_un, flips_un + excl_un):
            cid = str(n + i); ja[cid] = "answerable"; jb[cid] = "ambiguous"
        return a.adjudicate(cases, {"A": ja, "B": jb})

    def test_clean_suite_original_admissible(self):
        self.assertEqual(self.build(5, 5)["admission"], "ORIGINAL_ADMISSIBLE")

    def test_msmarco_shape_adjudicated_only(self):
        r = self.build(35, 25)
        self.assertEqual(r["admission"], "ADJUDICATED_ONLY")
        self.assertEqual(r["perDatasetLabel"]["unanswerable"], {"cases": 100, "kept": 40, "flipped": 35, "excluded": 25})

    def test_too_few_minority_cases_unusable(self):
        # 25 unanswerables survive (< 30) while exclusions stay at 12.5 %: the per-label floor alone rejects it.
        self.assertEqual(self.build(50, 25)["admission"], "UNUSABLE")

    def test_too_many_exclusions_unusable(self):
        # 200 cases: 50 exclusions (25 %) is the admissible edge, 51 is not.
        self.assertEqual(self.build(0, 50)["admission"], "ADJUDICATED_ONLY")
        self.assertEqual(self.build(0, 51)["admission"], "UNUSABLE")

    def test_incomplete_judge_rejected(self):
        cases = [case(1, "answerable"), case(2, "unanswerable")]
        with self.assertRaises(ValueError):
            a.adjudicate(cases, {"A": {"1": "answerable"}, "B": {"1": "answerable", "2": "unanswerable"}})

    def test_sha_independent_of_case_order(self):
        cases = [case(1, "answerable"), case(2, "unanswerable")]
        j = {"1": "answerable", "2": "unanswerable"}
        self.assertEqual(a.adjudicate(cases, {"A": j, "B": j})["casesSha256"],
                         a.adjudicate(cases[::-1], {"A": j, "B": j})["casesSha256"])


class Export(unittest.TestCase):
    def test_blind_parts_have_no_label_and_cover_all(self):
        cases = [case(i, "answerable" if i % 2 else "unanswerable") for i in range(10)]
        parts = a.export(cases, 3, 1)
        flat = [r for p in parts for r in p]
        self.assertEqual(sorted(r["id"] for r in flat), sorted(c["id"] for c in cases))
        self.assertTrue(all(set(r) == {"id", "question", "documents"} for r in flat))


class Scoring(unittest.TestCase):
    def test_balanced_and_interval(self):
        labels = {"1": "answerable", "2": "answerable", "3": "unanswerable", "4": "unanswerable"}
        preds = {"1": "answerable", "2": "unanswerable", "3": "unanswerable", "4": "unanswerable"}
        s = a.score(preds, labels)
        self.assertAlmostEqual(s["balancedAccuracy"], 0.75)
        lo, hi = s["balancedInterval95"]
        self.assertTrue(lo < 0.75 < hi)

    def test_missing_prediction_rejected(self):
        with self.assertRaises(ValueError):
            a.score({}, {"1": "answerable"})


if __name__ == "__main__":
    unittest.main()
