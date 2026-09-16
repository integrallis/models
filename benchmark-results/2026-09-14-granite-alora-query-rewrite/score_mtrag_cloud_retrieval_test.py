import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path


MODULE = Path(__file__).with_name("score_mtrag_cloud_retrieval.py")
SPEC = importlib.util.spec_from_file_location("score", MODULE)
score = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(score)


class MtragRetrievalScoreTest(unittest.TestCase):
    def test_scores_shared_generated_rewrite_against_qrels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite = {"cases": [{"id": "one", "queryId": "q1", "messages": [{"role": "user", "text": "tell me about it"}], "expectedRewrite": "granite cloud"}]}
            report = {"cases": [{"id": "one", "structured": True, "physicallyShared": True, "actualRewrite": "granite cloud"}]}
            (root / "suite.json").write_text(json.dumps(suite))
            (root / "report.json").write_text(json.dumps(report))
            (root / "qrels.tsv").write_text("query-id\tcorpus-id\tscore\nq1\tdoc-1\t1\n")
            with zipfile.ZipFile(root / "corpus.zip", "w") as archive:
                archive.writestr("corpus.jsonl", json.dumps({"_id": "doc-1", "text": "Granite cloud model"}) + "\n")
            result = score.score(root / "suite.json", root / "report.json", root / "qrels.tsv", root / "corpus.zip")
            self.assertTrue(result["screen"]["passed"])
            self.assertEqual(1.0, result["variants"]["generated"]["recallAt"]["1"])

    def test_rejects_unshared_or_unstructured_results(self):
        suite = {"cases": [{"id": "one", "queryId": "q1", "messages": [{"role": "user", "text": "x"}], "expectedRewrite": "x"}]}
        report = {"cases": [{"id": "one", "structured": False, "physicallyShared": True, "actualRewrite": "x"}]}
        with self.assertRaisesRegex(ValueError, "shared structured"):
            score.join_cases(suite, report)


if __name__ == "__main__":
    unittest.main()
