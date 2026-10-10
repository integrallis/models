"""Regression tests for committed, exact candidate evidence matching."""
import importlib.util
import json
import pathlib
import subprocess
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "candidate_inventory", pathlib.Path(__file__).with_name("candidate-inventory.py"))
inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inventory)


class CandidateInventoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.git("init", "-q")
        self.git("config", "user.name", "Inventory test")
        self.git("config", "user.email", "inventory@example.invalid")
        self.git("config", "commit.gpgsign", "false")
        self.git("commit", "-qm", "empty evidence", "--allow-empty")

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.root)

    def report(self, path, payload, commit=True):
        target = self.root / "benchmark-results" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload))
        if commit:
            self.git("add", str(target.relative_to(self.root)))
            self.git("commit", "-qm", "add report")
        return target

    def matches(self, *ids):
        return inventory.evidence_ids(self.root, ids)[0]

    def test_generic_filename_reads_payload_and_records_full_source(self):
        self.report("campaign/models-pure-java.json", {"modelId": "model_q4"})
        found, where = inventory.evidence_ids(self.root, ["model_q4"])
        self.assertEqual({"model_q4"}, found)
        self.assertEqual({"campaign/models-pure-java.json"}, where["model_q4"])

    def test_nested_verdict_and_hyphen_spelling(self):
        self.report("verdict.json", {"candidate": {"modelId": "model-q4"}})
        self.assertEqual({"model_q4"}, self.matches("model_q4"))

    def test_exact_legacy_filename_without_model_id(self):
        self.report("model-q4.json", {"perProbe": [{"cosine": 0.99}]})
        self.assertEqual({"model_q4"}, self.matches("model_q4"))

    def test_short_id_does_not_match_longer_quantization(self):
        self.report("model_q4_k_m.json", {"modelId": "model_q4_k_m"})
        self.assertEqual({"model_q4_k_m"}, self.matches("model_q4", "model_q4_k_m"))

    def test_filename_cannot_override_payload_identity(self):
        self.report("model_q4.json", {"modelId": "other_q8"})
        self.assertEqual(set(), self.matches("model_q4"))

    def test_untracked_and_staged_files_do_not_count(self):
        target = self.report("model_q4.json", {"modelId": "model_q4"}, commit=False)
        self.assertEqual(set(), self.matches("model_q4"))
        self.git("add", str(target.relative_to(self.root)))
        self.assertEqual(set(), self.matches("model_q4"))

    def test_uncommitted_edits_cannot_change_committed_identity(self):
        target = self.report("report.json", {"modelId": "model_q4"})
        target.write_text('{"modelId": "other_q8"}')
        self.assertEqual({"model_q4"}, self.matches("model_q4", "other_q8"))

    def test_invalid_committed_json_does_not_count(self):
        target = self.report("model_q4.json", {}, commit=False)
        target.write_text("not json")
        self.git("add", ".")
        self.git("commit", "-qm", "invalid report")
        self.assertEqual(set(), self.matches("model_q4"))

    def test_qualified_in_one_manifest_wins_over_rejected_elsewhere(self):
        catalog = self.root / "catalog"
        catalog.mkdir()
        (catalog / "qualifications.json").write_text(json.dumps({"entries": [
            {"modelId": "model_q4", "qualified": True}]}))
        (catalog / "tool-qualifications.json").write_text(json.dumps({"entries": [
            {"modelId": "model_q4", "summary": {"qualified": False}}]}))
        self.assertEqual("QUALIFIED", inventory.manifest_rows(catalog)["model_q4"][0])


if __name__ == "__main__":
    unittest.main()
