#!/usr/bin/env python3
"""Tests for workload-coverage.py. Stdlib only.

Run: python3 -m unittest discover -s scripts/fleet -p 'workload_coverage_test.py'
"""
import importlib.util
import io
import json
import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

_spec = importlib.util.spec_from_file_location(
    "workload_coverage", pathlib.Path(__file__).with_name("workload-coverage.py"))
wc = importlib.util.module_from_spec(_spec)
sys.modules["workload_coverage"] = wc
_spec.loader.exec_module(wc)

REPO = pathlib.Path(__file__).resolve().parents[2]

# A provenance-bound snapshot keeps these cases mandatory in standalone Models checkouts.
CATALOG = pathlib.Path(__file__).parent / "fixtures" / "candidate-catalog"


class ParsesTheRealEnum(unittest.TestCase):
    def test_reads_every_workload_the_enum_declares(self):
        workloads, unreadable = wc.parse_workloads(REPO)
        self.assertEqual(unreadable, [],
                         "a workload the parser cannot read is a silent hole in the report")
        # The enum is the authority; this asserts the parser agrees with it rather than asserting
        # a list of names that would itself drift.
        text = (REPO / wc.ENUM).read_text()
        body = text.split("public enum RagWorkload {", 1)[1].split(";", 1)[0]
        declared = sum(1 for line in body.splitlines()
                       if line.strip() and line.strip()[0].isupper()
                       and "(" in line and not line.strip().startswith("//"))
        self.assertGreaterEqual(len(workloads), 10)
        self.assertGreaterEqual(len(workloads), declared - 2,
                                "parsed far fewer workloads than the enum appears to declare")

    def test_handles_a_constant_wrapped_across_lines(self):
        # SUMMARIZATION and TRANSPORTATION are wrapped by the formatter in the real file. A parser
        # that only matched single-line constants would drop exactly the two zero-coverage
        # workloads this report exists to surface.
        workloads, _ = wc.parse_workloads(REPO)
        self.assertIn("summarization", workloads)
        self.assertIn("transportation", workloads)

    def test_ignores_commented_out_text(self):
        workloads, _ = wc.parse_workloads(REPO)
        # The enum carries a prose comment mentioning the router and summarization; none of its
        # words may become a workload id.
        self.assertNotIn("router", workloads)
        self.assertNotIn("coverage", workloads)

    def test_resource_paths_come_back_with_the_id(self):
        workloads, _ = wc.parse_workloads(REPO)
        docs, cases = workloads["sql"]
        self.assertEqual(docs, "/rag/sql/documents.json")
        self.assertEqual(cases, "/rag/sql/cases.json")


class SeparatesNoCorpusFromNoQualifiedModel(unittest.TestCase):
    def test_a_real_corpus_reports_its_case_count(self):
        state = wc.corpus_state(REPO, "/rag/sql/documents.json", "/rag/sql/cases.json")
        self.assertEqual(state["cases"], 9)
        self.assertEqual(state["documents"], 12)

    def test_a_missing_corpus_reports_none_rather_than_zero(self):
        # None means "could not read"; 0 would claim the file exists and is empty. The report
        # branches on these differently, so they must not collapse.
        state = wc.corpus_state(REPO, "/rag/nope/documents.json", "/rag/nope/cases.json")
        self.assertIsNone(state["cases"])
        self.assertIsNone(state["documents"])

    def test_unparseable_corpus_is_none_not_a_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            d = root / wc.RESOURCES / "rag" / "broken"
            d.mkdir(parents=True)
            (d / "cases.json").write_text("{not json")
            (d / "documents.json").write_text("[]")
            state = wc.corpus_state(root, "/rag/broken/documents.json", "/rag/broken/cases.json")
        self.assertIsNone(state["cases"])
        self.assertEqual(state["documents"], 0)


class Reporting(unittest.TestCase):
    """Behaviour is asserted against a SYNTHETIC catalog, never the live one.

    Asserting "sql has no qualified model" against the real catalog would encode today's gap as an
    invariant, and the test would fail at the exact moment the gap was closed -- a test that fails
    on success. The live catalog is used only for the assertions that stay true whatever the
    coverage is.
    """

    def _fake_catalog(self, root, qualified_workloads, rejected_workloads=()):
        cat = pathlib.Path(root) / "catalog"
        cat.mkdir(parents=True, exist_ok=True)
        (cat / "models.json").write_text(json.dumps([
            {"id": "m1", "sizeBytes": 1_000_000_000},
            {"id": "m2", "sizeBytes": 2_000_000_000},
            {"id": "rej", "sizeBytes": 500_000_000},
        ]))
        entries = [{"modelId": f"m{i+1}" if i < 2 else "m1", "workload": wl,
                    "promptTemplate": "chatml", "qualified": True}
                   for i, wl in enumerate(qualified_workloads)]
        entries += [{"modelId": "rej", "workload": wl, "promptTemplate": "chatml",
                     "qualified": False, "verdict": "FAILED_MODEL_CONTRIBUTION_GATE"}
                    for wl in rejected_workloads]
        (cat / "qualifications.json").write_text(json.dumps({"entries": entries}))
        return str(cat)

    def _run(self, catalog, extra=None):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = wc.main(["--catalog", catalog] + (extra or []))
        return rc, buf.getvalue()

    def test_a_workload_with_a_usable_corpus_and_no_model_is_a_closable_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Everything qualified except sql: the report must call sql out, and must not call it
            # "no corpus", because the sql corpus exists in this repo.
            every = [w for w in wc.parse_workloads(REPO)[0] if w != "sql"]
            cat = self._fake_catalog(tmp, every)
            rc, out = self._run(cat)
        self.assertEqual(rc, 0, "without --fail-on-zero the report is informational")
        self.assertIn("NO QUALIFIED MODEL", out)
        gap = [l for l in out.splitlines() if l.startswith("no qualified model but a usable")]
        self.assertTrue(gap, "the summary must name the closable gap")
        self.assertIn("sql", gap[0])
        self.assertNotIn("no corpus (", out.lower().replace("no corpus (no model", "x"))

    def test_fail_on_zero_exits_nonzero_only_while_a_gap_is_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            every = [w for w in wc.parse_workloads(REPO)[0] if w != "sql"]
            rc_open, _ = self._run(self._fake_catalog(tmp, every), ["--fail-on-zero"])
        self.assertEqual(rc_open, 1)
        with tempfile.TemporaryDirectory() as tmp:
            allw = list(wc.parse_workloads(REPO)[0])
            rc_closed, out = self._run(self._fake_catalog(tmp, allw), ["--fail-on-zero"])
        self.assertEqual(rc_closed, 0, "with every workload covered it must pass")
        self.assertIn("every declared workload with a corpus has at least one qualified", out)

    def test_a_rejected_row_is_not_counted_as_coverage(self):
        """qualifications.json holds BOTH outcomes; only one of them is coverage.

        Counting every row inflated general from 51 to 52 and the catalog's qualified total from
        101 to 102, because h2o-danube3-500m sits in `entries` with
        verdict FAILED_MODEL_CONTRIBUTION_GATE. A rejection is evidence; it is not coverage.
        """
        with tempfile.TemporaryDirectory() as tmp:
            every = [w for w in wc.parse_workloads(REPO)[0] if w != "sql"]
            # sql has ONLY a rejected row: it must still read as a gap, not as covered.
            cat = self._fake_catalog(tmp, every, rejected_workloads=["sql"])
            rc, out = self._run(cat, ["--fail-on-zero"])
        self.assertEqual(rc, 1, "a workload whose only row is a rejection is still uncovered")
        gap = [l for l in out.splitlines() if l.startswith("no qualified model but a usable")]
        self.assertTrue(gap and "sql" in gap[0])
        self.assertIn("rejected, and therefore NOT counted as coverage", out)
        self.assertIn("FAILED_MODEL_CONTRIBUTION_GATE", out)
        sql_row = [l for l in out.splitlines() if l.startswith("sql ")][0]
        self.assertRegex(sql_row, r"^sql\s+0\s", "sql must show zero qualified, not one")

    def test_a_workload_with_exactly_one_model_is_called_thin(self):
        with tempfile.TemporaryDirectory() as tmp:
            allw = list(wc.parse_workloads(REPO)[0])
            _, out = self._run(self._fake_catalog(tmp, allw))
        self.assertIn("thin: one model", out)

    def test_a_qualification_naming_an_undeclared_workload_is_surfaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            cat = self._fake_catalog(tmp, list(wc.parse_workloads(REPO)[0]) + ["astrology"])
            _, out = self._run(cat)
        self.assertIn("astrology", out)
        self.assertIn("the enum does not declare", out)


class DocCheck(unittest.TestCase):
    def test_passes_on_the_committed_doc(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = wc.main(["--catalog", str(CATALOG),
                          "--check-doc", str(REPO / "docs" / "WORKLOAD-COVERAGE.md")])
        self.assertEqual(rc, 0, buf.getvalue()[-600:])
        self.assertIn("match this run", buf.getvalue())

    def test_a_stale_table_in_the_doc_is_caught(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc = pathlib.Path(tmp, "DOC.md")
            doc.write_text("```\nworkload   qualified\nsql   999\n```\n")
            problems = wc.check_doc(REPO, str(doc), "the real table", wc.parse_workloads(REPO)[0])
        self.assertTrue(any("pasted coverage table" in x for x in problems))

    def test_a_missing_doc_is_reported_rather_than_ignored(self):
        problems = wc.check_doc(REPO, "/nope/DOC.md", "t", {})
        self.assertTrue(problems)

    def test_blockquote_markers_do_not_break_a_wrapped_quote(self):
        # A quoted case question legitimately wraps across lines; the "> " continuation is
        # formatting, not content, and must be stripped before comparing.
        cases = json.loads((REPO / wc.RESOURCES / "rag" / "sql" / "cases.json").read_text())
        q = cases[0]["question"]
        half = len(q) // 2
        with tempfile.TemporaryDirectory() as tmp:
            doc = pathlib.Path(tmp, "DOC.md")
            doc.write_text("`sql` does **not** ask for SQL.\n\n"
                           f"> case - {q[:half]}\n> {q[half:]}\n")
            problems = wc.check_doc(REPO, str(doc), "", wc.parse_workloads(REPO)[0])
        self.assertFalse([x for x in problems if "quoted case question" in x],
                         "a wrapped blockquote must still match")


class AgainstTheLiveCatalog(unittest.TestCase):
    """Only assertions that hold whatever the live coverage happens to be."""

    def test_the_report_runs_and_covers_every_declared_workload(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = wc.main(["--catalog", str(CATALOG)])
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        for wl in wc.parse_workloads(REPO)[0]:
            self.assertRegex(out, rf"(?m)^{wl}\s", f"{wl} is missing from the report")
        self.assertNotIn("WARNING:", out, "an unparseable enum constant would hole the report")


if __name__ == "__main__":
    unittest.main(verbosity=2)
