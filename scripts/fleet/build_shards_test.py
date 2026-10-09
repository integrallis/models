#!/usr/bin/env python3
"""Tests for build-shards.py. Stdlib only, no dependencies.

Run: python3 -m unittest discover -s scripts/fleet -p 'build_shards_test.py' -v
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
    "build_shards", pathlib.Path(__file__).with_name("build-shards.py"))
bs = importlib.util.module_from_spec(_spec)
sys.modules["build_shards"] = bs
_spec.loader.exec_module(bs)


CATALOG = {
    "a": {"id": "a", "format": "gguf", "downloadUri": "https://x/a.gguf",
          "sizeBytes": 1_000_000_000, "architecture": "qwen2",
          "capabilities": ["chat", "math"]},
    "b": {"id": "b", "format": "gguf", "downloadUri": "https://x/b.gguf",
          "sizeBytes": 4_000_000_000, "architecture": "qwen2",
          "capabilities": ["chat", "code-completion"]},
    "st": {"id": "st", "format": "safetensors", "downloadUri": "https://x/st", "sizeBytes": 10,
           "architecture": "qwen2", "capabilities": ["chat"]},
    "nosize": {"id": "nosize", "format": "gguf", "downloadUri": "https://x/n.gguf",
               "architecture": "qwen2", "capabilities": ["chat"]},
}


def rec(workload, tpl, mt=256, threads=16, decode=8, pool=16,
        stamp="2026-10-09T00:00:00Z", backend="models@dev"):
    env = {}
    if decode is not None:
        env["native-kernel-decode-threads"] = decode
    if pool is not None:
        env["native-kernel-threads"] = pool
    return {"generatedAt": stamp, "source": f"/r/{workload}.json", "backendVersion": backend,
            "settings": {"workload": workload, "promptTemplate": tpl, "maxOutputTokens": mt,
                         "threads": threads},
            "diagnostics": env}


def report(model_id, workload, tpl, stamp, mt=256, decode=8, pool=16):
    return json.dumps({"modelId": model_id, "generatedAt": stamp,
                       "settings": {"workload": workload, "promptTemplate": tpl,
                                    "maxOutputTokens": mt, "threads": pool},
                       "backendDiagnostics": {"environment": {
                           "native-kernel-decode-threads": decode,
                           "native-kernel-threads": pool}}})


class CopiesRecordedSettings(unittest.TestCase):
    def test_copies_every_field_from_the_prior_report(self):
        missing = []
        jobs = bs.build_jobs(["a"], CATALOG,
                             {"a": rec("math", "chatml-no-think", mt=768, decode=4)},
                             {}, {}, missing)
        self.assertEqual(missing, [])
        self.assertEqual(jobs[0]["wl"], "math")
        self.assertEqual(jobs[0]["tpl"], "chatml-no-think")
        self.assertEqual(jobs[0]["mt"], 768)
        self.assertEqual(jobs[0]["dt"], 4)
        self.assertEqual(jobs[0]["gb"], 1.0)

    def test_prefers_the_newest_prior_report_for_a_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = pathlib.Path(tmp, "old"); new = pathlib.Path(tmp, "new")
            old.mkdir(); new.mkdir()
            (old / "a.json").write_text(report("a", "general", "chatml", "2026-01-01T00:00:00Z"))
            (new / "a.json").write_text(
                report("a", "math", "chatml-no-think", "2026-10-09T00:00:00Z", mt=768))
            prior = bs.index_prior_reports([old, new])
        self.assertEqual(prior["a"]["settings"]["workload"], "math")
        self.assertEqual(prior["a"]["settings"]["maxOutputTokens"], 768)

    def test_ignores_verdict_and_comparator_files(self):
        # The comparator is the baseline arm; its settings are derived from the candidate's, so
        # reading it could only restate what the candidate already said. Here it carries a later
        # timestamp and a wrong workload on purpose.
        with tempfile.TemporaryDirectory() as tmp:
            d = pathlib.Path(tmp)
            (d / "a.verdict.json").write_text(json.dumps({"modelId": "a", "settings": {}}))
            (d / "a.comparator.json").write_text(
                report("a", "WRONG", "x", "2027-01-01T00:00:00Z"))
            (d / "a.json").write_text(report("a", "math", "chatml", "2026-10-09T00:00:00Z"))
            prior = bs.index_prior_reports([d])
        self.assertEqual(prior["a"]["settings"]["workload"], "math")

    def test_unreadable_json_is_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = pathlib.Path(tmp)
            (d / "broken.json").write_text("{not json")
            (d / "a.json").write_text(report("a", "math", "chatml", "2026-10-09T00:00:00Z"))
            prior = bs.index_prior_reports([d])
        self.assertIn("a", prior)


class DecodeThreadsAreNotTheThreadBudget(unittest.TestCase):
    """dt is -Dmodels.native.kernels.decodeThreads; settings.threads is --threads.

    Conflating them was this tool's first bug, found by running it against real reports: it wrote
    dt=16 from settings.threads where the prior runs had actually pinned decodeThreads=8.
    """

    def test_dt_comes_from_the_diagnostics_not_from_settings_threads(self):
        missing = []
        jobs = bs.build_jobs(["a"], CATALOG,
                             {"a": rec("math", "chatml", threads=16, decode=8, pool=16)},
                             {}, {}, missing)
        self.assertEqual(jobs[0]["dt"], 8, "dt must be the decode-thread count, not --threads")
        self.assertNotEqual(jobs[0]["dt"], 16)

    def test_dt_is_omitted_when_the_prior_run_used_the_whole_pool(self):
        # The job field is optional and omitting it inherits the worker's default. Pinning a number
        # the prior run did not pin would be inventing a setting.
        missing = []
        jobs = bs.build_jobs(["a"], CATALOG,
                             {"a": rec("math", "chatml", decode=16, pool=16)}, {}, {}, missing)
        self.assertNotIn("dt", jobs[0])

    def test_dt_is_omitted_when_the_report_does_not_record_it(self):
        missing = []
        jobs = bs.build_jobs(["a"], CATALOG,
                             {"a": rec("math", "chatml", decode=None, pool=None)},
                             {}, {}, missing)
        self.assertNotIn("dt", jobs[0])

    def test_dt_is_an_int_because_the_diagnostics_record_it_as_a_string(self):
        missing = []
        jobs = bs.build_jobs(["a"], CATALOG,
                             {"a": rec("math", "chatml", decode="8", pool="16")},
                             {}, {}, missing)
        self.assertEqual(jobs[0]["dt"], 8)
        self.assertIsInstance(jobs[0]["dt"], int,
                              "the job schema uses numbers; a quoted 8 would work by accident")

    def test_a_string_pool_equal_to_the_decode_count_still_omits_dt(self):
        missing = []
        jobs = bs.build_jobs(["a"], CATALOG,
                             {"a": rec("math", "chatml", decode="16", pool="16")},
                             {}, {}, missing)
        self.assertNotIn("dt", jobs[0])

    def test_a_non_numeric_decode_value_is_treated_as_absent(self):
        missing = []
        jobs = bs.build_jobs(["a"], CATALOG,
                             {"a": rec("math", "chatml", decode="pool", pool="16")},
                             {}, {}, missing)
        self.assertNotIn("dt", jobs[0])

    def test_dt_survives_the_round_trip_through_a_real_report_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = pathlib.Path(tmp)
            (d / "a.json").write_text(
                report("a", "math", "chatml", "2026-10-09T00:00:00Z", decode=8, pool=16))
            prior = bs.index_prior_reports([d])
            missing = []
            jobs = bs.build_jobs(["a"], CATALOG, prior, {}, {}, missing)
        self.assertEqual(missing, [])
        self.assertEqual(jobs[0]["dt"], 8)


class RefusesRatherThanGuesses(unittest.TestCase):
    def test_refuses_a_model_with_no_prior_report(self):
        # The whole point: capabilities do not predict workload and architecture does not predict
        # template, so an unseen model is a question for a person.
        missing = []
        jobs = bs.build_jobs(["a"], CATALOG, {}, {}, {}, missing)
        self.assertEqual(jobs, [])
        self.assertEqual(missing, [("a", "no prior report records its settings")])

    def test_refuses_a_non_gguf_entry(self):
        missing = []
        bs.build_jobs(["st"], CATALOG, {"st": rec("general", "chatml")}, {}, {}, missing)
        self.assertIn("format is safetensors", missing[0][1])

    def test_refuses_when_the_catalog_has_no_size(self):
        missing = []
        bs.build_jobs(["nosize"], CATALOG, {"nosize": rec("general", "chatml")}, {}, {}, missing)
        self.assertIn("sizeBytes", missing[0][1])

    def test_refuses_an_unknown_id(self):
        missing = []
        bs.build_jobs(["ghost"], CATALOG, {}, {}, {}, missing)
        self.assertEqual(missing, [("ghost", "not in the catalog")])

    def test_refuses_a_prior_report_missing_the_workload(self):
        missing = []
        broken = {"generatedAt": "z", "source": "s", "backendVersion": "v",
                  "settings": {"promptTemplate": "chatml"}}
        jobs = bs.build_jobs(["a"], CATALOG, {"a": broken}, {}, {}, missing)
        self.assertEqual(jobs, [])
        self.assertIn("no workload", missing[0][1])


class Sharding(unittest.TestCase):
    def test_one_workload_per_shard(self):
        jobs = [{"id": "a", "wl": "math", "gb": 1.0}, {"id": "b", "wl": "coding", "gb": 4.0},
                {"id": "c", "wl": "math", "gb": 2.0}]
        shards = bs.shard(jobs, 700)
        for group in shards.values():
            self.assertEqual(len({j["wl"] for j in group}), 1,
                             "a shard must carry exactly one workload")
        self.assertEqual(len(shards), 2)

    def test_smallest_model_first_within_a_shard(self):
        jobs = [{"id": "big", "wl": "math", "gb": 7.0}, {"id": "small", "wl": "math", "gb": 0.5}]
        group = next(iter(bs.shard(jobs, 700).values()))
        self.assertEqual([j["id"] for j in group], ["small", "big"])

    def test_splits_a_workload_over_the_per_shard_cap(self):
        jobs = [{"id": f"m{i}", "wl": "general", "gb": 1.0}
                for i in range(bs.MAX_JOBS_PER_SHARD + 1)]
        shards = bs.shard(jobs, 700)
        self.assertEqual(len(shards), 2)
        self.assertEqual(sum(len(g) for g in shards.values()), bs.MAX_JOBS_PER_SHARD + 1)


class Overrides(unittest.TestCase):
    def test_overrides_are_applied(self):
        missing = []
        jobs = bs.build_jobs(["a"], CATALOG, {"a": rec("math", "chatml", mt=256)},
                             {"a": 2048}, {"a": "finance"}, missing)
        self.assertEqual(jobs[0]["mt"], 2048)
        self.assertEqual(jobs[0]["wl"], "finance")

    def test_an_override_naming_a_model_not_requested_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = pathlib.Path(tmp)
            (d / "ids.txt").write_text("a\n")
            (d / "models.json").write_text(json.dumps(list(CATALOG.values())))
            (d / "rep").mkdir()
            with self.assertRaises(SystemExit):
                bs.main(["--ids", str(d / "ids.txt"), "--reports", str(d / "rep"),
                         "--catalog", str(d / "models.json"), "--out-dir", str(d),
                         "--start-shard", "900", "--mt-override", "zzz=1"])


class Precedent(unittest.TestCase):
    def test_precedent_reports_what_comparable_models_ran_with(self):
        prior = {"b": rec("coding", "chatml"), "other": rec("general", "llama3")}
        catalog = dict(CATALOG)
        catalog["other"] = {"id": "other", "format": "gguf", "downloadUri": "u", "sizeBytes": 1,
                            "architecture": "llama", "capabilities": ["chat"]}
        p = bs.precedent(catalog, prior, "a")
        self.assertEqual(p["architecture"], "qwen2")
        self.assertEqual(p["templates"], {"chatml": 1},
                         "only same-architecture models inform the template")
        # 'a' shares the 'chat' capability with both, so both workloads are offered -- and the
        # ambiguity is the point: it is why the tool refuses instead of picking one.
        self.assertEqual(p["workloads"], {"coding": 1, "general": 1})


class ShardFilesAreImmutable(unittest.TestCase):
    def test_refuses_to_overwrite_an_existing_shard_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = pathlib.Path(tmp)
            (d / "fleet-shard-700.json").write_text("[]")
            (d / "ids.txt").write_text("a\n")
            (d / "models.json").write_text(json.dumps(list(CATALOG.values())))
            rep = d / "rep"; rep.mkdir()
            (rep / "a.json").write_text(report("a", "math", "chatml", "2026-10-09T00:00:00Z"))
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = bs.main(["--ids", str(d / "ids.txt"), "--reports", str(rep),
                              "--catalog", str(d / "models.json"), "--out-dir", str(d),
                              "--start-shard", "700"])
            self.assertEqual(rc, 1)
            self.assertIn("immutable", buf.getvalue())

    def test_writes_the_expected_job_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = pathlib.Path(tmp)
            (d / "ids.txt").write_text("a\nb\n")
            (d / "models.json").write_text(json.dumps(list(CATALOG.values())))
            rep = d / "rep"; rep.mkdir()
            (rep / "a.json").write_text(report("a", "math", "chatml", "2026-10-09T00:00:00Z"))
            (rep / "b.json").write_text(report("b", "coding", "chatml", "2026-10-09T00:00:00Z"))
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = bs.main(["--ids", str(d / "ids.txt"), "--reports", str(rep),
                              "--catalog", str(d / "models.json"), "--out-dir", str(d),
                              "--start-shard", "800"])
            self.assertEqual(rc, 0)
            written = json.loads((d / "fleet-shard-800.json").read_text())
            self.assertEqual(sorted(written[0]),
                             ["arch", "dt", "gb", "id", "mt", "tpl", "uri", "wl"],
                             "the job file must carry exactly the worker's fields")
            for job in written:
                for key in job:
                    self.assertFalse(key.startswith("_"),
                                     "provenance fields are for the console, not the job file")


if __name__ == "__main__":
    unittest.main(verbosity=2)
