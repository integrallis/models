"""Regression tests for fail-closed qualification evidence handling (no model inference)."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from compare import compare


class ComparisonTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.candidate = Path(self.temp.name) / "candidate"
        self.oracle = Path(self.temp.name) / "oracle"
        self.report = dict(schema="vision-integration-v1", host="worker", threads=8,
                           mmprojSha256="projector", imageSha256="image", imageSize=2,
                           tokens=2, dimension=2, warmups=1, repetitions=3,
                           encoderProjectorMs=[1.0, 2.0, 3.0], candidateRevision="a" * 40)
        for directory in (self.candidate, self.oracle):
            directory.mkdir()
            (directory / "report.json").write_text(json.dumps(self.report))
            np.arange(1, 13, dtype="<f4").tofile(directory / "pixels.f32")
            np.arange(1, 5, dtype="<f4").tofile(directory / "embeddings.f32")

    def test_equal_vectors_pass_numerics_but_never_admit_a_catalog_model(self):
        result = compare(self.candidate, self.oracle)
        self.assertTrue(result["numericalPass"])
        self.assertFalse(result["catalogQualified"])
        self.assertEqual(result["candidateToOracleLatencyRatio"], 1.0)

    def test_wrong_vectors_fail_even_with_correct_shape(self):
        np.array([4, 3, 2, 1], dtype="<f4").tofile(self.candidate / "embeddings.f32")
        self.assertFalse(compare(self.candidate, self.oracle)["numericalPass"])

    def test_latency_gate_is_independent_of_numerical_fidelity(self):
        (self.candidate / "report.json").write_text(json.dumps(
            self.report | {"encoderProjectorMs": [2, 4, 6]}))
        result = compare(self.candidate, self.oracle, 1.10)
        self.assertTrue(result["numericalPass"])
        self.assertFalse(result["performancePass"])
        self.assertEqual(result["candidateToOracleLatencyRatio"], 2.0)
        for limit in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                compare(self.candidate, self.oracle, limit)

    def test_mismatched_provenance_and_partial_timings_are_rejected(self):
        for key, value in (("host", "other"), ("threads", 4), ("imageSha256", "different"),
                           ("mmprojSha256", "different"), ("encoderProjectorMs", [1.0]),
                           ("candidateRevision", "unrecorded")):
            with self.subTest(key=key):
                (self.candidate / "report.json").write_text(json.dumps(self.report | {key: value}))
                with self.assertRaises(ValueError):
                    compare(self.candidate, self.oracle)

    def test_missing_truncated_and_nonfinite_vectors_are_rejected(self):
        for values in ([], [1.0], [1, 2, float("nan"), 4]):
            np.array(values, dtype="<f4").tofile(self.candidate / "embeddings.f32")
            with self.assertRaises(ValueError):
                compare(self.candidate, self.oracle)
        (self.candidate / "embeddings.f32").unlink()
        with self.assertRaises(FileNotFoundError):
            compare(self.candidate, self.oracle)


if __name__ == "__main__":
    unittest.main()
