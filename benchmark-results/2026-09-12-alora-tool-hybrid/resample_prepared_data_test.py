import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from resample_prepared_data import resample


def _record(index: int, calls: bool) -> dict:
    invocation = [{"name": "tool", "arguments": {"value": index}}] if calls else []
    assistant = (
        '<tool_call>\n{"name":"tool","arguments":{"value":%d}}\n</tool_call>' % index
        if calls
        else "<tool_call>\n[]\n</tool_call>"
    )
    return {
        "sourceLine": index,
        "user": f"request {index}",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "tool",
                    "description": "fixture",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "calls": invocation,
        "assistant": assistant,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ResamplePreparedDataTest(unittest.TestCase):
    def test_resamples_exact_disjoint_strata_and_binds_the_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "parent"
            output = root / "output"
            parent.mkdir()
            rows = [_record(index, index % 2 == 0) for index in range(12)]
            train_hash = _write_jsonl(parent / "train.jsonl", rows[:8])
            validation_hash = _write_jsonl(parent / "validation.jsonl", rows[8:])
            manifest = {
                "schemaVersion": 2,
                "seed": 1,
                "trainingSource": {
                    "repository": "test/source",
                    "revision": "a" * 40,
                    "path": "source.jsonl",
                    "sha256": "b" * 64,
                },
                "irrelevanceSource": {
                    "repository": "test/no-call",
                    "revision": "c" * 40,
                    "path": "irrelevance.json",
                    "sha256": "d" * 64,
                },
                "excludedEvaluationSources": [],
                "train": {"count": 8, "noCallCount": 4, "sha256": train_hash},
                "validation": {
                    "count": 4,
                    "noCallCount": 2,
                    "sha256": validation_hash,
                },
            }
            (parent / "manifest.json").write_text(json.dumps(manifest))

            result = resample(
                parent=parent,
                output=output,
                train_count=6,
                validation_count=2,
                no_call_fraction=0.5,
                seed=20260913,
            )

            train = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
            validation = [
                json.loads(line) for line in (output / "validation.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(train), 6)
            self.assertEqual(sum(not row["calls"] for row in train), 3)
            self.assertEqual(len(validation), 2)
            self.assertEqual(sum(not row["calls"] for row in validation), 1)
            self.assertTrue(
                {row["sourceLine"] for row in train}.isdisjoint(
                    row["sourceLine"] for row in validation
                )
            )
            self.assertEqual(result["schemaVersion"], 2)
            self.assertEqual(result["noCallFraction"], 0.5)
            self.assertEqual(
                result["resampling"]["parentManifestSha256"],
                hashlib.sha256((parent / "manifest.json").read_bytes()).hexdigest(),
            )
            self.assertEqual(result["resampling"]["parentTrainSha256"], train_hash)
            self.assertEqual(
                result["resampling"]["parentValidationSha256"], validation_hash
            )

    def test_fails_when_a_parent_split_hash_differs(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "parent"
            parent.mkdir()
            _write_jsonl(parent / "train.jsonl", [_record(1, True)])
            validation_hash = _write_jsonl(parent / "validation.jsonl", [_record(2, False)])
            (parent / "manifest.json").write_text(
                json.dumps(
                    {
                        "schemaVersion": 2,
                        "train": {"sha256": "0" * 64},
                        "validation": {"sha256": validation_hash},
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "parent train split SHA-256 differs"):
                resample(
                    parent=parent,
                    output=Path(directory) / "output",
                    train_count=1,
                    validation_count=1,
                    no_call_fraction=0.5,
                    seed=1,
                )


if __name__ == "__main__":
    unittest.main()
