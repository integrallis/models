import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from merge_prepared_data import merge_prepared


def _record(source: str, index: int, calls: bool) -> dict:
    invocation = [{"name": "tool", "arguments": {"value": index}}] if calls else []
    assistant = (
        '<tool_call>\n{"name":"tool","arguments":{"value":%d}}\n</tool_call>' % index
        if calls
        else "<tool_call>\n[]\n</tool_call>"
    )
    return {
        "sourceLine": index,
        "source": source,
        "user": f"request {source} {index}",
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


def _write_parent(root: Path, train: list[dict], validation: list[dict]) -> dict:
    root.mkdir()

    def write(split: str, rows: list[dict]) -> str:
        path = root / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "schemaVersion": 2,
        "train": {"count": len(train), "sha256": write("train", train)},
        "validation": {
            "count": len(validation),
            "sha256": write("validation", validation),
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    return manifest


class MergePreparedDataTest(unittest.TestCase):
    def test_merges_exact_disjoint_strata_and_binds_both_parents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            positive = root / "positive"
            negative = root / "negative"
            positive_manifest = _write_parent(
                positive,
                [_record("positive", index, True) for index in range(8)],
                [_record("positive", index, True) for index in range(8, 12)],
            )
            negative_manifest = _write_parent(
                negative,
                [_record("negative", index, False) for index in range(6)],
                [_record("negative", index, False) for index in range(6, 9)],
            )

            output = root / "output"
            result = merge_prepared(
                positive=positive,
                negative=negative,
                output=output,
                train_call_count=5,
                train_no_call_count=3,
                validation_call_count=2,
                validation_no_call_count=2,
                seed=20260913,
            )

            train = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
            validation = [
                json.loads(line) for line in (output / "validation.jsonl").read_text().splitlines()
            ]
            self.assertEqual((len(train), sum(not row["calls"] for row in train)), (8, 3))
            self.assertEqual((len(validation), sum(not row["calls"] for row in validation)), (4, 2))
            train_keys = {(row["source"], row["sourceLine"]) for row in train}
            validation_keys = {(row["source"], row["sourceLine"]) for row in validation}
            self.assertTrue(train_keys.isdisjoint(validation_keys))
            self.assertEqual(result["schemaVersion"], 2)
            self.assertEqual(result["noCallFraction"], 5 / 12)
            self.assertEqual(result["parents"]["positive"]["trainSha256"], positive_manifest["train"]["sha256"])
            self.assertEqual(result["parents"]["negative"]["validationSha256"], negative_manifest["validation"]["sha256"])
            self.assertEqual(
                result["parents"]["positive"]["manifestSha256"],
                hashlib.sha256((positive / "manifest.json").read_bytes()).hexdigest(),
            )

    def test_filters_each_parent_to_its_role_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            positive = root / "positive"
            negative = root / "negative"
            _write_parent(positive, [_record("positive", 1, False)], [_record("positive", 2, True)])
            _write_parent(negative, [_record("negative", 1, False)], [_record("negative", 2, False)])
            result = merge_prepared(
                positive=positive,
                negative=negative,
                output=root / "filtered",
                train_call_count=0,
                train_no_call_count=1,
                validation_call_count=1,
                validation_no_call_count=1,
                seed=1,
            )
            self.assertEqual(result["statistics"]["positiveTrainExcludedNoCalls"], 1)

            positive = root / "positive-good"
            _write_parent(positive, [_record("positive", 1, True)], [_record("positive", 2, True)])
            (negative / "train.jsonl").write_text("tampered\n")
            with self.assertRaisesRegex(ValueError, "negative train split SHA-256 differs"):
                merge_prepared(
                    positive=positive,
                    negative=negative,
                    output=root / "tampered",
                    train_call_count=1,
                    train_no_call_count=1,
                    validation_call_count=1,
                    validation_no_call_count=1,
                    seed=1,
                )


if __name__ == "__main__":
    unittest.main()
