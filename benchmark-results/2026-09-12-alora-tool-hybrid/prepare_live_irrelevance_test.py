import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from prepare_live_irrelevance import prepare


def _row(identifier: str, query: str, tool: str = "weather") -> dict:
    return {
        "id": identifier,
        "question": [[{"role": "user", "content": query}]],
        "function": [
            {
                "name": tool,
                "description": "Get weather",
                "parameters": {
                    "type": "dict",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
    }


class PrepareLiveIrrelevanceTest(unittest.TestCase):
    def test_prepares_disjoint_deterministic_no_call_splits_and_binds_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "live.jsonl"
            excluded = root / "excluded.jsonl"
            output = root / "prepared"
            rows = [_row(str(index), f"unrelated request {index}") for index in range(8)]
            rows.append(_row("overlap", "held out query"))
            rows.append(
                {
                    **_row("conversation", "ignored"),
                    "question": [[
                        {"role": "user", "content": "first"},
                        {"role": "assistant", "content": "answer"},
                        {"role": "user", "content": "second"},
                    ]],
                }
            )
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            excluded.write_text(json.dumps(_row("held-out", "held out query")) + "\n")
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

            manifest = prepare(
                source=source,
                excluded_paths=[excluded],
                output=output,
                train_count=6,
                validation_count=2,
                seed=20260915,
                expected_source_sha256=source_hash,
            )

            train = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
            validation = [
                json.loads(line) for line in (output / "validation.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(train), 6)
            self.assertEqual(len(validation), 2)
            self.assertTrue(all(record["calls"] == [] for record in train + validation))
            self.assertNotIn("held out query", {record["user"] for record in train + validation})
            self.assertNotIn("second", {record["user"] for record in train + validation})
            self.assertTrue(
                {record["sourceLine"] for record in train}.isdisjoint(
                    record["sourceLine"] for record in validation
                )
            )
            self.assertEqual(manifest["source"]["sha256"], source_hash)
            self.assertEqual(manifest["excludedEvaluationSources"][0]["sha256"], hashlib.sha256(excluded.read_bytes()).hexdigest())
            self.assertEqual(manifest["statistics"]["overlap"], 1)
            self.assertEqual(manifest["statistics"]["multiMessage"], 1)

    def test_rejects_changed_source_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "live.jsonl"
            excluded = root / "excluded.jsonl"
            source.write_text(json.dumps(_row("1", "query")) + "\n")
            excluded.write_text("")

            with self.assertRaisesRegex(ValueError, "source SHA-256 differs"):
                prepare(
                    source=source,
                    excluded_paths=[excluded],
                    output=root / "prepared",
                    train_count=1,
                    validation_count=1,
                    seed=1,
                    expected_source_sha256="0" * 64,
                )


if __name__ == "__main__":
    unittest.main()
