import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from freeze_qualification_window import (
    assert_windows_disjoint,
    select_unseen_cases,
    verify_prepared_exclusions,
)


def _case(identifier: str, query: str) -> dict:
    return {
        "id": identifier,
        "question": [[{"role": "user", "content": query}]],
    }


class FreezeQualificationWindowTest(unittest.TestCase):
    def test_rejects_a_query_reused_from_the_exposed_window(self):
        exposed = [_case("old", "same query")]
        candidate = [_case("new", "same query")]

        with self.assertRaisesRegex(ValueError, "query overlap"):
            assert_windows_disjoint(exposed, candidate)

    def test_requires_each_preparation_to_bind_every_qualification_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / "manifest.json"
            expected = {"simple.json": "a" * 64, "multiple.json": "b" * 64}
            prepared.write_text(
                json.dumps(
                    {
                        "schemaVersion": 2,
                        "excludedEvaluationSources": [
                            {"path": "simple.json", "sha256": "a" * 64}
                        ],
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "does not bind every qualification source"):
                verify_prepared_exclusions([prepared], expected)

            value = json.loads(prepared.read_text())
            value["excludedEvaluationSources"].append(
                {"path": "multiple.json", "sha256": "b" * 64}
            )
            prepared.write_text(json.dumps(value))

            verified = verify_prepared_exclusions([prepared], expected)
            self.assertEqual(verified[0]["sha256"], hashlib.sha256(prepared.read_bytes()).hexdigest())

    def test_unseen_selector_skips_different_ids_with_exposed_or_duplicate_queries(self):
        cases = [
            _case("exposed", "already viewed"),
            _case("duplicate-id", "already viewed"),
            _case("new-a", "new query a"),
            _case("new-a-copy", "new query a"),
            _case("new-b", "new query b"),
            _case("new-c", "new query c"),
        ]

        exposed, selected = select_unseen_cases(cases, count=3, seed=9, exposed_count=1)

        self.assertEqual(len(exposed), 1)
        self.assertEqual(len(selected), 3)
        assert_windows_disjoint(exposed, selected)
        query_sets = [{*case["question"][0][0]["content"].splitlines()} for case in selected]
        self.assertEqual(len({next(iter(value)) for value in query_sets}), 3)


if __name__ == "__main__":
    unittest.main()
