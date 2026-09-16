import json
import tempfile
import unittest
from pathlib import Path

from prepare_answerability_suites import prepare, rank_key


def mtrag_row(task_id: str, label: str, turns=None) -> str:
    turns = turns or [{"speaker": "user", "text": f"question {task_id}"}]
    return json.dumps(
        {
            "task_id": task_id,
            "Answerability": [label],
            "input": turns,
            "contexts": [{"document_id": "d", "text": f"passage {task_id}"}],
        }
    )


def squad_source(possible: int, impossible: int) -> dict:
    qas = []
    for index in range(possible):
        qas.append({"id": f"p{index}", "question": f"q{index}", "is_impossible": False,
                    "answers": [{"text": "x", "answer_start": 0}]})
    for index in range(impossible):
        qas.append({"id": f"i{index}", "question": f"u{index}", "is_impossible": True, "answers": []})
    return {"version": "v2.0", "data": [{"paragraphs": [{"context": "ctx", "qas": qas}]}]}


class PrepareAnswerabilitySuitesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.mtrag = self.root / "RAG.jsonl"
        self.squad = self.root / "dev-v2.0.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_mtrag(self, rows: list[str]) -> None:
        self.mtrag.write_text("\n".join(rows) + "\n")

    def test_selects_every_unanswerable_task_and_an_equal_answerable_stratum(self) -> None:
        rows = [mtrag_row(f"a{i}", "ANSWERABLE") for i in range(10)]
        rows += [mtrag_row(f"u{i}", "UNANSWERABLE") for i in range(3)]
        rows += [mtrag_row("p0", "PARTIAL"), mtrag_row("c0", "CONVERSATIONAL")]
        self.write_mtrag(rows)
        self.squad.write_text(json.dumps(squad_source(120, 110)))

        window = prepare(self.mtrag, self.squad, "a" * 40)

        mtrag, squad = window["suites"]
        self.assertEqual({"answerable": 3, "unanswerable": 3}, mtrag["selectedByLabel"])
        self.assertEqual({"answerable": 10, "unanswerable": 3}, mtrag["eligibleByLabel"])
        self.assertEqual({"answerable": 100, "unanswerable": 100}, squad["selectedByLabel"])
        self.assertEqual("a" * 40, mtrag["source"]["revision"])
        self.assertNotIn("revision", squad["source"])
        self.assertEqual(64, len(window["windowSha256"]))

    def test_selection_is_deterministic_and_hash_ordered_not_file_ordered(self) -> None:
        rows = [mtrag_row(f"a{i}", "ANSWERABLE") for i in range(10)]
        rows += [mtrag_row("u0", "UNANSWERABLE")]
        self.write_mtrag(rows)
        self.squad.write_text(json.dumps(squad_source(100, 100)))

        first = prepare(self.mtrag, self.squad, "a" * 40)
        second = prepare(self.mtrag, self.squad, "a" * 40)

        self.assertEqual(first["windowSha256"], second["windowSha256"])
        expected = sorted((f"a{i}" for i in range(10)), key=rank_key)[:1]
        chosen = [c["id"] for c in first["suites"][0]["cases"] if c["label"] == "answerable"]
        self.assertEqual(expected, chosen)
        ids = [c["id"] for c in first["suites"][0]["cases"]]
        self.assertEqual(sorted(ids, key=rank_key), ids)

    def test_carries_the_full_conversation_and_documents_for_each_case(self) -> None:
        turns = [
            {"speaker": "user", "text": "first"},
            {"speaker": "agent", "text": "reply"},
            {"speaker": "user", "text": "follow-up"},
        ]
        self.write_mtrag([mtrag_row("u0", "UNANSWERABLE", turns), mtrag_row("a0", "ANSWERABLE")])
        self.squad.write_text(json.dumps(squad_source(100, 100)))

        window = prepare(self.mtrag, self.squad, "a" * 40)

        case = next(c for c in window["suites"][0]["cases"] if c["id"] == "u0")
        self.assertEqual(
            [{"role": "user", "text": "first"}, {"role": "assistant", "text": "reply"},
             {"role": "user", "text": "follow-up"}],
            case["messages"],
        )
        self.assertEqual([{"doc_id": 1, "text": "passage u0"}], case["documents"])
        self.assertEqual("unanswerable", case["label"])

    def test_rejects_a_conversation_that_does_not_end_with_the_user(self) -> None:
        turns = [{"speaker": "user", "text": "q"}, {"speaker": "agent", "text": "a"}]
        self.write_mtrag([mtrag_row("u0", "UNANSWERABLE", turns), mtrag_row("a0", "ANSWERABLE")])
        self.squad.write_text(json.dumps(squad_source(100, 100)))

        with self.assertRaisesRegex(ValueError, "end with a user turn"):
            prepare(self.mtrag, self.squad, "a" * 40)

    def test_rejects_squad_labels_that_contradict_their_answers(self) -> None:
        self.write_mtrag([mtrag_row("u0", "UNANSWERABLE"), mtrag_row("a0", "ANSWERABLE")])
        source = squad_source(100, 100)
        source["data"][0]["paragraphs"][0]["qas"][0]["answers"] = []
        self.squad.write_text(json.dumps(source))

        with self.assertRaisesRegex(ValueError, "carries no answers"):
            prepare(self.mtrag, self.squad, "a" * 40)

    def test_rejects_a_squad_source_that_is_too_small_for_the_frozen_strata(self) -> None:
        self.write_mtrag([mtrag_row("u0", "UNANSWERABLE"), mtrag_row("a0", "ANSWERABLE")])
        self.squad.write_text(json.dumps(squad_source(100, 99)))

        with self.assertRaisesRegex(ValueError, "only 99 unanswerable cases"):
            prepare(self.mtrag, self.squad, "a" * 40)


if __name__ == "__main__":
    unittest.main()
