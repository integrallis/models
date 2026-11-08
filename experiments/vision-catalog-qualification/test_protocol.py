import copy
import json
import tempfile
import unittest
from pathlib import Path

import compare
from prepare import select


class Evaluator:
    def answer_processor(self, answer):
        return answer.strip()

    def _compute_answer_scores(self, answers):
        return {answer: 1.0 for answer in answers}


class ProtocolTest(unittest.TestCase):
    def test_selection_is_order_independent_and_ignores_answers(self):
        shards = [[dict(question_id=i, answers=["yes"]) for i in range(start, start + 20)] for start in [0, 20]]
        first = select(shards, "seed", 12)
        changed = [[dict(row, answers=["no"]) for row in reversed(shard)] for shard in shards]
        second = select(changed, "seed", 12)
        self.assertEqual([r["question_id"] for r in first], [r["question_id"] for r in second])
        self.assertEqual(6, sum(r["question_id"] < 20 for r in first))

    def fixture(self, root):
        manifest = json.loads(Path(__file__).with_name("manifest.json").read_text())
        configuration = dict(host="fixture", threads=8, context=1024, maximumOutputTokens=32, revision="a" * 40, kvType="f32")
        configuration["diagnostics"] = dict(environment={"vision-projections": "rust-fixture", "decoder-attention": "NativeCausalAttention"})
        variant = "Q4_K_S"
        destination = root / variant
        (destination / "candidate").mkdir(parents=True)
        (root / "encoder").mkdir()
        (destination / "candidate/configuration.json").write_text(json.dumps(configuration))
        receipt = dict(configuration, corpusSha256="b" * 64)
        (root / "receipt.json").write_text(json.dumps(receipt))
        encoder = dict(configuration, corpusSha256="b" * 64,
                       mmprojSha256=manifest["files"]["llava-v1.5-7b-mmproj-model-f16.gguf"]["sha256"])
        (root / "encoder/configuration.json").write_text(json.dumps(encoder))
        references = [dict(configuration=configuration)]
        cases = []
        for dataset in ["vqav2", "textvqa"]:
            for i in range(32):
                for blank in ([False, True] if i < 8 else [False]):
                    identity = f"{dataset}-{i}" + ("-blank" if blank else "")
                    perf = i < 2 and not blank
                    case = dict(id=identity, dataset=dataset, blank=blank, answers=["yes"] * 10,
                                warmups=1 if perf else 0, repetitions=3 if perf else 1)
                    cases.append(case)
                    candidate, encodings = [], []
                    for run in range(-case["warmups"], case["repetitions"]):
                        answer = "no" if blank else "yes"
                        candidate.append(dict(run=run, answer=answer, tokens=[1], prefillMs=10, decodeMs=1, totalMs=11))
                        encodings.append(dict(run=run, encoderMs=1))
                        references.append(dict(id=identity, run=run, answerHex=answer.encode().hex(), tokens=[1], prefillMs=9, decodeMs=1))
                    for directory, rows in [(destination / "candidate" / identity, candidate), (root / "encoder" / identity, encodings)]:
                        directory.mkdir()
                        (directory / "runs.json").write_text(json.dumps(rows))
        (destination / "reference.jsonl").write_text("\n".join(map(json.dumps, references)))
        return dict(cases=cases), manifest, variant

    def test_java_fallback_cannot_be_claimed_as_native_qualification(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            corpus, manifest, variant = self.fixture(root)
            path = root / "receipt.json"
            receipt = json.loads(path.read_text()); receipt.update(schema=2, referenceFlashAttention="auto")
            path.write_text(json.dumps(receipt))
            path = root / variant / "reference.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["configuration"]["flashAttention"] = "auto"
            path.write_text("\n".join(map(json.dumps, rows)))
            path = root / variant / "candidate/configuration.json"
            config = json.loads(path.read_text())
            config["diagnostics"]["environment"] = {"vision-projections": "vector-api", "decoder-attention": "NoKernel"}
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "requires native"):
                compare.evaluate(corpus, manifest, root, variant, Evaluator())

    def test_complete_paired_evidence_passes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            corpus, manifest, variant = self.fixture(root)
            result = compare.evaluate(corpus, manifest, root, variant, Evaluator())
            self.assertTrue(result["taskQualified"])
            self.assertEqual(4, len(result["performance"]))
            self.assertEqual(1, result["datasets"]["vqav2"]["imageContribution"])

    def test_fidelity_loss_and_slow_prefill_cannot_qualify(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            corpus, manifest, variant = self.fixture(root)
            path = root / variant / "candidate/vqav2-0/runs.json"
            rows = json.loads(path.read_text())
            for row in rows:
                row.update(tokens=[2], answer="wrong", prefillMs=20, totalMs=21)
            path.write_text(json.dumps(rows))
            result = compare.evaluate(corpus, manifest, root, variant, Evaluator())
            self.assertFalse(result["taskQualified"])
            self.assertFalse(result["gates"]["accuracyLoss"])
            self.assertFalse(result["gates"]["performance"])
            self.assertFalse(result["gates"]["tokenAgreement"])

    def test_missing_reference_and_wrong_host_fail_closed(self):
        for corrupt in ["missing", "host"]:
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                corpus, manifest, variant = self.fixture(root)
                if corrupt == "missing":
                    path = root / variant / "reference.jsonl"
                    path.write_text("\n".join(path.read_text().splitlines()[:-1]))
                else:
                    path = root / "encoder/configuration.json"
                    config = json.loads(path.read_text()); config["host"] = "other"
                    path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    compare.evaluate(corpus, manifest, root, variant, Evaluator())

    def test_nan_and_missing_repetitions_rejected(self):
        with self.assertRaises(ValueError):
            compare.median([dict(latency=float("nan"))], "latency")
        with self.assertRaises(ValueError):
            compare.measured([dict(run=0)], dict(id="x", warmups=1, repetitions=3))

    def test_new_receipt_requires_the_explicit_frozen_attention_route(self):
        for mode in [None, "auto", "disabled"]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                corpus, manifest, variant = self.fixture(root)
                path = root / "receipt.json"
                receipt = json.loads(path.read_text())
                receipt.update(schema=2, referenceFlashAttention="auto")
                path.write_text(json.dumps(receipt))
                path = root / variant / "reference.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                if mode is not None:
                    rows[0]["configuration"]["flashAttention"] = mode
                path.write_text("\n".join(map(json.dumps, rows)))
                if mode == "auto":
                    self.assertTrue(compare.evaluate(corpus, manifest, root, variant, Evaluator())["taskQualified"])
                else:
                    with self.assertRaisesRegex(ValueError, "frozen AUTO"):
                        compare.evaluate(corpus, manifest, root, variant, Evaluator())


if __name__ == "__main__":
    unittest.main()
