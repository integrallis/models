import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

import reference_fusion as ref

VOCAB, STEPS, WEIGHTS = 7, 3, [0.5, 0.2, 0.3]


def scalar_log_softmax(row):
    """Deliberately scalar (math-module) implementation, independent of the vectorised one."""
    m = max(row)
    z = math.log(sum(math.exp(v - m) for v in row)) + m
    return [v - z for v in row]


def scalar_rule(rule, rows, weights):
    lsm = [scalar_log_softmax(r) for r in rows]
    out = []
    for t in range(len(rows[0])):
        if rule == "poe":
            out.append(sum(w * l[t] for w, l in zip(weights, lsm)))
        elif rule == "mixture":
            out.append(math.log(sum(w * math.exp(l[t]) for w, l in zip(weights, lsm) if w > 0)))
        else:
            out.append(sum(w * r[t] for w, r in zip(weights, rows)))
    return out


def write_dump(directory: Path, weights=WEIGHTS):
    rng = np.random.default_rng(7)
    members = ["A", "B", "C"]
    items = []
    for index in range(2):
        logits = {m: (rng.normal(size=(STEPS, VOCAB)) * 6).astype("<f4") for m in members}
        entry = {"id": f"item-{index}", "steps": STEPS, "members": {}, "fused": {}, "fusedRaw": {}}
        for m in members:
            name = f"item{index:03d}-{m}.f32"
            logits[m].tofile(directory / name)
            entry["members"][m] = name
        for rule in ref.RULES:
            raw = [scalar_rule(rule, [logits[m][s].astype(float).tolist() for m in members], weights)
                   for s in range(STEPS)]
            fused = [scalar_log_softmax(r) for r in raw]
            for kind, values in (("fusedRaw", raw), ("fused", fused)):
                name = f"item{index:03d}-{kind.lower()}-{rule}.f32"
                np.asarray(values, dtype="<f4").tofile(directory / name)
                entry[kind][rule] = name
        items.append(entry)
    manifest = {"schemaVersion": 1, "vocab": VOCAB, "members": members, "weights": weights,
                "rules": list(ref.RULES), "dtype": "float32-le", "items": items}
    (directory / "manifest.json").write_text(json.dumps(manifest))


class ReferenceFusionTest(unittest.TestCase):
    def run_main(self, *args):
        with redirect_stdout(io.StringIO()) as out:
            code = ref.main(list(args))
        return code, out.getvalue()

    def test_independent_dump_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_dump(d)
            code, out = self.run_main("--dump", tmp, "--report", str(d / "g2.json"))
            self.assertEqual(code, 0, out)
            self.assertTrue(out.startswith("PASS G2"))
            report = json.loads((d / "g2.json").read_text())
            self.assertTrue(report["passed"])
            self.assertEqual(report["itemCount"], 2)

    def test_zero_weight_member_is_excluded_from_mixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_dump(Path(tmp), weights=[1.0, 0.0, 0.0])
            code, out = self.run_main("--dump", tmp)
            self.assertEqual(code, 0, out)

    def test_perturbed_fused_file_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_dump(d)
            target = d / "item001-fused-mixture.f32"
            values = np.fromfile(target, dtype="<f4")
            values[4] += 1e-3
            values.tofile(target)
            code, out = self.run_main("--dump", tmp)
            self.assertEqual(code, 1)
            self.assertTrue(out.startswith("FAIL G2"))

    def test_wrong_shape_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_dump(d)
            np.zeros(5, dtype="<f4").tofile(d / "item000-A.f32")
            with self.assertRaises(ValueError):
                ref.check(d, 1e-4)

    def test_rules_match_scalar_math(self):
        rows = [[1.0, 2.0, -3.0], [0.5, 0.0, 4.0]]
        for rule in ref.RULES:
            got = ref.fuse(rule, [np.array([r]) for r in rows], [0.25, 0.75])[0]
            np.testing.assert_allclose(got, scalar_rule(rule, rows, [0.25, 0.75]), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
