#!/usr/bin/env python3
"""How close the fallback model comes to the publisher's eSpeak fallback on the words that matter:
the words of both corpora that the lexicon path leaves unresolved (from evidence/phonemizer/).

For each such word, misaki's EspeakFallback (as the publisher's pipeline wires it) gives the
reference and the exported ONNX model, run through onnxruntime here, gives ours; reports exact
match and phoneme error rate per accent. The Java executor's parity with the ONNX file is a
separate test; this is about the model, not the port.

  .venv/bin/python ../g2p/compare_with_espeak.py ../g2p/out ../evidence/phonemizer > ../g2p/out/espeak-agreement.json
"""
import json, pathlib, re, sys
import numpy as np, onnxruntime as ort
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from train_g2p import UPSAMPLE, levenshtein
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "comparator"))
import pins
pins.prepare_espeak()
from misaki import en, espeak


class Tok:
    def __init__(self, text): self.text = text


def main(out_dir, report_dir):
    out_dir, report_dir = pathlib.Path(out_dir), pathlib.Path(report_dir)
    result = {}
    for accent, british in (("us", False), ("gb", True)):
        card = json.load(open(out_dir / f"g2p-{accent}.json"))
        letter_ix = {c: i + 1 for i, c in enumerate(card["letters"])}
        symbols = card["symbols"]
        sess = ort.InferenceSession(str(out_dir / f"g2p-{accent}.onnx"), providers=["CPUExecutionProvider"])
        fallback = espeak.EspeakFallback(british=british)
        words = []
        for corpus in ("wikitext103-heldout-200", "harvard-200"):
            rep = json.load(open(report_dir / f"{corpus}-{accent}.json"))
            for row in rep["rows"]:
                for w in row["unknownWords"]:
                    if re.fullmatch(r"[A-Za-z']+", w):
                        words.append((corpus, w))
        rows = []; exact = 0; edits = 0; total = 0
        for corpus, w in words:
            ref, _ = fallback(Tok(w))
            ids = [letter_ix[c] for c in w.lower() if c in letter_ix]
            logits = sess.run(["logits"], {"letters": np.array([ids], dtype=np.int64)})[0][0]
            path = logits.argmax(-1); prev = 0; ours = []
            for k in path:
                if k != prev and k != 0: ours.append(symbols[k - 1])
                prev = k
            ours = "".join(ours)
            same = ours == ref; exact += same; d = levenshtein(ours, ref or ""); edits += d; total += len(ref or "")
            rows.append({"corpus": corpus, "word": w, "espeak": ref, "model": ours, "same": same, "edits": d})
        result[accent] = {"words": len(words), "exactMatch": exact / len(words) if words else None,
                          "phonemeErrorRateVsEspeak": edits / total if total else None, "rows": rows}
        print(f"{accent}: {len(words)} unresolved words, exact match with eSpeak {exact / len(words):.3f}, PER vs eSpeak {edits / total:.3f}", file=sys.stderr)
    json.dump(result, sys.stdout, ensure_ascii=False, indent=1); print()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
