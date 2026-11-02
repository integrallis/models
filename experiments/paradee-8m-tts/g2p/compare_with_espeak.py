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


SIBILANTS = "szʃʒʧʤ"
VOICELESS = "ptkfθ"


def dictionary_lookup(dictionary, word):
    # The same rules as LetterToPhonemeFallback.lookup: the word, else a possessive or plural of one.
    w = word.lower()
    if w in dictionary:
        return dictionary[w]
    base = None
    if w.endswith("'s"): base = dictionary.get(w[:-2])
    elif w.endswith("s'"): base = dictionary.get(w[:-2])
    elif w.endswith("es"):
        stem = dictionary.get(w[:-2])
        if stem and stem[-1] in SIBILANTS: return stem + "ᵻz"
        base = dictionary.get(w[:-1])
    elif w.endswith("s"): base = dictionary.get(w[:-1])
    if not base: return None
    return base + ("ᵻz" if base[-1] in SIBILANTS else "s" if base[-1] in VOICELESS else "z")


def main(out_dir, report_dir, model_dir=None):
    out_dir, report_dir = pathlib.Path(out_dir), pathlib.Path(report_dir)
    model_dir = pathlib.Path(model_dir) if model_dir else out_dir
    result = {}
    for accent, british in (("us", False), ("gb", True)):
        card = json.load(open(model_dir / f"g2p-{accent}.json"))
        dictionary = json.load(open(out_dir / f"cmudict-{accent}.json")) if (out_dir / f"cmudict-{accent}.json").exists() else {}
        letter_ix = {c: i + 1 for i, c in enumerate(card["letters"])}
        symbols = card["symbols"]
        sess = ort.InferenceSession(str(model_dir / f"g2p-{accent}.onnx"), providers=["CPUExecutionProvider"])
        fallback = espeak.EspeakFallback(british=british)
        words = []
        for corpus in ("wikitext103-heldout-200", "harvard-200"):
            rep = json.load(open(report_dir / f"{corpus}-{accent}.json"))
            for row in rep["rows"]:
                for w in row["unknownWords"]:
                    if re.fullmatch(r"[A-Za-z']+", w):
                        words.append((corpus, w))
        rows = []; exact = 0; edits = 0; total = 0; by_source = {"dictionary": [0, 0, 0, 0], "model": [0, 0, 0, 0]}
        for corpus, w in words:
            ref, _ = fallback(Tok(w))
            ours = dictionary_lookup(dictionary, w)
            source = "dictionary" if ours else "model"
            if ours is None:
                ids = [letter_ix[c] for c in w.lower() if c in letter_ix]
                logits = sess.run(["logits"], {"letters": np.array([ids], dtype=np.int64)})[0][0]
                path = logits.argmax(-1); prev = 0; out = []
                for k in path:
                    if k != prev and k != 0: out.append(symbols[k - 1])
                    prev = k
                ours = "".join(out)
            same = ours == ref; d = levenshtein(ours, ref or ""); exact += same; edits += d; total += len(ref or "")
            b = by_source[source]; b[0] += 1; b[1] += same; b[2] += d; b[3] += len(ref or "")
            rows.append({"corpus": corpus, "word": w, "espeak": ref, "ours": ours, "source": source, "same": same, "edits": d})
        result[accent] = {"words": len(words), "exactMatch": exact / len(words) if words else None,
                          "phonemeErrorRateVsEspeak": edits / total if total else None,
                          "bySource": {k: {"words": v[0], "exactMatch": v[1] / v[0] if v[0] else None, "phonemeErrorRateVsEspeak": v[2] / v[3] if v[3] else None} for k, v in by_source.items()},
                          "rows": rows}
        print(f"{accent}: {len(words)} unresolved words, exact match with eSpeak {exact / len(words):.3f}, PER vs eSpeak {edits / total:.3f}; " + ", ".join(f"{k}: {v[0]} words exact {v[1]/v[0] if v[0] else 0:.3f} PER {v[2]/v[3] if v[3] else 0:.3f}" for k, v in by_source.items()), file=sys.stderr)
    json.dump(result, sys.stdout, ensure_ascii=False, indent=1); print()


if __name__ == "__main__":
    main(*sys.argv[1:4])
