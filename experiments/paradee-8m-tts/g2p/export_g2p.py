#!/usr/bin/env python3
"""Export the best checkpoint train_g2p.py has saved so far: ONNX graph, card, held-out sample.

Rebuilds the same split (same seed, same ordering) and evaluates the checkpoint on it, so a run
can be stopped once accuracy plateaus without losing the export.

  .venv/bin/python ../g2p/export_g2p.py us OUT_DIR [--epochs-run N] [--seed S]
"""
import argparse, json, pathlib, random, sys, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from train_g2p import G2p, LETTERS, UPSAMPLE, encode_words, greedy_decode, levenshtein, load_words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("accent", choices=["us", "gb"]); ap.add_argument("out")
    ap.add_argument("--epochs-run", type=int, default=None); ap.add_argument("--seed", type=int, default=20261102)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    torch.manual_seed(a.seed); random.seed(a.seed)
    words = load_words(a.accent)
    symbols = sorted({c for ps in words.values() for c in ps})
    letter_ix = {c: i + 1 for i, c in enumerate(LETTERS)}
    items = sorted(words.items()); random.shuffle(items)
    dev_n = max(2000, len(items) // 50)
    dev, train = items[:dev_n], items[dev_n:]
    model = G2p(len(LETTERS), len(symbols))
    model.load_state_dict(torch.load(out / f"g2p-{a.accent}.pt")); model.eval()
    # One word per run, as the Java side and the phonemizer call it: a padded batch would let the
    # backward LSTM direction see padding first and change the real positions' outputs.
    correct = 0; edits = 0; total = 0; predictions = {}
    with torch.no_grad():
        for w, ref in dev:
            x, n = encode_words([w], letter_ix)
            h = greedy_decode(model(x), n, symbols)[0]
            predictions[w] = h
            correct += h == ref; total += len(ref); edits += levenshtein(h, ref)
    acc, per = correct / len(dev), edits / total
    dummy = torch.zeros(1, 8, dtype=torch.int64)
    torch.onnx.export(model, dummy, out / f"g2p-{a.accent}.onnx", opset_version=17, input_names=["letters"],
                      output_names=["logits"], dynamic_axes={"letters": {1: "length"}, "logits": {1: "frames"}})
    log = (out / f"train-{a.accent}.log").read_text().strip().splitlines() if (out / f"train-{a.accent}.log").exists() else []
    json.dump({"accent": a.accent, "letters": LETTERS, "symbols": symbols, "upsample": UPSAMPLE, "blank": 0,
               "trainingWords": len(train), "devWords": len(dev), "devWordAccuracy": acc, "devPhonemeErrorRate": per, "devEvaluation": "one word per run, greedy",
               "source": "misaki 0.9.4 gold+silver dictionaries (Apache-2.0), DEFAULT pronunciations, words of letters and apostrophes",
               "seed": a.seed, "epochsRun": a.epochs_run if a.epochs_run is not None else len(log), "trainingLog": log,
               "architecture": f"Embedding(96) x{UPSAMPLE} upsample, BiLSTM(128, 2 layers), Linear, CTC; greedy collapse",
               "license": "Apache-2.0"},
              open(out / f"g2p-{a.accent}.json", "w"), indent=1)
    sample = dev[:200]
    json.dump([{"word": w, "reference": ref, "model": predictions[w]} for w, ref in sample],
              open(out / f"g2p-{a.accent}-dev-sample.json", "w"), ensure_ascii=False, indent=0)
    print(f"exported {out / f'g2p-{a.accent}.onnx'}: dev word-accuracy {acc:.4f}, PER {per:.4f}, {(out / f'g2p-{a.accent}.onnx').stat().st_size} bytes")


if __name__ == "__main__":
    sys.exit(main())
