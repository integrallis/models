#!/usr/bin/env python3
"""Train the out-of-vocabulary letter-to-phoneme fallback for the misaki port, one accent at a time.

Data: misaki 0.9.4's gold and silver dictionaries (Apache-2.0), DEFAULT pronunciation, words of
letters and apostrophes only, gold winning over silver. Model: character embedding, each letter
repeated UPSAMPLE times, a two-layer bidirectional LSTM, a linear layer over the phoneme symbols
plus the CTC blank. Trained with CTC so the exported graph has no decoder loop: the Java side runs
the graph once per word and collapses the argmax path. Exported to ONNX opset 17 for
models-backend-onnx; the symbol tables go beside it.

  .venv/bin/python ../g2p/train_g2p.py us OUT_DIR [--epochs N] [--threads T]
"""
import argparse, json, pathlib, random, re, sys, time
import torch, torch.nn as nn

HERE = pathlib.Path(__file__).resolve().parent
# misaki's dictionaries: the comparator venv's copy here, or wherever MISAKI_DATA points (a bench
# box installs misaki==0.9.4 from PyPI and points this at its data directory).
import os
DATA = pathlib.Path(os.environ.get("MISAKI_DATA") or (HERE.parent / "comparator" / ".venv" / "lib" / "python3.12" / "site-packages" / "misaki" / "data"))
UPSAMPLE = 2
LETTERS = list("abcdefghijklmnopqrstuvwxyz'")


def load_words(accent):
    words = {}
    for name in ("gold", "silver"):
        d = json.load(open(DATA / f"{accent}_{name}.json", encoding="utf-8"))
        for w, v in d.items():
            ps = v if isinstance(v, str) else (v.get("DEFAULT") if isinstance(v, dict) else None)
            if not ps:
                continue
            wl = w.lower()
            if not re.fullmatch(r"[a-z']+", wl) or wl in words:
                continue
            words[wl] = ps
    return words


class G2p(nn.Module):
    def __init__(self, letters, symbols, emb=96, hidden=128, layers=2, dropout=0.15):
        super().__init__()
        self.embedding = nn.Embedding(letters + 1, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, num_layers=layers, bidirectional=True, batch_first=True, dropout=dropout)
        self.out = nn.Linear(2 * hidden, symbols + 1)  # index 0 is the CTC blank

    def forward(self, letters, lengths=None):  # letters: int64 [batch, length]
        x = self.embedding(letters)
        x = x.repeat_interleave(UPSAMPLE, dim=1)
        if lengths is not None and letters.shape[0] > 1:
            # Training and batched evaluation: pack so the backward direction starts at each
            # word's own last letter rather than at the padding; the export path (batch 1) is
            # the plain call and sees exactly the same computation.
            packed = nn.utils.rnn.pack_padded_sequence(x, (lengths * UPSAMPLE).cpu(), batch_first=True, enforce_sorted=False)
            y, _ = self.lstm(packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(y, batch_first=True, total_length=x.shape[1])
        else:
            x, _ = self.lstm(x)
        return self.out(x)  # logits [batch, UPSAMPLE*length, symbols+1]


def encode_words(words, letter_ix):
    lengths = [len(w) for w in words]
    t = torch.zeros(len(words), max(lengths), dtype=torch.int64)
    for i, w in enumerate(words):
        t[i, :len(w)] = torch.tensor([letter_ix[c] for c in w])
    return t, torch.tensor(lengths)


def greedy_decode(logits, lengths, symbols):
    path = logits.argmax(-1)
    out = []
    for row, n in zip(path.tolist(), lengths.tolist()):
        prev, s = 0, []
        for k in row[: n * UPSAMPLE]:
            if k != prev and k != 0:
                s.append(symbols[k - 1])
            prev = k
        out.append("".join(s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("accent", choices=["us", "gb"]); ap.add_argument("out")
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=512); ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--extra", default=None, help="JSON of word -> phonemes added to the training split only (never to dev)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed); random.seed(a.seed)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    words = load_words(a.accent)
    symbols = sorted({c for ps in words.values() for c in ps})
    letter_ix = {c: i + 1 for i, c in enumerate(LETTERS)}
    symbol_ix = {c: i + 1 for i, c in enumerate(symbols)}
    items = sorted(words.items()); random.shuffle(items)
    dev_n = max(2000, len(items) // 50)
    dev, train = items[:dev_n], items[dev_n:]
    extra_n = 0
    if a.extra:
        extra = json.load(open(a.extra, encoding="utf-8"))
        dev_words = {w for w, _ in dev}
        for w, ps in extra.items():
            if w in dev_words or w in words or not re.fullmatch(r"[a-z']+", w) or not all(c in symbol_ix for c in ps):
                continue
            train.append((w, ps)); extra_n += 1
    print(f"{a.accent}: {len(words)} words, {len(symbols)} symbols, train {len(train)} (extra {extra_n}), dev {len(dev)}", flush=True)

    model = G2p(len(LETTERS), len(symbols))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * (len(train) // a.batch + 1))
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    best = None
    log = open(out / f"train-{a.accent}.log", "a")

    def evaluate():
        model.eval(); correct = 0; edits = 0; total = 0
        with torch.no_grad():
            for i in range(0, len(dev), 512):
                chunk = dev[i:i + 512]
                x, n = encode_words([w for w, _ in chunk], letter_ix)
                hyp = greedy_decode(model(x, n), n, symbols)
                for (w, ref), h in zip(chunk, hyp):
                    correct += h == ref; total += len(ref)
                    edits += levenshtein(h, ref)
        model.train()
        return correct / len(dev), edits / total

    step = 0
    for epoch in range(1, a.epochs + 1):
        random.shuffle(train); t0 = time.time(); loss_sum = 0.0; batches = 0
        for i in range(0, len(train), a.batch):
            chunk = train[i:i + a.batch]
            x, n = encode_words([w for w, _ in chunk], letter_ix)
            targets = [torch.tensor([symbol_ix[c] for c in ps]) for _, ps in chunk]
            tlen = torch.tensor([len(t) for t in targets])
            logits = model(x, n)
            logp = logits.log_softmax(-1).transpose(0, 1)  # [time, batch, classes]
            loss = ctc(logp, torch.cat(targets), n * UPSAMPLE, tlen)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); sched.step()
            loss_sum += loss.item(); batches += 1; step += 1
        acc, per = evaluate()
        line = f"epoch {epoch} loss {loss_sum / batches:.4f} dev word-accuracy {acc:.4f} per {per:.4f} ({time.time() - t0:.0f}s)"
        print(line, flush=True); log.write(line + "\n"); log.flush()
        if best is None or acc > best:
            best = acc
            torch.save(model.state_dict(), out / f"g2p-{a.accent}.pt")

    model.load_state_dict(torch.load(out / f"g2p-{a.accent}.pt")); model.eval()
    acc, per = evaluate()
    dummy = torch.zeros(1, 8, dtype=torch.int64)
    torch.onnx.export(model, dummy, out / f"g2p-{a.accent}.onnx", opset_version=17, input_names=["letters"],
                      output_names=["logits"], dynamic_axes={"letters": {1: "length"}, "logits": {1: "frames"}})
    json.dump({"accent": a.accent, "letters": LETTERS, "symbols": symbols, "upsample": UPSAMPLE,
               "blank": 0, "trainingWords": len(train), "devWords": len(dev), "devWordAccuracy": acc,
               "devPhonemeErrorRate": per, "source": "misaki 0.9.4 gold+silver dictionaries (Apache-2.0), DEFAULT pronunciations",
               "seed": a.seed, "epochs": a.epochs, "extraTrainingWords": extra_n, "extraSource": a.extra, "architecture": f"Embedding(96) x{UPSAMPLE} upsample, BiLSTM(128, 2 layers, packed sequences), Linear, CTC; greedy collapse"},
              open(out / f"g2p-{a.accent}.json", "w"), indent=1)
    # Dev predictions for the Java parity test: 200 held-out words with the model's own greedy output.
    sample = dev[:200]
    x, n = encode_words([w for w, _ in sample], letter_ix)
    with torch.no_grad():
        hyp = greedy_decode(model(x), n, symbols)
    json.dump([{"word": w, "reference": ref, "model": h} for (w, ref), h in zip(sample, hyp)],
              open(out / f"g2p-{a.accent}-dev-sample.json", "w"), ensure_ascii=False, indent=0)
    print(f"exported {out / f'g2p-{a.accent}.onnx'}: dev word-accuracy {acc:.4f}, PER {per:.4f}", flush=True)


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


if __name__ == "__main__":
    sys.exit(main())
