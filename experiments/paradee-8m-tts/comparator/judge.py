#!/usr/bin/env python3
"""Whisper-base round-trip WER for every NNN.wav + NNN.txt pair in each directory given.

The protocol is training/scripts/onnx_eval.py verbatim: whisper.load_model("base"), audio resampled
24 kHz -> 16 kHz with torchaudio.functional.resample, transcribe(language="en", fp16=False), the
publisher's regex normalizer, jiwer.wer over all sentences. Whisper's own EnglishTextNormalizer is
reported beside it. Writes judge.json into each directory.

usage: judge.py DIR [DIR ...]
"""
import hashlib, json, pathlib, re, sys, warnings
warnings.filterwarnings("ignore")
import soundfile as sf, torch, torchaudio, whisper, jiwer
from whisper.normalizers import EnglishTextNormalizer

publisher_norm = lambda s: " ".join(re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split())


def model_digest(name: str) -> str:
    root = pathlib.Path.home() / ".cache" / "whisper"
    path = root / f"{name}.pt"
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "unknown"


def main(dirs):
    torch.set_num_threads(max(1, torch.get_num_threads()))
    model = whisper.load_model("base")
    whisper_norm = EnglishTextNormalizer()
    for d in dirs:
        d = pathlib.Path(d)
        refs_p, hyps_p, refs_w, hyps_w, rows = [], [], [], [], []
        for wav in sorted(d.glob("*.wav")):
            x, sr = sf.read(wav, dtype="float32")
            x = torchaudio.functional.resample(torch.tensor(x), sr, 16000).numpy()
            text = model.transcribe(x, language="en", fp16=False)["text"]
            ref = wav.with_suffix(".txt").read_text(encoding="utf-8").strip()
            rp, hp, rw, hw = publisher_norm(ref), publisher_norm(text), whisper_norm(ref), whisper_norm(text)
            refs_p.append(rp); hyps_p.append(hp); refs_w.append(rw); hyps_w.append(hw)
            m = jiwer.process_words(rp, hp)
            rows.append({"file": wav.name, "reference": ref, "hypothesis": text,
                         "errors": m.substitutions + m.deletions + m.insertions, "words": len(rp.split())})
        report = {
            "judge": "openai-whisper 20250625, model base", "modelSha256": model_digest("base"),
            "protocol": "training/scripts/onnx_eval.py: resample 24k->16k (torchaudio), transcribe(language=en, fp16=False), regex normalizer, jiwer.wer corpus-level",
            "sentences": len(rows), "words": sum(r["words"] for r in rows),
            "werPublisherNormalizer": jiwer.wer(refs_p, hyps_p) if rows else None,
            "werWhisperNormalizer": jiwer.wer(refs_w, hyps_w) if rows else None,
            "rows": rows,
        }
        (d / "judge.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{d}: WER {report['werPublisherNormalizer']*100:.2f}% (publisher norm) {report['werWhisperNormalizer']*100:.2f}% (whisper norm) over {len(rows)} sentences", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
