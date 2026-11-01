#!/usr/bin/env python3
"""Round-trip WER judge: transcribe synthesized audio and score it against the input text.

This validates the judge pipeline against Soprano, whose waveform parity is already proven by
`speech-oracle-streaming-latency-v1`. It is NOT a qualification: one 10-sentence set on one
machine scores the pipeline, not the model.

The judge is pinned by package version and by model identifier, because the judge is part of the
measurement. It is faster-whisper's `base` -- a CTranslate2 conversion of openai-whisper `base`,
not the same binary the Paradee card used -- and that difference is recorded rather than elided.
"""
import argparse, csv, json, pathlib, sys, wave

import numpy as np

WHISPER_RATE = 16_000


def load_mono_16k(path: pathlib.Path) -> np.ndarray:
    """Decode a 16-bit PCM wav to float32 mono at 16 kHz, without PyAV.

    faster-whisper's own file decoding goes through PyAV, and PyAV 19 dropped the
    `metadata_errors` argument faster-whisper 1.2.1 passes it, while `av<15` will not build
    here without ffmpeg headers. Passing a numpy array skips that path entirely, and it makes
    the resampling ours and stated rather than a library default: Soprano emits 32 kHz, so the
    ratio is exactly 2:1 and the filter is a 63-tap Hamming-windowed sinc at 0.5 Nyquist
    applied before decimation.
    """
    with wave.open(str(path), "rb") as handle:
        channels, width, rate, frames = (
            handle.getnchannels(), handle.getsampwidth(), handle.getframerate(),
            handle.getnframes())
        if width != 2:
            raise ValueError(f"{path}: expected 16-bit PCM, got {width * 8}-bit")
        raw = handle.readframes(frames)
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if rate == WHISPER_RATE:
        return samples
    if rate % WHISPER_RATE != 0:
        raise ValueError(f"{path}: {rate} Hz is not an integer multiple of {WHISPER_RATE}")
    factor = rate // WHISPER_RATE
    taps = 63
    n = np.arange(taps) - (taps - 1) / 2
    sinc = np.sinc(n / factor)
    kernel = (sinc * np.hamming(taps)).astype(np.float32)
    kernel /= kernel.sum()
    return np.convolve(samples, kernel, mode="same")[::factor].astype(np.float32)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=pathlib.Path, default=pathlib.Path("wav/manifest.tsv"))
    ap.add_argument("--model", default="base")
    ap.add_argument("--beam-size", type=int, default=5)
    args = ap.parse_args()

    from faster_whisper import WhisperModel
    from whisper_normalizer.english import EnglishTextNormalizer
    import jiwer

    normalizer = EnglishTextNormalizer()
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8"), delimiter="\t"))
    results, refs, hyps = [], [], []
    for row in rows:
        wav = args.manifest.parent / row["wav"]
        segments, _info = model.transcribe(
            load_mono_16k(wav), language="en", beam_size=args.beam_size)
        hypothesis = " ".join(s.text for s in segments).strip()
        reference, hypothesis_n = normalizer(row["text"]), normalizer(hypothesis)
        wer = jiwer.wer(reference, hypothesis_n) if reference else float("nan")
        refs.append(reference)
        hyps.append(hypothesis_n)
        results.append({
            "index": int(row["index"]), "wav": row["wav"],
            "realTimeFactor": float(row["realTimeFactor"]),
            "reference": reference, "hypothesis": hypothesis_n,
            "raw": hypothesis, "wer": wer,
        })
        print(f'{row["index"]}  WER {wer:6.3f}  "{hypothesis_n}"')

    corpus_wer = jiwer.wer(refs, hyps)
    out = {
        "judge": {
            "implementation": "faster-whisper",
            "model": args.model,
            "computeType": "int8",
            "beamSize": args.beam_size,
            "device": "cpu",
            "normalizer": "whisper-normalizer EnglishTextNormalizer",
            "note": "a CTranslate2 conversion of openai-whisper base, not that binary",
            "resampling": "ours: 63-tap Hamming-windowed sinc, then 2:1 decimation 32k -> 16k",
        },
        "sentences": len(results),
        "corpusWer": corpus_wer,
        "perSentenceWerMax": max(r["wer"] for r in results),
        "perSentenceWerZeroCount": sum(1 for r in results if r["wer"] == 0.0),
        "results": results,
    }
    pathlib.Path("judge-result.json").write_text(json.dumps(out, indent=2))
    print(f"\ncorpus WER {corpus_wer:.4f} over {len(results)} sentences; "
          f"{out['perSentenceWerZeroCount']} exact, worst {out['perSentenceWerMax']:.3f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
