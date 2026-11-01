#!/usr/bin/env python3
"""One pass of the comparator arm: the publisher's own package on the pinned artifact.

usage: comparator_arm.py --corpus ../corpus/X.json --voice af_heart --threads 1 --pass 1 --out DIR [--fp32]

Writes DIR/NNN.wav + NNN.txt per sentence and DIR/pass.json with per-sentence timing. The
real-time multiple follows training/scripts/onnx_eval.py: one untimed warm-up on the first
sentence, then sum(audio seconds) / sum(session.run seconds). Per-sentence real-time factor
(run / audio, the catalog's convention) is kept for p95.
"""
import argparse, json, pathlib, platform, sys, time
import numpy as np, soundfile as sf
import onnxruntime as ort
import pins

pins.prepare_espeak()
from paradee import Paradee  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True); ap.add_argument("--voice", required=True, choices=pins.VOICES)
    ap.add_argument("--threads", type=int, default=1); ap.add_argument("--pass", dest="pass_no", type=int, default=1)
    ap.add_argument("--out", required=True); ap.add_argument("--fp32", action="store_true")
    a = ap.parse_args()
    corpus = pins.load_corpus(a.corpus)
    model_name = "onnx/paradee.onnx" if a.fp32 else "onnx/paradee_int8.onnx"
    model_path, config_path = pins.artifact(model_name), pins.artifact("config.json")
    tts = Paradee(version="1.2", quantized=not a.fp32, model_path=str(model_path), config_path=str(config_path), threads=a.threads)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    rows = []
    sentences = corpus["sentences"]
    # Warm-up: the publisher's protocol runs the first sentence once untimed.
    for ps in tts._chunks(sentences[0], a.voice):
        tts.generate_from_phonemes(ps, 1.0, a.voice)
    for i, text in enumerate(sentences):
        t0 = time.perf_counter()
        chunks = list(tts._chunks(text, a.voice))
        g2p_seconds = time.perf_counter() - t0
        parts, run_seconds, ids_lengths, first_chunk_seconds = [], 0.0, [], None
        for ps in chunks:
            ids = [0] + [tts.vocab[c] for c in ps if c in tts.vocab][:510] + [0]
            ids_lengths.append(len(ids))
            t1 = time.perf_counter()
            y = tts.generate_from_phonemes(ps, 1.0, a.voice)
            dt = time.perf_counter() - t1
            run_seconds += dt
            if first_chunk_seconds is None:
                first_chunk_seconds = dt
            parts.append(y)
        audio = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        finite = bool(np.isfinite(audio).all())
        sf.write(out / f"{i:03d}.wav", audio, pins.SAMPLE_RATE, subtype="PCM_16")
        (out / f"{i:03d}.txt").write_text(text + "\n", encoding="utf-8")
        audio_seconds = len(audio) / pins.SAMPLE_RATE
        rows.append({
            "index": i, "text": text, "phonemes": chunks, "idsLengths": ids_lengths,
            "audioSeconds": audio_seconds, "runSeconds": run_seconds, "g2pSeconds": g2p_seconds,
            "timeToFirstAudioSeconds": (g2p_seconds + (first_chunk_seconds or 0.0)),
            "realTimeFactor": (run_seconds / audio_seconds) if audio_seconds else None,
            "finite": finite, "samples": int(len(audio)),
        })
        print(f"{i:03d} {audio_seconds:5.2f}s audio {run_seconds*1000:7.1f} ms run  rtf {rows[-1]['realTimeFactor']:.4f}", file=sys.stderr)
    total_audio = sum(r["audioSeconds"] for r in rows); total_run = sum(r["runSeconds"] for r in rows)
    report = {
        "arm": "comparator", "runtime": "paradee package @ aae3cc46657bc69c6dbf9573bba556fb233f6bc0 on onnxruntime",
        "onnxruntimeVersion": ort.__version__, "intraOpThreads": a.threads, "interOpThreads": 1,
        "model": model_name, "modelSha256": pins.FILES[model_name], "revision": pins.REVISION,
        "voice": a.voice, "voiceIndex": pins.VOICES.index(a.voice), "corpus": corpus["name"], "corpusSource": corpus.get("source"),
        "pass": a.pass_no, "python": platform.python_version(), "machine": platform.platform(),
        "sentences": len(rows), "totalAudioSeconds": total_audio, "totalRunSeconds": total_run,
        "realTimeMultiple": (total_audio / total_run) if total_run else None,
        "allFinite": all(r["finite"] for r in rows), "rows": rows,
    }
    (out / "pass.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{corpus['name']} {a.voice} t{a.threads} pass{a.pass_no}: {total_audio:.1f}s audio in {total_run:.2f}s -> {report['realTimeMultiple']:.1f}x realtime", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
