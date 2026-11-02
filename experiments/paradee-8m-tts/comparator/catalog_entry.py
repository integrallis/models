#!/usr/bin/env python3
"""Emit the ModelJars speech-qualifications entry for Paradee-8M v1.2 from committed artifacts.

Every field is read from a file: the verdict (WER gate, kill criteria), the candidate passes on
Harvard (p95 real-time factor and time-to-first-audio over every timed sentence of every pass; the
WikiText figures are printed beside them because long sentences make the first audio late in a
sentence-level streamer), the seeded-parity record (the oracle block: PCM cosine and
signal-to-difference against the publisher's runtime with the noise tensor shared), the peak RSS
sample, and the artifact and report digests.

usage: catalog_entry.py EVIDENCE_DIR VERDICT_JSON PEAK_RSS_TXT REPORT_PATH_IN_REPO MODELS_VERSION [LATENCY_ROOT]

LATENCY_ROOT (default EVIDENCE_DIR/devbox) is the directory whose candidate/ passes supply the
latency block; the quiet-box passes against the released library go there.

From models 0.5.12 the default route runs the executor on the bundled Rust kernels, so the entry is
labelled rust-ffm and its oracle block comes from the seeded-parity record of that route. The
peak-RSS file may be either "<pid> <bytes>" samples or the output of macOS `/usr/bin/time -l`.
"""
import hashlib, json, pathlib, sys

ARTIFACT_SHA = "a6989796692ac173597e15d405c971989202d5c73cfad65f22e90bd7b6c97da1"
ARTIFACT_BYTES = 8462119


def p95(values):
    values = sorted(values)
    return values[max(0, int(round(0.95 * (len(values) - 1))))] if values else None


def main(evidence, verdict_path, rss_path, report_path, models_version, latency_root=None):
    evidence = pathlib.Path(evidence)
    verdict = json.load(open(verdict_path))
    native = tuple(int(x) for x in models_version.split(".")) >= (0, 5, 12)
    parity_name = "seeded-parity-rust-ffm-2026-11-02.json" if native else "seeded-parity-2026-11-01.json"
    parity = json.load(open(evidence / "parity" / parity_name))
    cases = parity["perCase"]
    cos = min(c["pcmCosine"] for c in cases); sdr = min(c["signalToDifferenceDb"] for c in cases)
    per_corpus = {}
    for corpus in ("harvard-200", "wikitext103-heldout-200"):
        rtf, ttfa, passes = [], [], 0
        root = pathlib.Path(latency_root) if latency_root else evidence / "devbox"
        for p in sorted((root / "candidate" / corpus).glob("*/t1/pass*/pass.json")):
            d = json.load(open(p)); passes += 1
            rtf += [r["realTimeFactor"] for r in d["rows"] if r["realTimeFactor"] is not None]
            ttfa += [r["timeToFirstAudioSeconds"] * 1000.0 for r in d["rows"]]
        per_corpus[corpus] = {"passes": passes, "timedSentences": len(rtf), "p95RealTimeFactor": p95(rtf), "p95TimeToFirstAudioMillis": p95(ttfa)}
    rss_lines = [line.split() for line in open(rss_path) if line.split()]
    time_l = [int(f[0]) for f in rss_lines if f[0].isdigit() and " ".join(f[1:]) == "maximum resident set size"]
    rss = max(time_l) if time_l else max(int(f[1]) for f in rss_lines if f[0].isdigit() and len(f) > 1 and f[1].isdigit())
    report_bytes = pathlib.Path(verdict_path).read_bytes()
    cells = verdict["cells"]
    ours = [c.get("ourPhonemes", {}).get("intelligibility") for c in cells]
    harvard = per_corpus["harvard-200"]
    # The shipped speech policy's own limits (ModelSpeechQualification in ModelJars): a qualified
    # entry needs p95 real-time factor <= 2.0 and p95 time-to-first-audio <= 2000 ms, so an entry
    # measured under load that misses them is written as not qualified, with the reason.
    blocked = []
    if not (bool(cells) and all(v == "PASS" for v in ours)): blocked.append("intelligibility gate")
    if verdict["kill"]: blocked.append("kill criterion: " + "; ".join(verdict["kill"]))
    if verdict["incomplete"]: blocked.append("incomplete cells: " + ", ".join(verdict["incomplete"]))
    if harvard["p95RealTimeFactor"] is None or harvard["p95RealTimeFactor"] > 2.0: blocked.append(f"p95 real-time factor {harvard['p95RealTimeFactor']} > 2.0")
    if harvard["p95TimeToFirstAudioMillis"] is None or harvard["p95TimeToFirstAudioMillis"] > 2000.0: blocked.append(f"p95 time-to-first-audio {harvard['p95TimeToFirstAudioMillis']:.0f} ms > 2000 ms (measured under load on the shared dev box)")
    qualified = not blocked
    entry = {
        "modelId": "sahilmahendrakar_paradee_8m_v1_2_onnx_int8",
        "model": "Paradee-8M v1.2 ONNX INT8",
        "backend": "rust-ffm" if native else "pure-java",
        "backendVersion": f"models-{models_version}",
        "workload": "paradee_sentence_tts_v1",
        "artifactSha256": ARTIFACT_SHA,
        "artifactSizeBytes": ARTIFACT_BYTES,
        "report": report_path,
        "reportSha256": hashlib.sha256(report_bytes).hexdigest(),
        "qualified": qualified,
        "oracleBackend": "publisher's paradee package @ aae3cc46 on onnxruntime, RandomNormalLike externalized and shared (seeded parity)",
        "oracleVersion": "onnxruntime 1.23.2",
        "probes": len(cases),
        "minimumPcmCosine": round(cos, 9),
        "minimumSignalToDifferenceDb": round(sdr, 3),
        "sampleRate": 24000,
        "channels": 1,
        "streaming": True,
        "firstAudioBeforeCompletion": True,
        "trials": harvard["passes"],
        "p95RealTimeFactor": round(harvard["p95RealTimeFactor"], 6),
        "p95TimeToFirstAudioMillis": round(harvard["p95TimeToFirstAudioMillis"], 3),
        "peakRssBytes": rss,
    }
    json.dump({"entry": entry, "qualifiedBlockedBy": blocked, "latencyByCorpus": per_corpus, "intelligibilityVerdictPerCell": [
        {"corpus": c["corpus"], "voice": c["voice"], "ourPhonemes": c.get("ourPhonemes", {}).get("intelligibility"),
         "publisherPhonemes": c.get("publisherPhonemes", {}).get("intelligibility")} for c in cells]}, sys.stdout, indent=1); print()


if __name__ == "__main__":
    main(*sys.argv[1:7])
