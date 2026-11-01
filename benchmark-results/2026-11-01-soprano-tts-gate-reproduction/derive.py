#!/usr/bin/env python3
"""Derive the performance block of a speech-oracle-streaming-latency-v1 report from raw trial logs.

The two Soprano qualification reports under `benchmark-results/audio/soprano-1.1-80m-java/`
carry a `performance` block in this exact shape, and nothing in the repository computes it:
`grep -rn p95RealTimeFactor` matches the two reports and one fixture, and no script. So the
numbers in those reports were transcribed by hand from the test's stdout. This script closes
that gap for the half of the gate the integration test can measure. It does not touch the
oracle block -- PCM cosine and signal-to-difference come from a separate oracle comparison.

    python3 derive.py trials-*.log
"""

import json
import pathlib
import re
import statistics
import sys

LINE = re.compile(
    r"SOPRANO_QUALIFICATION_TRIAL backend=(?P<backend>\S+) trial=(?P<trial>\d+) "
    r"ttfaMillis=(?P<ttfa>[\d.]+) totalMillis=(?P<total>[\d.]+) audioMillis=(?P<audio>\d+) "
    r"realTimeFactor=(?P<rtf>[\d.]+) chunks=(?P<chunks>\d+) blockingCosine=(?P<cosine>[\d.]+)"
)


def percentile95(values: list[float]) -> float:
    """The reports' own convention: with five trials the 95th percentile is the maximum."""
    return max(values)


def main(paths: list[str]) -> int:
    runs: dict[str, list[dict]] = {}
    for path in paths:
        # Key on the log file, not on the backend, so two passes of the same backend stay
        # separate. The across-pass spread is the point: one pass cannot carry a verdict.
        key = pathlib.Path(path).stem.removeprefix("trials-")
        with open(path, encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                match = LINE.search(raw)
                if not match:
                    continue
                fields = match.groupdict()
                runs.setdefault(key, []).append(
                    {
                        "trial": int(fields["trial"]),
                        "ttfaMillis": float(fields["ttfa"]),
                        "totalMillis": float(fields["total"]),
                        "audioMillis": int(fields["audio"]),
                        "realTimeFactor": float(fields["rtf"]),
                        "chunks": int(fields["chunks"]),
                        "blockingPcmCosine": float(fields["cosine"]),
                    }
                )
    if not runs:
        print("no SOPRANO_QUALIFICATION_TRIAL lines found", file=sys.stderr)
        print("the test prints to stdout, which Gradle shows only at -i", file=sys.stderr)
        return 1

    report = {}
    for backend, trials in sorted(runs.items()):
        trials.sort(key=lambda t: t["trial"])
        factors = [t["realTimeFactor"] for t in trials]
        ttfas = [t["ttfaMillis"] for t in trials]
        cosines = [t["blockingPcmCosine"] for t in trials]
        report[backend] = {
            "trials": len(trials),
            "p95RealTimeFactor": round(percentile95(factors), 6),
            "medianRealTimeFactor": round(statistics.median(factors), 6),
            "p95TimeToFirstAudioMillis": round(percentile95(ttfas), 3),
            "medianTimeToFirstAudioMillis": round(statistics.median(ttfas), 3),
            "minimumBlockingPcmCosine": min(cosines),
            "firstAudioBeforeCompletion": all(t["ttfaMillis"] < t["totalMillis"] for t in trials),
            "audioMillis": sorted({t["audioMillis"] for t in trials}),
            "chunks": sorted({t["chunks"] for t in trials}),
            "measurements": trials,
        }

    passes = sorted(k for k in report if k.startswith("rust-ffm"))
    if len(passes) > 1:
        factors = [report[k]["p95RealTimeFactor"] for k in passes]
        report["acrossPassSpread"] = {
            "note": "same artifact, same backend, same machine, same five-trial protocol",
            "passes": passes,
            "p95RealTimeFactor": factors,
            "worstOverBest": round(max(factors) / min(factors), 3),
        }

    if "pure-java" in report and passes:
        # One ratio per rust-ffm pass, because the pass spread above moves the ratio too.
        # A single figure here would read as more settled than the data is.
        report["backendRatio"] = {
            "note": "same artifact, same machine, same five-trial protocol",
            "medianRealTimeFactorPureJavaOverRustFfm": {
                key: round(
                    report["pure-java"]["medianRealTimeFactor"]
                    / report[key]["medianRealTimeFactor"],
                    3,
                )
                for key in passes
            },
            "medianTimeToFirstAudioPureJavaOverRustFfm": {
                key: round(
                    report["pure-java"]["medianTimeToFirstAudioMillis"]
                    / report[key]["medianTimeToFirstAudioMillis"],
                    3,
                )
                for key in passes
            },
        }

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["trials-pure-java.log", "trials-rust-ffm.log"]))
