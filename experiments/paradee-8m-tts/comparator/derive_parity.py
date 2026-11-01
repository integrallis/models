#!/usr/bin/env python3
"""Derive the seeded-parity record from the JUnit XML the integration test wrote. Nothing hand-typed.

usage: derive_parity.py evidence/parity/seeded-parity-DATE.junit.xml > evidence/parity/seeded-parity-DATE.json
"""
import json, re, sys, xml.etree.ElementTree as ET

LINE = re.compile(
    r"PARADEE_PARITY (?P<tag>\S+)\s+samples ours=(?P<ours>\d+) oracle=(?P<oracle>\d+) "
    r"cosine=(?P<cosine>[\d.]+) sdr=(?P<sdr>[-\d.]+|Infinity)dB maxAbs=(?P<maxAbs>[\d.]+) run=(?P<run>\d+)ms (?P<verdict>PASS|FAIL)")


def main(path):
    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    out = (suite.findtext("system-out") or "")
    cases = []
    for m in LINE.finditer(out):
        cases.append({
            "tag": m["tag"], "graph": "int8" if "_int8_" in m["tag"] else "fp32",
            "voice": re.sub(r"^paradee(_int8)?_|_\d$", "", m["tag"]), "sentence": int(m["tag"].rsplit("_", 1)[1]),
            "samplesOurs": int(m["ours"]), "samplesOracle": int(m["oracle"]),
            "pcmCosine": float(m["cosine"]), "signalToDifferenceDb": float(m["sdr"]), "maxAbsDifference": float(m["maxAbs"]),
            "runMillis": int(m["run"]), "pass": m["verdict"] == "PASS",
        })
    record = {
        "policyThresholds": {"minimumPcmCosine": 0.995, "minimumSignalToDifferenceDb": 20.0, "source": "speech-oracle-streaming-latency-v1"},
        "method": "RandomNormalLike output captured from the publisher's onnxruntime run and injected into the Java executor; identical artifact pair, identical ids, speed and voice",
        "suite": suite.get("name"), "timestamp": suite.get("timestamp"), "tests": int(suite.get("tests", 0)), "failures": int(suite.get("failures", 0)),
        "cases": len(cases), "allPass": all(c["pass"] for c in cases) and bool(cases),
        "worstPcmCosine": min((c["pcmCosine"] for c in cases), default=None),
        "worstSignalToDifferenceDb": min((c["signalToDifferenceDb"] for c in cases), default=None),
        "worstMaxAbsDifference": max((c["maxAbsDifference"] for c in cases), default=None),
        "sampleCountsMatch": all(c["samplesOurs"] == c["samplesOracle"] for c in cases),
        "perCase": cases,
    }
    json.dump(record, sys.stdout, indent=1); print()
    return 0 if record["allPass"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
