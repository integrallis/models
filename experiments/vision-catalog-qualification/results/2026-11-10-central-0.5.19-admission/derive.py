#!/usr/bin/env python3
"""Recompute the admission verdict with the frozen finalizer and print its summary."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parent
expected = (root / 'admission-attempt.json').read_bytes()
assert expected == (root / 'campaign/results/admission-attempt.json').read_bytes()
with tempfile.TemporaryDirectory(prefix='vision-admission-') as temporary:
    campaign = Path(temporary) / 'campaign'
    shutil.copytree(root / 'campaign', campaign)
    result = subprocess.run([sys.executable, str(root.parent.parent / 'finalize.py'),
        str(campaign), str(root / 'integration'), str(root / 'integration-confirmation-3w7r')],
        capture_output=True, text=True)
    assert result.returncode in (0, 1), result.stderr
    assert (campaign / 'results/admission-attempt.json').read_bytes() == expected, 'Finalizer output changed'
report = json.loads(expected)
print('| Variant | Gates passed | Worst task latency ratio | Token agreement | Backend qualified |')
print('| --- | --- | --- | --- | --- |')
for candidate in report['candidates']:
    gates = candidate['gates']
    worst = max(value for case in candidate['performance'] for value in case['worstRatios'].values())
    print(f"| {candidate['variant']} | {sum(gates.values())}/{len(gates)} | {worst:.5f} | "
          f"{candidate['tokenAgreement']:.3f} | {str(candidate['backendQualified']).lower()} |")
print('\nShared Q4_K_M/cats integration confirmation decoder ratios:')
for decoder in report['confirmationIntegration']['decoders']:
    print(f"  {decoder['kv']}: decode {decoder['decodeRatio']:.5f}; "
          f"prefill {decoder['imageTextPrefillRatio']:.5f}; total {decoder['totalRatio']:.5f}; "
          f"tokens match {decoder['greedyTokensMatch']}")
