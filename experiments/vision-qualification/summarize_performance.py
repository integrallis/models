#!/usr/bin/env python3
"""Render performance evidence directly from the retained paired reports."""
import json
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
read = lambda path: json.loads(path.read_text())
probes = {name: read(root / name / 'verdict.json') for name in ('cats', 'wide-gradient', 'tall-bands')}
decoders = {kv: read(root / 'cats' / ('decoder-verdict-' + kv + '.json')) for kv in ('f32', 'f16')}
if not all(x['numericalPass'] and x['performancePass'] for x in [*probes.values(), *decoders.values()]):
    raise ValueError('Cannot report closure with failed numerical/performance gates')
a = decoders['f32']['candidate']
lines = [
    '# LLaVA 1.5 performance closure — measured 2026-11-07', '',
    'All three encoder probes and both paired decoder comparisons pass the frozen 1.10 maximum median-latency ratio and unchanged correctness gates. This closes the measured integration performance gap; it is not catalog admission or a VQA/OCR task score.', '',
    f"Candidate runtime commit: `{a['candidateRevision']}`. Host: `{a['host']}`; {a['threads']} threads in each arm; {a['warmups']} warmups and {a['repetitions']} measured runs. Models uses the original pinned GGUF weights, F32 KV, context {a['context']}, and its production image-prefill default. Reference settings and binary/library hashes are embedded in the decoder reports.", '',
    '| Probe | Models median ms | PyTorch median ms | Ratio | Cosine | Normalized RMSE | Pixel max error |',
    '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
]
for name, v in probes.items():
    lines.append(f"| {name} | {v['candidateMedianMs']:.3f} | {v['oracleMedianMs']:.3f} | {v['candidateToOracleLatencyRatio']:.4f} | {v['embeddings']['cosine']:.12f} | {v['embeddings']['normalizedRmse']:.8f} | {v['pixels']['maxAbs']:.8f} |")
lines += ['', 'Reference prefill below includes its separately measured PyTorch encoding/preprocessing plus llama.cpp decoder prefill. Models prefill measures the complete image/text call. Decode emits 32 greedy tokens (31 forward calls); every token matches the reference in every repetition.', '',
          '| Reference KV | Models prefill ms | Reference image + prefill ms | Prefill ratio | Models decode ms | Reference decode ms | Decode ratio | Total ratio |',
          '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
vision = probes['cats']
for kv, v in decoders.items():
    ref = v['reference']
    prefill = statistics.median(r['prefillMs'] for r in ref['runs']) + vision['oracleMedianMs'] + vision['oracle']['preprocessMs']
    lines.append(f"| {kv} | {statistics.median(a['imageTextPrefillMs']):.3f} | {prefill:.3f} | {v['imageTextPrefillRatio']:.4f} | {statistics.median(a['decodeMs']):.3f} | {statistics.median(r['decodeMs'] for r in ref['runs']):.3f} | {v['decodeRatio']:.4f} | {v['totalRatio']:.4f} |")
lines += ['', 'The F32 comparison matches cache precision; F16 also tests the faster stock reference default. Encoder vectors are checked independently against Transformers/PyTorch on the same converted weights. llama.cpp is an external comparator, never linked or bundled with Models.', '',
          'The historical pre-optimization run and its distinct worker/protocol remain in `../2026-11-07/`; no historical timing is used as the denominator of the paired closure gate.', '',
          'Qualification still requires frozen published VQA/OCR workloads, accuracy scoring, full-workload memory/cold-load measurements, and two-artifact catalog packaging. The three probes and one bounded caption establish integration fidelity and performance only.', '']
(root / 'README.md').write_text('\n'.join(lines))
