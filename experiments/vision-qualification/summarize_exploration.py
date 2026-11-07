#!/usr/bin/env python3
"""Keep rejected approaches visible without retaining their production code."""
import json
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
e = root / 'exploration'
read = lambda p: json.loads(p.read_text())
q8 = read(e / 'q8-comparison.json')
ordered = read(e / 'packed128.json')
lanes = read(e / 'packed-lanes.json')
original = read(e / 'flat128.json')
assert ordered['generatedTokens'] != original['generatedTokens']
assert lanes['generatedTokens'] == original['generatedTokens']
lines = [
 '# Rejected approaches and retained contracts', '',
 'These exploratory runs selected the final candidate. They are not the final paired qualification distributions. The archive retains the experimental source, payloads, vectors, timings and profiling records. Temporary instrumentation and losing implementations are absent from production.', '',
 f"The old load-time Q8 conversion measured {q8['medianMs']:.3f} ms per encode on the photographic probe, with cosine {q8['cosine']:.8f} and normalized RMSE {q8['normalizedRmse']:.6f}. It failed the unchanged cosine/RMSE gates and was slower than the final unconverted path. Its conversion method, alternative encoder branch, unused weight copies, and synthetic-only benchmark were removed. `q8-payload-sha256.txt` identifies the exact experimental payload; `lib-lanes4-final.so` is the saved library used in that run.", '',
 f"The first packed implementation forced ordered block reduction in the image decoder. Its median prefill was {statistics.median(ordered['imageTextPrefillMs']):.3f} ms, but it changed the caption from ‘located’ to ‘positioned’. That policy was rejected. The retained packing preserves each original partial sum and the final reduction tree; its preliminary median was {statistics.median(lanes['imageTextPrefillMs']):.3f} ms and all caption tokens matched the original. The production defaults and final paired evidence are recorded separately in `README.md`.", '',
 'Both existing ordinary and explicitly ordered Q4 contracts remain supported. Packed-vs-original comparisons were bit-identical on actual GGUF matrices and synthetic ragged ranges, including untouched output sentinels. Rust tests cover both modes, rows, batch tails, signed scales, and dispatch fallbacks. The ordered mode remains for existing callers that already require it; image inference does not select it.', '',
 'A final query-tile sweep compared real Q4 gate weights, eight threads, two warmups and three measured calls. Shape is 128 queries × 4096 columns × 11008 output rows. Smaller tiles were rejected:', '',
 '| Query tile | Median ms |', '| --- | ---: |',
]
for n, file in [(1,'quant-probe-lanes1.jsonl'),(2,'quant-probe-lanes2.jsonl'),(4,'quant-probe-packed-lanes.jsonl')]:
 rows = [json.loads(line) for line in (e/file).read_text().splitlines()]
 r = next(r for r in rows if r['matrix']=='blk.0.ffn_gate.weight' and r['shape']==[128,4096,11008])
 lines.append(f"| {n} | {r['medianMs']:.6f} |")
lines += ['', 'Earlier dense dequantization, row-pair packing, larger tiles and alternate loop layouts were also profiled and rejected; their raw microbenchmarks and sources remain in the archive. Full-run measurements, not microbenchmarks alone, determine acceptance.', '']
(root / 'EXPERIMENTS.md').write_text('\n'.join(lines))
