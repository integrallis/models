#!/usr/bin/env python3
"""Paired prompt/decode performance and greedy fidelity, without catalog admission."""
import argparse
import json
from pathlib import Path
import numpy as np
from compare import compare
from decoder_reference import sha


def compare_decoder(candidate, encoder, reference, limit=1.10):
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError('Invalid latency gate')
    vision = compare(candidate, encoder, limit)
    a = vision['candidate']
    b = json.loads((reference / 'report.json').read_text())
    for key in ('host', 'threads', 'textSha256', 'mmprojSha256', 'imageSha256', 'warmups', 'repetitions', 'prefixTokens', 'suffixTokens'):
        if a[key] != b[key]:
            raise ValueError('Incomparable decoder ' + key)
    if a['context'] != b['configuration']['context'] or a['maximumOutputTokens'] != b['configuration']['maximumOutputTokens'] or b['configuration']['threads'] != a['threads']:
        raise ValueError('Incomparable decoder settings')
    if b['embeddingsSha256'] != sha(encoder / 'embeddings.f32'):
        raise ValueError('Reference used different image embeddings')
    n = a['repetitions']
    if len(b['runs']) != n or len(a['generatedTokenRuns']) != n:
        raise ValueError('Incomplete decoder repetitions')
    times = [a['imageTextPrefillMs'], a['decodeMs'], [r['prefillMs'] for r in b['runs']], [r['decodeMs'] for r in b['runs']]]
    if any(len(t) != n or not all(np.isfinite(v) and v > 0 for v in t) for t in times):
        raise ValueError('Incomplete/nonfinite decoder timing')
    if any(not row['tokens'] for row in b['runs']):
        raise ValueError('Empty decoder output')
    tokens_match = all(a['generatedTokenRuns'][i] == b['runs'][i]['tokens'] for i in range(n))
    oracle_encoder = vision['oracleMedianMs'] + vision['oracle']['preprocessMs']
    prefill = float(np.median(times[0])) / (float(np.median(times[2])) + oracle_encoder)
    decode = float(np.median(times[1])) / float(np.median(times[3]))
    total = float(np.median(np.add(times[0], times[1]))) / (float(np.median(np.add(times[2], times[3]))) + oracle_encoder)
    x = np.fromfile(candidate / 'first-logits.f32', dtype='<f4').astype(np.float64)
    y = np.fromfile(reference / 'oracle-first-logits.f32', dtype='<f4').astype(np.float64)
    if x.size == 0 or x.size != y.size or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.linalg.norm(x) * np.linalg.norm(y):
        raise ValueError('Invalid decoder logits')
    return dict(encoderRatio=vision['candidateToOracleLatencyRatio'], imageTextPrefillRatio=prefill, decodeRatio=decode,
                totalRatio=total, maximumLatencyRatio=limit, greedyTokensMatch=tokens_match,
                logitCosine=float(x.dot(y) / (np.linalg.norm(x)*np.linalg.norm(y))),
                logitNormalizedRmse=float(np.linalg.norm(x-y)/np.linalg.norm(y)),
                numericalPass=vision['numericalPass'] and tokens_match,
                performancePass=vision['performancePass'] and max(prefill, decode, total) <= limit,
                referenceKvType=b['configuration']['kvType'], catalogQualified=False,
                candidate=a, reference=b)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'encoder', 'reference', 'output'):
        p.add_argument(name, type=Path)
    p.add_argument('--max-latency-ratio', type=float, default=1.10)
    args = p.parse_args()
    result = compare_decoder(args.candidate, args.encoder, args.reference, args.max_latency_ratio)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('candidate', 'reference')}))
    raise SystemExit(0 if result['numericalPass'] and result['performancePass'] else 1)
