#!/usr/bin/env python3
"""Rescore committed raw observations without loading model weights or image pixels."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from compare import evaluate, read, require, sha


def main():
    root, scorer = map(Path, sys.argv[1:])
    script = Path(__file__).resolve().parent
    manifest = read(script / 'manifest.json')
    require(sha(scorer) == manifest['scorer']['sha256'], 'Upstream scorer digest mismatch')
    corpus = read(root / 'campaign/cases.json')
    receipt = read(root / 'campaign/results/receipt.json')
    require(receipt['corpusSha256'] == sha(root / 'campaign/cases.json'), 'Corpus changed')
    require(receipt['manifestSha256'] == corpus['manifestSha256'] == sha(script / 'manifest.json'), 'Protocol changed')
    spec = importlib.util.spec_from_file_location('upstream_scorer', scorer)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scorer = module.TextVQAAccuracyEvaluator()
    variants = [evaluate(corpus, manifest, root / 'campaign/results', variant, scorer)
                for variant in manifest['variants']]
    existing = read(root / 'campaign/results/task-verdict.json')
    require(existing['receipt'] == receipt and existing['variants'] == variants, 'Task verdict differs from raw observations')
    # finalize writes a new report. Preserve and compare to the recorded bytes afterward.
    destination = root / 'campaign/results/admission-attempt.json'
    saved = destination.read_bytes()
    try:
        result = subprocess.run([sys.executable, str(script / 'finalize.py'), str(root / 'campaign'),
                                 str(root / 'integration'), str(root / 'integration-confirmation-3w7r')],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        require(result.returncode in (0, 1) and destination.read_bytes() == saved,
                'Admission verdict differs or finalizer failed: ' + result.stderr)
        require(json.loads(result.stdout) == json.loads(saved), 'Finalizer did not produce the expected report')
    finally:
        destination.write_bytes(saved)
    print('Task scores, paired timings and admission verdict reproduce from the recorded raw observations.')
    print('Numerical vector comparisons are retained in the separate checksummed archive; no inference ran.')


if __name__ == '__main__':
    main()
