#!/usr/bin/env python3
"""Collect the existing frozen protocol sequentially, retaining failed verdicts."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('existing', type=Path)
    args = parser.parse_args()
    source, output, existing = [p.resolve() for p in (args.source, args.output, args.existing)]
    runtime = json.loads((output / 'runtime-receipt.json').read_text())
    revision = runtime['revision']
    assert subprocess.check_output(['git', '-C', source, 'rev-parse', 'HEAD'], text=True).strip() == revision
    protocol = source / 'experiments/vision-catalog-qualification'
    integration = source / 'experiments/vision-qualification'
    campaign = output / 'campaign'
    campaign.mkdir(exist_ok=False)
    (campaign / 'cases.json').write_bytes((existing / 'corpus/cases.json').read_bytes())
    for name in ('images', 'artifacts', 'datasets'):
        (campaign / name).symlink_to(existing / 'corpus' / name, target_is_directory=True)
    python = str(existing / 'venv/bin/python')
    env = dict(os.environ, VISION_JAVA='/usr/lib/jvm/temurin-25-jdk-amd64/bin/java',
               VISION_PYTHON=python, VISION_THREADS='8',
               VISION_NATIVE_LIBRARY=str(output / 'libjmodels_kernels.so'),
               VISION_ARTIFACT_DIRECTORY=str(existing / 'corpus/artifacts'),
               VISION_DECODER_REFERENCE_DIRECTORY=str(existing / 'integration/decoder-reference'))
    dist = output / 'distribution'
    reference = existing / 'reference'
    stages = []

    def run(name, command, allowed=(0,), extra=None):
        print(datetime.datetime.now(datetime.timezone.utc).isoformat(), name, flush=True)
        started = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with (output / f'{name}.stdout.log').open('w') as stdout, (output / f'{name}.stderr.log').open('w') as stderr:
            result = subprocess.run(list(map(str, command)), env=env | (extra or {}), stdout=stdout, stderr=stderr)
        stages.append(dict(stage=name, startedAt=started, exitCode=result.returncode))
        (output / 'stage-status.json').write_text(json.dumps(stages, indent=2) + '\n')
        if result.returncode not in allowed:
            raise RuntimeError(f'{name} failed with {result.returncode}; see retained logs')

    run('task', [python, protocol / 'run.py', campaign, dist, reference, revision])
    run('task-score', [python, protocol / 'compare.py', campaign, existing / 'm4c_evaluator.py'], allowed=(0, 1))
    assert (campaign / 'results/task-verdict.json').is_file(), 'Missing task verdict'
    corpus = json.loads((campaign / 'cases.json').read_text())
    selected = {c['id'] for c in corpus['cases'] if c['repetitions'] == 3}
    assert len(selected) == 4
    for variant in ('Q4_K_S', 'Q4_K_M'):
        destination = campaign / 'results' / variant
        lines = [line for line in (destination / 'reference-cases.tsv').read_text().splitlines()
                 if line.split('\t')[0] in selected]
        assert len(lines) == 4
        tsv = destination / 'reference-performance-f16.tsv'
        tsv.write_text('\n'.join(lines) + '\n')
        receipt = dict(host=socket.gethostname(), kvType='f16', threads=8, context=1024,
                       maximumOutputTokens=32, corpusSha256=sha(campaign / 'cases.json'),
                       tsvSha256=sha(tsv), binarySha256=sha(reference / 'task-oracle'),
                       referenceCommit=subprocess.check_output(['git', '-C', reference / 'source', 'rev-parse', 'HEAD'], text=True).strip())
        (destination / 'reference-performance-f16-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        run(f'f16-{variant}', [reference / 'task-oracle', campaign / 'artifacts' / f'llava-v1.5-7b-{variant}.gguf', tsv, '8', 'f16'])
        (destination / 'reference-performance-f16.jsonl').write_bytes((output / f'f16-{variant}.stdout.log').read_bytes())
        (destination / 'reference-performance-f16.log').write_bytes((output / f'f16-{variant}.stderr.log').read_bytes())
    for name, warmups, repetitions in [('integration', '2', '5'), ('integration-confirmation-3w7r', '3', '7')]:
        run(name, ['bash', integration / 'run.sh', dist, revision, output / name], allowed=(0, 1),
            extra=dict(VISION_WARMUPS=warmups, VISION_REPETITIONS=repetitions))
    run('admission', [python, protocol / 'finalize.py', campaign, output / 'integration', output / 'integration-confirmation-3w7r'], allowed=(0, 1))
    verdict = json.loads((campaign / 'results/admission-attempt.json').read_text())
    assert len(verdict['candidates']) == 2
    (output / 'COLLECTION_COMPLETE').write_text('Both arms and admission evaluation complete.\n')
    (output / 'COMPLETE').write_text('Full frozen admission collected; inspect each candidate verdict.\n')
    print(json.dumps([dict(variant=c['variant'], qualified=c['backendQualified'], gates=c['gates']) for c in verdict['candidates']], indent=2), flush=True)


if __name__ == '__main__':
    main()
