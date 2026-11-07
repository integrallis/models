#!/usr/bin/env python3
"""Run the external decoder on the candidate's exact prompt and oracle patch embeddings."""
import argparse
import hashlib
import json
import platform
import resource
import socket
import subprocess
from pathlib import Path
import numpy as np

LLAMA_REVISION = '2cbca389f515df455412dafa07e2b1359cc72a6d'


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def run(args):
    if platform.system() != 'Linux':
        raise ValueError('Real-model measurements require remote Linux')
    candidate = json.loads((args.candidate / 'report.json').read_text())
    encoder = json.loads((args.encoder / 'report.json').read_text())
    for key in ('host', 'threads', 'mmprojSha256', 'imageSha256', 'tokens', 'dimension', 'warmups', 'repetitions'):
        if candidate[key] != encoder[key]:
            raise ValueError('Incomparable ' + key)
    if candidate['host'] != socket.gethostname() or candidate['textSha256'] != sha(args.model):
        raise ValueError('Wrong reference host or text artifact')
    revision = subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(args.source), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if revision != LLAMA_REVISION or dirty:
        raise ValueError('Reference requires the clean pinned llama.cpp checkout')
    args.output.mkdir(parents=True, exist_ok=False)
    for key in ('prefixTokens', 'suffixTokens'):
        np.asarray(candidate[key], dtype='<i4').tofile(args.output / (key + '.i32'))
    argv = [str(args.binary.resolve()), str(args.model.resolve()), str((args.encoder / 'embeddings.f32').resolve()),
            str((args.output / 'prefixTokens.i32').resolve()), str((args.output / 'suffixTokens.i32').resolve()),
            str(candidate['threads']), str(candidate['warmups']), str(candidate['repetitions']), '512', args.kv]
    with open(args.output / 'raw.jsonl', 'w') as out, open(args.output / 'stderr.log', 'w') as err:
        subprocess.run(argv, cwd=args.output, stdout=out, stderr=err, check=True)
    rows = [json.loads(line) for line in (args.output / 'raw.jsonl').read_text().splitlines()]
    configuration = rows[0]['configuration']
    measured = rows[1:]
    if [row['run'] for row in measured] != list(range(candidate['repetitions'])):
        raise ValueError('Missing/reordered reference repetitions')
    report = {key: candidate[key] for key in ('host', 'threads', 'textSha256', 'mmprojSha256', 'imageSha256',
              'warmups', 'repetitions', 'prefixTokens', 'suffixTokens')}
    report.update(schema='vision-decoder-reference-v1', referenceRevision=revision, binarySha256=sha(args.binary),
                  embeddingsSha256=sha(args.encoder / 'embeddings.f32'), command=argv, configuration=configuration,
                  librarySha256={p.name: sha(p) for p in sorted((args.source / 'build/bin').glob('*.so*')) if p.is_file()},
                  maxRssKiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, runs=measured)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'source', 'model', 'candidate', 'encoder', 'output'):
        p.add_argument(name, type=Path)
    p.add_argument('--kv', choices=['f32', 'f16'], default='f32')
    run(p.parse_args())
