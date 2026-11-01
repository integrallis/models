"""Shared artifact verification; no model inference code belongs here."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_snapshot(directory):
    manifest = json.loads((HERE / 'snapshot.json').read_text())
    for entry in manifest['files']:
        path = Path(directory) / entry['path']
        if path.stat().st_size != entry['sizeBytes'] or digest(path) != entry['sha256']:
            raise ValueError('snapshot identity mismatch: ' + entry['path'])
    return manifest


def read_cases(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [row['id'] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError('empty cohort or duplicate case ids')
    return rows
