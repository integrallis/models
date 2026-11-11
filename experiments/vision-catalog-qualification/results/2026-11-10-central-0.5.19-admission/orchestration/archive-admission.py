#!/usr/bin/env python3
"""Retain completed numerical evidence without model weights or runtime binaries."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('root', type=Path)
parser.add_argument('archive', type=Path)
args = parser.parse_args()
root = args.root.resolve()
assert (root / 'COMPLETE').is_file() and (root / 'exit-code').read_text().strip() == '0'
assert not args.archive.exists(), 'Refusing to replace a previous archive'
omitted = {'distribution', 'supervisor-started'}
with tarfile.open(args.archive, 'w:gz') as archive:
    def retain(info):
        path = Path(info.name)
        relative = path.parts[1:]
        if not relative or info.issym() or info.islnk():
            return None
        if relative[0] in omitted or path.suffix in {'.so', '.jar', '.gguf'}:
            return None
        if len(relative) > 1 and relative[0] == 'campaign' and relative[1] in {'artifacts', 'images', 'datasets'}:
            return None
        return info
    for path in sorted(root.iterdir()):
        archive.add(path, arcname=root.name + '/' + path.name, filter=retain)
with args.archive.open('rb') as stream:
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
args.archive.with_name(args.archive.name + '.sha256').write_text(digest + '  ' + args.archive.name + '\n')
print(json.dumps(dict(file=str(args.archive), bytes=args.archive.stat().st_size, sha256=digest), indent=2))
