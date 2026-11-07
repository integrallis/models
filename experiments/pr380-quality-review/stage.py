#!/usr/bin/env python3
"""Stage exactly the published Java modules scored by the pinned MFCQI workflow."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

checkout, stage = map(Path, sys.argv[1:])
modules = ['models-api', 'models-runtime', 'models-rag', 'models-semantic-order', 'backend-java',
           'backend-native', 'models-langchain4j', 'models-spring-ai', 'models-spring-boot-starter', 'models-embedding']
stage.mkdir(parents=True, exist_ok=False)
for module in modules:
    source = checkout / module / 'src/main/java'
    for path in source.rglob('*.java'):
        target = stage / path.relative_to(source)
        if target.exists():
            raise ValueError('Duplicate staged path: ' + str(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
manifest = {str(p.relative_to(stage)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(stage.rglob('*.java'))}
(stage.parent / (stage.name + '-manifest.json')).write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(dict(modules=modules, files=len(manifest))))
