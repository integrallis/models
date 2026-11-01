#!/usr/bin/env python3
"""Copy pinned dependency notices for the six shipped native targets; no model downloads."""
import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/'backend-native/src/main/rust/model-kernels/Cargo.toml'
OUTPUT = ROOT/'backend-native/src/main/resources/META-INF/licenses/rust-dependencies.txt'
TARGETS = ['x86_64-unknown-linux-gnu', 'aarch64-unknown-linux-gnu',
           'x86_64-apple-darwin', 'aarch64-apple-darwin',
           'x86_64-pc-windows-msvc', 'aarch64-pc-windows-msvc']


def generate():
    selected = set()
    for target in TARGETS:
        output = subprocess.check_output(['cargo', 'tree', '--locked', '--manifest-path', str(MANIFEST),
            '--target', target, '--edges', 'normal,build', '--prefix', 'none', '--format', '{p}'], text=True)
        selected.update((m[1], m[2]) for line in output.splitlines() if (m := re.match(r'(\S+) v(\S+)', line)))
    metadata = json.loads(subprocess.check_output(['cargo', 'metadata', '--locked', '--format-version', '1',
                                                   '--manifest-path', str(MANIFEST)], text=True))
    parts = ['Rust native dependency license notices\n\nDependencies reachable on the six advertised native targets, including build-time crates.\nGenerated from the pinned Cargo.lock; license texts are copied verbatim from the crate archives.\n']
    for package in sorted(metadata['packages'], key=lambda p: (p['name'], p['version'])):
        if not package['source'] or (package['name'], package['version']) not in selected:
            continue
        root = Path(package['manifest_path']).parent
        files = sorted(f for f in root.iterdir() if f.is_file() and f.name.upper().startswith(
            ('LICENSE', 'LICENCE', 'COPYING', 'NOTICE', 'COPYRIGHT')))
        if package['name'] == 'onig_sys':
            files.append(root/'oniguruma/COPYING')
        if not files:
            raise ValueError('missing dependency license: '+package['name'])
        parts.append('\n'+'='*72+'\n'+package['name']+' '+package['version']+'\nSPDX: '+package['license']+
                     '\nRepository: '+str(package['repository'])+'\n')
        for file in files:
            parts.append('\n--- '+str(file.relative_to(root))+' ---\n'+file.read_text())
    return '\n'.join(parts)+'\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    generated = generate()
    if args.check:
        if OUTPUT.read_text() != generated:
            raise SystemExit('Rust dependency notices are stale; run scripts/generate-native-rust-notices.py')
        print('Rust dependency notices match the pinned native dependency graph')
    else:
        OUTPUT.write_text(generated)
