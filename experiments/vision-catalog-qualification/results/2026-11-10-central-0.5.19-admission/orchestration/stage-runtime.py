#!/usr/bin/env python3
"""Stage Central runtime bytes with only the unpublished benchmark harness built from source."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request
import zipfile


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('revision')
    parser.add_argument('--version', default='0.5.19')
    parser.add_argument('--vectors-version', default='0.1.34')
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    actual = subprocess.check_output(['git', '-C', source, 'rev-parse', 'HEAD'], text=True).strip()
    assert actual == args.revision and len(actual) == 40, 'Source revision differs'
    assert not subprocess.check_output(['git', '-C', source, 'status', '--porcelain'], text=True).strip(), 'Dirty source'
    assert f'version = {args.version}' in (source / 'gradle.properties').read_text()
    original = source / 'models-bench/build/install/models-bench/lib'
    assert (original / f'models-bench-{args.version}.jar').is_file()
    output.mkdir(parents=True, exist_ok=False)
    library = output / 'distribution/lib'
    library.mkdir(parents=True)
    rows = []
    for jar in sorted(original.glob('*.jar')):
        if jar.name.startswith('models-kernels-'):
            continue  # The published backend-native JAR carries all native platforms.
        target = library / jar.name
        row = dict(file=jar.name)
        if jar.name.startswith('models-bench-'):
            shutil.copyfile(jar, target)
            row.update(origin='unpublished benchmark harness', revision=actual)
        elif jar.name.startswith(('models-', 'backend-', 'vectors-')):
            version = args.vectors_version if jar.name.startswith('vectors-') else args.version
            suffix = f'-{version}.jar'
            assert jar.name.endswith(suffix), f'Unexpected runtime version: {jar.name}'
            artifact = jar.name[:-len(suffix)]
            url = f'https://repo.maven.apache.org/maven2/com/integrallis/{artifact}/{version}/{jar.name}'
            with urllib.request.urlopen(url, timeout=120) as response, target.open('wb') as stream:
                shutil.copyfileobj(response, stream)
            row.update(origin='Maven Central', url=url, coordinate=f'com.integrallis:{artifact}:{version}')
        else:
            shutil.copyfile(jar, target)
            row.update(origin='third-party dependency resolved by Gradle')
        row.update(sha256=digest(target), sizeBytes=target.stat().st_size)
        rows.append(row)
    native_jar = library / f'backend-native-{args.version}.jar'
    member = 'META-INF/models/native/linux-x86_64/libjmodels_kernels.so'
    native = output / 'libjmodels_kernels.so'
    with zipfile.ZipFile(native_jar) as archive:
        native.write_bytes(archive.read(member))
        bridge = archive.read('com/integrallis/models/backend/nativekernel/NativeDebertaKernels.class')
        assert b'jmodels_vision_causal_attention_f16_store' in bridge, 'Published bridge does not select F16 storage'
    symbols = subprocess.check_output(['nm', '-D', native], text=True)
    assert 'jmodels_vision_causal_attention_f16_store' in symbols, 'Published native library lacks F16 storage'
    receipt = dict(schema='central-vision-runtime-v1', modelsVersion=args.version,
                   vectorsVersion=args.vectors_version, revision=actual, jars=rows,
                   native=dict(jar=native_jar.name, member=member, sha256=digest(native)),
                   note='Benchmark harness is source-built; all Integrallis runtime JARs are downloaded from Central. The task harness override names the unmodified native library extracted from the published JAR.')
    (output / 'runtime-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
