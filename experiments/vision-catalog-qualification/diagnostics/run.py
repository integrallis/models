#!/usr/bin/env python3
"""Run the crossed encoder/decoder diagnostic after all qualification timing has ended."""
import hashlib
import json
import os
import platform
import subprocess
import sys
import zipfile
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(command, log):
    with log.open('w') as stream:
        subprocess.run(list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    if platform.system() != 'Linux':
        raise SystemExit('Remote Linux only')
    root, campaign, distribution, reference, output = map(Path, sys.argv[1:])
    # Never run alongside qualification workloads, including a failed or unfinished repeat.
    for marker in [root / 'COLLECTION_COMPLETE', campaign / 'results/COLLECTION_COMPLETE']:
        if not marker.is_file():
            raise ValueError('Qualification has not completed: ' + str(marker))
    output.mkdir(exist_ok=False)
    receipt = read(campaign / 'results/receipt.json')
    cases = {c['id']: c for c in read(campaign / 'cases.json')['cases'] if not c['blank']}
    task = read(campaign / 'results/task-verdict.json')
    source = Path(__file__).with_name('DecoderAblation.java')
    java = Path(os.environ['VISION_JAVA'])
    native = Path(os.environ['VISION_NATIVE_LIBRARY'])
    if sha(native) != receipt['nativeSha256']:
        raise ValueError('Diagnostic native library differs from the measured binary')
    jars = sorted(p for p in (distribution / 'lib').glob('*.jar') if not p.name.startswith('._'))
    if not jars or any(not zipfile.is_zipfile(p) or sha(p) != receipt['jars'][p.name] for p in jars):
        raise ValueError('Diagnostic classpath differs from the measured executable JARs')
    classpath = ':'.join(map(str, jars))
    classes = output / 'classes'
    classes.mkdir()
    run([java.with_name('javac'), '-cp', classpath, '-d', classes, source], output / 'compile.log')
    common = [java, '--add-modules=jdk.incubator.vector', '--enable-native-access=ALL-UNNAMED', '-Xmx12g',
              '-Dvectors.gguf.threads=8', '-Dmodels.purejava.maxContextLength=1024',
              f'-Dmodels.native.kernels.library={native}', f'-Dmodels.qualification.revision={receipt["revision"]}',
              '-cp', classpath + ':' + str(classes)]
    metadata = dict(receipt=receipt, sourceSha256=sha(source), scriptSha256=sha(Path(__file__)),
                    executableJars=[p.name for p in jars], results=[])
    for variant in task['variants']:
        quant = variant['variant']
        model = campaign / 'artifacts' / f'llava-v1.5-7b-{quant}.gguf'
        for identity in variant['disagreements']:
            if identity not in cases:
                continue
            directory = output / quant / identity
            directory.mkdir(parents=True)
            candidate = campaign / 'results' / quant / 'candidate' / identity
            native_encoder = output / 'encoders' / identity
            if not native_encoder.exists():
                native_encoder.parent.mkdir(exist_ok=True)
                run([*common, 'com.integrallis.models.bench.VisionQualificationCli',
                     campaign / 'artifacts/llava-v1.5-7b-mmproj-model-f16.gguf',
                     campaign / cases[identity]['image'], native_encoder, '1', '3'],
                    native_encoder.parent / (identity + '.log'))
            vectors = {'models': native_encoder / 'embeddings.f32',
                       'reference': campaign / 'results/encoder' / identity / 'embeddings.f32'}
            tsv = directory / 'reference-cases.tsv'
            tsv.write_text(''.join('\t'.join([name, str(embedding), str(candidate / 'prefix.i32'),
                                              str(candidate / 'suffix.i32'), '0', '1']) + '\n'
                                    for name, embedding in vectors.items()))
            for name, embedding in vectors.items():
                run([*common, 'DecoderAblation', model, candidate / 'prefix.i32', candidate / 'suffix.i32',
                     embedding, directory / f'models-decoder-{name}-encoder.json'], directory / f'models-{name}.log')
            with (directory / 'reference.jsonl').open('w') as stdout, (directory / 'reference.log').open('w') as stderr:
                subprocess.run([str(reference / 'task-oracle'), str(model), str(tsv), '8'],
                               stdout=stdout, stderr=stderr, check=True)
            rows = [json.loads(line) for line in (directory / 'reference.jsonl').read_text().splitlines()][1:]
            result = dict(variant=quant, caseId=identity, embeddings={name: sha(path) for name, path in vectors.items()}, cells={})
            original = next(c for c in variant['cases'] if c['id'] == identity)
            for name in vectors:
                ours = read(directory / f'models-decoder-{name}-encoder.json')
                other = next(r for r in rows if r['id'] == name)
                result['cells'][f'modelsDecoder_{name}Encoder'] = dict(tokens=ours['tokens'], answer=ours['answer'])
                result['cells'][f'referenceDecoder_{name}Encoder'] = dict(tokens=other['tokens'], answer=bytes.fromhex(other['answerHex']).decode())
            result['diagonalAnswersReproduce'] = (
                result['cells']['modelsDecoder_modelsEncoder']['answer'] == original['candidateAnswer'] and
                result['cells']['referenceDecoder_referenceEncoder']['answer'] == original['referenceAnswer'])
            metadata['results'].append(result)
            (output / 'report.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (output / 'report.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
