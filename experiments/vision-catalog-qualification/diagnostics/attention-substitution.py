#!/usr/bin/env python3
"""Remote, untimed component ablation; not qualification or a runtime backend."""
import ctypes
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(command, log):
    with log.open('w') as stream:
        subprocess.run(list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT, check=True)


def validate_bridge(library):
    """Analytic uniform-attention cases exercise packing, masking and split KV."""
    import numpy as np
    fn = ctypes.CDLL(str(library)).oracle_attention
    fn.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*6
    fn.restype = ctypes.c_int
    cases = []
    for start, batch in [(0, 1), (0, 9), (3, 9), (599, 1)]:
        heads, width, capacity = 2, 128, 1024
        query = np.zeros((batch, heads, width), dtype=np.float32)
        keys = np.zeros((heads, capacity, width), dtype=np.float32)
        values = np.zeros_like(keys)
        for h in range(heads):
            values[h, :, :] = h+1
        # Poison future values to prove the causal mask is applied.
        values[:, start+batch:, :] = 100
        out = np.empty_like(query)
        status = fn(*(x.ctypes.data for x in [query, keys, values, out]),
                    batch, start, heads, width, capacity, 8)
        assert status == 0, status
        expected = np.broadcast_to(np.array([1, 2], dtype=np.float32)[None, :, None], out.shape)
        assert np.max(np.abs(out-expected)) <= 2e-6, (start, batch, out)
        cases.append(dict(start=start, batch=batch, maximumError=float(np.max(np.abs(out-expected)))))
    return cases


def main():
    if platform.system() != 'Linux':
        raise SystemExit('Remote Linux only')
    root = Path('/opt/vision')
    # The parent worker writes this only after all queued qualification timing exits.
    if not Path('/home/ubuntu/attention-dispatch-worker-exit.txt').exists():
        raise ValueError('Timing worker has not exited')
    source = Path(__file__).resolve().parent
    inputs = root/'attention-oracle-inputs'
    output = root/'attention-oracle-diagnostic'
    output.mkdir(exist_ok=False)
    for name, digest in json.loads((inputs/'sha256.json').read_text()).items():
        assert sha(inputs/name) == digest, name
    reference = root/'reference/source'
    revision = subprocess.check_output(['git', '-C', str(reference), 'rev-parse', 'HEAD'], text=True).strip()
    assert revision == '2cbca389f515df455412dafa07e2b1359cc72a6d'
    assert not subprocess.check_output(['git', '-C', str(reference), 'status', '--porcelain'], text=True).strip()
    library = output/'liboracle-attention.so'
    run(['c++', '-O3', '-std=c++17', '-fPIC', '-shared', '-I', reference/'ggml/include',
         source/'oracle-attention.cpp', '-L', reference/'build/bin',
         '-Wl,-rpath,'+str(reference/'build/bin'), '-lggml-cpu', '-lggml-base', '-o', library], output/'native-compile.log')
    synthetic = validate_bridge(library)
    classes = output/'classes'
    classes.mkdir()
    jars = sorted(p for p in (root/'models-bench-quick/lib').glob('*.jar') if not p.name.startswith('._'))
    cp = ':'.join(map(str, jars))
    java = Path('/opt/java/bin/java')
    run([java.with_name('javac'), '-cp', cp, '-d', classes, source/'DecoderAblation.java',
         source/'OracleAttention.java'], output/'java-compile.log')
    base = [java, '--add-modules=jdk.incubator.vector', '--enable-native-access=ALL-UNNAMED', '-Xmx12g',
            '-Dvectors.gguf.threads=8', '-Dmodels.purejava.maxContextLength=1024',
            '-Dmodels.native.kernels.library='+str(root/'native-baseline.so'),
            '-Dvision.oracle.attention.library='+str(library),
            '-Dmodels.qualification.revision=3f31646c743647449310de2c5ff57fbd55d9e957',
            '-cp', cp+':'+str(classes)]
    report = dict(scope='Diagnostic only; reference attention substitution, not admission or performance',
                  referenceCommit=revision, synthetic=synthetic, librarySha256=sha(library),
                  sources={p.name: sha(p) for p in source.iterdir() if p.is_file()},
                  jars={p.name: sha(p) for p in jars}, nativeSha256=sha(root/'native-baseline.so'), results=[])
    selected = [('Q4_K_M', 'textvqa-38337'), ('Q4_K_S', 'textvqa-36979'),
                ('Q4_K_S', 'textvqa-35501'), ('Q4_K_S', 'textvqa-39587'),
                ('Q4_K_S', 'vqav2-454129000'), ('Q4_K_M', 'vqav2-359592004'),
                ('Q4_K_M', 'textvqa-36979'), ('Q4_K_M', 'textvqa-37016')]
    cases = {row['id']: row for row in json.loads((root/'campaign/cases.json').read_text())['cases']}
    for quant, identity in selected:
        directory = output/quant/identity
        directory.mkdir(parents=True)
        native_encoder = output/'encoders'/identity
        if not native_encoder.exists():
            native_encoder.parent.mkdir(exist_ok=True)
            image = root/'campaign'/cases[identity]['image']
            assert sha(image) == cases[identity]['imageSha256']
            run([*base, 'com.integrallis.models.bench.VisionQualificationCli',
                 root/'campaign/artifacts/llava-v1.5-7b-mmproj-model-f16.gguf', image, native_encoder, '1', '3'],
                native_encoder.parent/(identity+'.log'))
        vectors = dict(models=native_encoder/'embeddings.f32',
                       reference=inputs/'encoder'/identity/'embeddings.f32')
        token_path = inputs/quant/'candidate'/identity
        model = root/'campaign/artifacts'/f'llava-v1.5-7b-{quant}.gguf'
        row = dict(variant=quant, caseId=identity, embeddings={name:sha(p) for name,p in vectors.items()}, cells={})
        for name, embedding in vectors.items():
            for implementation in ['native', 'oracle']:
                destination = directory/f'{implementation}-{name}.json'
                cmd = [*base, 'DecoderAblation', model, token_path/'prefix.i32', token_path/'suffix.i32',
                       embedding, destination]
                if implementation == 'oracle':
                    cmd.append('OracleAttention')
                run(cmd, destination.with_suffix('.log'))
                result = json.loads(destination.read_text())
                row['cells'][implementation+'Attention_'+name+'Encoder'] = result
        baseline = json.loads((token_path/'runs.json').read_text())[0]['tokens']
        row['baselineReproduces'] = baseline == row['cells']['nativeAttention_modelsEncoder']['tokens']
        if not row['baselineReproduces']:
            raise ValueError('Baseline does not reproduce: '+identity)
        tsv = directory/'reference-cases.tsv'
        tsv.write_text(''.join('\t'.join([name, str(p), str(token_path/'prefix.i32'),
                                         str(token_path/'suffix.i32'), '0', '1'])+'\n' for name,p in vectors.items()))
        with (directory/'reference.jsonl').open('w') as stdout, (directory/'reference.log').open('w') as stderr:
            subprocess.run([str(root/'reference-attention-dispatch/task-oracle'), str(model), str(tsv), '8'],
                           stdout=stdout, stderr=stderr, check=True)
        refs = [json.loads(line) for line in (directory/'reference.jsonl').read_text().splitlines()][1:]
        row['referenceDecoder'] = {r['id']: dict(tokens=r['tokens'], answer=bytes.fromhex(r['answerHex']).decode()) for r in refs}
        report['results'].append(row)
        (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(dict(variant=quant, caseId=identity,
                             cells={k:v['answer'] for k,v in row['cells'].items()},
                             reference=row['referenceDecoder'])), flush=True)
    (output/'COMPLETE').write_text('Component ablation complete; no admission claim\n')


if __name__ == '__main__':
    main()
