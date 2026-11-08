#!/usr/bin/env python3
"""Compare disputed image preprocessing and embeddings without another model forward pass."""
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'vision-qualification'))
from oracle import gguf, np, prepare_encoder, sha


def difference(actual, expected):
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('Invalid compared arrays')
    a, b = actual.astype(np.float64), expected.astype(np.float64)
    return dict(maxAbs=float(np.max(np.abs(a - b))),
                normalizedRmse=float(np.linalg.norm(a - b) / np.linalg.norm(b)),
                differingValues=int(np.count_nonzero(a != b)))


def main():
    if platform.system() != 'Linux':
        raise SystemExit('Run on the remote worker')
    root = Path(sys.argv[1])
    campaign, diagnostics = root / 'campaign', root / 'disagreement-diagnostics'
    report = json.loads((diagnostics / 'report.json').read_text())
    corpus = json.loads((campaign / 'cases.json').read_text())
    cases = {c['id']: c for c in corpus['cases']}
    reader = gguf.GGUFReader(str(campaign / 'artifacts/llava-v1.5-7b-mmproj-model-f16.gguf'))
    size = int(reader.get_field('clip.vision.image_size').contents())
    mean = np.array(reader.get_field('clip.vision.image_mean').contents(), dtype=np.float32)
    std = np.array(reader.get_field('clip.vision.image_std').contents(), dtype=np.float32)
    state = None, None, None, None, None, size, mean, std, None
    results = []
    for result in report['results']:
        identity = result['caseId']
        image = campaign / cases[identity]['image']
        if sha(image) != cases[identity]['imageSha256']:
            raise ValueError('Image changed')
        directory = diagnostics / 'encoders' / identity
        pixels, _, _ = prepare_encoder(state, image)
        pixels.astype('<f4').tofile(directory / 'reference-pixels.f32')
        actual = np.fromfile(directory / 'pixels.f32', dtype='<f4')
        native = np.fromfile(directory / 'embeddings.f32', dtype='<f4')
        reference = np.fromfile(campaign / 'results/encoder' / identity / 'embeddings.f32', dtype='<f4')
        results.append(dict(variant=result['variant'], caseId=identity,
                            pixels=difference(actual, pixels.flatten()),
                            embeddings=difference(native, reference)))
    destination = diagnostics / 'input-differences.json'
    if destination.exists():
        raise ValueError('Preserve the prior diagnostic')
    destination.write_text(json.dumps(dict(scriptSha256=sha(Path(__file__)), results=results), indent=2) + '\n')


if __name__ == '__main__':
    main()
