"""Thin CPU-only publisher worker. Diagnostic hooks observe, never replace inference."""
import argparse
import contextlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from common import HERE, verify_snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    policy = json.loads((HERE / 'policy.json').read_text())
    revision = subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != policy['referenceCommit']:
        raise ValueError('wrong publisher source revision')
    if subprocess.check_output(['git', '-C', str(args.source), 'status', '--porcelain'], text=True).strip():
        raise ValueError('publisher checkout must be clean')
    manifest = verify_snapshot(args.snapshot)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    sys.path.insert(0, str(args.source))
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from gliner2 import AutoExtractor
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        started = time.perf_counter()
        model = AutoExtractor.from_pretrained(str(args.snapshot), local_files_only=True,
                                             attention_backend='standard', map_location='cpu')
        model.eval()
        load_ms = (time.perf_counter() - started) * 1000
    placements = sorted({str(p.device) for p in model.parameters()})
    dtypes = sorted({str(p.dtype) for p in model.parameters()})
    if placements != ['cpu'] or dtypes != ['torch.float32']:
        raise ValueError(('unexpected placement/dtype', placements, dtypes))
    versions = {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'tokenizers', 'safetensors']}
    if versions['tokenizers'] != policy['tokenizersVersion']:
        raise ValueError('tokenizers version differs from native shim')
    emit(dict(type='ready', backend='publisher-gliner2', snapshot=manifest, sourceRevision=revision,
              device=placements, dtype=dtypes, threads=torch.get_num_threads(), versions=versions,
              hostname=platform.node(), loadMs=load_ms))
    for line in sys.stdin:
        envelope = json.loads(line)
        try:
            request = envelope['request']
            schema = {'classifications': [dict(task=t['name'], labels=[l['name'] for l in t['labels']],
                      label_descriptions={l['name']: l['description'] for l in t['labels'] if l['description']},
                      multi_label=t['multiLabel'], cls_threshold=t['threshold']) for t in request['tasks']]}
            observed = {'logits': [], 'tokenIds': []}
            hooks = []
            if envelope.get('mode') == 'diagnostic':
                def encoder_input(module, positional, keyword):
                    observed['tokenIds'] = keyword['input_ids'][0].tolist()
                def head_output(module, positional, output):
                    observed['logits'].extend(output.detach().reshape(-1).tolist())
                hooks = [model.encoder.register_forward_pre_hook(encoder_input, with_kwargs=True),
                         model.classifier.register_forward_hook(head_output)]
            try:
                start = time.perf_counter()
                with torch.inference_mode(), contextlib.redirect_stdout(sys.stderr):
                    result = model.extract(request['text'], schema, format_results=False, include_confidence=True)
                elapsed = (time.perf_counter() - start) * 1000
            finally:
                for hook in hooks:
                    hook.remove()
            selected = {}
            for task in request['tasks']:
                value = result[task['name']]
                selected[task['name']] = [v[0] for v in value] if task['multiLabel'] else [value[0]]
            emit(dict(id=envelope['id'], selected=selected, raw=result, elapsedMs=elapsed, **observed))
        except Exception as failure:
            import traceback
            traceback.print_exc(file=sys.stderr)
            emit(dict(id=envelope.get('id'), error=str(failure)))


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
