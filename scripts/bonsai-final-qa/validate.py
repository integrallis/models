"""Reject missing pairs, mixed runtime manifests, or misplaced prompt/case identities."""
import json
from urllib.parse import urlsplit
from pathlib import Path


def validate_header(header, protocol, *, models):
    if header['caseSha256'] != protocol['inputs']['public-qa-cases.json'] or header['context'] != protocol['context']:
        raise ValueError('changed cases or context')
    if header['generation'] != {'temperature': 0, 'topK': 1, 'topP': 1, 'seed': 42, 'repetitionPenalty': 1, 'rawPrompt': True, 'promptCache': False}:
        raise ValueError('changed generation controls')
    if header['tokenizationMode'] != 'trusted-frozen-chatml-v2':
        raise ValueError('changed prompt template')
    if models:
        if header['processors'] != protocol['processors'] or header['runtimeIdentity'] != protocol['modelsRevision'] or header['javaVersion'] != protocol['javaVersion']:
            raise ValueError('changed candidate runtime settings')
        for key, prefix in [('core', 'vectors-core-0.1.30.jar'), ('models', 'backend-java-'), ('native', 'backend-native-'), ('runtime', 'models-runtime-')]:
            path = Path(urlsplit(header['classOrigins'][key]).path)
            if path.parent.name != 'lib' or not path.name.startswith(prefix):
                raise ValueError('unexpected class origin')
    elif header['backendVersion'] != 'PrismML-Eng/llama.cpp@' + protocol['referenceRevision']:
        raise ValueError('changed publisher source')


def validate_indices(rows, expected):
    indices = [row['index'] for row in rows]
    if len(indices) != len(set(indices)) or set(indices) != set(expected):
        raise ValueError('missing, duplicate or unexpected cases')
