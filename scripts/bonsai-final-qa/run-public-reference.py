"""Execute every frozen case on the publisher server; preserve raw responses and failures."""
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

cases_path, output_path, endpoint = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
raw = cases_path.read_bytes()
cases = json.loads(raw)['cases']
expected_tokens = {r['index']: r for r in map(json.loads, cases_path.with_name('public-qa-reference-tokens.jsonl').read_text().splitlines())}
first = int(sys.argv[4]) if len(sys.argv) > 4 else 0
count = int(sys.argv[5]) if len(sys.argv) > 5 else len(cases)
if first < 0 or count < 1 or first + count > len(cases):
    raise ValueError('case range')
with output_path.open('x') as out:
    out.write(json.dumps({'type': 'manifest', 'caseSha256': hashlib.sha256(raw).hexdigest(),
        'backend': 'llama.cpp', 'backendVersion': 'PrismML-Eng/llama.cpp@e8fd209c975c80aa7fb36a8423571ff3b1b7e955',
        'context': 2048, 'count': count, 'first': first, 'endpoint': endpoint,
        'tokenizationMode': 'trusted-frozen-chatml-v2',
        'generation': {'temperature': 0, 'topK': 1, 'topP': 1, 'seed': 42,
                       'repetitionPenalty': 1, 'rawPrompt': True, 'promptCache': False}}) + '\n')
    out.flush()
    for index in range(first, first + count):
        case = cases[index]
        row = {key: case[key] for key in ['dataset', 'id', 'promptSha256', 'maxTokens']}
        row['index'] = index
        body = {'prompt': case['prompt'], 'n_predict': case['maxTokens'], 'temperature': 0,
                'top_k': 1, 'top_p': 1, 'seed': 42, 'repeat_penalty': 1,
                'cache_prompt': False, 'stream': False}
        start = time.monotonic()
        try:
            request = urllib.request.Request(endpoint + '/tokenize', data=json.dumps({'content': case['prompt'],
                'add_special': True, 'parse_special': True}).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=60) as response:
                tokens = json.load(response)['tokens']
            assert tokens == expected_tokens[index]['inputTokenIds'], 'publisher server token IDs differ from pinned oracle'
            row['inputTokenIds'] = tokens
            request = urllib.request.Request(endpoint + '/completion', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=1800) as response:
                result = json.load(response)
            row['rawResponse'] = result
            row['output'] = result['content']
            row['stopReason'] = 'MAX_TOKENS' if result.get('stopped_limit') or result.get('stop_type') == 'limit' else 'EOS' if result.get('stopped_eos') or result.get('stop_type') == 'eos' else 'STOP_SEQUENCE' if result.get('stopped_word') or result.get('stop_type') == 'word' else 'UNKNOWN'
            if result.get('truncated'):
                row['error'] = 'reference truncated the input context'
        except Exception as error:
            row['error'] = str(error)
            row['output'] = ''
        row['totalMillis'] = (time.monotonic() - start) * 1000
        out.write(json.dumps(row, ensure_ascii=False) + '\n')
        out.flush()
        print(f"case={index} dataset={case['dataset']} stop={row.get('stopReason')} error={row.get('error')} millis={row['totalMillis']:.0f}", flush=True)
