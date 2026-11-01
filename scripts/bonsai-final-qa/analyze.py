"""Require all frozen paired blocks before applying unchanged descriptive quality gates."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from validate import validate_header, validate_indices

here = Path(__file__).resolve().parent
p = json.loads((here / 'protocol.json').read_text())
evidence, output = map(Path, sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)
summary = {'status': 'INCOMPLETE', 'protocol': p}
try:
    if (p['shards'], p['casesPerShard'], p['blocks'], p['casesPerBlock']) != (20, 10, 10, 1):
        raise ValueError('changed final single-case schedule')
    cases = json.loads((here / 'public-qa-cases.json').read_text())['cases']
    if len(cases) != 200:
        raise ValueError('expected exactly 200 frozen public cases')
    expected_tokens = {r['index']: r['inputTokenIds'] for r in map(json.loads, (here / 'public-qa-reference-tokens.jsonl').read_text().splitlines())}
    arms = {'models': [], 'publisher': []}
    identities = set()
    provenance = []
    manifest_by_shard = {}
    for path in evidence.rglob('manifest.json'):
        manifest = json.loads(path.read_text())
        shard = manifest['shard']
        if shard in manifest_by_shard or manifest['protocol'] != p:
            raise ValueError('duplicate or changed host protocol')
        if manifest['jars'].get('vectors-core-0.1.30.jar') != p['coreSha256']:
            raise ValueError('candidate artifact changed')
        manifest_by_shard[shard] = (manifest, hashlib.sha256(path.read_bytes()).hexdigest())
    if set(manifest_by_shard) != set(range(p['shards'])):
        raise ValueError('incomplete host cohort')
    if len({(m['runId'], m['harnessRevision']) for m, _ in manifest_by_shard.values()}) != 1:
        raise ValueError('mixed runs or harness sources')
    if len({json.dumps(m['jars'],sort_keys=True) for m,_ in manifest_by_shard.values()}) != 1:
        raise ValueError('runtime JAR inventory differs across hosts')
    if len({m['nativeSha256'] for m, _ in manifest_by_shard.values()}) != 1:
        raise ValueError('native library differs across hosts')
    if len({(m['rustc'], m['cargo']) for m, _ in manifest_by_shard.values()}) != 1:
        raise ValueError('native compiler differs across hosts')
    for identity_file in sorted(evidence.rglob('identity.json')):
        directory = identity_file.parent
        identity = json.loads(identity_file.read_text())
        shard, block = identity['shard'], identity['block']
        key = (shard, block)
        if key in identities or shard not in range(p['shards']) or block not in range(p['blocks']):
            raise ValueError('duplicate or unexpected block')
        identities.add(key)
        manifest, manifest_hash = manifest_by_shard[shard]
        if identity['manifestSha256'] != manifest_hash or any(identity[k] != manifest[k] for k in ['runId','runnerName']):
            raise ValueError('pair moved hosts or changed artifacts')
        first = shard * p['casesPerShard'] + block * p['casesPerBlock']
        count = p['casesPerBlock']
        if identity['first'] != first or identity['count'] != count or (directory/'STATUS').read_text().strip() != 'COMPLETE':
            raise ValueError('incomplete or changed block')
        command = json.loads((directory/'reference-command.json').read_text())
        for flag,value in [('-t','4'),('-tb','4'),('-c','2048'),('-ngl','0'),('--parallel','1')]:
            if command[command.index(flag)+1] != value:
                raise ValueError('publisher execution controls changed')
        for arm in arms:
            path = directory / f'{arm}.jsonl'
            header,*rows = map(json.loads,path.read_text().splitlines())
            validate_header(header,p,models=arm=='models')
            if header['first'] != first or header['count'] != count:
                raise ValueError('header case range changed')
            validate_indices(rows,range(first,first+count))
            execution=json.loads((directory/f'{arm}.execution.json').read_text())
            if execution['exitCode'] != 0:
                raise ValueError('arm failed')
            for row in rows:
                case=cases[row['index']]
                if any(row[k] != case[k] for k in ['dataset','id','promptSha256','maxTokens']) or row['inputTokenIds'] != expected_tokens[row['index']]:
                    raise ValueError('case identity or prompt changed')
            arms[arm].extend(rows)
            provenance.append({'shard':shard,'block':block,'arm':arm,'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    if identities != {(s,b) for s in range(p['shards']) for b in range(p['blocks'])}:
        raise ValueError('missing paired blocks')
    for arm, rows in arms.items():
        validate_indices(rows,range(len(cases)))
        merged=output/f'{arm}.jsonl'
        merged.write_text(json.dumps({'caseSha256':p['inputs']['public-qa-cases.json']})+'\n'+''.join(json.dumps(r)+'\n' for r in sorted(rows,key=lambda r:r['index'])))
    subprocess.run(['python3',str(here/'score-public-qa.py'),str(here/'public-qa-cases.json'),str(output/'models.jsonl'),str(output/'publisher.jsonl'),str(output/'scores.json')],check=True)
    scores=json.loads((output/'scores.json').read_text())
    publisher={r['index']:r for r in arms['publisher']}
    summary.update(hosts=len(manifest_by_shard), pairedBlocks=len(identities), pairedCases=len(arms['models']), status='PASS' if scores['descriptiveRegressionRulePassed'] else 'FAILED',rawAnswerDisagreements=[r['index'] for r in arms['models'] if r['output']!=publisher[r['index']]['output']],scores=scores,observations=provenance)
except Exception as error:
    summary['error']=repr(error)
finally:
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(summary['status'], summary.get('error',''))
if summary['status'] != 'PASS':
    raise SystemExit('Final public-QA gate did not pass')
