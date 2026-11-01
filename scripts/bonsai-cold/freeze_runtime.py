"""Freeze only reviewed runtime identities; never changes workload or numerical/performance gates."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

HERE=Path(__file__).resolve().parent
REFERENCE_SHA='13f0bd516ede8e4306541c72ef368e94c9bddddd80a99631ff5aa65992fdbba8'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def entries(path):
    with zipfile.ZipFile(path) as archive:
        return {name:hashlib.sha256(archive.read(name)).hexdigest() for name in sorted(archive.namelist()) if not name.endswith('/')}
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')

def freeze(revision,candidate,expected_sha,reference,phase):
    if not re.fullmatch('[0-9a-f]{40}',revision):raise ValueError('immutable source revision required')
    if digest(candidate)!=expected_sha:raise ValueError('candidate artifact checksum mismatch')
    if digest(reference)!=REFERENCE_SHA:raise ValueError('original qualified Qwen artifact required')
    before,after=entries(reference),entries(candidate)
    changes={name:{'prior':before.get(name),'candidate':after.get(name)} for name in sorted(before.keys()|after.keys()) if before.get(name)!=after.get(name)}
    protocol=json.loads((HERE/'protocol.json').read_text())
    protocol.update(vectorsRevision=revision,candidateCoreSha256=expected_sha,phase=phase)
    # These are the only permitted protocol mutations, checked before any write.
    old=json.loads((HERE/'protocol.json').read_text())
    assert {key for key in old if old[key]!=protocol[key]} <= {'vectorsRevision','candidateCoreSha256','phase'}
    manifests=[]
    for architecture in ['x86','arm']:
        path=HERE/f'expected-{architecture}.json';value=json.loads(path.read_text())
        names=[name for name in value['jarEntries']['candidate'] if name.startswith('vectors-core-')]
        assert len(names)==1
        value['jarEntries']['candidate'][names[0]]=after
        manifests.append((path,value))
    save(HERE/'protocol.json',protocol)
    for path,value in manifests:save(path,value)
    save(HERE/'candidate-correspondence.json',{'priorCoreSha256':REFERENCE_SHA,'candidateCoreSha256':expected_sha,'vectorsRevision':revision,'changedOrAddedJarEntries':list(changes),'entryChanges':changes,'scope':'Entry correspondence only. Review changed method reachability before transferring unaffected-path evidence; this is not new qualification.'})
    print(json.dumps({'revision':revision,'candidateSha256':expected_sha,'changedEntries':list(changes),'launch':'HELD until supported-host component screen passes'},indent=2))

def freeze_models(revision, backend, expected_sha, change_stage="dot-and-decoder"):
    if not re.fullmatch('[0-9a-f]{40}',revision):raise ValueError('immutable Models source revision required')
    if digest(backend)!=expected_sha:raise ValueError('Models artifact checksum mismatch')
    after=entries(backend)
    stages={
        'dot-and-decoder': {'com/integrallis/models/backend/purejava/ops/PinnedReduction.class',
             'com/integrallis/models/backend/purejava/quant/Q5_1Dequantizer.class',
             'com/integrallis/models/backend/purejava/quant/Q5_1Dequantizer$Narrow.class'},
        'grouped-attention-and-ptq': {'com/integrallis/models/backend/purejava/ops/GroupedQueryAttentionKernel.class',
             'com/integrallis/models/backend/purejava/quant/TernaryDequantizer.class'}}
    if change_stage not in stages:raise ValueError('unreviewed source-change stage')
    allowed=stages[change_stage]
    manifests=[];changes=[]
    for architecture in ['x86','arm']:
        path=HERE/f'expected-{architecture}.json';value=json.loads(path.read_text())
        for arm in ('baseline','candidate'):
            names=[name for name in value['jarEntries'][arm] if name.startswith('backend-java-')]
            assert len(names)==1
            before=value['jarEntries'][arm][names[0]]
            delta={name for name in before.keys()|after.keys() if before.get(name)!=after.get(name)}
            assert delta==allowed,('unreviewed Models runtime change',delta)
            changes.append({'architecture':architecture,'arm':arm,'changedEntries':sorted(delta)})
            value['jarEntries'][arm][names[0]]=after
        manifests.append((path,value))
    protocol=json.loads((HERE/'protocol.json').read_text());previous=protocol['modelsRevision']
    protocol['modelsRevision']=revision
    save(HERE/'protocol.json',protocol)
    for path,value in manifests:save(path,value)
    save(HERE/'models-correspondence.json',{'previousModelsRevision':previous,'modelsRevision':revision,
         'backendJavaSha256':expected_sha,'changeStage':change_stage,'changes':changes,
         'scope':'Both Vectors-comparison arms use this exact composed Models runtime. This isolates Vectors changes; it is not a comparison to the previous published Models runtime.'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--revision',required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--phase',required=True)
    args=p.parse_args();freeze(args.revision,args.candidate,args.sha256,args.reference,args.phase)
