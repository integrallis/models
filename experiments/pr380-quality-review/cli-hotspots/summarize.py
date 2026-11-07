#!/usr/bin/env python3
"""Render measured CLI scores, parser coverage and selected hotspots from raw artifacts."""
import gzip,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
R=HERE/'results'
mods=['backend-java','backend-native','models-api','models-bench']
def rows(name):
 with gzip.open(R/(name+'.tsv.gz'),'rt') as f:return [s.rstrip('\n').split('\t') for s in f]
default,modern,main=[rows(n) for n in ['default-details','modern-details','main-modern-details']]
meta=json.loads((R/'provenance.json').read_text())
lines=['# CLI scores and hotspots','','Generated from MFCQI 0.7.0 CLI JSON and the same version’s cognitive/smell analyzers. Scores are out of 100. Cognitive complexity is a raw count; its default hotspot threshold is 15.','','| Module | CLI score | Source files | Default parser failures | Modern parser failures |','|---|---:|---:|---:|---:|']
for m in mods:
 d=json.loads((R/(m+'.json')).read_text())
 n=[sum(r[0]=='PARSE' and r[1].startswith(m+'/') for r in arm) for arm in [default,modern]]
 lines.append(f'| {m} | {100*d["mfcqi_score"]:.4f} | {meta["source_counts"][m]} | {n[0]} | {n[1]} |')
lines+=['','## Selected PR hotspots with modern syntax enabled','','Main values match by file and method name; overloads are listed together. “Absent” means no same-named method in that file on main, not necessarily entirely new logic.','','| Method | Branch line | Main cognitive | Branch cognitive |','|---|---:|---:|---:|']
chosen={('ClipImagePreprocessor.java','preprocess'):40,('VisionQualificationCli.java','measureDecoder'):28,('LlamaForwardPass.java','advancePrefillBatch'):47,('LlamaForwardPass.java','forwardSequenceInternal'):42,('LlamaForwardPass.java','prefillEmbeddings'):16,('ClipVisionWeights.java','fromGgufFile'):15,('InferenceBenchmarkCli.java','main'):71,('RustGgufBatchedMatrixKernel.java','multiply'):11,('NativeCausalAttention.java','attend'):9}
for (file,method),score in chosen.items():
 found=[r for r in modern if r[0]=='COGNITIVE' and Path(r[2]).name==file and r[4]==method and int(r[5])==score]
 assert len(found)==1,(file,method,found)
 r=found[0]
 prev=[x[5] for x in main if x[0]=='COGNITIVE' and x[2]==r[2] and x[4]==method]
 lines.append(f'| `{file}:{method}` | {r[3]} | {", ".join(prev) or "Absent"} | {r[5]} |')
lines+=['','## Highest cognitive complexity per module (whole module, including existing code)','','| Module | Method | Line | Cognitive |','|---|---|---:|---:|']
for m in mods:
 for r in sorted([r for r in modern if r[0]=='COGNITIVE' and r[1].startswith(m+'/')],key=lambda r:int(r[5]),reverse=True)[:5]:
  lines.append(f'| {m} | `{Path(r[2]).name}:{r[4]}` | {r[3]} | {r[5]} |')
lines+=['','The default parser rejects modern syntax and the analyzer silently skips affected files. Modern-mode diagnostics change only JavaParser’s language level to BLEEDING_EDGE; they are supplemental findings, not newly calculated aggregate CLI scores. No LLM service was invoked.','']
(HERE/'FINDINGS.md').write_text('\n'.join(lines))
