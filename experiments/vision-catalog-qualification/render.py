#!/usr/bin/env python3
"""Render the admission summary from the machine report; no hand-transcribed scores."""
import json
import sys
from pathlib import Path


def render(report):
    lines = ['# LLaVA catalog admission attempt', '',
             'Generated from `campaign/results/admission-attempt.json` by `render.py`.', '',
             f"Measured Models runtime: `{report['receipt']['revision']}`.",
             f"Host: `{report['receipt']['host']}`. Corpus: `{report['receipt']['corpusSha256']}`.", '',
             'These are bounded 32-question subsets of each public dataset, plus paired blank controls.',
             'They are not full publisher benchmark scores. Lower latency ratios are better.', '',
             '| Quantization | Backend admission | VQAv2 (candidate / reference) | TextVQA (candidate / reference) | Token agreement |',
             '| --- | --- | --- | --- | --- |']
    for c in report['candidates']:
        d = c['datasets']
        lines.append(f"| {c['variant']} | {'PASS' if c['backendQualified'] else 'FAIL'} | "
                     f"{d['vqav2']['candidateScore']:.6f} / {d['vqav2']['referenceScore']:.6f} | "
                     f"{d['textvqa']['candidateScore']:.6f} / {d['textvqa']['referenceScore']:.6f} | {c['tokenAgreement']:.6f} |")
    for c in report['candidates']:
        lines += ['', f"## {c['variant']}", '', '| Gate | Result |', '| --- | --- |']
        lines += [f"| {k} | {'PASS' if v else 'FAIL'} |" for k, v in c['gates'].items()]
        lines += ['', '| Dataset | Paired blank controls | Image contribution |', '| --- | --- | --- |']
        lines += [f"| {name} | {d['blankCases']} | {d['imageContribution']:.6f} |" for name, d in c['datasets'].items()]
        lines += ['', 'Worst of the matched F32 and faster F16 reference ratios; ceiling **1.10**.', '',
                  '| Case | Prefill | Decode | Total | Generation lengths match |', '| --- | --- | --- | --- | --- |']
        for p in c['performance']:
            r=p['worstRatios']
            lines.append(f"| {p['id']} | {r['prefillMs']:.6f} | {r['decodeMs']:.6f} | {r['totalMs']:.6f} | {p['generationLengthsMatch']} |")
        lines += ['', 'Token disagreements: ' + (', '.join(c['disagreements']) or 'none') + '.']
        r = c['resources']
        lines += ['', '| Resource observation | Value |', '| --- | --- |',
                  f"| Candidate peak process RSS | {r['candidatePeakRssKiB'] / 1048576:.3f} GiB |",
                  f"| Reference decoder peak process RSS | {r['referenceDecoderPeakRssKiB'] / 1048576:.3f} GiB |",
                  f"| Candidate two-file load | {r['candidateLoadMs']:.3f} ms |",
                  f"| Reference decoder load | {r['referenceDecoderLoadMs']:.3f} ms |",
                  f"| Reference encoder load | {r['referenceEncoderLoadMs']:.3f} ms |", '',
                  'Loads use the warm filesystem. The reference encoder and decoder ran separately;',
                  'encoder RSS was not measured, so this is not a combined-pipeline memory comparison.']
    lines += ['', '## Integration repeats', '',
              'The initial one-warmup/three-run result and the declared three-warmup/seven-run confirmation are both retained.', '',
              '| Attempt | Passed | Missing reports |', '| --- | --- | --- |']
    for key in ['initialIntegration', 'confirmationIntegration']:
        r=report[key]
        lines.append(f"| {r['directory']} | {r['passed']} | {', '.join(r['missingEvidence']) or 'none'} |")
    lines += ['', '| Attempt | KV | Prefill ratio | Decode ratio | Total ratio | Numerical / performance |', '| --- | --- | --- | --- | --- | --- |']
    for key in ['initialIntegration', 'confirmationIntegration']:
        r=report[key]
        for d in r['decoders']:
            lines.append(f"| {r['directory']} | {d['kv']} | {d['imageTextPrefillRatio']:.6f} | {d['decodeRatio']:.6f} | {d['totalRatio']:.6f} | {d['numericalPass']} / {d['performancePass']} |")
    lines += ['', '## Publication boundary', '',
              f"Catalog published: **{report['catalogPublished']}**. Released runtime verified: **{report['releasedRuntimeVerified']}**. "
              f"ModelJars runtime verified by this backend report: **{report['modeljarsRuntimeVerified']}**.", '',
              'ModelJars consumer evidence is recorded separately. A branch backend pass does not authorize public marker publication.', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    print(render(json.loads(Path(sys.argv[1]).read_text())), end='')
