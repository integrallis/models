"""Reproduce the supplied checkpoint's complete VPS measurements from retained raw logs."""
import json
from pathlib import Path

from common import HERE, digest, read_cases
from compare_performance import compare_pair


def main():
    root = HERE.parent.parent/'benchmark-results/2026-11-01-gliner25-decide'
    run = root/'results-restart-1'
    case_path = HERE/'performance-cases.jsonl'
    cohort = json.loads((HERE/'performance-cohort.json').read_text())
    if digest(case_path) != cohort['caseSha256']:
        raise ValueError('frozen performance inputs changed')
    cases = read_cases(case_path)
    paths = {arm:run/f'performance-1-{arm}.jsonl' for arm in ['models','reference']}
    report = compare_pair(cases,paths['models'],paths['reference'])
    historical = json.loads((run/'performance-1-report.json').read_text())
    if report != historical:
        raise ValueError('recalculation differs from the historical report')
    snapshot = json.loads((HERE/'snapshot.json').read_text())
    for key in ['candidateRuntime','referenceRuntime']:
        if report[key]['snapshot'] != snapshot:
            raise ValueError('this is not the supplied base checkpoint')
    if set(report['buckets']) != set(cohort['buckets']):
        raise ValueError('incomplete bucket coverage')
    rows = {arm:[json.loads(line) for line in path.read_text().splitlines()]
            for arm,path in paths.items()}
    for arm,data in rows.items():
        if len(data)-1 != cohort['cases'] or any('error' in row for row in data[1:]):
            raise ValueError('incomplete or failed '+arm+' run')
    observations = {
        arm:dict(requests=len(data)-1,errors=0,loadMs=data[0]['loadMs'],
                 peakRssGiB=max(row['peakRssKiB'] for row in data[1:])/1024**2)
        for arm,data in rows.items()
    }
    deployment = json.loads((root/'deployment.json').read_text())
    sources = [case_path,HERE/'performance-cohort.json',*paths.values(),
               run/'performance-1-report.json',run/'lscpu.txt',run/'memory.txt',
               root/'deployment.json',Path(__file__)]
    audit = dict(
        checkpoint=snapshot['id'],revision=snapshot['revision'],
        measuredRuntimeSource=deployment['payloadManifest']['sourceRevision'],
        benchmarkCompleted=True,performanceGatePassed=report['passed'],catalogQualified=False,
        reportReproducedExactly=True,pairedRuns=1,
        buckets=len(report['buckets']),requestsPerRuntime=cohort['cases'],
        observations=observations,failedGateBuckets=report['failures'],
        sources={str(path.relative_to(HERE.parent.parent)):digest(path) for path in sources},
        note='Audit of existing remote measurements; no new inference run. Load is one worker-reported observation, not controlled cold-start latency.',
    )
    (root/'base-performance-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    lines = [
        '# Supplied GLiNER2.5-Decide: completed VPS performance measurements',
        '',
        '**The benchmark completed. The Models runtime failed the relative speed gate.**',
        'Those are separate outcomes. Both runtimes successfully executed all 429 requests across',
        'all 13 registered shapes, with zero inference errors. Each shape has 3 warmups and 30 timed',
        'requests (390 timed requests per runtime). Recalculating from the raw logs exactly reproduces',
        'the historical report. This audit does not rerun inference or incur another rental.',
        '',
        '## Environment and artifact',
        '',
        '- Date: 2026-11-01; EC2 m6a.4xlarge, AMD EPYC 7R13, 16 vCPU.',
        '- Both arms ran serially on the same host, using 8 inference threads, F32, batch 1, concurrency 1.',
        '- Supplied checkpoint: `fastino/GLiNER2.5-Decide`, revision `'+snapshot['revision']+'`.',
        '- This is the original checkpoint, not the separately trained variant.',
        '- Models source: `'+audit['measuredRuntimeSource']+'`; Java 25.0.3, Vectors 0.1.32.',
        '- Publisher source: `'+report['referenceRuntime']['sourceRevision']+'`; PyTorch 2.9.1+cpu.',
        '- Models computes the encoder/head in Java and uses a Rust tokenizer. The later native projection experiment is not this run.',
        '',
        '## Absolute timings and throughput',
        '',
        'Times are end-to-end request wall times in milliseconds, including the local JSONL driver.',
        'Throughput is timed request count divided by total timed wall time, not the reciprocal of p50.',
        'Inputs are the frozen synthetic shape corpus; this measures speed, not answer accuracy.',
        '',
        '| Tokens / labels / tasks | Publisher p50 ms | Models p50 ms | Publisher p95 ms | Models p95 ms | Publisher req/s | Models req/s |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for name,bucket in report['buckets'].items():
        shape=name.replace('tokens','').replace('-labels',' / ').replace('-tasks',' / ')
        ref,models=bucket['reference'],bucket['candidate']
        if ref['samples']!=30 or models['samples']!=30:
            raise ValueError('unexpected timed sample count')
        lines.append(f"| {shape} | {ref['p50Ms']:.1f} | {models['p50Ms']:.1f} | {ref['p95Ms']:.1f} | {models['p95Ms']:.1f} | {ref['requestsPerSecond']:.3f} | {models['requestsPerSecond']:.3f} |")
    ref,models=observations['reference'],observations['models']
    lines += [
        '', '## Load and memory observations', '',
        '| Metric | Publisher | Models |', '| --- | ---: | ---: |',
        f"| Worker-reported model load | {ref['loadMs']/1000:.2f} s | {models['loadMs']/1000:.2f} s |",
        f"| Process peak RSS | {ref['peakRssGiB']:.2f} GiB | {models['peakRssGiB']:.2f} GiB |",
        '| Request errors | 0 / 429 | 0 / 429 |', '',
        'Load times are one observation per runtime around their model-loading calls. They exclude',
        'model downloads and process/interpreter startup; filesystem caches were not cleared.',
        'They are not controlled cold-start measurements. Peak RSS is the process high-water mark,',
        'including runtime overhead, not just tensor storage or per-request allocation.', '',
        '## What failed, and what did not', '',
        '- **Measurement completed:** all requested shapes and both runtimes are present.',
        '- **Runtime performance failed:** 12 of 13 shapes miss at least one fixed relative gate',
        '  (throughput >= 0.80x publisher and p95 latency <= 1.50x publisher).',
        '- **Qualification is incomplete:** this is one measured pair; final qualification needs',
        '  three passing pairs, passing accuracy, and released-runtime integration checks.',
        '- The later native projection experiment stopped early and has its own partial report.',
        '  It does not invalidate or replace this complete baseline.', '',
        '## Reproduction and evidence', '',
        '```sh', 'python3 experiments/gliner25-decide/report_base_performance.py', '```', '',
        '- [Audit and source digests](base-performance-audit.json)',
        '- [Complete paired report](results-restart-1/performance-1-report.json)',
        '- [Publisher raw requests](results-restart-1/performance-1-reference.jsonl)',
        '- [Models raw requests](results-restart-1/performance-1-models.jsonl)',
        '- [Runtime artifact hashes and worker cleanup](deployment.json)', '',
    ]
    (root/'BASE-PERFORMANCE.md').write_text('\n'.join(lines))
    print(json.dumps({k:audit[k] for k in ['benchmarkCompleted','reportReproducedExactly','requestsPerRuntime','buckets','performanceGatePassed']}))


if __name__ == '__main__':
    main()
