"""Registered thread sweep followed by two fresh, order-reversed confirmations."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def answer_signature(report):
    # RAG reports retain raw text and token counts, not the token-ID stream.
    return [(r['caseId'], r['promptSha256'], r['grounding']['rawText'],
             r['generation']['outputTokens'], r['grounding']['decision'], r['evaluation'])
            for r in report['runs']]


def select_winner(rows):
    eligible = [r for r in rows if r['answersStable'] and r['qualityPassed']]
    if not eligible:
        raise ValueError('no quality-preserving configuration')
    # Freeze selection before confirmations. Never pick the best confirmation.
    return max(eligible, key=lambda r: (r.get('qualified', False), r['decodeRatio'], -r['latencyRatio'], -r['decodeThreads']))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    protocol = json.loads((a.work / 'protocol.json').read_text())
    baseline = None
    rows = []

    def trial(name, configuration, reference_first):
        nonlocal baseline
        job = dict(protocol['job'], **configuration, referenceFirst=reference_first)
        threads = job['decodeThreads']
        native_threads = job.get('nativeThreads', job['threads'])
        job_path = a.output / (name + '.job.json')
        job_path.write_text(json.dumps(job, indent=2) + '\n')
        subprocess.run(['runuser', '-u', 'root', '--', 'ollama', 'stop', 'qualification'], check=True)
        directory = a.output / name
        subprocess.run([sys.executable, str(a.work / 'run-released-pair.py'),
            '--payload', str(a.work / 'released'), '--job', str(job_path),
            '--model', str(a.work / 'model.gguf'), '--java', str(a.work / 'jdk/bin/java'),
            '--output', str(directory), '--reference-version', 'ollama@0.40.13',
            '--reference-model', 'qualification'], check=True)
        report = json.loads((directory / 'models-rag.json').read_text())
        env = report['backendDiagnostics']['environment']
        assert int(env['native-kernel-decode-threads']) == threads
        assert int(env['native-kernel-threads']) == native_threads
        assert job['threads'] == 16
        assert report['summary']['totalAttempts'] == report['summary']['successfulAttempts'] == 27
        signature = answer_signature(report)
        if baseline is None:
            assert name.startswith('screen-')
            baseline = signature
        q = json.loads((directory / 'verdict.json').read_text())['qualification']
        assert len(q['comparisons']) == 1 and not q['exclusions']
        comparison = q['comparisons'][0]
        row = dict(trial=name, decodeThreads=threads, nativeThreads=native_threads, referenceFirst=reference_first,
            answersStable=signature == baseline, qualityPassed=q['modelAnswerRate'] >= 1/3 and q['modelAnswerCorrectRate'] >= .9,
            qualified=q['qualified'], verdict=q['verdict'],
            decodeTokensPerSecond=report['summary']['p50DecodeTokensPerSecond'],
            decodeRatio=comparison['decodeThroughputRatio'], latencyRatio=comparison['endToEndLatencyRatio'],
            modelAnswerRate=q['modelAnswerRate'], modelAnswerCorrectRate=q['modelAnswerCorrectRate'])
        rows.append(row)
        (a.output / 'summary.json').write_text(json.dumps(dict(trials=rows, complete=False), indent=2) + '\n')
        print(json.dumps(row), flush=True)
        return row

    configurations = protocol.get('screenConfigurations')
    if configurations is None:
        configurations = [dict(name=str(t), decodeThreads=t) for t in protocol['screenOrder']]
    for configuration in configurations:
        trial('screen-' + configuration['name'], {k: v for k, v in configuration.items() if k != 'name'}, False)
    winner = select_winner(rows)
    (a.output / 'selection.json').write_text(json.dumps(winner, indent=2) + '\n')
    chosen = dict(decodeThreads=winner['decodeThreads'], nativeThreads=winner['nativeThreads'])
    confirmations = [trial('confirm-' + str(i + 1), chosen, first)
                     for i, first in enumerate(protocol['confirmationReferenceFirst'])]
    confirmed = all(r['qualified'] and r['answersStable'] for r in [winner, *confirmations])
    (a.output / 'summary.json').write_text(json.dumps(dict(trials=rows, complete=True,
        selectedDecodeThreads=winner['decodeThreads'], selectedNativeThreads=winner['nativeThreads'], confirmed=confirmed,
        marginTargetMet=confirmed and all(r['decodeRatio'] >= .85 for r in [winner, *confirmations]),
        stabilityScope='Identical raw generated text, output-token counts, prompts and grounding/evaluation per case; token IDs are not retained by this RAG harness.'), indent=2) + '\n')


if __name__ == '__main__':
    main()
