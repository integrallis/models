#!/usr/bin/env python3
# Copyright 2025-2026 Integrallis Software, LLC
# SPDX-License-Identifier: Apache-2.0
"""Read immutable campaign artifacts; never turn runtime evidence into publication approval."""
import argparse
import hashlib
import json
from pathlib import Path

MODEL_SHA = 'c3c4d93f9db92627a2fc47ea8fe8c64a094cabd3549db544bcdd2c00e9a9a347'
HELD_OUT_IDS = {
    'remote-permit-negation', 'office-permit-positive', 'amber-form-and-fee',
    'violet-form-and-fee', 'route-identifier', 'retry-header-and-expiry',
    'retention-units', 'evidence-format-and-deadline', 'delivery-weekdays',
    'access-approval', 'storage-locations', 'settlement-directions',
    'false-premise-permit', 'upload-negation', 'unknown-satellite', 'unknown-moon',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(root, name):
    return json.loads((root / name).read_text())


def bind_audit(root, label, audit):
    require(audit['sourceReportSha256'] == digest(root / (label + '.json')),
            'visible audit does not match original report')
    require(audit['artifactSha256'] == MODEL_SHA, 'wrong model artifact')
    require(audit['policyId'] == 'visible-answer-completion-and-bounded-facts-v1',
            'unexpected visible-answer policy')
    require(audit['assessmentGroundingPolicy'].endswith('-v24'),
            'unexpected grounding policy')
    rows = audit['assessments']
    require(audit['attempts'] == len(rows), 'incorrect attempt count')
    require(audit['distinctQuestions'] == len({r['caseId'] for r in rows}),
            'incorrect distinct-question count')
    require(audit['requiredFactsAndCitationsPasses'] == sum(
        r['requiredFactsAndCitationsPass'] is True for r in rows), 'incorrect pass count')


def check_held_out(audit):
    rows = audit['assessments']
    require(len(rows) == 16 and {r['caseId'] for r in rows} == HELD_OUT_IDS,
            'held-out coverage must contain each of the 16 questions exactly once')
    require(sum(r['requiredFactsAndCitationsPass'] is True for r in rows) >= 15,
            'fewer than 15/16 complete visible answers passed')
    require(all(r['polarityConsistent'] is True for r in rows),
            'detected polarity contradiction')
    for row in rows:
        if row['caseId'].startswith('unknown-'):
            require(row['requiredFactsAndCitationsPass'] is True
                    and row['visibleEvaluation']['abstained'] is True,
                    'both unknown questions must correctly abstain')


def assess(root):
    checks = {}
    hashes = {}

    def checked(name, action):
        try:
            action()
            checks[name] = {'passed': True}
        except (OSError, ValueError, KeyError, TypeError) as error:
            checks[name] = {'passed': False, 'reason': str(error)}

    def load(name):
        value = read(root, name)
        hashes[name] = digest(root / name)
        return value

    def parity():
        report = load('parity.json')
        require(report['kind'] == 'flint-native-parity-v1' and report['passed'] is True,
                'numerical parity failed')
        rows = report['cases']
        require(len(rows) == 4 and {(r['backend'], r['case']) for r in rows} == {
            (backend, case) for backend in ('pure-java', 'rust-ffm')
            for case in ('completion', 'chat')}, 'incomplete parity coverage')
        require(all(r['passed'] is True and r['tokenizerMatches'] is True
                    and r['continuationMatches'] is True and r['finiteLogits'] is True
                    and r['maxAbsoluteLogitDifferenceFromJava'] <= 0.05
                    and r['relativeRmsLogitDifferenceFromJava'] <= 0.002 for r in rows),
                'parity values exceed preregistered tolerances')

    def default_smoke():
        smoke = load('default-correctness/smoke.json')
        report = load('native-default.json')
        require(smoke['reportSha256'] == hashes['native-default.json']
                and smoke['artifactSha256'] == MODEL_SHA
                and smoke['backend'] == 'rust-ffm', 'default smoke binding mismatch')
        require(report['summary']['successfulAttempts'] == 9
                and report['summary']['totalAttempts'] == 9
                and report['summary']['correctAnswerRate'] == 1
                and report['summary']['abstentionAccuracy'] == 1
                and not report['failures'], 'default smoke did not pass')

    def profile():
        selection = load('selected-profile.json')
        cap = selection['maxOutputTokens']
        require(cap in (256, 512, 1024) and selection['pilotVisibleGatePassed'] is True
                and selection['heldOutConsulted'] is False, 'budget pilot did not pass')
        label = selection.get('pilotReportLabel', 'native-default' if cap == 256 else 'native-budget-' + str(cap))
        require(label == 'native-budget-' + str(cap) or (cap == 256 and label == 'native-default'),
                'invalid pilot report label')
        audit = load('visible/' + label + '.json')
        bind_audit(root, label, audit)
        if 'nativePoolThreads' in selection:
            threads = str(selection['nativePoolThreads'])
            for native_label in (label, 'native-candidate', 'native-reliability'):
                native_report = load(native_label + '.json')
                require(native_report['backendDiagnostics']['environment']['native-kernel-threads'] == threads,
                        'candidate native pool does not match frozen profile')
            if int(threads) != 16:
                parity_profile = load('parity-profile.json')
                require(parity_profile['passed'] is True and len(parity_profile['cases']) == 4
                        and all(row['passed'] is True for row in parity_profile['cases']),
                        'selected native pool did not pass numerical parity')
        require(audit['attempts'] == 9 and audit['distinctQuestions'] == 9
                and audit['requiredFactsAndCitationsPasses'] == 9, 'pilot was not 9/9')
        for label in ('native-candidate', 'ollama-candidate', 'native-reliability',
                      'ollama-reliability'):
            report = load(label + '.json')
            require(report['settings']['maxOutputTokens'] == cap,
                    label + ' uses a different output budget')

    def held_out():
        audit = load('visible/native-reliability.json')
        bind_audit(root, 'native-reliability', audit)
        require(audit['workload'] == 'answer-reliability', 'wrong held-out workload')
        check_held_out(audit)

    def qualification():
        report = load('qualification.json')
        require(report['policyId'] == 'production-rag-model-contribution-v6',
                'unexpected catalog policy')
        require(report['candidate']['sha256'] == digest(root / 'native-candidate.json'),
                'qualification candidate binding mismatch')
        require(any(c['sha256'] == digest(root / 'ollama-candidate.json')
                    for c in report['comparators']), 'qualification comparator mismatch')
        require(report['qualification']['qualified'] is True, 'catalog runtime gates failed')

    def provenance():
        manifest = load('inputs.json')
        revision = manifest['sourceRevision']
        require(len(revision) == 40, 'missing frozen revision')
        for label in ('native-default', 'native-candidate', 'native-reliability'):
            report = load(label + '.json')
            require(report['artifactSha256'] == MODEL_SHA and report['backend'] == 'rust-ffm'
                    and ('+flint-' + revision) in report['backendVersion'],
                    'runtime provenance mismatch: ' + label)

    for name, action in [('numericalParity', parity), ('defaultCorrectness', default_smoke),
                         ('frozenBudgetPilot', profile), ('heldOutAccuracy', held_out),
                         ('catalogRuntimePolicy', qualification), ('runtimeProvenance', provenance)]:
        checked(name, action)
    failed = [name for name, check in checks.items() if not check['passed']]
    return {'policyId': 'flint-improvement-evidence-v1', 'runtimeEvidencePassed': not failed,
            'catalogAdmitted': False, 'manualAnswerReviewRequired': True,
            'releasedLibraryEvidenceRequired': True, 'toolCapabilityAssessedSeparately': True,
            'checks': checks, 'failedChecks': failed, 'inputSha256': hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('new_report', type=Path)
    args = parser.parse_args()
    result = assess(args.results)
    with args.new_report.open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    return 0 if result['runtimeEvidencePassed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
