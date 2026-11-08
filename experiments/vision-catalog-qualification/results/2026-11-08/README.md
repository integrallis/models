# LLaVA catalog admission attempt

Generated from `campaign/results/admission-attempt.json` by `render.py`.

Measured Models runtime: `ff627e32a3910488d9d4a9c1d15d361fe3242e25`.
Host: `ip-172-31-13-143`. Corpus: `b0120176e445cf46e6d85d7aab6f96e37ec417ec73532adca6425466eb621e41`.

These are bounded 32-question subsets of each public dataset, plus paired blank controls.
They are not full publisher benchmark scores. Lower latency ratios are better.

| Quantization | Backend admission | VQAv2 (candidate / reference) | TextVQA (candidate / reference) | Token agreement |
| --- | --- | --- | --- | --- |
| Q4_K_S | FAIL | 0.687500 / 0.687500 | 0.415625 / 0.415625 | 0.984375 |
| Q4_K_M | FAIL | 0.696875 / 0.675000 | 0.384375 / 0.384375 | 0.968750 |

## Q4_K_S

| Gate | Result |
| --- | --- |
| accuracyLoss | PASS |
| vqaFloor | PASS |
| textVqaFloor | PASS |
| imageContribution | PASS |
| tokenAgreement | FAIL |
| performance | PASS |
| strongestReferencePerformance | FAIL |
| integrationConfirmation | FAIL |

| Dataset | Paired blank controls | Image contribution |
| --- | --- | --- |
| vqav2 | 8 | 0.537500 |
| textvqa | 8 | 0.375000 |

Worst of the matched F32 and faster F16 reference ratios; ceiling **1.10**.

| Case | Prefill | Decode | Total | Generation lengths match |
| --- | --- | --- | --- | --- |
| vqav2-132415014 | 1.073926 | 1.167027 | 1.075438 | True |
| vqav2-423731001 | 1.051876 | 1.135140 | 1.053487 | True |
| textvqa-36063 | 1.043766 | 1.153549 | 1.044763 | True |
| textvqa-37476 | 1.069290 | 1.113025 | 1.069723 | True |

Token disagreements: textvqa-39587.

| Resource observation | Value |
| --- | --- |
| Candidate peak process RSS | 10.078 GiB |
| Reference decoder peak process RSS | 7.925 GiB |
| Candidate two-file load | 2071.182 ms |
| Reference decoder load | 4271.270 ms |
| Reference encoder load | 3525.590 ms |

Loads use the warm filesystem. The reference encoder and decoder ran separately;
encoder RSS was not measured, so this is not a combined-pipeline memory comparison.

## Q4_K_M

| Gate | Result |
| --- | --- |
| accuracyLoss | PASS |
| vqaFloor | PASS |
| textVqaFloor | PASS |
| imageContribution | PASS |
| tokenAgreement | FAIL |
| performance | PASS |
| strongestReferencePerformance | FAIL |
| integrationConfirmation | FAIL |

| Dataset | Paired blank controls | Image contribution |
| --- | --- | --- |
| vqav2 | 8 | 0.537500 |
| textvqa | 8 | 0.375000 |

Worst of the matched F32 and faster F16 reference ratios; ceiling **1.10**.

| Case | Prefill | Decode | Total | Generation lengths match |
| --- | --- | --- | --- | --- |
| vqav2-132415014 | 1.083576 | 1.137717 | 1.084400 | True |
| vqav2-423731001 | 1.078951 | 1.109387 | 1.080046 | True |
| textvqa-36063 | 1.063622 | 1.118504 | 1.063896 | True |
| textvqa-37476 | 1.064460 | 1.087030 | 1.064702 | True |

Token disagreements: vqav2-359592004, textvqa-38337.

| Resource observation | Value |
| --- | --- |
| Candidate peak process RSS | 10.318 GiB |
| Reference decoder peak process RSS | 7.745 GiB |
| Candidate two-file load | 2082.538 ms |
| Reference decoder load | 3891.480 ms |
| Reference encoder load | 3525.590 ms |

Loads use the warm filesystem. The reference encoder and decoder ran separately;
encoder RSS was not measured, so this is not a combined-pipeline memory comparison.

## Integration repeats

The initial one-warmup/three-run result and the declared three-warmup/seven-run confirmation are both retained.

| Attempt | Passed | Missing reports |
| --- | --- | --- |
| integration | False | none |
| integration-confirmation-3w7r | False | none |

| Attempt | KV | Prefill ratio | Decode ratio | Total ratio | Numerical / performance |
| --- | --- | --- | --- | --- | --- |
| integration | f32 | 1.049207 | 0.873357 | 0.999391 | True / True |
| integration | f16 | 1.055195 | 1.105637 | 1.067977 | True / False |
| integration-confirmation-3w7r | f32 | 1.023943 | 0.891887 | 0.986523 | True / True |
| integration-confirmation-3w7r | f16 | 1.028903 | 1.102956 | 1.048204 | True / False |

## Publication boundary

Catalog published: **False**. Released runtime verified: **False**. ModelJars runtime verified by this backend report: **False**.

ModelJars consumer evidence is recorded separately. A branch backend pass does not authorize public marker publication.
