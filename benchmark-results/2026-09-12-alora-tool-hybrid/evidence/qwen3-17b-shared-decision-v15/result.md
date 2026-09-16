# V15 result: rejected after the decision screen

V15 completed its frozen sequential Java decision profile. The independently held 45-case screen
passed its own advance gate, but the fixed threshold produced only 45/50 correct call decisions
over the complete exposed development set. The downstream V15 gate requires at least 47/50 before
dual generation. V15 is therefore rejected; no generation, sealed qualification, packaging, or
release work was run.

| Metric | Calibration | Held-out screen | All exposed cases | Required next gate |
| --- | ---: | ---: | ---: | ---: |
| Correct calls | 17/20 | 28/30 | 45/50 | at least 47/50 |
| Correct no-calls | 9/10 | 15/15 | 24/25 | at least 24/25 |
| Balanced accuracy | 0.8750 | 0.9667 | 0.9300 | not a substitute for the count floors |
| Physical shared-prefix identity | 30/30 | 45/45 | 75/75 | 75/75 |

The threshold was `4.8179874`, selected only from the 30 calibration margins by the frozen rule.
No held-out margin, threshold, partition, token ID, prompt, adapter, or gate was changed after
output. The screen report's `PASS` means only that its predeclared 28/30 call, 14/15 no-call, and
0.93 balanced-accuracy screen passed. It does not mean that V15 passed the later release train.

The first 75-case run completed scoring but failed while Jackson serialized the adapter's optional
training-selection provenance. A regression test reproduced that report-layer defect; commit
`2fab0dc6e4a52ea6f0b98e257f9f6e62946984a2` registered Jackson's JDK 8 datatype module and added
the missing dependency. The corrected run used identical model, adapter, records, scoring, and
gates. Both logs and the first partial report are retained rather than hidden.

## Preserved evidence

| File | SHA-256 |
| --- | --- |
| `decision-profile-attempt1.partial.json` | `fe58482a9b02922436ecc164b8573645d1feebfb642f831dfb678edb45d99f65` |
| `decision-profile-attempt1.log` | `44fbe6be7dd32b9b8b7065758a1496fef0e1bca0e3452a10fe2a4e4e927957d7` |
| `decision-profile.json` | `799b1645ec510b6765ad50057ef379bd549e9dabc127516289189484e159643a` |
| `decision-profile.log` | `9b075c9df6f1f242d9a893a1540c853f5f51f84d1722a81ed46ade034ebea2a5` |

The valid report binds Models revision `2fab0dc6e4a52ea6f0b98e257f9f6e62946984a2`, source SHA-256
`944d137ba1e325e0f3daa922a73108de0da5ccf0292bcf4dbe4a565aa11fcb4c`, model SHA-256
`061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a`, and adapter SHA-256
`f1a4a1bc15e3942ed5a3fbe2c4750f39a8b24804e20566242ec740c5ce851c21`.

The V11 300-case qualification window remained sealed. V16 is a new, separately frozen experiment
that preserves V15's physical-sharing design but replaces the inadequate two-logit discriminator
with a trained hidden-state applicability head.
