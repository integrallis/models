# Qwen3 1.7B production-Q4 transfer V18 preflight

Status: **frozen before any V18 model score was collected**.

V17 proved physical immutable-prefix sharing in 73/73 observed cases but rejected its threshold on
the untouched screen. All 75 static cases are consequently exposed. V18 treats the entire static
window as calibration data and evaluates its fixed result on a separately pinned BFCL-live screen.
V18 does not reinterpret V17 or call its former screen untouched.

## Immutable model inputs

- Base model: `Qwen/Qwen3-1.7B`
- Hugging Face revision: `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
- Production GGUF SHA-256:
  `061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a`
- Activated adapter SHA-256:
  `f1a4a1bc15e3942ed5a3fbe2c4750f39a8b24804e20566242ec740c5ce851c21`
- V16 head file SHA-256:
  `67b1f5fd231f924554d02daf450ce049fbccd24c084f7730d4ce2724ca234d1e`
- V16 canonical head artifact SHA-256:
  `acece5d8e0188b874ef8a919597ba7a359d876bc20fd828af5fbfab00984445a`
- Exposed static records SHA-256:
  `944d137ba1e325e0f3daa922a73108de0da5ccf0292bcf4dbe4a565aa11fcb4c`

No model, adapter, head feature, weight, prompt template, activation sequence, quantization, or
physical-sharing implementation changes from V17.

## Frozen live screen

The live window is the lowest 25 SHA-256-ranked cases per kind under seed `20260915` from the
pinned BFCL-live files at Gorilla revision `c15b2a151662cac9839c96d7dfb1493b5329c975`.

- Frozen records SHA-256:
  `4d70a040341480ccbae9af5a00d83bc4e7365ebead67e390cb5c3763ed83b30a`
- Frozen manifest SHA-256:
  `f1815fa298e8b4e205b061a26ed8a1043e7aa688f7a4b40d034e7c329d14507a`
- Case-set SHA-256:
  `ce93b078003095c0db03d4cb2099f2b1e0ddeac7df6c8e819260ade29a1690fc`
- Counts: 25 simple, 25 multiple, 25 irrelevance
- Query overlap with all static BFCL qualification sources: zero
- Query overlap with the complete pinned BFCL fine-tuning and Hammer irrelevance sources: zero

The screen retains paired applicability cases when the same user words are presented with
different advertised tool sets. They are different inference inputs and directly test whether the
model reasons about the available capabilities rather than the query alone. Exact duplicate
messages-plus-tools inputs are prohibited.

## Phase 1: exposed production-Q4 calibration

The Java 25 in-process runtime scores all 75 exposed static cases. Every score is the unchanged V16
affine head over the final normalized activated hidden state at `<tool_call>\n`, obtained from the
same activated turn whose base and specialist branches physically share their immutable KV prefix.

Candidate thresholds and tie-breaking are unchanged from V17: maximize balanced accuracy, then
correct no-calls, then correct calls, then choose the higher threshold. A call requires
`score > threshold`.

Calibration advances only with all of:

- 75/75 observations and physical shared-prefix identity;
- at least 47/50 correct call decisions;
- at least 24/25 correct no-call decisions; and
- balanced accuracy at least 0.95.

The complete calibration report and SHA-256 are committed before the live screen begins.

## Phase 2: untouched production-Q4 live screen

The screen command accepts the calibration report, calibration hash, live records, live manifest,
and live manifest hash. It recomputes and verifies every frozen identity and uses the calibration
threshold without adjustment.

V18 passes this phase only with all of:

- 75/75 observations and physical shared-prefix identity;
- at least 47/50 correct call decisions;
- at least 24/25 correct no-call decisions; and
- balanced accuracy at least 0.95.

Any threshold, head, data, prompt, or gate change after a live score is a new experiment.

## Gates after the live screen

A passing screen permits the already frozen oracle-free dual-branch generation policy to run on
the 75 exposed static cases. It requires 75/75 parseable syntax, 75/75 schema validity, at least
47/50 exact positive calls, at most 1/25 false calls, and physical shared-prefix identity for every
hybrid case. Base-only and adapter-only arms must use the same bytes and policy.

Only then may the original sealed 300-case qualification window be opened. Its unchanged gate is
300/300 syntax, 300/300 schema validity, at least 170/200 exact positive calls and no worse than the
same-prompt base, and at most 5/100 false calls.

The 14-case product suite, zipcode regression, six-turn conversation, natural-language tool-result
synthesis, exact base behavior, disabled-adapter no-op, projection oracle, Spring AI, LangChain4j,
long-context, physical identity, memory/native-memory, performance/crossover, packaging,
clean-host, and published-artifact tests remain mandatory.

Models and ModelJars publication is prohibited until every gate passes.
