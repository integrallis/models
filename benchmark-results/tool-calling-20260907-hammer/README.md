# Hammer 2.1 pure-Java qualification — 2026-09-07

This experiment asked whether a small Hammer 2.1 checkpoint could become the next ModelJars
tool-calling artifact. Models implemented the official prompt and response protocol in Java and
fixed the GGUF YaRN contract before judging model behavior. No external inference engine is used at
runtime; llama.cpp build b9960 was used only as an independent token-output oracle.

## Artifacts

| Candidate | Q4_K_M SHA-256 | Size | Retained result |
| --- | --- | ---: | --- |
| Hammer 2.1 0.5B | `190676fe7ac430ac6680b32c216d4eb7413002e2af0c140b8280a0f74fcc6a4d` | 397,544,192 bytes | stopped after 5 cases; 1/5 passed |
| Hammer 2.1 1.5B | `c3447bf9d0dbbedcff33f4f64769f3add82877790a23940d554b5099056a0110` | 985,701,312 bytes | stopped after 11 cases; 6/11 passed |
| Hammer 2.1 3B | `a42369a669dbab0be10f1e04d755e55eb103b4b1bb7b4dec8c0ce4cf6c12b98d` | 1,929,442,112 bytes | complete 14-case run; 8/14 passed |

The 0.5B report is exploratory because its YaRN implementation was not yet represented by the
`modelsRevision` recorded in the report. The 1.5B process was interrupted after the result was
already mathematically unable to meet the policy; it is retained as partial negative evidence and
is not labeled complete. The complete 3B report is bound to Models commit
`d38d38b31d64f21cb1c6212d137c77df880bd192`.

## Correctness findings

The Java tokenizer and forward pass are not the reason for rejection. For the pinned 0.5B weather
prompt, the 379 input token IDs and all 30 greedy output token IDs match llama.cpp after Models
implemented the GGUF YaRN parameters. The 3B artifact also passed the Spring zipcode canary.

The complete 3B run nevertheless reached only 85.7% exact tool selection, 71.4% schema validity,
and 64.9% expected-argument accuracy. Independent hard failures included:

- returning no booking call for an explicit booking-record request;
- copying review text into a required sentiment enum instead of classifying it; and
- producing `dim` for an `on|off` home-control enum.

The suite also exposed three expectations worth correcting in a future versioned fixture: an
ambiguous home-state instruction, URL normalization (`github.com` versus
`https://github.com`), and ambiguous email-subject wording. Those cases were not changed during
this run. The booking and sentiment failures reject 3B without relying on any disputed case.

## Parser finding

Hammer sometimes combines Python-style single-quoted objects with JSON-style lowercase booleans.
The original safe literal normalizer accepted `True`, `False`, and `None` but rejected lowercase
`true`, `false`, and `null`. Commit `1e0c90d` adds the missing literal subset without evaluating
model output. The retained one-case replay confirms that the previously unparsed flight call is now
structured, schema-valid, and 4/4 on expected arguments. This parser correction does not change the
model-level rejection.

## Decision

Do not publish Hammer 2.1 0.5B, 1.5B, or 3B in ModelJars. Keep the protocol and YaRN support because
they are independently correct and useful for future Qwen2.5-derived artifacts. Do not package a
Hammer-based composite to mask a specialist that failed its own selection contract.

## Evidence

- `hammer2.1-0.5b-partial.json`
- `hammer2.1-1.5b-partial.json`
- `hammer2.1-3b-full.json`
- `hammer2.1-3b-flight-parser.json`
