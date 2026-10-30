# Flint improvement qualification — stopped at numerical parity

Frozen runtime and worker: `23e4e083ec311a216a8d19d78084dfefe82f3133`.
The full build, tests, formatting, and compliance checks passed; see `validation/`.

The disposable m6a.4xlarge worker `i-07dc7d39ef36fa1bd` launched at
2026-10-30T02:52:17Z with a four-hour shutdown bound and termination on shutdown.
Root volume `vol-04ad1cc1a973265eb` is configured for deletion on termination.
After the parity failure, the worker requested shutdown. Follow-up EC2 queries returned
neither the instance nor its root volume; see the final resource check.

The immutable deployment manifest records artifact provenance and the preregistered
budget-selection and held-out criteria. See `../../docs/flint.md` for the protocol.
The numerical parity gate failed and correctly prevented all quality and performance
qualification stages. No catalog admission is claimed.

`baseline-reaudit/` applies the separate visible-answer audit to unchanged previous
reports. It performs no inference and does not replace their original scores.
Java passed 15/27 attempts (five of nine distinct questions), and Ollama passed
18/27 (six of nine). Repetitions do not increase the independent accuracy sample.

Results upload to `s3://models-qual-077051030817/results/flint-improvement-20261030-23e4e083/`.

## Numerical result

Java matched all 16 pinned reference continuation IDs. Rust changed the third completion
token (expected 2635, observed 63509); the chat continuation IDs still matched.
Rust's maximum absolute logit differences were 4.52323 (completion) and 2.81813 (chat),
above the preregistered 0.05 limit; relative RMS errors were 0.02287 and 0.00998,
above the 0.002 limit. All logits were finite. These timings are a parity diagnostic,
not a warmed or comparable speed measurement.

The failed report is retained unchanged. The Q4_K Java and Rust paths use different
floating-point reduction orders; a separate format-ablation and identical-input
projection diagnostic is required before assigning the end-to-end discrepancy to it.
