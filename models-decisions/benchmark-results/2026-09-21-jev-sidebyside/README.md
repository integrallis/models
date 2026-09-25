# Side by side — Harriet vs TypeSafe Jev

Video: `sidebyside.mp4` (14.9 s), `sidebyside.gif`, raw terminal capture `sidebyside.cast`.
Re-run it with `run_demo.sh`. Both arms answer the same ten questions about the same 404-token
master services agreement, and each timing is taken inside the process that answers, so neither
side is credited or charged with the other's network path.

**Recorded 2026-09-25. Both arms were re-run for it; nothing here is a carried-over number.**

## Result

| | TypeSafe Jev | Harriet |
| --- | ---: | ---: |
| correct of 10 | **10** | 7 |
| total, cold start | **2.71 s** | 10.36 s |
| one-off prefill | n/a | 5.89 s |
| marginal per question | **0.271 s** | 0.447 s |
| document sent to a server | 10x | **0** |
| input tokens billed | 7,292 | **0** |
| runs offline | no | **yes** |
| prefix physically shared | n/a | yes, 10/10 |
| model | `jev-1.13.0` | Qwen3.5-4B Q4_K_M, frozen |
| kernel | hosted | rust-ffm, native abi 6 |

**We lose on both measured axes: three answers and 3.8x on wall clock.**

## Disclosure: we do not know what Jev runs on

Not the device, not how many, not whether requests were batched, not what else shared the machine.
Its timings include the network round trip from a laptop in Arizona to its API, which *penalises* it
on network and credits it on nothing. Harriet's host is named and it is a CPU: a RunPod `cpu3c`,
32 vCPU on an AMD EPYC 9655P (Zen 5), 16 worker threads, model on local disk.

So the wall-clock column is two systems on unknown-versus-known hardware. It is not a like-for-like
comparison and must not be read as one. The rows below the line in the video — documents sent,
tokens billed, offline — are the ones that do not depend on whose machine is bigger.

## What changed since the 2026-09-21 recording, and why it had to be re-made

The old video is kept as `sidebyside-2026-09-21-granite-head.{mp4,gif,cast}`. It showed
**Granite 4.1 3B plus an 82 KB squad2-fitted Noul head**, and *neither part ships any more*. Harriet
is frozen Qwen3.5-4B read through letter logits with no trained head at all, so the old arm could not
be re-run — the old video advertised a product that does not exist.

Against our own history, on the hardware class the old one used (4 Milan cores), the new stack is
**slower**: 22.7 s cold-start against the old 18.9 s, because Qwen3.5-4B is a bigger model than
Granite 3B. It buys that back on the benchmark that matters — JevBench Intelligence went 72.2 to
**88.0** — and on a wider box. Both figures are stated rather than the flattering one chosen.

| Harriet, same ten questions | prefill | marginal | cold total |
| --- | ---: | ---: | ---: |
| 4 Milan cores (the old video's hardware class) | 13.10 s | 0.954 s | 22.65 s |
| 16 threads, Zen 5 (recorded) | 5.89 s | 0.447 s | **10.36 s** |

## Accuracy: 7 of 10, and the same two clauses as before

Misses are **Q3** (may Acme use customer data to train models), **Q4** (what is the liability cap),
**Q5** (does the cap apply to data breaches).

Q3 and Q5 are settled by *exception or negation* — "shall not use", "this cap does not apply to
Section 4". The old video missed those too and blamed the squad2 head, which teaches whether a
matching span is present. **That explanation was incomplete**: the head is gone and the misses
remain, so the failure is not a property of the discarded component. Q4 sits at p=0.4687, a hair
under the threshold, and is the one genuinely borderline answer. Q7, which the old arm missed, is
now correct.

Output is deterministic: two runs on the Milan host and two on the Zen 5 host returned
byte-identical probabilities `[0.7332, 0.617, 0.3739, 0.4687, 0.2452, 0.6743, 0.5752, 0.7734,
0.055, 0.0401]`, so the 7/10 is a property of the model and not of a scheduling accident.

## Why the gap is prefill, and what would close it

Of the 10.36 s, **5.89 s is the one-off prefill** of the document and 4.47 s is the ten questions.
Per question we are 1.6x behind Jev, not 3.8x. Jev has no prefill because it re-sends the document
on every call, which is what the 7,292 billed tokens are.

Hardware does not close it. Measured on the recorded host, this demo saturates at 16 threads:

| threads | 4 | 8 | 16 | 32 |
| --- | ---: | ---: | ---: | ---: |
| cold total | 15.40 s | 15.33 s | **10.30 s** | 10.80 s |

4 to 8 buys nothing and 16 to 32 is worse, so 96 or 192 cores would not have helped either. Two
measured-but-unbuilt software levers are what remain: persisted prefix state, which deletes the
5.89 s outright, and an AVX-512+VNNI Q4_K path, measured at 2.26x on the inner loop at this batch
width. See `../../../benchmark-results/2026-09-24-decision-latency/latency/NOTES.md`.

## Honest scope

Ten questions, one document, one family, gold labels written by us. **A demo, not a benchmark.** The
measured benchmark position is in `../2026-09-21-jevbench-baseline/`.

## Traps this demo has walked into

- **A kernel misreport, twice.** The 2026-09-21 run published a prediction from the fallback path
  without checking which kernel had loaded. The rebuilt arm then guessed the kernel from
  `runtime.toString()` and wrote `pure-java` for a run that was demonstrably on `rust-ffm`. It now
  asks `runtime.backend().name()` and prints the native ABI, and the JSON carries both.
- **Prefill measured, never subtracted.** The one-off prefill is its own timed call against a fresh
  state, after a warmup on a *different* document. Differencing two averages to get it is how this
  project earned a retraction.
- **A truncated recording.** The first capture derived the scp target from a string substitution,
  hung under a pty, and produced a cast missing the entire comparison table — including the
  disclosure. Host and port are explicit now, and the cast is checked for every section before it is
  converted.
