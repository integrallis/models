# Host portability of the LLaVA decode arithmetic — 2026-11-10

**The arithmetic is host-portable. Four hosts produce byte-identical token sequences.** This
directory exists to retract a claim made on 2026-11-09, not to support it.

## Result

Runtime `54b086c7`, LLaVA 1.5 7B Q4_K_S, 8 threads, context 1024, F32 KV, same pinned corpus
(`corpusSha256 = b0120176e445cf46…`):

| comparison | cases | differing |
|---|---|---|
| Genoa native AVX-512 vs Genoa with `-XX:MaxVectorSize=32` | 80 | **0** |
| Genoa native vs EC2 `m6a.4xlarge` EPYC 7R13 | 80 | **0** |
| Genoa capped vs EC2 `m6a.4xlarge` EPYC 7R13 | 80 | **0** |

Hetzner `ccx33` (EPYC-Milan, AVX2) also reported `tokenAgreement = 1.000` against the same
reference. So EC2 EPYC 7R13, Hetzner Milan and Genoa in both JVM configurations all agree.

## What was retracted

A single run on a Hetzner `cpx62` Genoa box on 2026-11-09 reported `tokenAgreement` of 0.96875
(Q4_K_S) and 0.984375 (Q4_K_M). That was written up as "token agreement is not host-portable" and
"the cause is not identified", in the qualification README and the 0.5.18 changelog.

**That conclusion was wrong and is withdrawn.** It rested on one run. The same CPU model now
reproduces the qualified tokens exactly, in both JVM configurations, so nothing about that hardware
accounts for the difference.

## What is ruled out, by measurement rather than reasoning

- **Instruction set.** Genoa with AVX-512 fully exposed to the JVM gives identical tokens.
  `active-vector-bits` is 256 on this hardware either way: the runtime caps itself, and
  `-XX:MaxVectorSize=32` changes only the reported `preferred` value (512 to 256).
- **The `MaxVectorSize` flag**, which had been named as the prime suspect. It changes nothing.
- **Vector width** generally: the active species was 256 bits on every host compared here.
- **Thread count.** Pinned at 8 in every run, and the rayon pool is built from that Java-supplied
  value (`deberta_kernels.rs`: `ThreadPoolBuilder::new().num_threads(threads)`), not from the
  machine.
- **Corpus.** Identical `corpusSha256` and `manifestSha256` across the hosts whose receipts were
  retained, so `prepare.py` is deterministic.

## Why the cause of that one run cannot be established

Its artifacts were not retained. The box was deleted without pulling its results, so its
`receipt.json` — which would have carried its `corpusSha256`, jar hashes and native library hash —
does not exist. The one run that could not be explained is the one whose evidence was discarded.

That is the reason this directory keeps both Genoa arms in full rather than only the summary.

## A real finding that survives, unrelated to the retraction

`vision_attention.rs` takes a chunked online-softmax path whenever `decode && padded >= 512`, which
LLaVA's 576 image tokens always satisfy. Chunk width is `padded.div_ceil(threads)` and the partial
results are merged by rescaling with `exp(mᵢ − m_global)`, which is not associative in floating
point. **So the vision attention output depends on the thread count by construction**, and the
manifest's `threads: 8` pin is load-bearing for numerics, not only for speed. Nothing here
contradicts that; every run compared above used 8 threads.
