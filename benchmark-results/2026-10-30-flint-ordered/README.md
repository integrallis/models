# Flint ordered Q4_K qualification — comparator setup failed

Runtime and worker revision: `3336aed192786be17cca9360fdf65ee5ede731c7`.
Full build, tests, formatting, compliance, and consumer checks passed before deployment.
This reruns the preregistered improvement protocol after correcting Q4_K accumulation order.
Parity tolerances, budget selection, held-out accuracy, and catalog gates are unchanged.

Worker `i-02ad875e7f94d81e3` launched at 2026-10-30T14:44:58Z, with a four-hour shutdown
bound and termination on shutdown. Both the instance and root volume were absent from the EC2 inventory at the final cleanup check.
The comparison is incomplete; no qualification or catalog admission is claimed.

Results: `s3://models-qual-077051030817/results/flint-ordered-20261030-3336aed1/`.
Prior failed parity and the causal format ablation are retained in their separate evidence folders.

## Intermediate findings

Numerical parity passed: all 16 pinned continuation IDs match, and the Java/Rust logit difference
is exactly zero in both reference cases. The nine Java/native default raw answers also match
byte-for-byte. These checks establish execution agreement, not general answer accuracy.

| Arm | Output cap | Complete correct visible answers | Median decode tokens/s | Model contribution |
| --- | ---: | ---: | ---: | ---: |
| Java default | 256 | 5/9 | 3.95 | 3/9 |
| Native default | 256 | 5/9 | 7.60 | 3/9 |
| Native budget pilot | 512 | 9/9 | 7.48 | 6/9 |

These are zero-warmup, one-iteration measurements, not the final comparative arms. All three
pipelines score 9/9 with fallback; the visible-answer column credits only complete raw visible
answers, including the correct abstention. Four answers exhaust the 256-token budget. At 512,
the longest observed output is 405 tokens and none truncates. Manual review agrees with the pilot
audit. Travel and idempotency remain grounding false negatives, retained in the original reports.

The worker selected 512 tokens before consulting the held-out workload. The warmed native arm
completed 27 attempts with 27/27 complete correct visible answers, byte-identical to the reviewed
pilot for each repeated question. Its median decode speed is 7.74 tokens/s, median end-to-end
latency 35.41 s, and p95 end-to-end latency 63.28 s. Model contribution is 18/27 and correctness
among accepted model answers is 100%.

The held-out arm passed **16/16 complete correct visible answers**, including both required
abstentions and all polarity checks. Manual inspection against retained evidence agrees. There
is no truncation. The original pipeline reports 7/16 model contribution, seven conservative
fallbacks, and two abstentions. These are 16 new questions; the general-workload repetitions do
not add distinct accuracy samples. Tools completed at 10/14; profiling completed. Ollama imported and primed successfully, but
runner discovery expected `ollama runner` while this release launches `/usr/local/lib/ollama/llama-server`.
The worker stopped at 16:19:21Z before the comparative arms. No catalog qualification decision follows from these intermediate results.

Later branch work is recorded separately: `3ec2572e` adds an opt-in tool grammar prototype, which
is not present in this campaign. Its full build/compliance checks passed with a fresh 1 GB Gradle
heap after documentation verification exhausted the original 512 MB heap. The validation files
retain both attempts. Scripted adapter continuation tests likewise do not constitute real-model
tool accuracy evidence.

The frozen runtime's generic optimization labels are inaccurate for Flint's custom graph: the
grouping and attention flags are planner placeholders. Later diagnostics correct those labels.
This campaign's original reports are retained unchanged; use the actual kernel implementation,
source routing, numerical parity, and invocation tests when interpreting the execution path.
