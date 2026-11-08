# LLaVA fidelity and decode follow-up — 2026-11-08

Written before provisioning or new inference. The rejected admission attempt remains immutable.

| Field | Value |
| --- | --- |
| Experiment | Layer-localize the three retained disagreements, profile short-answer decode, retain a general production improvement only after a matched A/B, then repeat frozen admission |
| Branch | `fix/llava-catalog-fidelity`, based on merged main e59dbc821af3ac3867be250f70d1865aa0e9cadd; each deployed payload records immutable revision and SHA-256 |
| Provider / region / plan | AWS us-east-1, one m6a.4xlarge, 16 vCPU / 64 GiB, 80 GiB gp3, Ubuntu 24.04 ami-0d6dd9e8265d48906 |
| Duration / spend | Expected 2 hours, maximum 4 hours; USD 6 total within existing qualification authorization |
| Creation / deletion deadlines | Create by 2026-11-08 05:00 UTC; delete by 2026-11-08 08:00 UTC and no later than four hours after launch; instance shutdown behavior terminate |
| Correctness | Original frozen manifest, independent reference, both public datasets, same weights/input tokens/image vectors; no gate changes |
| Performance | Same host, sequential candidate/reference arms, eight threads, original workload and warmup/repetition schedule; layer tracing is diagnostic and never timing evidence |
| Diagnostic order | Q4_K_M textvqa-38337 with identical reference embeddings first; then Q4_K_S textvqa-39587 and Q4_K_M vqav2-359592004; record per-layer final-prompt hidden states and full logits before selecting implementation changes |
| Acceptance of a fix | No case-specific branch or answer override; isolated operator fixture where possible; full original workload and short/long performance repeats before admission |
| Stop conditions | Corrupt inputs, nonfinite values, unsupported architecture, budget or deadline; archive failures and observations before termination |
| Secrets on box | No provider or service credentials; ephemeral SSH public key only |
| Local evidence | `audit-artifacts/pr380-vision-fidelity-2026-11-08/`; compact reproducible evidence committed under this experiment |

The live Standard EC2 quota was 192 vCPU and there were no pending, running,
stopping or shutting-down instances immediately before this record. Recheck on launch.
Only this experiment's instance, key and security group may be deleted. SSH is limited
to the controller IP. The root volume deletes on termination. No real inference,
model downloads or real-image preprocessing runs on the developer Mac.

The initial diagnostic reuses the measured ff627e32 JAR/kernel payload and retained
embeddings to reproduce the discrepancy. This is explicitly not a released-runtime
qualification. New implementation candidates and published 0.5.15 are identified by
separate receipts, never substituted into the original admission evidence.

## Comparator precision diagnosis — declared before the corrected repeat

The pinned reference's `build_attn_mha` explicitly casts F32 K/V tensors to F16
when Flash Attention is active. `task-oracle` left `flash_attn_type` at AUTO, so its
reported F32 cache does not mean F32 attention arithmetic. On identical full first-layer
Q/K/V inputs in textvqa-38337, independently computed float64 attention differs from
Models by 3.84e-7 normalized RMSE and from that reference by 0.00255. This is a
precision mismatch to investigate, not permission to alter an answer or admission gate.

Next diagnostic: disable Flash Attention explicitly in the F32 reference, preserving
the original outputs. If this removes the mismatch, repeat correctness with genuinely
F32 attention and document this as a comparator correction. Keep AUTO/F16 as the
faster performance arm and retain every existing accuracy/performance threshold.

## Encoder precision experiment

The corrected F32 reference still differs on the two encoder-sensitive cases. The
shipping CLIP layer-normalization implementation sums 1,024 values and squared
centered values sequentially in F32. Test wider moment accumulation against the
same independent encoder and all three retained images, without changing image
preprocessing, weights, prompts or decoder. The hypothesis is lower normalization
roundoff; it is not established until intermediate/output arrays are compared.
Retain a change only if it improves fidelity over multiple images and passes the
full frozen task and performance workload. No lookup or case-specific branch.

## Decode SIMD scale-broadcast experiment

A CPU-timer profile of the original caption decode, using an optimized symbol-bearing
build whose generated tokens and every top-two margin match the original binary,
places 55.2% of samples in Q4_K dot products, 16.6% in Q6_K and 14.8% in attention
score/value products. Test keeping Q4_K scale broadcasts in SIMD registers instead
of extracting eight scalar lanes per block. Integer products and floating reduction
order remain unchanged. Require synthetic reduction parity and real-output parity,
then same-host sequential A/B latency before retaining it. Profile samples themselves
are not throughput evidence. Do not promote a slowdown or retain an alternate mode.

## Preserve the frozen admission reference

The explicit-F32 run is an operator diagnostic. Inspection confirms the F16 casts
are deliberate Flash Attention behavior, not corrupted tensors or an incorrect
weight mapping. Disabling that optimization changes the reference's numerical
execution and its answers. Do not substitute the new counterfactual for the frozen
admission reference: retain AUTO with the original F32 cache for task admission,
AUTO/F16 for faster-reference performance, and disclose the effective attention
precision. Preserve the OFF traces separately. The original 99% sequence gate,
accuracy floors/loss bound, corpus and performance thresholds remain unchanged.
This declaration precedes the full corpus run of the normalization candidate.

For decode optimization A/B use the original caption inputs, eight threads, three
warmups and seven measured generations in baseline/candidate/baseline order.
Compare every token sequence; report both baseline medians and raw distributions.
A/B timing is diagnostic; final admission retains its original schedules.

## SIMD result and full-corpus candidate

The scale-broadcast experiment preserves all 30 generated sequences but does not
improve decode: candidate median 4012.566 ms versus 4002.542 ms and 4002.233 ms
in the two baselines (ratios 1.002504 and 1.002582). Remove the experimental
production code; preserve the raw A/B evidence. The next full frozen workload
uses only wider CLIP normalization moments with the original native kernels.
Reuse the exact archived 80-image corpus, verifying its manifest and image hashes,
rather than selecting or preprocessing new inputs. Collect both quantizations,
both reference KV types and the declared integration schedules on this worker.

The first full-repeat payload omitted the optional native benchmark runtime. Its
startup diagnostics showed `vector-api` and `NoKernel`; stop and retain that
invalid attempt before any task case completed. This is a collection error, not
a model verdict. Rebuild using `-PmodelsBenchNative=true`, require the native
execution plan before inference, and start a fresh result directory with the
same inputs. The evidence evaluator independently rejects a fallback plan.

## Attention span result

The four-row experiment preserves all 30 token sequences. Caption decode medians
are 4007.679 / 3999.094 / 4009.435 ms for baseline/candidate/baseline: only
0.21–0.26% improvement, within the observed run variation. Prefill fluctuates
between ratios 0.9902 and 1.0166. This does not justify the added production
complexity or establish closure of the admission gap. Retain the original native
kernels for the full normalization qualification. Keep the experiment's raw data
and source as diagnostics, outside the shipping branch.

## Cache-layout diagnostic, to run only after admission collection stops

Both arithmetic-level experiments produced negligible decode benefit. Test whether
F32 attention's token-major cache stride limits throughput before changing the
shipping cache. An ignored remote-only synthetic benchmark compares token-major,
16-float row padding, head-major, and token-major repeat with identical 32-head,
128-wide inputs at 64/606/1024 positions and eight workers. All outputs must be
bit-identical. Record seven samples of thirty calls after twenty warmups, both
with warm caches and a separately labeled controller-only 64 MiB eviction (not
a guarantee of eviction from every worker cache). This is component evidence,
never model qualification. Public ABI and default layout remain unchanged. Only
a substantial measured gain warrants integrating a new layout and rerunning the
full workload; do not overlap this experiment with the current admission run.

The isolated worker below supersedes that initial same-host scheduling plan;
no additional workload runs on the admission host.

## Isolated component worker — declared before provisioning

- Experiment: synthetic attention-cache layout A/B; only if material, prototype
  the winning layout and run original/candidate/original real-decoder A/B.
- Branch: `experiment/llava-attention-layout`; every payload records its commit.
- Provider: AWS us-east-1, one additional m6a.4xlarge, 16 vCPU, 64 GiB RAM,
  Ubuntu 24.04 ami-0d6dd9e8265d48906, 40 GiB disposable gp3 root.
- Expected duration: 20 minutes; maximum one hour, automatic terminate on shutdown.
- Spend: within the original USD 6 combined experiment ceiling; total maximum
  five instance-hours across the four-hour primary and one-hour auxiliary worker.
- Create before 2026-11-08 05:00 UTC; delete no later than 06:00 UTC and one hour
  after creation. Keep the primary admission host idle except its frozen workload.
- Correctness: exact synthetic output parity across all layouts/context lengths;
  a real-decoder candidate must also preserve every generated token.
- Performance: same auxiliary host, original/candidate/original; component
  results are not model admission. Full admission stays on the primary worker.
- Stop on corrupt payloads, nonfinite values, exhausted budget or deadline.
- No service credentials on the worker; ephemeral SSH public key only.
- Local evidence: `audit-artifacts/pr380-attention-layout-2026-11-08/`. Record and
  delete only the auxiliary instance/key/security group created by this experiment.
- Re-read live quota and account instances before launch; the existing primary
  `i-04b38cce99ef0308c` must remain running and untouched.

The synthetic run preserves every output. At 606 positions, head-major warm
latency is 0.1666 ms versus 0.2373/0.2236 ms in the two token-major runs. The
eviction diagnostic is nearly neutral (0.3368 versus 0.3457/0.3272 ms); 1,024
positions under eviction regress. Do not claim a decoder gain. Prototype an
explicit head-major F32 cache through both prefill and decode, retaining the
same GEMM, dot-product and softmax arithmetic and worker pool. Preserve old C
entry points as compatibility wrappers, not a selectable Java implementation.
Require synthetic public-ABI parity/bounds tests and existing chunk/reset/rewind
Java tests before the same-host real caption A/B (3 warmups, 7 repetitions,
original/candidate/original). Reject an end-to-end regression or negligible gain.

## Head-major decoder result

The real caption A/B preserves all 30 token sequences. Decode medians are
4008.752 / 3888.977 / 4012.910 ms for original/head-major/original, improving
by 2.99% and 3.09% against the two baselines. Prefill ratios are 0.9685 and
0.9931. Retain this candidate for full frozen qualification; this diagnostic
does not establish catalog admission. All 63 active Rust tests, Clippy, and
seven Java native tests pass. The A/B reducer initially referenced a Python
virtualenv absent on the auxiliary host; its three timing arms completed.
Run the same standard-library reducer with system Python, preserving the
original exit-127 log and all timings. No inference was repeated for this fix.
## Q4_K fixed-block view experiment

The inner Q4_K decoder indexes scale bytes through a dynamically sized full-model
slice. Test validating the row once and iterating fixed-size 144-byte blocks
zipped with activation scales, allowing bounds checks to leave the block loop.
No unchecked indexing, quantization, integer math or float reduction change.
After the head-major A/B ends, compare head-major alone versus head-major plus
this isolated change with the same 3-warmup/7-repeat caption schedule. Require
all native tests and exact real token parity; remove the block-view change if
it does not produce a meaningful end-to-end gain. No overlap with timing runs.

The block-view change preserves all 30 output sequences and passes 63 active
Rust tests and Clippy. Decode medians are 3896.659 / 3857.314 / 3888.310 ms,
a 0.80–1.01% improvement against bracketing head-major baselines. Prefill ratios
are 0.9629 and 0.9699. Retain this bounds-check simplification with the cache
layout for full qualification; this modest measurement is not admission.

The full normalization-only Q4_K_S repeat fixes its original mismatch but finds
two new mismatches (vqav2-454129000 and textvqa-35501), matching 62/64 sequences.
Do not promote the three-case diagnostic to a full-corpus correctness claim.
Use the remaining isolated-worker time for crossed encoder/decoder diagnostics
on these cases after its timing finishes. Preserve the complete running primary
campaign and all gates unchanged.

## Short-answer confirmation and rejected normalization variants

On the same auxiliary worker, the combined cache/block-view changes reduce the
hardest retained Q4_K_S short-answer decode median from 255.400 to 244.735 ms
(4.18%). The AUTO/F16 reference takes 223.391 ms, giving a 1.09555 ratio; prefill
ratio is 1.05734. All token sequences match across 3 warmups and 7 repetitions.
Inputs use fixed independent image embeddings. This is a decoder diagnostic,
not an end-to-end admission result or a substitute for the frozen full repeat.

Five-case normalization diagnostics separately tested F32 reciprocal-square-root
rounding after double moments (e8998d62), then pairwise Welford moments matching
the independent encoder's published algorithm (a7a40f8c). Both trade mismatches
instead of resolving them; exclude both implementations from this branch.
The full normalization-only admission continues without interruption.

The isolated worker's 380-file archive, 106112899 bytes, has SHA-256
f328332ae339018d1466490db6a2e6708969fe9094b0a7de68dd27482dcf37f8.
It retains executable payloads, source, inputs, timings, answer traces and failure
records; excludes model weights, credentials and build caches. Verify the local
copy before terminating i-091c95dd56f03732a and deleting its key/security group.

## Fresh full-repeat worker — declared before provisioning

The current worker's four-hour deadline leaves too little time for another full
candidate/reference collection and safe evidence transfer. Keep every frozen
case and performance arm; use a fresh isolated worker instead of shortening the
campaign or reusing old timings. The completed normalization-only task scores
are Q4_K_S VQA 0.675 versus 0.6875 reference (accuracy-loss gate fails), TextVQA
0.415625 versus 0.415625; Q4_K_M VQA 0.675 versus 0.675, TextVQA 0.403125 versus
0.384375. Both match 62/64 complete nonblank token sequences. These are complete
task results, with integration and faster-reference performance still collecting.

- Experiment: full retained-candidate admission repeat after the queued arithmetic
  diagnostic; each payload identifies its immutable source and binary digests.
- Branch: `fix/llava-catalog-fidelity`; current prepared payload is c4d1c348.
  Arithmetic variants remain separate experiments unless independently justified.
- Provider: AWS us-east-1, one m6a.4xlarge (16 vCPU, 64 GiB), Ubuntu 24.04
  ami-0d6dd9e8265d48906, 80 GiB disposable gp3 root, delete on termination.
- Expected duration: 90 minutes; maximum two hours from creation, automatic shutdown
  with instance shutdown behavior terminate. Create by 2026-11-08 06:15 UTC;
  delete by 08:15 UTC and no later than two hours after creation.
- Spend: retain the USD 6 combined fidelity-experiment ceiling. The original
  primary's maximum four hours, deleted auxiliary's actual <1 hour, and this
  worker's maximum two hours total <7 instance-hours, leaving storage allowance.
  Terminate the original primary after collecting its outstanding evidence.
- Correctness: unchanged exact artifacts, public corpus, scoring, 99% sequence
  agreement, <=0.5 percentage-point score loss, absolute floors and integration.
- Performance: unchanged same-host sequential arms, 8 threads, all F32/AUTO and
  faster F16 reference comparisons, original warmups/repetitions and <=1.10 ratios.
- Stop on corrupt artifacts, nonfinite output, budget or deadline. Retain failures.
- Secrets: ephemeral SSH public key only; no service credentials on the worker.
- Local evidence: `audit-artifacts/pr380-vision-repeat-2026-11-08/`.
- Re-read live quota and active fleet. Do not modify any unrelated resource or the
  ongoing primary measurement. This worker has a distinct key and security group.
