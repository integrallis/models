# Flint Q4_K_M qualification — 2026-10-29

The complete tool suite failed. The repeated RAG run also falls below the model-contribution
gate: 3/27 accepted model answers (11.1%) against a required 33.3%. The completed Ollama comparison
has been scored with the unchanged policy: `FAILED_MODEL_CONTRIBUTION_GATE`. The supplementary
llama.cpp server exceeded the worker's readiness deadline and produced no benchmark report.
Flint is not admitted to the catalog. The worker has terminated and its root disk was deleted.
This is a source-build evaluation, not released-library catalog evidence.

## Provenance

- Runtime source: `7541b1a4676ffadcfd728aec39753258341f27f4` (`feat/flint`).
- Worker source: `8cba6fdaed76845a57b668752fb99b6f3865b81a`.
- Offline follow-up scorer: `98b0f4c44c1fa1a74912b7e6bfd01d4c07010f0b`.
- Checkpoint: `ggml-org/Flint-GGUF`, revision `468307a1b38259e9e4c9d4fe53d4e3b76386572c`, `Flint-Q4_K_M.gguf`.
- Artifact SHA-256: `c3c4d93f9db92627a2fc47ea8fe8c64a094cabd3549db544bcdd2c00e9a9a347` (23,387,911,136 bytes).
- Host: EC2 `m6a.4xlarge`, AMD EPYC 7R13, 8 cores / 16 threads, 64 GiB class memory, CPU only.
- Java: Temurin 25.0.3+9; Vectors 0.1.32; pure Java backend, default kernel settings.
- Comparators: Ollama 0.40.9 and llama.cpp `0893ec98f5eafad36a43a72cd1c3723d72b0d03a`.

The immutable payload and worker digests are in [the run manifest](qualification-attempt/manifest.json).
The [bootstrap](qualification-attempt/bootstrap.sh) enforces a four-hour shutdown; EC2 shutdown
terminates the instance and deletes its root volume. The worker uploads intermediate evidence.

The first host stopped before inference because `aws s3 sync` required a permission its restricted
role does not have. The corrected worker uses direct recursive copies. The failed setup receipt and
log remain under [setup-attempt](setup-attempt/run.json); no IAM permissions were expanded.

## Tool conformance

[Original report](qualification-attempt/results/tool-qualification.json), policy
`small-model-tool-conformance-v1`, greedy decoding, 256 output tokens, all 14 cases complete:

| Metric | Result |
| --- | ---: |
| Fully passing cases | 9 / 14 |
| Structured output | 92.9% |
| Exact tool selection | 78.6% |
| Schema validity | 92.9% |
| Expected argument accuracy | 70.3% |
| Refusal accuracy | 100% |
| p95 end-to-end latency | 236.008 seconds |
| Overall verdict | FAIL |

Failures are distinct from protocol integration failures:

- `home`: expected lock_door.door="front door".
- `robot`: expected tool sequence [move, rotate, gripper] but received [move].
- `device`: expected speak.text="hello".
- `gallery`: expected tool sequence [create_album, move_photos] but received [create_album].
- `document`: malformed tool-call output; expected tool sequence [record_booking] but received [].

`robot` and `gallery` chose stepwise execution instead of emitting every requested call in one
response. `document` spent most of the budget reasoning and stopped inside an argument at the
256-token cap. `home` and `device` changed literal values (`front` versus `front door`, and
`Hello!` versus `hello`). The gate and token budget were not relaxed after observing these failures.

The original scorer incorrectly marked the 96-token follow-up as passing: both weather facts
occurred inside `<think>`, while the visible answer ended at “shows some interesting”. Its raw
output and original PASS remain in the report. The corrected v2 scorer assesses only visible text;
[the separate offline audit](qualification-attempt/follow-up-audit.json) records **FAIL**, with
`newInference: false` and the original report's digest. This does not change the suite's already
failing overall verdict or misrepresent a new inference run.

To reproduce the offline audit from scorer commit `98b0f4c44c1fa1a74912b7e6bfd01d4c07010f0b`:

```sh
./gradlew :models-bench:installDist
javac --release 25 -cp 'models-bench/build/install/models-bench/lib/*' \
  -d /tmp/flint-follow-up-audit scripts/flint/FlintFollowUpAudit.java
java --add-modules=jdk.incubator.vector \
  -cp '/tmp/flint-follow-up-audit:models-bench/build/install/models-bench/lib/*' \
  com.integrallis.models.bench.FlintFollowUpAudit \
  benchmark-results/2026-10-29-flint/qualification-attempt/results/tool-qualification.json \
  /tmp/flint-follow-up-audit.json 98b0f4c44c1fa1a74912b7e6bfd01d4c07010f0b
```

## RAG protocol

All arms use the `general` workload, Flint template, context 2048, 16 threads, retrieval top-k 1,
and 256 output tokens. Default correctness has zero warmups and one iteration. Comparative arms
have one warmup and three measured iterations; they run sequentially on this host and import the
same verified file. The existing `production-rag-model-contribution-v6` policy is unchanged.
Model contribution and model-answer correctness must be read alongside the pipeline score.

### Default correctness

The [library-default smoke](qualification-attempt/results/default-correctness/smoke.json) passed its
pipeline criteria: 9/9 successful cases, 100% correctness and abstention accuracy, and no failures.
That is **not** the model-contribution admission gate.

| Default-smoke metric | Result |
| --- | ---: |
| Accepted model answers | 1 / 9 (11.1%) |
| Correct among accepted model answers | 100% |
| Extractive fallback | 7 / 9 (77.8%) |
| Retrieval abstention | 1 / 9 |
| Token-limit truncation | 4 / 9 (44.4%) |
| Median decode rate | 3.082 tokens/s |
| Median end-to-end latency | 98.036 seconds |
| Peak process RSS | 23,702,806,528 bytes |

These are cold-smoke observations, not the warmed comparative verdict. Four fallbacks followed
truncated generations. Three other completed, cited answers (`auto-glass-deadline`,
`telemedicine-benefit`, `idempotency`) were rejected by the conservative grounding policy.
`claim-payment-times` was accepted as a model answer. The unsupported lunar-vehicle question
correctly abstained. The raw-correctness field includes raw completion text, so it must not be
presented as visible-answer accuracy when a reasoning block contains the facts.

An offline inspection of the unchanged grounding policy isolates its supported-claim word check:
the three completed answers above contain, respectively, `you`, `no`, and `use`, which do not
match the question or source vocabulary under that check. Their identifier, question-clause
anchor, and trusted-citation checks pass. This is a limitation of the conservative lexical
policy, not evidence that every rejected answer is factually incorrect. No policy exception
was added for Flint, and no saved score was rewritten.

### Repeated candidate run

The [candidate report](qualification-attempt/results/candidate.json) contains 27/27 successful
measured attempts after a full-workload warmup. The outcomes repeat the default smoke exactly:
3 accepted model answers, 21 extractive fallbacks, 3 retrieval abstentions, and 12 truncated
generations. Pipeline correctness is 100%; correctness among accepted model answers is 100%.
The model-answer rate is 11.1%, below the existing one-third admission minimum.

Median decode rate is 3.085 tokens/s, median end-to-end latency 93.240 seconds, p95 end-to-end
latency 99.226 seconds, and peak process RSS 23,718,465,536 bytes. The absolute latency tier is
`PERFORMANCE_REDUCED`; the model-contribution gate is the admission blocker.

### Ollama comparison

The [Ollama report](qualification-attempt/results/ollama.json) completes all 27 measured trials.
Artifact digest/size, host environment, and every paired prompt hash match the Java run.
Median decode is 15.063 tokens/s and p95 end-to-end latency is 18.482 seconds. Java delivers
20.5% of that decode throughput and takes 5.37 times the p95 latency. These descriptive ratios
also miss the relative thresholds, but the formal policy stops earlier at model contribution.

Ollama itself receives `FAILED_QUALITY`: pipeline correctness is 88.9%, model contribution
44.4%, correctness among accepted model answers 75%, fallback 44.4%, and truncation 33.3%.
The repeated `priority-one` answer stops after “and if unresolved”; the grounding policy attaches
a citation and accepts it despite the missing escalation route. Faster execution alone does
not establish sufficient answer quality. The 256-token cap remains material in both engines.

The report's Ollama RSS/CPU fields sample the daemon PID, not the inference runner or complete
process tree. They are not valid total model-memory or inference-CPU measurements and must not
be compared with the Java process totals. Future workers should collect the runner/tree too.

[Qualification against Ollama](qualification-attempt/qualification-ollama.json) was produced
offline from the two unchanged reports with the existing `models-rag-bench qualify` CLI; it
performed no inference. The policy is unchanged between runtime commit `7541b1a4` and scorer
commit `98b0f4c4`. Its verdict is `FAILED_MODEL_CONTRIBUTION_GATE`.

### Supplementary comparator and cleanup

The pinned llama.cpp server remained in model loading through the worker's 180-second readiness
loop. The empty `llama-health.json` is the original failed curl output, not a valid health report.
The final health request returned HTTP 503, terminating the worker with exit code 22 at
2026-10-29T22:27:15Z. No llama.cpp inference benchmark was started and no three-engine qualification
report was generated. This is a startup-deadline failure, not a measured model-quality result.
The offline two-engine verdict above uses the complete authoritative Ollama comparison; it does
not invent or substitute a llama.cpp report. A future worker should allow a longer, still bounded
readiness interval and preserve startup resource diagnostics before retrying this comparator.

EC2 confirmed instance `i-078d9fa5473d3e210` terminated and volume `vol-0a8f0068de01d6ace` absent at
2026-10-29T22:28:22Z. The first setup attempt was also terminated. No worker is left running.
The [run receipt](qualification-attempt/run.json) records this outcome. Logs are retained as
lossless `.log.gz` archives; raw originals remain locally and in S3. `SHA256SUMS` covers the
committed evidence files. The [full validation receipt](validation/build.json) and build log
record successful formatting, build, and compliance checks after the scorer fix; these software
checks do not imply model qualification.

### Visible-answer review (manual, not an admission verdict)

The candidate's raw completions are identical across all three repetitions. Reviewing only text
after the closed reasoning block against the workload's required facts gives the following result:

| Outcome | Distinct questions | Cases |
| --- | ---: | --- |
| Complete answer with requested facts and valid citation | 4 | auto-glass-deadline, telemedicine-benefit, claim-payment-times, idempotency |
| Correct abstention | 1 | unsupported-lunar-vehicle |
| Requested facts appear, but answer is truncated and uncited | 1 | travel-limit |
| No visible answer; reasoning never closes | 3 | chargeback-evidence, break-glass, priority-one |

Thus 5/9 responses are complete and correct under this manual review, including abstention;
4/8 answerable questions receive a complete factual answer. The travel answer ends at
“3,500 dollars per”. No factual error was identified in the four completed answers, but nine
distinct questions are insufficient to estimate general accuracy or frontier-model parity.
Neither the pipeline's 100% score nor the policy's 11.1% contribution score measures this quantity.
The 27 measured trials are repetitions, not 27 independent questions.
