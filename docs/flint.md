# Flint GGUF execution

The pinned Flint Q4_K_M artifact passes `production-rag-model-contribution-v6` with Rust FFM
on the measured Linux CPU host. The full 27-attempt run reaches 11.93 median tokens/second
versus Ollama's 14.30 (83.39%); p95 end-to-end latency is 37.91 seconds versus 26.14 seconds
(1.450 times). Its absolute tier is `PERFORMANCE_REDUCED`. The result qualifies this grounded
RAG workload and hardware profile; it does not establish frontier-model equivalence.

All 27 raw visible answers are complete and correct across nine distinct questions. The
grounding layer accepts 18 model answers, uses six extractive fallbacks, and abstains three times.
The profile uses a 512-token output budget, a 2,048-token context, and eight native workers on
an m6a.4xlarge (16 vCPUs, 64 GiB RAM); Ollama retains 16 workers. The untuned 256-token smoke
passes pipeline correctness but truncates four visible answers, so use the separately validated
output budget for this workload. A fresh 16-question held-out run passes all complete visible
answers, including both abstentions, with manual review of facts, polarity, units, and identifiers.
Released-library checks are being completed before catalog admission. See the
[immutable reports and review](../benchmark-results/2026-10-30-flint-qualification/README.md).

Tool protocol support is implemented, but task accuracy has not passed qualification. Earlier
failed runs below are retained as development history and do not replace the current verdict.

The decoder reads `general.architecture=flint` and executes the leading dense layers, sigmoid
top-k routed experts plus the shared expert, per-head Q/K RMS normalization, learned attention
sinks, and independent per-layer sliding-window and RoPE patterns. All large tensors remain
memory-mapped. Full-attention layers have linear KV storage; sliding layers use bounded ring
buffers. Sessions are independent. Rewinding replays the prefix only when the required sliding
window has already been overwritten. Prefill runs sequentially and skips intermediate vocabulary
projections.

`ChatTemplate.FLINT` renders the publisher's text and tool envelopes. Retain `<think>...</think>` in
assistant history. Native `<arg_key>`/`<arg_value>` calls require the declared `ToolSpec` schemas
to distinguish raw strings from JSON values. Tool results use the observation role. The shared
scanner supports parallel calls and rejects malformed, truncated, unknown, or duplicate arguments;
calls inside reasoning are not executed. LangChain4j and Spring AI use the same schema-aware path.
This is protocol support, not evidence that the checkpoint passes tool qualification.
The tokenizer implements Flint's sequential digit/CJK/word splitting and recognizes its EOS
and user, developer, and observation transitions as generation terminators.

## Checkpoint

The integration fixture is the following exact artifact:

- Repository: `ggml-org/Flint-GGUF`
- Revision: `468307a1b38259e9e4c9d4fe53d4e3b76386572c`
- File: `Flint-Q4_K_M.gguf`
- Size: `23387911136` bytes
- SHA-256: `c3c4d93f9db92627a2fc47ea8fe8c64a094cabd3549db544bcdd2c00e9a9a347`
- Source: [Recursive Flint](https://huggingface.co/RecursiveSI/flint), Apache-2.0

Its 49 layers comprise 2 leading dense layers and 47 MoE layers; each routed layer selects 6
of 256 experts. Query width is 4096, distinct from the 2048 model width. There are 35 sliding
layers with a 513-token window and 14 full-attention layers. Only sliding layers apply RoPE.
The fused shared tensors have width 1024; the GGUF shared-count value of 2 describes two
512-wide expert units, not two separately routed shared branches.

Set an explicit context budget before loading on a laptop, for example
`-Dmodels.purejava.maxContextLength=4096`. The model advertises 262144 tokens, and allocating
the full context requires much more memory than the weights alone. A 32 GB Intel laptop is
a constrained correctness-test host for this 23.4 GB file, not the performance target.

Use the local GGUF directly from a build of this branch with `backend-java` and `models-runtime`
on the classpath. Launch Java 25+ with `--add-modules=jdk.incubator.vector` and the context property
above, then render the prompt explicitly:

```java
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.runtime.InferencePipeline;
import com.integrallis.models.runtime.chat.ChatMessage;
import com.integrallis.models.runtime.chat.ChatTemplate;
import java.nio.file.Path;
import java.util.List;

var prompt = ChatTemplate.FLINT.render(List.of(ChatMessage.user("What is 2 + 2?")));
var options = SamplingOptions.builder().temperature(0).maxTokens(128).build();
try (var pipeline = new InferencePipeline(PureJavaBackend.load(Path.of("Flint-Q4_K_M.gguf")))) {
    System.out.println(pipeline.generate(prompt, options));
}
```

## Validation

Fast tests, requiring no checkpoint download:

```sh
./gradlew :backend-java:test --tests '*Flint*' \
  :models-runtime:test --tests '*Flint*' --tests '*ChatTemplateEndOfTurnTest'
```

The F32 fixture's full logit vectors are independently generated with llama.cpp commit
`0893ec98f5eafad36a43a72cd1c3723d72b0d03a`, CPU inference, F32 KV, and flash attention disabled.
The reference helper disables extra repacked weight buffers to keep the real checkpoint mapped.
The test covers leading dense and routed layers, selection bias, shared experts, sinks, Q/K
normalization, selective RoPE, and positions past the sliding window. The absolute tolerance is
`2e-5`; the oracle values are not captured from the Java implementation.

Reproduce the reference in a separate llama.cpp checkout pinned to that commit:

```sh
cmake -S /path/to/llama.cpp -B /path/to/llama.cpp/build \
  -DGGML_METAL=OFF -DGGML_BLAS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build /path/to/llama.cpp/build -j 4 --target llama
c++ -std=c++17 -O2 scripts/flint/reference-logits.cpp \
  -I/path/to/llama.cpp/include -I/path/to/llama.cpp/ggml/include \
  -L/path/to/llama.cpp/build/bin -Wl,-rpath,/path/to/llama.cpp/build/bin \
  -lllama -lggml -lggml-base -o /tmp/flint-reference-logits
./gradlew :backend-java:flintReferenceFixture
/tmp/flint-reference-logits backend-java/build/flint-reference/flint-f32.gguf 3,4,5,6,7,8
```

The real-checkpoint gate downloads and verifies the pinned file before running, and does not
silently skip if the artifact is absent:

```sh
./gradlew :backend-java:flintIntegrationTest
```

Use `-Dmodels.fixtures.directory=/path/to/models` to reuse an existing verified download.
The test uses a 128-token context and a 2 GB heap; the mapped weights are additional memory.
It verifies the tokenized prompts and eight greedy continuation tokens for each of a completion
and a chat prompt. The independent expectations are in
`backend-java/src/test/resources/flint/reference.properties`. Reproduce them with:

```sh
/tmp/flint-reference-logits /path/to/Flint-Q4_K_M.gguf 'text:The quick brown fox' 7
/tmp/flint-reference-logits /path/to/Flint-Q4_K_M.gguf \
  'text:<|user|>What is 2 + 2?<|assistant|>' 7
```

For each run, take the `top` values starting at the last prompt row. These short continuations
exercise execution; they do not assess answer quality or completed reasoning.

### Development-laptop observation

On 2026-10-29 the pinned Q4_K_M test passed on a six-core 2.6 GHz Intel Core i7 MacBook Pro
with 32 GB RAM, Java 25.0.3, 256-bit Vector API operations, and the default 12-worker vector
executor. Both prompts matched the reference tokenization and all 16 continuation tokens matched.

| Prompt | Prompt tokens | Prefill | Subsequent tokens | Decode time | Decode rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Completion | 4 | 11.098 s | 7 | 10.024 s | 0.70 tokens/s |
| Chat | 10 | 12.419 s | 7 | 9.192 s | 0.76 tokens/s |

This was one short, uninstrumented test invocation, with a 128-token context and 2 GB heap.
The file had already been read for checksum verification and reference generation. There was
no warmup protocol, isolated-host control, or sustained throughput measurement. The observation
shows that this laptop can execute the checkpoint correctly on these cases; interactive latency
is high. It is not a performance qualification or evidence of frontier-LLM answer quality.

## Publication work remaining

The [2026-10-29 qualification](../benchmark-results/2026-10-29-flint/README.md) completed all 14
tool cases: 9 passed, so the tool gate failed. The default RAG correctness smoke passed its
pipeline criteria, but only 1/9 answers were accepted as model contributions; seven used
extractive fallback. The warmed, three-iteration run repeated this outcome: 3/27 accepted model
answers (11.1%), below the one-third minimum. Flint remains outside the catalog.
The original follow-up scorer credited facts inside reasoning; a separately
recorded v2 audit corrects that false positive without modifying the original report.

Longer-context validation, acceleration measurements, and released-library evidence remain
separate work. No claim about all Flint quantizations, frontier-model parity, or catalog
admission follows from the implementation or short numerical comparisons.

## Tool protocol and qualification

The tool prompt fixture is independently rendered using Jinja 3.1.6 and the template embedded in
the pinned GGUF. Its provenance is in `models-runtime/src/test/resources/flint/README.md`.
The scanner checks structural types, required fields, nested properties/items, enum/const and
`additionalProperties: false`. It is not a full JSON Schema validator. `$ref`, `oneOf`, and `anyOf`
are unsupported; ambiguous schemas or values degrade to plain text. Default qualification measures natural output; the later opt-in grammar prototype is described below.

For experimental tool use, supply the same declarations when rendering and scanning. For example,
inside an open `InferencePipeline` scope:

```java
var tools = List.of(new com.integrallis.models.api.ToolSpec(
    "get_weather", "Get the weather for a city",
    "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}"));
var toolPrompt = ChatTemplate.FLINT.render(
    List.of(ChatMessage.user("What is the weather in Austin?")), tools);
var output = pipeline.generate(toolPrompt,
    SamplingOptions.builder().temperature(0).maxTokens(256).build());
var result = com.integrallis.models.runtime.chat.ToolCallScanner.scan(
    output, ChatTemplate.FLINT.toolSyntax(), tools);
// Inspect result.toolCalls(); the scanner itself does not execute any function.
```

The dedicated `scripts/flint/qualification-worker.sh` runs on a disposable Linux host with 64 GB
RAM and 16 vCPUs. It accepts a JSON manifest with `sourceRevision` (full commit), `payloadUri`,
`payloadSha256`, and `resultsUri`. The payload contains the Gradle `installDist` directories
`models-bench`, `models-rag-bench`, and `source-revision.txt`. Bootstrap installs AWS CLI, curl,
jq, git, CMake, a C++ compiler, zstd and libcurl development headers before running it as a root
login shell. The bootstrap must schedule unconditional termination after four hours and set EC2
instance-initiated shutdown to terminate. The worker also shuts down on exit and uploads logs and
partial reports every 30 seconds. Use a fresh, immutable S3 prefix for each run.

The preregistered configuration is the full shared tool suite at 256 output tokens, then the
general RAG workload with context 2048, 16 threads, retrieval top-k 1 and 256 output tokens.
Default correctness uses zero warmups and one iteration. All comparative arms use one warmup
and three iterations on the same host and artifact. Comparators are Ollama 0.40.9 and the pinned
llama.cpp revision above; Java is Temurin 25.0.3+9. Cache drops and sequential arms avoid concurrent
inference. The existing admission policy is unchanged. Phase timeouts preserve incomplete evidence
and do not count as passes. A failed correctness smoke prevents admission even if a comparative
verdict passes. Source-build results cannot stand in for released-library catalog evidence.

## Plan toward catalog admission

The first target is qualified text/RAG use. ModelJars has separate capability manifests, so a
text qualification can admit an artifact while tool calling remains unqualified. Tool capability
must earn its own passing evidence. The current failed reports remain the baseline.

1. **Establish answer quality independently of fallback.** Retain raw output, reasoning, visible
   answer, grounding decision, and fallback separately. Add held-out factual, unanswerable,
   negation, numeric, identifier, and citation cases. Report complete correct answers, incomplete
   answers, hallucinations, and appropriate abstentions. The current manual review found four
   complete factual answers plus one correct abstention in nine distinct questions; repeated
   trials do not expand that accuracy sample. Investigate lexical grounding false negatives with
   model-independent regression tests and adversarial counterexamples. Never make negation a
   blanket ignored word or add a Flint-specific vocabulary exemption. Any policy correction gets
   a new version and fresh evidence across affected models. Also reject incomplete clause coverage:
   the Ollama baseline's truncated escalation answer was accepted with a derived citation despite
   omitting the requested route. Improving acceptance must preserve rejection of incomplete answers.
2. **Fix completion reliability.** Preregister bounded reasoning/output-budget experiments and
   compare the same settings across engines. Four of nine RAG cases hit the 256-token limit;
   the tool follow-up is capped at 96 tokens. Test whether a larger budget or a supported concise
   generation profile yields complete answers without factual regressions. A larger cap is an
   explicitly different configuration, not a reinterpretation of this failed run. Measure its
   latency and memory cost. The pinned publisher template exposes no non-thinking switch.
3. **Remove avoidable Java work and connect existing Rust kernels.** The baseline `multiply`
   used the convenience `TensorOps.ggufMatmul` overload, allocating quantization scratch for each
   product. The improvement implementation moves scratch to session ownership and groups
   Q/K/V and gate/up projections. `GgufBatchedMatrixKernel` now reaches the forward pass through
   `loadFlint`; the baseline passed it only to planning. The implementation uses the
   existing Q4_K/Q6_K Rust FFM kernels; verify actual invocation, fallback behavior, numerical
   parity, session isolation, and native resource lifetime before claiming acceleration.
4. **Profile and optimize the remaining hot paths.** Measure matrix products, quantization,
   thread scheduling, allocation, attention, and prefill separately. Use grouped Q/K/V and
   gate/up operations, then grouped selected-expert execution with independent scratch and
   deterministic accumulation. Reuse existing grouped-kernel patterns from Gemma 4. Avoid nested
   thread pools and excessive FFM crossings. Add batched prefill with token-to-expert grouping
   only after the single-token path is correct; the present adapter advertises batch size one.
   Consider new Rust fused operations or bounded expert repacking only where profiles justify
   them. Account for all extra resident memory on the 32 GB laptop. Retain sliding-window,
   attention-sink, rewind, and longer-context reference checks. Measure native runner processes as
   well as server daemons; the present Ollama RSS/CPU fields cover only its daemon PID.
   Allow a longer bounded comparator startup interval: the initial llama.cpp arm exceeded its
   180-second readiness deadline before producing a report.
5. **Qualify tool behavior separately.** Preserve exact argument literals, test dependent
   sequential calls as well as independent parallel calls, and exercise real adapter continuation
   with simulated tool responses. Prototype schema-constrained Flint syntax for malformed or
   truncated calls. Grammar cannot by itself choose the right tool or argument value. Keep the
   existing 14-case result, add explicit multi-turn coverage, and rerun the full suite plus visible
   follow-up grading after changes. Do not silently normalize wrong literal arguments into passes.
6. **Rerun admission and publish only supported claims.** Freeze code, artifact, profile, context,
   and policy before fresh same-host Java/Rust/Ollama comparisons. Require the default smoke,
   at least one-third model contribution, at least 90% correctness among accepted model answers,
   and the unchanged relative performance gates (Ollama: at least 80% decode throughput and
   at most 1.5 times p95 end-to-end latency). Record the actual latency tier and tested context.
   Obtain released-library evidence and matching ModelJars metadata, profiles, hashes, and
   capability manifests before publication. Laptop usability needs its own measurements; a
   passing EPYC result does not establish laptop speed or API-frontier answer quality.

The first implementation slice includes reusable Java scratch, existing Rust kernel dispatch,
grouped expert execution, grounding v24, and a separate visible-answer audit. Its measured quality
and speed remain subject to the following controlled campaign.

## Preregistered improvement campaign

`scripts/flint/improvement-worker.sh` runs the pinned Q4_K_M artifact on a fresh m6a.4xlarge with
the same 2048-token context and 16-processor setting. Rust 1.96.0 builds the repository's own
dependency-free kernels with the lockfile and offline mode. The Java/Rust real-weight probe must
match all 16 pinned continuation IDs and keep the maximum absolute logit difference at or below
0.05 and relative RMS difference at or below 0.002 before quality measurements start. These are
short-context numerical checks, not general accuracy evidence.

The campaign retains zero-warmup, one-iteration Java and Rust correctness reports at 256 output
tokens. It selects the smallest of 256, 512, and 1024 tokens for which all nine general-workload
pilot responses pass `visible-answer-completion-and-bounded-facts-v1`. If none passes, 1024 is
used for diagnosis and `pilotVisibleGatePassed` remains false. The selected budget is frozen
before the held-out workload runs. Neither the publisher template nor sampling is changed.

The Rust and Ollama comparative arms use that same budget, one full-workload warmup, and three
measured iterations. Both also run the independent 16-question `answer-reliability` workload once,
without using its outputs to select the budget. This campaign requires at least 15/16 complete
visible fact/citation passes, both unanswerable cases correctly abstaining, and no detected
polarity contradictions on the Rust held-out arm, in addition to the existing catalog gates.
The audit is a deterministic screen, not a general semantic-entailment judge; retained raw
answers still require inspection. Repeated trials do not increase the distinct-question count.

Tools are assessed separately with 512 tokens for calls and an explicitly recorded 512-token
follow-up budget. A separate 64-token single-case JFR run diagnoses CPU/allocation costs and is
excluded from comparative timings. Ollama's runner is primed before measurement so its PID can
be sampled; the old daemon-only memory figures are not reused. The worker uploads intermediate
evidence, shuts down on exit, and requires the bootstrap's independent four-hour termination bound.
No catalog entry or capability claim follows from a changed scorer or passing unit tests alone.

After collecting a complete campaign, run the additional offline evidence gate:

```bash
python3 scripts/flint/assess-improvement.py RESULTS_DIR NEW_ASSESSMENT.json
python3 -m unittest discover -s scripts/flint -p test_assess_improvement.py
```

The gate binds audits and qualification decisions to their original report hashes, checks the
frozen budget, and enforces all 16 distinct held-out questions, at least 15 passes, both
abstentions, and no detected polarity contradictions. Missing evidence fails closed. A passing
runtime assessment still requires manual raw-answer review and released-library evidence;
it never grants catalog admission or tool capability by itself. It does not modify the worker
or runtime of an already deployed campaign.

### Numerical correction after the first improvement attempt

The first native run stopped at parity: Java matched all 16 reference IDs, while Rust changed
one completion ID and exceeded both logit tolerances. The format ablation reproduced the failure
with Q4_K native execution alone and with grouping disabled. Q8_0-only native execution was
bit-identical to Java. Identical-input projection checks found zero Q8_0 differences; Q4_K
projection differences reached 0.00048828125 and propagated into much larger final logit errors.

Flint now requires sequential Q4_K floating-point block accumulation. The Rust implementation
keeps AVX2 integer products, folds each block's integer lanes, and performs the two floating-point
updates in Java's order. This is an additive native capability (bit 30, format 10); other models
retain their existing reduction mode. Older native libraries decline Flint's Q4_K projections
and use the Java fallback. Provider configuration does not change a shared worker context.
Synthetic tests require bitwise equality across single, grouped, and independent projections;
fresh real-weight parity and the full qualification campaign are still required.

The Spring AI and LangChain4j adapter tests also replay a dependent two-call Flint conversation:
the first weather result supplies the backup city for the second call, and both results reach
the final answer prompt. They assert exact recovered arguments, observation envelopes, ordering,
and retained reasoning. These use scripted model outputs and prove adapter behavior only;
they do not change the real-model tool qualification or its deployed runtime.

### Opt-in tool grammar prototype

`FlintToolCallConstraint.compile(tokenizer, tools)` builds a request-scoped grammar from declared
tool schemas alone. The benchmark exposes it with `--tool-constraints flint-schema` and records
`flint-schema-v1` in generation controls. The default remains unconstrained Flint generation;
the tool-result follow-up remains unconstrained in either mode. The ordered-Q4_K campaign at
revision `3336aed192786be17cca9360fdf65ee5ede731c7` does not contain this later prototype.

The grammar permits reasoning, ordinary answers and refusal before the first call. It ignores
call examples inside reasoning, prevents EOS inside an unfinished call, and permits multiple
calls before EOS. It enforces declared names, required/optional fields, primitive types, nested
objects/arrays, and enum/const values. Strings are raw only at Flint's argument boundary; nested
strings remain JSON-quoted. Objects use declaration order and only declared properties. Literal
`<` in string output is excluded by this prototype; `format` is annotation-only. Unsupported
validation keywords such as numeric bounds, string patterns, references and union types fail
compilation. It is a bounded schema subset, not a full JSON Schema validator.

Fixture tests cover refusal, parallel calls, partial and multi-marker token fragments, wrong
types, missing/duplicate keys, nested JSON, and protocol injection. Every declaration in the
shared tool suite compiles. These checks do not establish real-model accuracy or decoding speed.
The prototype must earn separate remote evidence before becoming a recommended profile, and
cannot fix a semantically wrong value or guarantee completion within the output budget.

Flint-specific diagnostics now report mapped-shape eligibility for injected QKV, dense/shared
gate-up, and routed-expert groups, plus the actual sequential Java attention/prefill paths.
They replace generic placeholder labels that incorrectly advertised batched attention or said
all grouping was unsupported. Eligibility counts are not invocation counters or speed evidence.
The frozen `3336aed1` campaign retains the old generic labels; its reports remain unchanged.

The separate `scripts/flint/tool-grammar-worker.sh` comparison freezes two full 14-case tool arms:
default and `flint-schema-v1`, sequentially on one fresh m6a.4xlarge. Both use greedy generation,
512 tokens for calls, and an unconstrained 512-token visible follow-up. It first repeats reference
parity, preserves incomplete reports, and requires an independent 90-minute termination bound.
Its payload uses the same pinned model, JDK, and Rust versions as the ordered-Q4_K campaign.
Only declared schemas reach the grammar; it never receives expected argument values. The existing
follow-up fixture supplies a scripted prior weather call and result, then grades the generated
visible answer. It is a result-synthesis check, not a live multi-step agent evaluation.
Tool capability remains unqualified until the full suite and visible follow-up actually pass.

Greedy constrained sampling also avoids grammar evaluation for scores that cannot beat the best
allowed token. Exhaustive-mask regressions preserve tie and non-finite-score behavior. A synthetic
150,016-token vocabulary probe selected the same token five times while reducing predicate calls
from 750,080 to 15. This is a sampler overhead diagnostic, not a real-model speed measurement.

### Completed ordered-Q4_K and grammar campaigns

The ordered-Q4_K campaign passed numerical parity (zero Java/Rust logit differences), the
512-token pilot (9/9 complete visible answers), and held-out accuracy (16/16, including both
abstentions). Native median decode was 7.74 tokens/s in the warmed arm. Its Ollama comparison
was incomplete: the worker assumed an `ollama runner` process, but 0.40.9 launched a child
`/usr/local/lib/ollama/llama-server`. Discovery now recognizes both forms and checks the parent
PID. No runtime qualification is inferred from the incomplete comparison.

The separate grammar campaign completed both arms: default and schema-constrained each passed
10/14, with the same omitted actions and wrong literal arguments. Tool capability remains
unqualified. Both disposable instances and their root volumes have been removed. Reports are
retained under `benchmark-results/2026-10-30-flint-ordered` and `2026-10-30-flint-tools`.

JFR recorded 566 Java execution samples inside attention, among 699 samples containing
`FlintForwardPass.executeLayer`. Most other samples were artifact hashing. Java execution
sampling does not attribute native-kernel CPU time and is not a wall-time breakdown. The next
implementation vectorizes value columns with separate multiply/add and distributes independent
heads once the work reaches 1,048,576 elements. Key reductions, sink softmax, chronological
value order, and sliding spans remain unchanged. Per-head session scratch adds
`numHeads * contextLength * 4` bytes (256 KiB at 32 heads and context 2048). Synthetic scalar
comparisons cover short, full, and wrapped histories, grouped heads, vector tails, and the
parallel threshold with bitwise equality. Real-weight parity and fresh same-host comparison
remain mandatory before any speed claim or admission.

The worker also accepts `diagnosticOnly: true` for a bounded, single-case, same-host
512-token native/Ollama speed comparison (one warmup, three measured iterations).
This mode omits admission entirely and is used to reject ineffective optimizations before
spending a full campaign. It cannot qualify a model, replace default correctness, or
replace the held-out workload.

The first attention diagnostic measured 9.37 native versus 14.53 Ollama decode tokens/s
(64.45%, below the 80% target), while native p95 end-to-end was 15.29 s versus 31.67 s.
Both produced correct answers, but native emitted 143 tokens versus Ollama's 454; lower
wall time does not waive the decode floor. This was one repeated question, not qualification.
The next bounded diagnosis uses `diagnosticProfiles: "workers"`: native decode workers
16 (default), 8, and 4, with the unchanged 16-thread Ollama control. CPU, artifact, prompt,
sampling and budgets stay fixed. Smaller worker counts use the existing public native setting;
no reduction order or grading rule changes. A selected profile still needs full fresh evidence.
New hosts provision 500 MiB/s gp3 throughput (within the instance's 625 MB/s EBS baseline)
to reduce artifact I/O outside the warmed measurements; each comparator shares that same volume.

### Worker scheduling follow-up

The worker-count diagnosis keeps the runtime payload at `9c453e9d` and varies only active native
workers (16, 8, 4). A separate scheduling diagnosis compares the public pool setting
`models.native.kernels.threads=8` and the existing experimental polling setting. The latter is
not a supported ModelJars recommendation. `FlintKernelTiming.java` measures elapsed time around
native matrix calls, including FFM, copies, scheduling, reflection and timer overhead; it is
instrumented diagnosis, not qualification evidence.

A full campaign may freeze `candidateNativeThreads` in its manifest after these development
measurements. Library-default correctness stays untuned; a different candidate pool receives
its own real-weight parity check and all budget pilots, candidate and held-out runs. The
comparator retains 16 threads. The selected profile and diagnostics bind the actual pool size.
The formal candidate/comparator verdict now runs before held-out and tool work, so those
separate stages cannot delay discovering a performance failure. All original gates remain.

### Bounded router snapshots

Flint's Java F32 router path originally copied every row from mapped weights for every token.
The candidate now retains immutable router arrays once per loaded graph, shared by sessions,
under a total 128 MiB limit. Non-F32 routers, over-budget matrices, and eligible native routes
retain their existing path. The pinned artifact's 47 routers require 94 MiB. Diagnostics report
`flint-router-cache-bytes`; this is extra resident heap, not a reduction in model weight size.
The exact same `PinnedReduction.dot` scores each row. Fixture tests compare every result bit
at widths 4, 17, 32, 128 and 2048 across changing inputs, and cover budget/type fallbacks.
The synthetic routing comparison is supporting development evidence only; whole-model gains
must be measured remotely before qualification.
